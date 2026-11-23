from __future__ import annotations

import asyncio
import json
import math
import os
import re
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx
from joblib import Parallel, delayed
from loguru import logger as loguru_logger

from mai_gym.agents.errors import AgentAPIError
from mai_gym.client.env_client import APIError, EnvClient, EnvTransportError, PollTimeoutError
from mai_gym.models.config import MobileGymConfig
from mai_gym.models.evaluation import EvaluationScore, task_succeeded
from mai_gym.runner.base import TaskConfig
from mai_gym.runner.rollout import RolloutAgent, ScreenshotTransportError
from mai_gym.server.trajectory import model_slug


class RolloutSetupError(RuntimeError):
    pass


@dataclass
class RolloutRunResult:
    scores: dict[str, float | None] = field(default_factory=dict)
    n_steps: dict[str, int] = field(default_factory=dict)
    elapsed_s: float = 0.0
    log_dir: str = ""
    train_id: str | None = None


_MAX_TASK_ATTEMPTS = 3


def _env_number(name: str, default: float) -> float:
    try:
        value = float(os.environ.get(name, "") or default)
    except ValueError:
        return default
    return value if math.isfinite(value) else default


_STEP_RETRIES = max(0, int(_env_number("MAI_GYM_STEP_RETRIES", 3)))
_LEASE_KEEPALIVE_S = max(0.0, _env_number("MAI_GYM_LEASE_KEEPALIVE_S", 120.0))
_ENV_GONE_404_CODES = frozenset({"session_not_found", "fleet_request_not_found"})


_SERVER_RESOLVED_TASK_ENVS = frozenset(
    {
        "mobileworld",
        "mobileworld_k8s",
        "osworld",
        "osworld_cua",
        "osworld_k8s",
    }
)


def _is_transient(exc: BaseException) -> bool:

    if isinstance(exc, ScreenshotTransportError):
        return True
    if isinstance(exc, AgentAPIError):
        return True
    if isinstance(exc, (PollTimeoutError, EnvTransportError)):
        return True
    if isinstance(exc, APIError):
        if exc.status_code == 404:
            body = exc.body if isinstance(exc.body, dict) else {}
            detail = body.get("detail")
            if isinstance(detail, dict):
                body = detail
            error = body.get("error")
            return isinstance(error, str) and error in _ENV_GONE_404_CODES
        return exc.status_code == 409 or exc.status_code >= 500
    if isinstance(exc, httpx.HTTPStatusError):
        return exc.response.status_code >= 500

    return isinstance(exc, httpx.TransportError)


def _resolve_tasks(
    client: EnvClient,
    task_filter: str | None,
    *,
    env_type: str,
    resolve_metadata: bool = False,
    benchmark: str | None = None,
) -> list[dict]:

    wanted = [tid.strip() for tid in task_filter.split(",") if tid.strip()] if task_filter else []
    if wanted and env_type in _SERVER_RESOLVED_TASK_ENVS and not resolve_metadata:
        loguru_logger.info(
            "Using {} configured task ID(s) for {}; skipping manager /tasks",
            len(wanted),
            env_type,
        )
        return [{"task_id": tid} for tid in wanted]

    resp = client.list_tasks()
    all_tasks = [t.model_dump(exclude_none=True) for t in resp.tasks]
    if resolve_metadata:
        from mai_gym.models.environment import matches_environment

        all_tasks = [
            t
            for t in all_tasks
            if matches_environment(
                env_type,
                t.get("env_type", env_type),
                infra_type=t.get("infra_type"),
                profile=t.get("environment_profile"),
            )
            and t.get("benchmark", benchmark) == benchmark
        ]
    if not task_filter:
        return all_tasks
    by_id = {t["task_id"]: t for t in all_tasks}
    return [by_id.get(tid, {"task_id": tid}) for tid in wanted]


def _resolve_run_tasks(
    client: EnvClient,
    cfg: MobileGymConfig,
    task_filter: str | None,
    env_type: str,
    tasks_dir: str | None,
    domains: set[str] | None,
) -> list[dict[str, Any]]:
    if tasks_dir:
        if task_filter:
            print("Warning: --tasks-dir overrides --tasks; ignoring --tasks filter.")
        tasks = _discover_generated_tasks(tasks_dir, domains=domains)
    else:
        wanted = [tid.strip() for tid in (task_filter or "").split(",") if tid.strip()]
        if wanted and all(tid in cfg.rollout.task_metadata for tid in wanted):
            tasks = [{"task_id": tid} for tid in wanted]
        else:
            tasks = _resolve_tasks(
                client,
                task_filter,
                env_type=env_type,
                resolve_metadata=cfg.rollout.resolve_task_metadata,
                benchmark=cfg.rollout.benchmark.suite,
            )
    for task in tasks:
        task_id = task["task_id"]
        task.update(cfg.rollout.task_metadata.get(task_id, {}))
        if task["task_id"] != task_id:
            raise RolloutSetupError("Task metadata cannot change its task_id")
        limit = task.get("max_steps")
        if limit is not None and (
            not isinstance(limit, int) or isinstance(limit, bool) or limit < 1
        ):
            raise RolloutSetupError(f"Task {task['task_id']!r} has an invalid max_steps")
        if cfg.rollout.require_task_steps and limit is None:
            raise RolloutSetupError(
                f"Task {task['task_id']!r} requires task-specific max_steps metadata"
            )
    if not tasks:
        raise RolloutSetupError(
            "No tasks found. Configure a task list or check the manager /tasks endpoint."
        )
    return tasks


def _discover_generated_tasks(
    tasks_dir: str,
    domains: set[str] | None = None,
) -> list[dict]:

    base = Path(tasks_dir)
    if not base.is_dir():
        raise RolloutSetupError(f"--tasks-dir not a directory: {tasks_dir}")

    out: list[dict] = []
    for seed_path in sorted(base.glob("*/*.json")):
        domain = seed_path.parent.name
        if domains is not None and domain not in domains:
            continue
        try:
            data = json.loads(seed_path.read_text())
        except Exception as exc:
            loguru_logger.warning(
                "Skipping {}: failed to parse JSON: {}",
                seed_path,
                exc,
            )
            continue
        seed_uuid = data.get("seed_task_id") or seed_path.stem
        seed_domain = data.get("seed_domain") or domain
        seed_task_id = f"{seed_domain}/{seed_uuid}"
        gens = data.get("generated_tasks") or []
        if not isinstance(gens, list):
            loguru_logger.warning(
                "Skipping {}: generated_tasks is not a list",
                seed_path,
            )
            continue
        for gen in gens:
            gen_id = gen.get("id")
            instruction = gen.get("instruction")
            if gen_id is None or not instruction:
                continue
            out.append(
                {
                    "task_id": f"{seed_domain}/{seed_uuid}/gen-{gen_id}",
                    "seed_task_id": seed_task_id,
                    "instruction": instruction,
                    **({"max_steps": gen["max_steps"]} if "max_steps" in gen else {}),
                }
            )
    return out


_SCORE_RE = re.compile(r"^score:\s*([\d.+\-eE]+)\s*$")
_BARE_SCORE_RE = re.compile(r"^[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?$")


def _parse_result_score(path: str) -> float | None:
    try:
        with open(path, encoding="utf-8", errors="replace") as result:
            lines = result.readlines()
            success = next(
                (
                    line.strip().split(":", 1)[1].strip() == "true"
                    for line in lines
                    if line.strip() in {"success: true", "success: false"}
                ),
                None,
            )
            for line in lines:
                stripped = line.strip()
                match = _SCORE_RE.match(stripped)
                if match or _BARE_SCORE_RE.match(stripped):
                    score = float(match.group(1) if match else stripped)
                    return EvaluationScore(score, success) if math.isfinite(score) else None
    except (OSError, ValueError):
        pass
    return None


def _write_local_result_marker(task_log_dir: str, score: float) -> None:
    result_path = os.path.join(task_log_dir, "result.txt")
    tmp_path = f"{result_path}.tmp.{uuid.uuid4().hex}"
    try:
        os.makedirs(task_log_dir, exist_ok=True)
        with open(tmp_path, "w", encoding="utf-8") as result:
            result.write(f"score: {score}\nreason: rollout-side resume marker\n")
            success = getattr(score, "success", None)
            if success is not None:
                result.write(f"success: {str(success).lower()}\n")
        os.replace(tmp_path, result_path)
    except OSError as exc:
        loguru_logger.warning("Cannot save resume marker {}: {}", result_path, exc)
    finally:
        try:
            os.unlink(tmp_path)
        except FileNotFoundError:
            pass
        except OSError as exc:
            loguru_logger.warning("Cannot remove temporary resume marker {}: {}", tmp_path, exc)


def _resolve_generation_cap(runtime_conf: dict) -> int | None:
    for key in ("max_new_tokens", "max_response_len", "max_tokens"):
        value = runtime_conf.get(key)
        if value is not None:
            return int(value)
    return None


def _sampling_params(runtime_conf: dict) -> dict[str, object]:
    params: dict[str, object] = {
        "temperature": runtime_conf.get("temperature", 0.0),
        "top_p": runtime_conf.get("top_p", 1.0),
    }
    cap = _resolve_generation_cap(runtime_conf)
    if cap is not None:
        params["max_new_tokens"] = cap
    for key in (
        "top_k",
        "repetition_penalty",
        "presence_penalty",
        "frequency_penalty",
        "stop",
        "no_stop_trim",
    ):
        if key in runtime_conf:
            params[key] = runtime_conf[key]
    return params


def _ensure_capacity_with_wait(
    client: EnvClient, num_envs: int, env_type: str, *, job_id: str, mode: str
) -> int:
    ready = client.ensure_capacity(num_envs, env_type, job_id=job_id, mode=mode)
    wait_s = max(0.0, _env_number("MAI_GYM_POOL_WAIT_TIMEOUT_S", 300.0))
    poll_s = max(0.01, _env_number("MAI_GYM_POOL_WAIT_POLL_S", 10.0))
    deadline = time.monotonic() + wait_s
    while ready <= 0:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            break
        time.sleep(min(poll_s, remaining))
        try:
            ready = client.ensure_capacity(num_envs, env_type, job_id=job_id, mode=mode)
        except Exception as exc:
            if not _is_transient(exc):
                raise
            loguru_logger.warning("Pool readiness retry failed: {}", exc)
    return ready


def _scan_finished_tasks(log_dir: str, task_ids: list[str]) -> dict[str, float]:

    finished: dict[str, float] = {}
    for tid in task_ids:
        task_dir = os.path.join(log_dir, tid)

        candidates = [os.path.join(task_dir, "result.txt")]

        if os.path.isdir(task_dir):
            subs = sorted(
                (
                    e
                    for e in os.listdir(task_dir)
                    if os.path.isfile(os.path.join(task_dir, e, "result.txt"))
                ),
            )
            if subs:
                candidates.insert(0, os.path.join(task_dir, subs[-1], "result.txt"))

        def result_mtime(path: str) -> int:
            try:
                return os.stat(path).st_mtime_ns
            except OSError:
                return -1

        candidates.sort(key=result_mtime, reverse=True)
        for rt in candidates:
            score = _parse_result_score(rt)
            if score is not None:
                finished[tid] = score
                break
    return finished


def _run_task(
    task_data: dict,
    *,
    idx: int,
    train_id: str | None,
    cfg: MobileGymConfig,
    server: str,
    env_type: str,
    log_dir: str,
    attempt_rollout: int = 0,
    fleet_request_id: str | None = None,
) -> tuple[str, int, float | None]:

    task_id = task_data["task_id"]
    thread_client = EnvClient(
        host=server,
        screenshot_transport=cfg.rollout.screenshot_transport,
        fleet_request_id=fleet_request_id,
        env_type=env_type,
    )
    try:
        task_log_dir = os.path.join(log_dir, task_id)
        os.makedirs(task_log_dir, exist_ok=True)

        thread_id = threading.current_thread().ident
        debug_log_file = os.path.join(task_log_dir, f"thread_{thread_id}.log")

        def _thread_filter(record: dict) -> bool:
            return record["extra"].get("thread_id") == thread_id

        handler_id = loguru_logger.add(
            debug_log_file,
            format=("{time:YYYY-MM-DD HH:mm:ss} | {level} | {name}:{function}:{line} | {message}"),
            level="DEBUG",
            enqueue=True,
            filter=_thread_filter,
        )

        try:
            with loguru_logger.contextualize(thread_id=thread_id):
                loguru_logger.info("Starting task '{}'", task_id)
                try:
                    from mai_gym.agents import create_policy

                    assert cfg.agent is not None
                    agent_config = cfg.agent.model_copy(
                        update={
                            "max_steps": task_data.get("max_steps") or cfg.agent.max_steps,
                        }
                    )
                    policy = create_policy(agent_config, env_type=env_type)
                    agent = RolloutAgent(
                        env_client=thread_client,
                        policy=policy,
                        env_type=env_type,
                        oss_creds=cfg.manager.oss,
                        model_name=cfg.agent.model_name,
                        step_retries=_STEP_RETRIES,
                        lease_keepalive_s=_LEASE_KEEPALIVE_S,
                        preserve_action_sequences=cfg.rollout.require_score,
                    )
                    task = TaskConfig(
                        category=env_type,
                        task_id=task_id,
                        goal=task_data.get("instruction"),
                        max_steps=agent_config.max_steps,
                        seed_task_id=task_data.get("seed_task_id"),
                        train_id=train_id,
                        attempt_id=f"eval_r{attempt_rollout:05d}_s{idx:03d}",
                        evaluation_mode=cfg.rollout.evaluation_mode,
                        is_eval=cfg.rollout.is_eval,
                        require_score=cfg.rollout.require_score,
                    )

                    async def _run_and_close():

                        try:
                            last_exc: BaseException | None = None
                            for attempt in range(_MAX_TASK_ATTEMPTS):
                                try:
                                    async with thread_client.asession(
                                        release_on=(ScreenshotTransportError,),
                                    ) as sid:
                                        task.attempt_id = (
                                            f"eval_r{attempt_rollout:05d}_s{idx:03d}_a{attempt:02d}"
                                        )
                                        loguru_logger.info(
                                            "Task '{}' leased session {} (attempt {}/{})",
                                            task_id,
                                            sid,
                                            attempt + 1,
                                            _MAX_TASK_ATTEMPTS,
                                        )
                                        return await agent.run(sid, task, tear_down=False)
                                except BaseException as exc:
                                    if not _is_transient(exc):
                                        raise
                                    last_exc = exc
                                    loguru_logger.warning(
                                        "Task '{}' transient env error (attempt {}/{}): {}",
                                        task_id,
                                        attempt + 1,
                                        _MAX_TASK_ATTEMPTS,
                                        exc,
                                    )
                                    await asyncio.sleep(min(2**attempt, 8))
                            raise last_exc
                        finally:
                            try:
                                cm = getattr(policy, "chat_model", None)
                                aclose = getattr(cm, "aclose", None)
                                if aclose is not None:
                                    await aclose()
                            finally:
                                from mai_gym.io.oss_io import close_async_fs

                                await close_async_fs()

                    traj = asyncio.run(_run_and_close())
                    loguru_logger.info(
                        "Task '{}' done: steps={}, score={:.3f}",
                        task_id,
                        len(traj.steps),
                        traj.reward,
                    )
                    print(f"  {task_id}: steps={len(traj.steps)}, score={traj.reward:.3f}")
                    score = EvaluationScore(traj.reward, getattr(traj, "success", None))
                    _write_local_result_marker(task_log_dir, score)
                    return (task_id, len(traj.steps), score)
                except Exception as exc:
                    loguru_logger.exception("Task '{}' failed: {}", task_id, exc)
                    if _is_transient(exc):
                        print(
                            f"  {task_id}: ENV-FAILED after {_MAX_TASK_ATTEMPTS} "
                            f"attempts ({type(exc).__name__})"
                        )
                        return (task_id, 0, None)
                    print(f"  {task_id}: ERROR ({type(exc).__name__}: {exc})")
                    if cfg.rollout.require_score:
                        return (task_id, 0, None)
                    _write_local_result_marker(task_log_dir, 0.0)
                    return (task_id, 0, 0.0)
        finally:
            loguru_logger.remove(handler_id)
    finally:
        thread_client.close()


def run_batch_rollout(
    cfg: MobileGymConfig,
    *,
    server: str,
    num_envs: int,
    env_type: str,
    task_filter: str | None,
    no_resume: bool,
    tasks_dir: str | None = None,
    domains: set[str] | None = None,
    pass_index: int = 0,
    pass_total: int = 1,
) -> RolloutRunResult:

    if cfg.agent is None:
        raise RolloutSetupError("Config must have an 'agent' section for rollout.")

    t0 = time.monotonic()

    client = EnvClient(host=server)
    run_train_id = cfg.rollout.train_id or f"eval_{model_slug(cfg.agent.model_name)}"

    ready = _ensure_capacity_with_wait(
        client,
        num_envs,
        env_type,
        job_id=run_train_id,
        mode=cfg.rollout.fleet_request_mode,
    )
    print(f"Pool: {ready} READY session(s)")
    if ready <= 0:
        client.close()
        raise RolloutSetupError(
            "No sessions available — server pool empty and provisioning yielded none."
        )
    concurrency = min(num_envs, ready)

    try:
        task_list = _resolve_run_tasks(client, cfg, task_filter, env_type, tasks_dir, domains)
    except Exception:
        client.close()
        raise

    base_log_dir = cfg.manager.traj_log_dir or "./traj_logs"

    log_dir = f"{base_log_dir.rstrip('/')}/{model_slug(cfg.agent.model_name)}"

    if pass_total > 1:
        log_dir = f"{log_dir}/pass{pass_index + 1}"
    print(f"Trajectory log dir: {log_dir}")

    from mai_gym.models.run_config import build_cli_run_config

    _sampling = _sampling_params(cfg.agent.runtime_conf or {})
    client.register_run_config(
        run_train_id,
        build_cli_run_config(
            cfg,
            train_id=run_train_id,
            env_type=env_type,
            num_envs=num_envs,
            sampling_params=_sampling,
            tasks=[t["task_id"] for t in task_list],
        ),
        model_name=cfg.agent.model_name,
    )

    finished_scores: dict[str, float] = {}
    if not no_resume:
        finished_scores = _scan_finished_tasks(
            log_dir,
            [t["task_id"] for t in task_list],
        )
        if finished_scores:
            print(
                f"Resume: skipping {len(finished_scores)} already-finished task(s) "
                f"(use --no-resume to force re-run)"
            )
            task_list = [t for t in task_list if t["task_id"] not in finished_scores]

    if not task_list:
        print("All requested tasks already finished — nothing to do.")
        scored = list(finished_scores.values())
        if scored:
            print(
                f"Resumed total: {len(scored)} task(s), mean score={sum(scored) / len(scored):.3f}"
            )
        client.close()
        return RolloutRunResult(
            scores=dict(finished_scores),
            n_steps={},
            elapsed_s=time.monotonic() - t0,
            log_dir=log_dir,
            train_id=run_train_id,
        )

    task_ids_preview = [t["task_id"] for t in task_list]
    if len(task_ids_preview) > 20:
        preview = task_ids_preview[:20] + [f"... (+{len(task_ids_preview) - 20} more)"]
    else:
        preview = task_ids_preview
    print(f"Running {len(task_list)} task(s): {preview}")
    print(f"Run id (dashboard group): {run_train_id}")

    try:
        results = Parallel(n_jobs=concurrency, backend="threading")(
            delayed(_run_task)(
                td,
                idx=i,
                train_id=run_train_id,
                cfg=cfg,
                server=server,
                env_type=env_type,
                log_dir=log_dir,
                attempt_rollout=pass_index,
                fleet_request_id=client.fleet_request_id,
            )
            for i, td in enumerate(task_list)
        )
    finally:
        client.close()

    env_failed = sum(1 for r in results if r is not None and r[2] is None)
    new_scores = [r[2] for r in results if r is not None and r[2] is not None]
    all_scores = new_scores + list(finished_scores.values())
    print(f"\nCollected {len(results)} new trajectory(ies)")
    if finished_scores:
        print(f"Resumed scores: {len(finished_scores)}")
    if env_failed:
        print(f"Env-failed (excluded from SR): {env_failed}")
    if all_scores:
        success = sum(1 for s in all_scores if task_succeeded(s))
        print(
            f"Aggregate: {len(all_scores)} scored task(s), "
            f"mean={sum(all_scores) / len(all_scores):.3f}, "
            f"success_rate={success}/{len(all_scores)}={success / len(all_scores):.1%}"
        )
    print(f"Trajectories saved to: {log_dir}")

    scores: dict[str, float | None] = dict(finished_scores)
    n_steps: dict[str, int] = {}
    for r in results:
        if r is None:
            continue
        task_id, steps, score = r
        scores[task_id] = score
        n_steps[task_id] = steps
    return RolloutRunResult(
        scores=scores,
        n_steps=n_steps,
        elapsed_s=time.monotonic() - t0,
        log_dir=log_dir,
        train_id=run_train_id,
    )
