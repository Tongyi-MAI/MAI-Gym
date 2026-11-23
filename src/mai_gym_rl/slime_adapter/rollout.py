from __future__ import annotations

import asyncio
import atexit
import contextlib
import logging
import queue
import re
import threading
import time
from argparse import Namespace
from pathlib import Path
from typing import Any

import httpx
from slime.rollout.base_types import RolloutFnEvalOutput, RolloutFnTrainOutput
from slime.rollout.sglang_rollout import GenerateState
from slime.utils.types import Sample

from mai_gym.agents.implementations.general_e2e import (
    SYSTEM_PROMPT as QWEN35_SYSTEM_PROMPT,
)
from mai_gym.agents.implementations.mai_gym_mobile_use import (
    SYSTEM_PROMPT as MAI_UI_SYSTEM_PROMPT,
)
from mai_gym.agents.implementations.mai_gym_mobile_use import (
    MAIGymMobileUsePolicy,
)
from mai_gym.agents.implementations.qwen35_policy import Qwen35SLIMEPolicy
from mai_gym.client.env_client import APIError, EnvClient
from mai_gym.io import oss_io
from mai_gym.runner.base import TaskConfig, Trajectory
from mai_gym.runner.rollout import RolloutAgent, ScreenshotTransportError

from .helpers.async_worker import _get_async_worker
from .helpers.rollout_metrics import (
    _build_train_output,
    _compute_rollout_metrics,
    _percentile,
    _sample_metadata,
)

__all__ = [
    "generate_rollout",
    "generate_rollout_async",
    "_run_single_task",
    "_compute_rollout_metrics",
    "_percentile",
]
from .config import SlimeAdapterConfig
from .eval_tasks import load_eval_tasks
from .helpers.chat_model import CustomChatModel
from .step_builder import build_step_samples

logger = logging.getLogger(__name__)


_config: SlimeAdapterConfig | None = None


_env_clients: dict[str, EnvClient] = {}
_provisioned: set[str] = set()
_client_lock = threading.Lock()


_registered_sets: set[str] = set()
_register_lock = threading.Lock()


_run_config_sent: set[str] = set()


def _close_env_clients() -> None:

    for client in list(_env_clients.values()):
        with contextlib.suppress(Exception):
            client.close()


atexit.register(_close_env_clients)


def _derive_set_id(zip_path: str) -> str:

    sid = re.sub(r"[^A-Za-z0-9_]", "_", Path(zip_path).stem)
    if not sid or not re.match(r"[A-Za-z_]", sid[0]):
        sid = "_" + sid
    return sid


def _ensure_task_sets_registered(
    samples: list[Sample],
    env_client: EnvClient,
) -> None:

    from mai_gym.client.sdk_client import SDKManagerClient

    if isinstance(env_client, SDKManagerClient):
        return
    wanted: dict[str, str] = {}
    for s in samples:
        meta = _sample_metadata(s)
        zip_path = str(meta.get("zip_path") or "").strip()
        if not zip_path:
            continue
        set_id = str(meta.get("task_set") or "").strip() or _derive_set_id(zip_path)
        wanted.setdefault(set_id, zip_path)

    for set_id, zip_path in wanted.items():
        with _register_lock:
            if set_id in _registered_sets:
                continue
            try:
                result = env_client.register_task_set(set_id, zip_path)
                logger.info(
                    "Registered task set %s from %s: registered=%s skipped=%s "
                    "failed=%s backends=%s",
                    set_id,
                    zip_path,
                    result.get("num_registered"),
                    result.get("num_skipped"),
                    result.get("num_failed"),
                    result.get("num_backends"),
                )
                if result.get("num_failed", 0) == 0:
                    _registered_sets.add(set_id)
                else:
                    logger.warning(
                        "Task set %s: %s backend(s) failed registration; will retry on next batch",
                        set_id,
                        result.get("num_failed"),
                    )
            except Exception:
                logger.exception(
                    "Failed to register task set %s from %s; tasks from this "
                    "set may fail task_init",
                    set_id,
                    zip_path,
                )


def _get_config() -> SlimeAdapterConfig:
    global _config
    if _config is None:
        _config = SlimeAdapterConfig.load()
    return _config


def _pool_spec(config: SlimeAdapterConfig, env_type: str) -> tuple[str, int]:

    pool = (config.env_pools or {}).get(env_type)
    if pool:
        url = pool.get("manager_url") or pool.get("controller_url") or config.manager_url
        n = int(pool.get("num_sessions") or config.num_sessions)
    else:
        url = config.manager_url
        n = config.num_sessions
    return url, n


def _ensure_client(config: SlimeAdapterConfig, env_type: str) -> EnvClient:

    url, n = _pool_spec(config, env_type)
    total = max(1, n + config.num_backup_sessions)
    with _client_lock:
        client = _env_clients.get(env_type)
        if client is None:
            client_type = EnvClient
            if config.rollout_agent == "sdk":
                from mai_gym.client.sdk_client import SDKManagerClient

                config.validate_sdk()
                client_type = SDKManagerClient
            client = client_type(
                host=url,
                max_async_connections=2 * total,
                screenshot_transport=config.screenshot_transport,
            )
            _env_clients[env_type] = client
        if env_type not in _provisioned and config.rollout_agent != "sdk":
            client.ensure_capacity(
                total,
                env_type=env_type,
                job_id=config.train_id,
                mode=config.fleet_request_mode,
            )
            _provisioned.add(env_type)
        return client


def _get_env_client(config: SlimeAdapterConfig) -> EnvClient:

    if config.env_pools:
        for et in config.env_pools:
            _ensure_client(config, et)
        primary = (
            config.env_type if config.env_type in config.env_pools else next(iter(config.env_pools))
        )
        return _env_clients[primary]
    return _ensure_client(config, config.env_type)


def _get_env_client_for(config: SlimeAdapterConfig, env_type: str | None) -> EnvClient:

    if not config.env_pools:
        return _ensure_client(config, config.env_type)
    client = _env_clients.get(env_type or "")
    if client is None:
        raise ValueError(
            f"task env_type={env_type!r} has no configured pool; add it to "
            f"env_pools (configured: {sorted(config.env_pools)})"
        )
    return client


def _extract_task(sample: Sample, default_env_type: str | None = None) -> TaskConfig:

    meta = _sample_metadata(sample)
    return TaskConfig(
        category=meta.get("category", "default"),
        task_id=meta.get("task_id"),
        goal=meta.get("goal"),
        target_apps=meta.get("target_apps"),
        max_steps=meta.get("max_steps", 15),
        env_type=meta.get("env_type") or default_env_type,
        use_judge=bool(meta.get("use_judge", False)),
    )


def _build_original_prompt(sample: Sample) -> list[dict[str, Any]]:

    if isinstance(sample.prompt, list):
        return list(sample.prompt)
    return [{"role": "user", "content": sample.prompt}]


async def _run_single_task(
    sample: Sample,
    config: SlimeAdapterConfig,
    env_client: EnvClient,
    args: Namespace,
    *,
    concurrency_sem: asyncio.Semaphore,
    evaluation: bool = False,
    attempt_id: str | None = None,
    rollout_id: int | None = None,
) -> list[Sample]:

    task = _extract_task(sample, default_env_type=config.env_type)
    task.max_steps = config.max_turns_for_env(task.env_type)
    task.attempt_id = attempt_id

    task.train_id = config.train_id or None

    task.rollout_id = rollout_id
    task.is_eval = evaluation

    trajectory: Trajectory | None = None
    state: GenerateState | None = None
    step_records: list[Any] = []

    failure_log = {
        "train_id": task.train_id,
        "task_id": task.task_id,
        "attempt_id": task.attempt_id,
        "rollout_id": task.rollout_id,
        "evaluation": evaluation,
        "model_name": None,
        "env_type": task.env_type,
    }

    try:
        async with (
            concurrency_sem,
            env_client.asession(
                target_apps=task.target_apps,
                failure_log=failure_log,
            ) as session_id,
        ):
            state = GenerateState(args)
            url = f"http://{args.sglang_router_ip}:{args.sglang_router_port}/generate"
            model_type = "llm" if args.multimodal_keys is None else "vlm"

            chat_model = CustomChatModel(
                url=url,
                tokenizer=state.tokenizer,
                processor=state.processor,
                model_type=model_type,
                tool_parser_name=(
                    "general_e2e" if config.policy_variant == "qwen35" else config.tool_parser
                ),
                routing_key=session_id,
                max_context_tokens=getattr(args, "rollout_max_context_len", None),
            )

            meta = _sample_metadata(sample)
            system_prompt = meta.get("system_prompt")
            if system_prompt is None:
                system_prompt = (
                    QWEN35_SYSTEM_PROMPT
                    if config.policy_variant == "qwen35"
                    else MAI_UI_SYSTEM_PROMPT
                )

            sampling_params = {
                "max_new_tokens": args.rollout_max_response_len,
                "temperature": args.rollout_temperature,
                "top_p": getattr(args, "rollout_top_p", 1.0),
            }

            messages_log_dir: Any = None
            if config.messages_log_root:
                short_sid = session_id.split("-")[0]
                leaf = f"{short_sid}_{task.task_id or 'notask'}"
                if oss_io.is_oss_path(config.messages_log_root):
                    messages_log_dir = oss_io.join(config.messages_log_root, leaf)
                else:
                    messages_log_dir = Path(config.messages_log_root) / leaf

            def _remote_log(step_idx: int, payload: dict[str, Any]) -> None:
                env_client.log_messages(session_id, step_idx, payload)

            policy_cls = (
                Qwen35SLIMEPolicy if config.policy_variant == "qwen35" else MAIGymMobileUsePolicy
            )
            policy = policy_cls(
                chat_model=chat_model,
                sampling_params=sampling_params,
                original_image_width=config.original_image_width,
                original_image_height=config.original_image_height,
                resize_width=config.resize_width,
                resize_height=config.resize_height,
                resize_longest_edge=config.resize_longest_edge,
                max_pixels=config.max_pixels,
                min_pixels=config.min_pixels,
                image_factor=config.image_factor,
                system_prompt=system_prompt,
                image_retention=config.image_retention,
                messages_log_dir=messages_log_dir,
                remote_log_callback=_remote_log,
                oss_creds=config.oss,
            )

            try:
                agent = RolloutAgent(
                    env_client,
                    policy,
                    env_type=config.env_type,
                    oss_creds=config.oss,
                )
                trajectory = await agent.run(
                    session_id,
                    task,
                    tear_down=False,
                )
            except (
                APIError,
                ScreenshotTransportError,
                httpx.RemoteProtocolError,
                httpx.ReadError,
                httpx.WriteError,
                httpx.ConnectError,
            ) as exc:
                logger.warning(
                    "Step API/network error for task %s on %s: %s",
                    task.task_id,
                    session_id,
                    exc,
                )
            step_records = policy.step_records
    except Exception:
        logger.exception("Agent loop failed for task %s", task.task_id)

    if trajectory is None:
        sample.status = Sample.Status.FAILED
        sample.reward = 0.0
        sample.remove_sample = True
        return [sample]

    if evaluation:
        sample.reward = trajectory.reward
        sample.status = Sample.Status.COMPLETED
        sample.response = str(trajectory.steps)
        sample.response_length = 0
        return [sample]

    assert state is not None
    original_prompt = _build_original_prompt(sample)

    step_samples = await asyncio.to_thread(
        build_step_samples,
        trajectory=trajectory,
        step_records=step_records,
        original_prompt=original_prompt,
        config=config,
        tokenizer=state.tokenizer,
        processor=state.processor,
        system_prompt=system_prompt,
        instruction=getattr(policy, "_instruction", ""),
    )

    if not step_samples:
        sample.status = Sample.Status.FAILED
        sample.reward = 0.0
        sample.remove_sample = True
        return [sample]

    return step_samples


def generate_rollout(
    args: Namespace,
    rollout_id: int,
    data_source: Any,
    evaluation: bool = False,
) -> RolloutFnTrainOutput | RolloutFnEvalOutput:

    config = _get_config()
    env_client = _get_env_client(config)

    if evaluation:
        batch_size = getattr(args, "rl_prompts_per_eval", 32)

        _eval_tasks = load_eval_tasks(args)
        if _eval_tasks:
            import copy as _copy

            n_per = max(1, getattr(args, "n_samples_per_eval_prompt", 1))
            task_samples = [
                _copy.deepcopy(s) for s in _eval_tasks[:batch_size] for _ in range(n_per)
            ]
        else:
            task_groups = data_source.get_samples(batch_size)
            task_samples = [s for group in task_groups for s in group]
    else:
        batch_size = getattr(args, "rollout_batch_size", 32)
        task_groups = data_source.get_samples(batch_size)
        task_samples = [s for group in task_groups for s in group]

    _ensure_task_sets_registered(task_samples, env_client)

    if config.train_id and config.train_id not in _run_config_sent:
        from mai_gym.models.run_config import build_slime_run_config

        _sampling = {
            "max_new_tokens": args.rollout_max_response_len,
            "temperature": args.rollout_temperature,
            "top_p": getattr(args, "rollout_top_p", 1.0),
        }
        env_client.register_run_config(
            config.train_id,
            build_slime_run_config(
                train_id=config.train_id,
                model_name=getattr(args, "hf_checkpoint", None)
                or getattr(args, "model_name", None),
                agent_variant=config.agent_variant or config.policy_variant,
                sampling_params=_sampling,
                num_sessions=config.num_sessions,
                env_type=config.env_type,
                evaluation=evaluation,
            ),
            model_name=None,
        )
        _run_config_sent.add(config.train_id)

    attempt_prefix = f"eval_r{rollout_id:05d}" if evaluation else f"r{rollout_id:05d}"

    async def _run_all() -> list[list[Sample]]:

        sem = asyncio.Semaphore(max(1, config.num_sessions))
        tasks = [
            _run_single_task(
                s,
                config,
                env_client,
                args,
                concurrency_sem=sem,
                evaluation=evaluation,
                attempt_id=f"{attempt_prefix}_s{i:03d}",
                rollout_id=rollout_id,
            )
            for i, s in enumerate(task_samples)
        ]
        return await asyncio.gather(*tasks, return_exceptions=False)

    loop = asyncio.new_event_loop()
    try:
        all_step_samples = loop.run_until_complete(_run_all())
    finally:
        with contextlib.suppress(Exception):
            loop.run_until_complete(env_client.aclose())
        with contextlib.suppress(Exception):
            loop.run_until_complete(oss_io.close_async_fs())
        loop.close()

    if evaluation:
        rewards = []
        for step_list in all_step_samples:
            for s in step_list:
                rewards.append(s.reward if s.reward is not None else 0.0)
        eval_metrics = _compute_rollout_metrics(
            all_step_samples,
            prefix="eval/gui_agent",
        )
        logger.info(
            "Eval %d: %d trajectories, pass=%d/%d, mean reward %.4f",
            rollout_id,
            int(eval_metrics.get("eval/gui_agent/num_trajectories", 0)),
            int(eval_metrics.get("eval/gui_agent/num_successes", 0)),
            int(eval_metrics.get("eval/gui_agent/num_trajectories", 0)),
            eval_metrics.get("eval/gui_agent/mean_trajectory_reward", 0.0),
        )
        return RolloutFnEvalOutput(
            data={
                "gui_agent": {
                    "rewards": rewards,
                    "truncated": [False] * len(rewards),
                    "samples": [s for sl in all_step_samples for s in sl],
                    "custom_metrics": eval_metrics,
                },
            },
        )

    return _build_train_output(
        all_step_samples,
        len(task_samples),
        rollout_id,
    )


def generate_rollout_async(
    args: Namespace,
    rollout_id: int,
    data_source: Any,
    evaluation: bool = False,
) -> RolloutFnTrainOutput | RolloutFnEvalOutput:

    if evaluation:
        return generate_rollout(args, rollout_id, data_source, evaluation=True)

    config = _get_config()
    env_client = _get_env_client(config)
    worker = _get_async_worker(args, data_source, config, env_client)

    worker.current_rollout_id = rollout_id

    target_groups = max(1, getattr(args, "rollout_batch_size", 32))
    collected: list[tuple[int, list[list[Sample]]]] = []
    start = time.time()
    last_progress = start

    while len(collected) < target_groups:
        try:
            item = worker.output_queue.get(timeout=30)
        except queue.Empty:
            elapsed = time.time() - last_progress
            logger.warning(
                "Rollout %d (async): no group emitted in %.0fs; collected=%d/%d, queue=%d",
                rollout_id,
                elapsed,
                len(collected),
                target_groups,
                worker.output_queue.qsize(),
            )
            continue
        last_progress = time.time()
        collected.append(item)
        logger.info(
            "Rollout %d (async): group %d/%d (worker_gid=%d) collected in %.1fs cumulative",
            rollout_id,
            len(collected),
            target_groups,
            item[0],
            time.time() - start,
        )

    collected.sort(key=lambda x: x[0])

    flat_step_samples: list[list[Sample]] = []
    task_count = 0
    for _gid, group_result in collected:
        for step_list in group_result:
            task_count += 1
            flat_step_samples.append(step_list)

    return _build_train_output(
        flat_step_samples,
        task_count,
        rollout_id,
        label="async",
        elapsed=time.time() - start,
    )
