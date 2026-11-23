from __future__ import annotations

import asyncio
import atexit
import contextlib
import logging
import os
import queue
import threading
import time
from argparse import Namespace
from collections.abc import Callable, Iterable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import httpx
import pydantic
from slime.rollout.base_types import RolloutFnEvalOutput, RolloutFnTrainOutput
from slime.rollout.sglang_rollout import GenerateState
from slime.utils.types import Sample

from mai_gym.agents import _ctor_accepts
from mai_gym.agents.variants import (
    UNIFIED_AGENT_NAME,
    AgentVariant,
    get_agent,
    resolve_variant_for_env_type,
)
from mai_gym.agents.window_utils import aggregate_cache_metrics
from mai_gym.client.env_client import APIError, EnvClient, PollTimeoutError
from mai_gym.io import oss_io
from mai_gym.runner.base import Trajectory
from mai_gym.runner.rollout import RolloutAgent, ScreenshotTransportError

from .config import SlimeAdapterConfig
from .distributed_rollout import WorkerOptions, make_executor
from .eval_tasks import load_eval_tasks
from .helpers.chat_model import CustomChatModel
from .helpers.ppo_select import is_task_trainable_ppo, select_ppo_trajectories
from .helpers.rollout_metrics import _build_train_output, _compute_rollout_metrics, _sample_metadata
from .helpers.timing import TimedEnvClient, TrajectoryTiming
from .helpers.trained_manifest import build_trained_manifest
from .helpers.traj_saver import save_trajectory
from .rollout import (
    _build_original_prompt,
    _ensure_task_sets_registered,
    _extract_task,
    _get_config,
    _get_env_client,
    _get_env_client_for,
)
from .step_builder import (
    build_step_samples,
    build_window_samples,
    materialize_multimodal_inputs,
)

logger = logging.getLogger(__name__)


def _log_proc_mem(tag: str, **extras: Any) -> None:

    try:
        import psutil

        mi = psutil.Process().memory_info()
        rss_gb = mi.rss / 1e9
        vms_gb = mi.vms / 1e9
        shared_gb = getattr(mi, "shared", 0) / 1e9
    except Exception:
        rss_gb = vms_gb = shared_gb = -1.0

    rss_anon_gb = rss_file_gb = rss_shmem_gb = -1.0
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("RssAnon:"):
                    rss_anon_gb = int(line.split()[1]) / 1e6
                elif line.startswith("RssFile:"):
                    rss_file_gb = int(line.split()[1]) / 1e6
                elif line.startswith("RssShmem:"):
                    rss_shmem_gb = int(line.split()[1]) / 1e6
    except Exception:
        pass

    gc_objs = -1
    if os.environ.get("MAI_GYM_LOG_GC_OBJECTS") == "1":
        try:
            import gc

            gc_objs = len(gc.get_objects())
        except Exception:
            pass

    extra_str = " ".join(f"{k}={v}" for k, v in extras.items())
    logger.info(
        "[mem] %s pid=%d rss=%.2fGB vms=%.2fGB shared=%.2fGB "
        "anon=%.2fGB file=%.2fGB shmem=%.2fGB gc_objs=%d %s",
        tag,
        os.getpid(),
        rss_gb,
        vms_gb,
        shared_gb,
        rss_anon_gb,
        rss_file_gb,
        rss_shmem_gb,
        gc_objs,
        extra_str,
    )


_resolved_variant: AgentVariant | None = None


def _resolve_agent_variant(config: SlimeAdapterConfig, env_type: str | None = None) -> AgentVariant:

    if config.agent_variant == UNIFIED_AGENT_NAME:
        if not env_type:
            raise ValueError(
                "agent_variant 'unified' needs a per-task env_type to resolve "
                "the concrete variant; stamp env_type into the dataset metadata "
                "(or set a single-pool env_type)."
            )
        return resolve_variant_for_env_type(env_type, overrides=config.unified_variant_map)

    global _resolved_variant
    if _resolved_variant is not None:
        return _resolved_variant

    if config.agent_variant:
        variant = get_agent(config.agent_variant)
        logger.info(
            "[AgentVariant] using registry variant: %s (%s)",
            variant.name,
            variant.description,
        )
    else:
        name = "general_e2e" if config.policy_variant == "qwen35" else "mai_gym_mobile_use"
        variant = get_agent(name)
        logger.info(
            "[AgentVariant] legacy fallback: policy_variant=%s → %s",
            config.policy_variant,
            variant.name,
        )

    _resolved_variant = variant
    return variant


__all__ = [
    "generate_rollout",
    "generate_rollout_async",
]


_rollout_counter: int = 0


_traj_save_pool = ThreadPoolExecutor(max_workers=32, thread_name_prefix="traj_save")


def _deterministic_sampling() -> bool:

    return os.environ.get("DETERMINISTIC_SAMPLING", "0") == "1"


def _rollout_concurrency(config: Any) -> int:

    v = os.environ.get("ROLLOUT_CONCURRENCY")
    if v:
        try:
            return max(1, int(v))
        except ValueError:
            logger.warning("Invalid ROLLOUT_CONCURRENCY=%r, falling back to num_sessions", v)
    return max(1, getattr(config, "num_sessions", 1))


class _AdaptiveTaskScheduler:
    MIN_SIBLINGS = int(os.environ.get("SCHED_MIN_SIBLINGS", "4"))
    DEFAULT_SIBLINGS = int(os.environ.get("SCHED_DEFAULT_SIBLINGS", "8"))
    MAX_SIBLINGS = int(os.environ.get("SCHED_MAX_SIBLINGS", "16"))

    HOPELESS_SIBLINGS = int(os.environ.get("SCHED_HOPELESS_SIBLINGS", "4"))
    MASTERED_SIBLINGS = int(os.environ.get("SCHED_MASTERED_SIBLINGS", "2"))

    MAX_TOTAL_PER_TASK = int(os.environ.get("SCHED_MAX_TOTAL_PER_TASK", "16"))

    EMA_ALPHA = float(os.environ.get("SCHED_EMA_ALPHA", "0.3"))

    TIER_MIN_OBS = int(os.environ.get("SCHED_TIER_MIN_OBS", "6"))

    MASTERED_ACC_THRESHOLD = float(os.environ.get("SCHED_MASTERED_ACC", "0.8"))

    def __init__(self) -> None:
        self._tasks: list[Sample] | None = None
        self._task_ids: list[str] = []
        self._cursor: int = 0
        self._epoch: int = 0
        self._history: dict[str, dict[str, float]] = {}
        self._rollout_count: int = 0

        self._launched_per_task: dict[str, int] = {}

    def reset_launch_budget(self) -> None:

        self._launched_per_task = {}

    def cap_room(self, task_id: str) -> int:

        if self.MAX_TOTAL_PER_TASK <= 0:
            return 1 << 30
        return self.MAX_TOTAL_PER_TASK - self._launched_per_task.get(task_id, 0)

    def record_launch(self, task_id: str, n: int) -> None:

        if n > 0:
            self._launched_per_task[task_id] = self._launched_per_task.get(task_id, 0) + n

    def is_capped(self, task_id: str) -> bool:

        return self.MAX_TOTAL_PER_TASK > 0 and self.cap_room(task_id) <= 0

    def _init_tasks(self, data_source: Any) -> None:

        groups = data_source.get_samples(9999)
        raw = [s for group in groups for s in group]
        seen: dict[str, Sample] = {}
        for s in raw:
            meta = _sample_metadata(s)
            key = str(meta.get("task_id") or id(s))
            if key not in seen:
                seen[key] = s
        self._tasks = list(seen.values())
        self._task_ids = [str(_sample_metadata(s).get("task_id") or id(s)) for s in self._tasks]
        self._cursor = 0
        self._epoch = 0
        logger.info(
            "[AdaptiveScheduler] initialized with %d unique tasks",
            len(self._tasks),
        )

    def all_tasks(self, data_source: Any) -> list[Sample]:

        if self._tasks is None:
            self._init_tasks(data_source)
        assert self._tasks is not None
        return list(self._tasks)

    def _task_tier(self, task_id: str) -> str:

        if task_id not in self._history:
            return "new"
        hist = self._history[task_id]
        acc = hist["acc"]
        obs = hist["total"]
        has_enough_obs = obs >= self.TIER_MIN_OBS
        if acc > self.MASTERED_ACC_THRESHOLD and has_enough_obs:
            return "mastered"
        elif acc == 0 and has_enough_obs:
            return "hopeless"
        elif acc > 0:
            return "frontier"
        else:
            return "new"

    def get_task_siblings(self, task_id: str, default_n: int) -> int:

        tier = self._task_tier(task_id)
        if tier == "hopeless":
            return self.HOPELESS_SIBLINGS
        if tier == "mastered":
            return self.MASTERED_SIBLINGS

        return default_n

    def advance_rollout_count(self) -> None:
        self._rollout_count += 1

    def update_history(self, task_id: str, n_pass: int, n_total: int) -> None:

        if n_total == 0:
            return
        batch_acc = n_pass / n_total
        if task_id not in self._history:
            self._history[task_id] = {
                "acc": batch_acc,
                "total": n_total,
                "pass": n_pass,
                "last_acc": batch_acc,
                "last_n": n_total,
            }
        else:
            old = self._history[task_id]
            alpha = self.EMA_ALPHA
            old["acc"] = alpha * old["acc"] + (1 - alpha) * batch_acc
            old["total"] += n_total
            old["pass"] += n_pass
            old["last_acc"] = batch_acc
            old["last_n"] = n_total

    def export_state(self) -> dict:

        return {
            "version": 1,
            "history": {k: dict(v) for k, v in self._history.items()},
            "rollout_count": self._rollout_count,
        }

    def import_state(self, state: dict) -> None:

        if not isinstance(state, dict):
            return
        hist = state.get("history")
        if isinstance(hist, dict):
            self._history = {
                str(k): dict(v)
                for k, v in hist.items()
                if isinstance(v, dict) and "acc" in v and "total" in v
            }
        rc = state.get("rollout_count")
        if isinstance(rc, int) and rc >= 0:
            self._rollout_count = rc

    def _trainability_priority(self, task_id: str) -> int:

        tier = self._task_tier(task_id)
        if tier == "frontier":
            return 0
        elif tier == "new":
            return 1
        elif tier == "hopeless":
            return 2
        else:
            return 3

    def _frontier_score(self, task_id: str) -> float:

        if task_id not in self._history:
            return 0.0
        acc = self._history[task_id]["acc"]
        return acc * (1 - acc)

    def next_batch(
        self,
        data_source: Any,
        batch_size: int,
        exclude: set[str] | None = None,
    ) -> list[Sample]:

        if self._tasks is None:
            self._init_tasks(data_source)
        assert self._tasks is not None
        total = len(self._tasks)
        if total == 0:
            return []

        candidates: list[tuple[int, int, int]] = []
        for i in range(total):
            tid = self._task_ids[i]
            if exclude and tid in exclude:
                continue

            if self.is_capped(tid):
                continue
            pri = self._trainability_priority(tid)
            rotated = (i - self._cursor) % total
            candidates.append((pri, rotated, i))

        candidates.sort()
        unique_batch = [self._tasks[orig_idx] for _, _, orig_idx in candidates[:batch_size]]
        self._cursor = (self._cursor + len(unique_batch)) % max(total, 1)

        if len(unique_batch) < batch_size:
            logger.info(
                "[AdaptiveScheduler] next_batch: only %d/%d distinct tasks "
                "available for requested batch_size=%d (excluded=%d) — "
                "sampling all available without padding.",
                len(unique_batch),
                total,
                batch_size,
                len(exclude) if exclude else 0,
            )
        return unique_batch

    def build_adaptive_groups(
        self,
        unique_tasks: list[Sample],
        default_n: int,
    ) -> list[tuple[str, list[Sample], int]]:

        import copy as _copy

        groups: list[tuple[str, list[Sample], int]] = []
        for s in unique_tasks:
            meta = _sample_metadata(s)
            task_id = str(meta.get("task_id") or id(s))

            if _deterministic_sampling():
                n = default_n
            else:
                n = self.get_task_siblings(task_id, default_n)

                room = self.cap_room(task_id)
                if room <= 0:
                    continue
                n = min(n, room)
                if n <= 0:
                    continue

            self.record_launch(task_id, n)
            copies = [_copy.deepcopy(s) for _ in range(n)]
            groups.append((task_id, copies, n))
        return groups

    def log_schedule_summary(self, groups: list[tuple[str, list[Any], int]]) -> None:

        lines = [f"[AdaptiveScheduler] sampling plan (rollout #{self._rollout_count}):"]
        tier_counts: dict[str, int] = {}
        task_totals: dict[str, int] = {}
        for task_id, _, n in groups:
            tier = self._task_tier(task_id)
            tier_counts[tier] = tier_counts.get(tier, 0) + 1
            task_totals[task_id] = task_totals.get(task_id, 0) + n
            hist = self._history.get(task_id)
            if hist:
                acc_str = f"acc={hist['acc']:.2f}"
            else:
                acc_str = "new"
            lines.append(f"  {task_id}: n={n}, {acc_str}, tier={tier}")
        total = sum(n for _, _, n in groups)
        cap_str = str(self.MAX_TOTAL_PER_TASK) if self.MAX_TOTAL_PER_TASK > 0 else "off"
        lines.append(
            f"  TOTAL: {len(groups)} groups, {total} trajectories, "
            f"tiers={tier_counts}, per-task-cap={cap_str}"
        )

        if self.MAX_TOTAL_PER_TASK > 0:
            for tid, tot in task_totals.items():
                if tot >= self.MAX_TOTAL_PER_TASK:
                    lines.append(f"  ⚠ {tid}: hit per-task cap ({tot}/{self.MAX_TOTAL_PER_TASK})")
        logger.info("\n".join(lines))


_SCHED_BUILD_TAG = "sched-cap-at-source-v2"

_task_iterator = _AdaptiveTaskScheduler()


_SCHED_STATE_META_KEY = "adaptive_scheduler_state"
_sched_state_restored = False


def _resume_hist_acc_enabled() -> bool:

    return os.environ.get("MAI_RESUME_HIST_ACC", "1") != "0"


def _maybe_restore_scheduler_state(data_source: Any) -> None:

    global _sched_state_restored
    if _sched_state_restored or not _resume_hist_acc_enabled():
        return
    _sched_state_restored = True
    meta = getattr(data_source, "metadata", None)
    if not isinstance(meta, dict):
        return
    state = meta.get(_SCHED_STATE_META_KEY)
    if not state:
        logger.info("[resume-hist-acc] no saved scheduler history (fresh run)")
        return
    try:
        _task_iterator.import_state(state)
        logger.info(
            "[resume-hist-acc] restored scheduler history: %d tasks, rollout_count=%d",
            len(_task_iterator._history),
            _task_iterator._rollout_count,
        )
    except Exception:
        logger.exception("[resume-hist-acc] failed to restore scheduler history")


def _stash_scheduler_state(data_source: Any) -> None:

    if not _resume_hist_acc_enabled():
        return
    meta = getattr(data_source, "metadata", None)
    if not isinstance(meta, dict):
        return
    try:
        meta[_SCHED_STATE_META_KEY] = _task_iterator.export_state()
    except Exception:
        logger.exception("[resume-hist-acc] failed to stash scheduler history")


async def _run_single_task_mobile(
    sample: Sample,
    config: SlimeAdapterConfig,
    env_client: EnvClient,
    args: Namespace,
    *,
    concurrency_sem: asyncio.Semaphore,
    evaluation: bool = False,
    attempt_id: str | None = None,
    traj_root: str | None = None,
    rollout_id: int = 0,
    epoch_getter: Callable[[], int] | None = None,
    clients_by_env: dict[str, EnvClient] | None = None,
) -> list[Sample]:

    task = _extract_task(sample, default_env_type=config.env_type)
    task.max_steps = config.max_turns_for_env(task.env_type)
    task.attempt_id = attempt_id

    task.train_id = config.train_id or None

    task.rollout_id = rollout_id
    task.is_eval = evaluation

    if clients_by_env is not None:
        client = clients_by_env[task.env_type if config.env_pools else config.env_type]
    else:
        client = _get_env_client_for(config, task.env_type) if config.env_pools else env_client

    if config.rollout_agent == "sdk":
        from mai_gym_rl.slime_adapter.sdk_rollout import run_sdk_task

        return await run_sdk_task(
            sample,
            task,
            config,
            client,
            args,
            concurrency_sem=concurrency_sem,
            state_factory=GenerateState,
            timing=TrajectoryTiming(),
            evaluation=evaluation,
            epoch_getter=epoch_getter,
            timeout=_DYN_TIMEOUT_S or float(os.environ.get("TRAJ_TIMEOUT_S", "1200")),
        )

    timing = TrajectoryTiming()
    t_total_start = time.monotonic()

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
        t_sem = time.monotonic()
        async with concurrency_sem:
            timing.concurrency_wait = time.monotonic() - t_sem

            t_acq = time.monotonic()
            async with client.asession(
                target_apps=task.target_apps,
                failure_log=failure_log,
            ) as session_id:
                timing.env_acquire = time.monotonic() - t_acq

                state = GenerateState(args)
                url = f"http://{args.sglang_router_ip}:{args.sglang_router_port}/generate"
                model_type = "llm" if args.multimodal_keys is None else "vlm"

                agent_variant = _resolve_agent_variant(config, env_type=task.env_type)

                chat_model = CustomChatModel(
                    url=url,
                    tokenizer=state.tokenizer,
                    processor=state.processor,
                    model_type=model_type,
                    tool_parser_name=agent_variant.parser_name,
                    preserve_thinking=agent_variant.preserve_thinking,
                    token_native=bool(getattr(config, "token_native_history", False)),
                    routing_key=session_id,
                    max_context_tokens=getattr(args, "rollout_max_context_len", None),
                )

                meta = _sample_metadata(sample)
                system_prompt = meta.get("system_prompt")
                if system_prompt is None:
                    system_prompt = agent_variant.system_prompt

                eval_temp = getattr(args, "eval_temperature", 0.0)
                sampling_params = {
                    "max_new_tokens": args.rollout_max_response_len,
                    "temperature": eval_temp if evaluation else args.rollout_temperature,
                    "top_p": getattr(args, "rollout_top_p", 1.0),
                }

                if agent_variant.stop_sequences:
                    sampling_params["stop"] = list(agent_variant.stop_sequences)
                    sampling_params["no_stop_trim"] = True

                messages_log_dir: Any = None
                if config.messages_log_root:
                    short_sid = session_id.split("-")[0]
                    leaf = f"{short_sid}_{task.task_id or 'notask'}"

                    if oss_io.is_oss_path(config.messages_log_root):
                        messages_log_dir = oss_io.join(config.messages_log_root, leaf)
                    else:
                        messages_log_dir = Path(config.messages_log_root) / leaf

                def _remote_log(step_idx: int, payload: dict[str, Any]) -> None:
                    client.log_messages(session_id, step_idx, payload)

                policy_kwargs: dict[str, Any] = dict(
                    chat_model=chat_model,
                    sampling_params=sampling_params,
                    original_image_width=config.original_image_width,
                    original_image_height=config.original_image_height,
                    resize_width=config.resize_width,
                    resize_height=config.resize_height,
                    system_prompt=system_prompt,
                    image_retention=config.image_retention,
                    messages_log_dir=messages_log_dir,
                    remote_log_callback=_remote_log,
                    oss_creds=config.oss,
                )

                if agent_variant.extra.get("windowed"):
                    policy_kwargs["window_size"] = config.window_size
                    policy_kwargs["overlap"] = config.overlap

                for _pk in ("resize_longest_edge", "max_pixels", "min_pixels", "image_factor"):
                    _pv = getattr(config, _pk, None)
                    if _pv is not None and _ctor_accepts(agent_variant.policy_cls, _pk):
                        policy_kwargs[_pk] = _pv

                if epoch_getter is not None and _ctor_accepts(
                    agent_variant.policy_cls, "epoch_getter"
                ):
                    policy_kwargs["epoch_getter"] = epoch_getter
                policy = agent_variant.policy_cls(**policy_kwargs)

                timed_client = TimedEnvClient(client, timing)

                policy.timing = timing

                try:
                    agent = RolloutAgent(
                        timed_client,
                        policy,
                        env_type=task.env_type,
                        oss_creds=config.oss,
                    )

                    traj_timeout = (
                        _DYN_TIMEOUT_S
                        if _DYN_TIMEOUT_S
                        else float(os.environ.get("TRAJ_TIMEOUT_S", "1200"))
                    )
                    if traj_timeout > 0:
                        trajectory = await asyncio.wait_for(
                            agent.run(session_id, task, tear_down=False),
                            timeout=traj_timeout,
                        )
                    else:
                        trajectory = await agent.run(
                            session_id,
                            task,
                            tear_down=False,
                        )
                except TimeoutError:
                    logger.warning(
                        "Trajectory timeout (%.0fs) for task %s on %s n_steps_logged=%d",
                        traj_timeout,
                        task.task_id,
                        session_id,
                        len(policy.step_records) if policy is not None else -1,
                    )
                except (
                    APIError,
                    PollTimeoutError,
                    ScreenshotTransportError,
                    pydantic.ValidationError,
                    httpx.RemoteProtocolError,
                    httpx.ReadError,
                    httpx.WriteError,
                    httpx.ConnectError,
                    httpx.ConnectTimeout,
                    httpx.WriteTimeout,
                    httpx.PoolTimeout,
                ) as exc:
                    n_steps = len(policy.step_records) if policy is not None else 0
                    phase = (
                        "screenshot_transport"
                        if isinstance(exc, ScreenshotTransportError)
                        else "reset/areset"
                        if n_steps == 0
                        else f"step#{n_steps}/astep"
                    )
                    logger.warning(
                        "[rollout] ENV-FAIL phase=%s task=%s sid=%s "
                        "exc=%s: %r — releasing (not discarding)",
                        phase,
                        task.task_id,
                        session_id,
                        type(exc).__name__,
                        str(exc) or "(empty)",
                    )

                step_records = policy.step_records

                if traj_root and trajectory is not None and step_records:
                    subdir = "eval" if evaluation else f"train_iter_{rollout_id:05d}"
                    traj_dir = os.path.join(
                        traj_root,
                        subdir,
                        f"{task.task_id or 'notask'}_{attempt_id or 'noid'}",
                    )
                    instruction = getattr(policy, "_instruction", "")
                    logger.info(
                        "Saving trajectory: task=%s, reward=%.3f, steps=%d, dir=%s",
                        task.task_id,
                        trajectory.reward,
                        len(step_records),
                        traj_dir,
                    )
                    save_future = _traj_save_pool.submit(
                        save_trajectory,
                        step_records,
                        trajectory.reward,
                        task.task_id or "",
                        instruction,
                        traj_dir,
                        system_prompt=system_prompt,
                        oss_creds=config.oss,
                    )
                    if clients_by_env is not None:
                        await asyncio.wrap_future(save_future)
                elif traj_root:
                    logger.warning(
                        "Skipping traj save: task=%s, trajectory=%s, step_records=%d",
                        task.task_id,
                        "None" if trajectory is None else f"reward={trajectory.reward}",
                        len(step_records),
                    )

    except Exception:
        logger.exception("Agent loop failed for task %s", task.task_id)

    timing.total = time.monotonic() - t_total_start
    timing.n_steps = len(step_records)

    timing_dict = timing.to_dict()

    n = max(timing.n_steps, 1)
    logger.info(
        "[timing] task=%s n_steps=%d total=%.1fs | per-step env_step=%.1fs "
        "sglang=%.1fs cpu_mm=%.2fs [template=%.2f vision=%.2f png=%.2f "
        "processor=%.2f fast_mm=%.2f splice=%.2f] postproc=%.2fs | once "
        "acquire=%.1fs reset=%.1fs concurrency_wait=%.1fs eval=%.1fs",
        task.task_id,
        timing.n_steps,
        timing.total,
        timing.env_step / n,
        timing.sglang_http_post / n,
        timing.cpu_multimodal_total / n,
        timing.chat_template / n,
        timing.vision_extract / n,
        timing.image_encode / n,
        timing.hf_processor / n,
        timing.fast_mm_token_plan / n,
        timing.token_native_splice / n,
        timing.postprocess / n,
        timing.env_acquire,
        timing.env_reset,
        timing.concurrency_wait,
        timing.env_eval,
    )

    if trajectory is None:
        sample.status = Sample.Status.FAILED
        sample.reward = 0.0
        sample.remove_sample = True
        meta = _sample_metadata(sample)
        meta["env_failed"] = True
        meta["task_id"] = task.task_id or "unknown"
        meta["trajectory_reward"] = 0.0
        meta["n_steps"] = 0
        meta["timing"] = timing_dict
        sample.metadata = meta
        return [sample]

    if evaluation:
        sample.reward = trajectory.reward
        sample.status = Sample.Status.COMPLETED
        sample.response = str(trajectory.steps)
        sample.response_length = 0
        meta = _sample_metadata(sample)
        meta["trajectory_reward"] = trajectory.reward
        meta["task_id"] = task.task_id or "unknown"
        meta["n_steps"] = len(step_records)
        meta["env_failed"] = False
        meta["timing"] = timing_dict
        sample.metadata = meta
        return [sample]

    assert state is not None
    original_prompt = _build_original_prompt(sample)

    _windowed = bool(agent_variant.extra.get("windowed"))

    _windowwise = _windowed and bool(getattr(config, "windowwise_train", False))
    if _windowwise:
        step_samples = await asyncio.to_thread(
            build_window_samples,
            trajectory=trajectory,
            step_records=step_records,
            original_prompt=original_prompt,
            config=config,
            tokenizer=state.tokenizer,
            processor=state.processor,
            system_prompt=system_prompt,
            instruction=getattr(policy, "_instruction", ""),
            defer_multimodal=not evaluation,
            window_size=config.window_size,
            overlap=config.overlap,
        )
    else:
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
            defer_multimodal=not evaluation,
            window_size=config.window_size if _windowed else None,
            overlap=config.overlap,
        )

    if not step_samples:
        sample.status = Sample.Status.FAILED
        sample.reward = 0.0
        sample.remove_sample = True
        return [sample]

    if step_samples:
        for step_sample in step_samples:
            metadata = _sample_metadata(step_sample)
            metadata["env_type"] = task.env_type
            metadata["agent_variant"] = agent_variant.name
            step_sample.metadata = metadata
        first_meta = _sample_metadata(step_samples[0])
        first_meta["timing"] = timing_dict
        step_samples[0].metadata = first_meta

    return step_samples


def _filter_zero_reward_groups(
    groups: list[list[list[Sample]]],
) -> tuple[list[list[list[Sample]]], int]:

    n_filtered = 0
    for group in groups:
        all_zero = True
        for sibling_steps in group:
            if not sibling_steps:
                continue
            meta = _sample_metadata(sibling_steps[0])
            traj_reward = float(meta.get("trajectory_reward", 0.0) or 0.0)
            if traj_reward != 0.0:
                all_zero = False
                break

        if all_zero:
            n_filtered += 1
            for sibling_steps in group:
                for s in sibling_steps:
                    s.remove_sample = True

    return groups, n_filtered


_traj_root_cached: str | None = None


def _get_traj_root(config: SlimeAdapterConfig) -> str | None:

    global _traj_root_cached
    if _traj_root_cached is not None:
        return _traj_root_cached

    root = os.environ.get("TRAJ_LOG_ROOT")
    if root is not None:
        if not root:
            return None
        base = root
    else:
        base = "./traj_logs"

    from datetime import datetime

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    _traj_root_cached = os.path.join(base, timestamp)
    return _traj_root_cached


def _kept_task_ids(flat_step_samples: list[list[Any]]) -> set[str]:

    kept: set[str] = set()
    for steps in flat_step_samples:
        if steps:
            tid = _sample_metadata(steps[0]).get("task_id")
            if tid:
                kept.add(tid)
    return kept


def _upload_trained_manifest(
    env_client: Any,
    config: SlimeAdapterConfig,
    args: Namespace,
    *,
    rollout_id: int,
    degrade_level: int,
    num_groups: int,
    trainable_task_ids: Iterable[str],
    kept_task_ids: Iterable[str],
    task_rewards: dict[str, list[float]],
    num_trained_trajectories: int,
) -> None:

    if not config.train_id:
        return
    try:
        manifest = build_trained_manifest(
            rollout_id=rollout_id,
            evaluation=False,
            advantage_estimator=getattr(args, "advantage_estimator", "grpo"),
            degrade_level=degrade_level,
            num_groups=num_groups,
            trainable_task_ids=trainable_task_ids,
            kept_task_ids=kept_task_ids,
            task_rewards=task_rewards,
            num_trained_trajectories=num_trained_trajectories,
            min_samples_per_task=MIN_SAMPLES_PER_TASK,
            created_at=time.time(),
        )
        env_client.register_trained_tasks(
            config.train_id,
            rollout_id,
            manifest,
            model_name=None,
        )
    except Exception:
        logger.exception("Rollout %d: trained-manifest upload failed", rollout_id)


def _run_eval_with_retry(
    task_samples: list[Any],
    config: SlimeAdapterConfig,
    env_client: Any,
    args: Namespace,
    *,
    attempt_prefix: str,
    traj_root: str | None,
    rollout_id: int,
    max_retries: int = 4,
) -> list[list[Any]]:

    import copy as _copy

    results: list[list[Any]] = [[] for _ in range(len(task_samples))]
    pending_indices = list(range(len(task_samples)))

    for attempt in range(1 + max_retries):
        if not pending_indices:
            break

        attempt_label = f"{attempt_prefix}_a{attempt}"
        samples_to_run = [(i, task_samples[i]) for i in pending_indices]

        logger.info(
            "Eval %d attempt %d/%d: running %d tasks",
            rollout_id,
            attempt,
            max_retries,
            len(samples_to_run),
        )

        async def _run_batch() -> list[tuple[int, list[Any]]]:
            sem = asyncio.Semaphore(max(1, config.num_sessions))

            async def _run_one(idx: int, sample: Any) -> tuple[int, list[Any]]:
                result = await _run_single_task_mobile(
                    sample,
                    config,
                    env_client,
                    args,
                    concurrency_sem=sem,
                    evaluation=True,
                    attempt_id=f"{attempt_label}_s{idx:03d}",
                    traj_root=traj_root,
                    rollout_id=rollout_id,
                )
                return idx, result

            coros = [_run_one(i, _copy.deepcopy(s)) for i, s in samples_to_run]
            return await asyncio.gather(*coros, return_exceptions=False)

        loop = asyncio.new_event_loop()
        try:
            batch_results = loop.run_until_complete(_run_batch())
        finally:
            with contextlib.suppress(Exception):
                loop.run_until_complete(oss_io.close_async_fs())
            loop.close()

        still_pending = []
        for idx, step_list in batch_results:
            results[idx] = step_list

            env_failed = False
            if step_list:
                meta = _sample_metadata(step_list[0])
                env_failed = bool(meta.get("env_failed"))
            else:
                env_failed = True

            if env_failed and attempt < max_retries:
                still_pending.append(idx)
                task_meta = _sample_metadata(task_samples[idx])
                logger.warning(
                    "Eval %d: task %s env_failed on attempt %d, will retry",
                    rollout_id,
                    task_meta.get("task_id", "?"),
                    attempt,
                )

        pending_indices = still_pending

    if pending_indices:
        failed_tasks = [
            str(_sample_metadata(task_samples[i]).get("task_id", "?")) for i in pending_indices
        ]
        logger.warning(
            "Eval %d: %d tasks still env_failed after %d retries: %s",
            rollout_id,
            len(pending_indices),
            max_retries,
            ", ".join(failed_tasks),
        )

    return results


def _is_env_failed(result: list[Any]) -> bool:

    if not result:
        return True
    meta = _sample_metadata(result[0])
    return bool(meta.get("env_failed"))


async def _cancel_dispatch_tasks(pending: set[asyncio.Task]) -> None:
    tasks = list(pending)
    for task in tasks:
        task.cancel()
    if tasks:
        await asyncio.gather(*tasks, return_exceptions=True)


async def _run_training_pooled(
    adaptive_groups,
    data_source,
    config,
    env_client,
    args,
    **kwargs,
):
    if WorkerOptions.from_env().nodes == 0:
        return await _collect_training_pooled(
            adaptive_groups,
            data_source,
            config,
            env_client,
            args,
            **kwargs,
        )
    logger.info(
        "[distributed-rollout] timeout_mode=%s trajectory_timeout_s=%g collection_timeout_s=%g",
        "dynamic" if _DYN_TIMEOUT_S else "fixed",
        _DYN_TIMEOUT_S or float(os.environ.get("TRAJ_TIMEOUT_S", "1200")),
        _DYN_TIMEOUT_S or ROLLOUT_TIMEOUT_S,
    )
    async with make_executor(
        config,
        args,
        _rollout_concurrency(config),
        _DYN_TIMEOUT_S,
    ) as executor:
        pending: set[asyncio.Task] = set()
        try:
            return await _collect_training_pooled(
                adaptive_groups,
                data_source,
                config,
                env_client,
                args,
                executor=executor,
                task_registry=pending,
                **kwargs,
            )
        finally:
            await _cancel_dispatch_tasks(pending)


async def _collect_training_pooled(
    adaptive_groups: list[tuple[str, list[Any], int]],
    data_source: Any,
    config: SlimeAdapterConfig,
    env_client: Any,
    args: Any,
    *,
    n_per_prompt: int,
    traj_root: str | None,
    rollout_id: int,
    executor: Any = None,
    task_registry: set[asyncio.Task] | None = None,
) -> tuple[
    list[list[list[Any]]], set[str], dict[str, list[int]], dict[str, list[float]], list[int]
]:

    import copy as _copy

    is_ppo = getattr(args, "advantage_estimator", "grpo") == "ppo"

    pool_size = int(os.environ.get("ROLLOUT_THREAD_POOL_SIZE", str(max(64, config.num_sessions))))
    loop = asyncio.get_running_loop()
    loop.set_default_executor(ThreadPoolExecutor(max_workers=pool_size))
    logger.info(
        "Set default thread pool to %d workers (num_sessions=%d)", pool_size, config.num_sessions
    )

    concurrency = _rollout_concurrency(config)
    sem = asyncio.Semaphore(concurrency)
    logger.info(
        "Pooled rollout: in-flight concurrency cap=%d (num_sessions=%d, ROLLOUT_CONCURRENCY=%s)",
        concurrency,
        config.num_sessions,
        os.environ.get("ROLLOUT_CONCURRENCY", "<unset>"),
    )
    max_retries = int(os.environ.get("MAX_ROLLOUT_GROUP_RETRIES", "10"))

    group_results: list[list[list[Any] | None]] = []
    group_task_ids: list[str] = []
    group_valid: list[int] = []

    group_pass: list[int] = []
    group_finalized: list[bool] = []
    sampled_task_ids: set[str] = set()

    group_retry_target: list[int] = []
    group_n_launched: list[int] = []

    task_rewards: dict[str, list[float]] = {}
    task_gids: dict[str, list[int]] = {}
    trainable_task_set: set[str] = set()

    task_total_launched: dict[str, int] = {}

    eviction_budget = int(os.environ.get("MAX_TRAIN_STEP_SAMPLES") or "0")
    kept_groups: dict[int, tuple[float, int]] = {}
    kept_step_count = [0]

    def _add_group(task_id: str, copies: list[Any]) -> tuple[int, list[Any]]:

        cap = _task_iterator.MAX_TOTAL_PER_TASK
        if cap > 0:
            room = cap - task_total_launched.get(task_id, 0)
            if room <= 0:
                return -1, []
            if len(copies) > room:
                copies = copies[:room]
        gid = len(group_results)
        n = len(copies)
        group_results.append([None] * n)
        group_task_ids.append(task_id)
        group_n_launched.append(n)
        group_retry_target.append(max(3, n * 2 // 3))
        group_valid.append(0)
        group_pass.append(0)
        group_finalized.append(False)
        sampled_task_ids.add(task_id)
        task_gids.setdefault(task_id, []).append(gid)
        task_total_launched[task_id] = task_total_launched.get(task_id, 0) + n
        return gid, copies

    pending_tasks: set[asyncio.Task] = task_registry if task_registry is not None else set()

    async def _run_one(gid: int, sid: int, sample: Any, attempt: int = 0) -> None:

        try:
            s = _copy.deepcopy(sample) if attempt > 0 else sample
            async with sem:
                if executor is None:
                    result = await _run_single_task_mobile(
                        s,
                        config,
                        env_client,
                        args,
                        concurrency_sem=asyncio.Semaphore(999),
                        evaluation=False,
                        attempt_id=f"g{gid:05d}_s{sid:02d}_a{attempt}",
                        traj_root=traj_root,
                        rollout_id=rollout_id,
                    )
                else:
                    env_type = (
                        _extract_task(s, default_env_type=config.env_type).env_type
                        if config.env_pools
                        else config.env_type
                    )
                    result = await executor.run(
                        s,
                        env_type=env_type,
                        evaluation=False,
                        attempt_id=f"g{gid:05d}_s{sid:02d}_a{attempt}",
                        traj_root=traj_root,
                        rollout_id=rollout_id,
                    )
        except Exception:
            if executor is not None and executor.failure is not None:
                return
            logger.exception("_run_one failed: gid=%d sid=%d attempt=%d", gid, sid, attempt)
            result = []

        if group_results[gid] is None:
            return
        group_results[gid][sid] = result

        if _is_env_failed(result):
            if group_valid[gid] < group_retry_target[gid] and attempt < max_retries:
                t = asyncio.create_task(_run_one(gid, sid, sample, attempt + 1))
                pending_tasks.add(t)
                t.add_done_callback(pending_tasks.discard)

            return

        group_valid[gid] += 1
        _reward = float(_sample_metadata(result[0]).get("trajectory_reward", 0.0) or 0.0)
        if _reward > 0:
            group_pass[gid] += 1
        n_launched = group_n_launched[gid]
        finalize_threshold = max(3, n_launched * 2 // 3)
        if not group_finalized[gid]:
            if group_valid[gid] >= finalize_threshold:
                group_finalized[gid] = True
                _finalize_group(gid)
        else:
            _record_late_reward(gid, _reward)

    def _finalize_group(gid: int) -> None:

        results = [r for r in group_results[gid] if r is not None]
        task_id = group_task_ids[gid]

        group_rewards: list[float] = []
        for r in results:
            if not _is_env_failed(r):
                reward = float(_sample_metadata(r[0]).get("trajectory_reward", 0.0) or 0.0)
                group_rewards.append(reward)
        n_pass = sum(1 for r in group_rewards if r > 0)
        n_valid = len(group_rewards)

        task_rewards.setdefault(task_id, []).extend(group_rewards)
        all_rewards = task_rewards[task_id]

        if task_id not in trainable_task_set and _is_task_trainable(all_rewards, is_ppo=is_ppo):
            trainable_task_set.add(task_id)
            logger.info(
                "Rollout %d: task %s trainable [%d valid, %d pass, %d groups], "
                "trainable so far: %d",
                rollout_id,
                task_id,
                len(all_rewards),
                sum(1 for r in all_rewards if r > 0),
                len(task_gids.get(task_id, [])),
                len(trainable_task_set),
            )

        if eviction_budget > 0:
            if not _is_task_trainable(all_rewards, is_ppo=is_ppo):
                group_results[gid] = None
                return
            pr = n_pass / max(1, n_valid)
            n_steps = sum(len(s) for s in group_results[gid] if s)
            kept_groups[gid] = (abs(pr - 0.5), n_steps)
            kept_step_count[0] += n_steps
            while kept_step_count[0] > eviction_budget and len(kept_groups) > 1:
                worst = max(kept_groups, key=lambda g: kept_groups[g][0])
                kept_step_count[0] -= kept_groups[worst][1]
                del kept_groups[worst]
                group_results[worst] = None

    def _record_late_reward(gid: int, reward: float) -> None:

        task_id = group_task_ids[gid]
        task_rewards.setdefault(task_id, []).append(reward)
        all_rewards = task_rewards[task_id]
        if task_id not in trainable_task_set and _is_task_trainable(all_rewards, is_ppo=is_ppo):
            trainable_task_set.add(task_id)
            logger.info(
                "Rollout %d: task %s trainable [%d valid, %d pass, %d groups] "
                "(on a post-finalize sibling), trainable so far: %d",
                rollout_id,
                task_id,
                len(all_rewards),
                sum(1 for r in all_rewards if r > 0),
                len(task_gids.get(task_id, [])),
                len(trainable_task_set),
            )

    for task_id, copies, _n in adaptive_groups:
        gid, copies = _add_group(task_id, copies)
        if gid < 0:
            continue
        for sid, sample in enumerate(copies):
            t = asyncio.create_task(_run_one(gid, sid, sample))
            pending_tasks.add(t)
            t.add_done_callback(pending_tasks.discard)

    _budget_start = time.monotonic()
    _harvest_timeout = _DYN_TIMEOUT_S if _DYN_TIMEOUT_S else ROLLOUT_TIMEOUT_S
    while pending_tasks:
        if executor is not None:
            executor.raise_if_failed()
        remaining = _harvest_timeout - (time.monotonic() - _budget_start)
        if remaining <= 0:
            logger.warning(
                "Rollout %d: rollout timeout=%.0fs reached with %d trajectories "
                "still in flight — finalizing with collected results.",
                rollout_id,
                _harvest_timeout,
                len(pending_tasks),
            )
            break

        await asyncio.wait(
            pending_tasks,
            timeout=min(30.0, remaining),
            return_when=asyncio.FIRST_COMPLETED,
        )

    if executor is not None:
        executor.raise_if_failed()

    for gid in range(len(group_results)):
        if not group_finalized[gid]:
            group_finalized[gid] = True
            _finalize_group(gid)

    _n_hist = 0
    for gid in range(len(group_results)):
        if group_valid[gid] >= 1:
            _task_iterator.update_history(
                group_task_ids[gid],
                group_pass[gid],
                group_valid[gid],
            )
            _n_hist += 1
    logger.info(
        "Rollout %d: hist acc updated for %d/%d groups (%d valid trajectories, %d pass)",
        rollout_id,
        _n_hist,
        len(group_results),
        sum(group_valid),
        sum(group_pass),
    )

    for t in pending_tasks:
        t.cancel()
    if pending_tasks:
        try:
            await asyncio.wait_for(
                asyncio.gather(*pending_tasks, return_exceptions=True),
                timeout=120.0,
            )
        except TimeoutError:
            logger.warning(
                "Rollout %d: %d trajectories did not unwind within 120s after "
                "cancel; proceeding with collected results.",
                rollout_id,
                len(pending_tasks),
            )

    all_results = [group_results[gid] for gid in range(len(group_results))]
    all_n_launched = list(group_n_launched)
    return all_results, trainable_task_set, task_gids, task_rewards, all_n_launched


async def _run_eval_pooled(
    task_samples: list[Any],
    config: SlimeAdapterConfig,
    env_client: Any,
    args: Any,
    *,
    attempt_prefix: str,
    traj_root: str | None,
    rollout_id: int,
    max_retries: int = 4,
) -> list[list[Any]]:

    import copy as _copy

    pool_size = int(os.environ.get("ROLLOUT_THREAD_POOL_SIZE", str(max(64, config.num_sessions))))
    loop = asyncio.get_running_loop()
    loop.set_default_executor(ThreadPoolExecutor(max_workers=pool_size))

    sem = asyncio.Semaphore(max(1, config.num_sessions))
    results: list[list[Any]] = [[] for _ in range(len(task_samples))]
    pending_tasks: set[asyncio.Task] = set()

    async def _run_one(idx: int, sample: Any, attempt: int = 0) -> None:
        try:
            s = _copy.deepcopy(sample) if attempt > 0 else sample
            async with sem:
                result = await _run_single_task_mobile(
                    s,
                    config,
                    env_client,
                    args,
                    concurrency_sem=asyncio.Semaphore(999),
                    evaluation=True,
                    attempt_id=f"{attempt_prefix}_s{idx:03d}_a{attempt}",
                    traj_root=traj_root,
                    rollout_id=rollout_id,
                )
        except Exception:
            logger.exception("Eval _run_one failed: idx=%d attempt=%d", idx, attempt)
            result = []
        results[idx] = result

        if _is_env_failed(result) and attempt < max_retries:
            meta = _sample_metadata(result[0]) if result else {}
            logger.info(
                "Eval %d: task %s env_failed (attempt %d), retrying immediately",
                rollout_id,
                meta.get("task_id", "?"),
                attempt,
            )
            t = asyncio.create_task(_run_one(idx, sample, attempt + 1))
            pending_tasks.add(t)
            t.add_done_callback(pending_tasks.discard)

    for idx, sample in enumerate(task_samples):
        t = asyncio.create_task(_run_one(idx, sample))
        pending_tasks.add(t)
        t.add_done_callback(pending_tasks.discard)

    _eval_start = time.time()
    while pending_tasks:
        await asyncio.wait(
            pending_tasks,
            timeout=30.0,
            return_when=asyncio.FIRST_COMPLETED,
        )
        if time.time() - _eval_start >= ROLLOUT_TIMEOUT_S:
            logger.warning(
                "Eval: wall-clock deadline %.0fs reached with %d tasks pending — "
                "cancelling and returning partial results.",
                ROLLOUT_TIMEOUT_S,
                len(pending_tasks),
            )
            for t in list(pending_tasks):
                t.cancel()
            with contextlib.suppress(TimeoutError):
                await asyncio.wait_for(
                    asyncio.gather(*pending_tasks, return_exceptions=True),
                    timeout=120.0,
                )
            break

    return results


_GENERATE_STATE: GenerateState | None = None


def _get_generate_state(args: Any) -> GenerateState:

    global _GENERATE_STATE
    if _GENERATE_STATE is None:
        _GENERATE_STATE = GenerateState(args)
    return _GENERATE_STATE


_run_config_sent: set[str] = set()


_train_rollout_started = False


def _eval_prompt_count(args: Any) -> int:

    raw = os.environ.get("RL_PROMPTS_PER_EVAL", "").strip()
    if raw:
        try:
            return max(1, int(raw))
        except ValueError:
            logger.warning(
                "RL_PROMPTS_PER_EVAL=%r is not an int; falling back to args",
                raw,
            )
    return getattr(args, "rl_prompts_per_eval", 32)


def generate_rollout(
    args: Namespace,
    rollout_id: int,
    data_source: Any,
    evaluation: bool = False,
) -> RolloutFnTrainOutput | RolloutFnEvalOutput:

    global _train_rollout_started

    _skip = getattr(args, "skip_initial_eval", False) or os.environ.get("SKIP_INITIAL_EVAL") == "1"
    if evaluation and rollout_id == 0 and _skip and not _train_rollout_started:
        logger.info("Skipping initial eval (rollout_id=0, --skip-initial-eval set)")
        return RolloutFnEvalOutput(data={})
    if not evaluation:
        _train_rollout_started = True

    config = _get_config()
    if WorkerOptions.from_env().nodes != 0:
        WorkerOptions.from_env().validate(config, args)
    env_client = _get_env_client(config)
    traj_root = _get_traj_root(config)

    if config.train_id and config.train_id not in _run_config_sent:
        from mai_gym.models.run_config import build_slime_run_config

        _eval_temp = getattr(args, "eval_temperature", 0.0)
        _sampling: dict[str, Any] = {
            "max_new_tokens": args.rollout_max_response_len,
            "temperature": _eval_temp if evaluation else args.rollout_temperature,
            "top_p": getattr(args, "rollout_top_p", 1.0),
        }
        try:
            _variant = _resolve_agent_variant(config)
            if config.rollout_agent != "sdk" and _variant.stop_sequences:
                _sampling["stop"] = list(_variant.stop_sequences)
                _sampling["no_stop_trim"] = True
        except Exception:
            pass
        if config.rollout_agent == "sdk":
            _sampling["top_k"] = getattr(args, "rollout_top_k", -1)
        snapshot = build_slime_run_config(
            train_id=config.train_id,
            model_name=getattr(args, "hf_checkpoint", None) or getattr(args, "model_name", None),
            agent_variant=(
                "sdk"
                if config.rollout_agent == "sdk"
                else config.agent_variant or config.policy_variant
            ),
            sampling_params=_sampling,
            num_sessions=config.num_sessions,
            env_type=config.env_type,
            evaluation=evaluation,
        )
        if config.rollout_agent == "sdk":
            snapshot["sdk"] = {
                "coordinate_space": config.sdk_coordinate_space,
                "image_retention": config.image_retention,
                "max_turns": config.max_turns,
                "integration_package": "mai-ui-mobileworld>=0.2.1",
                "worker_api": "0.2",
                "reasoning_end_token": config.sdk_reasoning_end_token,
                "manager_protocol": "sdk_operations.v1",
            }
        env_client.register_run_config(
            config.train_id,
            snapshot,
            model_name=None,
        )
        _run_config_sent.add(config.train_id)

    import copy as _copy
    import random as _random

    n_samples_per_prompt = max(1, getattr(args, "n_samples_per_prompt", 8))

    is_ppo = getattr(args, "advantage_estimator", "grpo") == "ppo"

    if evaluation:
        batch_size = _eval_prompt_count(args)
        n_per_prompt = getattr(args, "n_samples_per_eval_prompt", 1)

        _rng_state = _random.getstate()
        _random.seed(42)
    else:
        batch_size = getattr(args, "rollout_batch_size", 32)
        n_per_prompt = n_samples_per_prompt

    if evaluation:
        _random.setstate(_rng_state)

        _eval_tasks = load_eval_tasks(args)
        unique_tasks = (_eval_tasks or _task_iterator.all_tasks(data_source))[:batch_size]

        task_samples = []
        for s in unique_tasks:
            for _ in range(n_per_prompt):
                task_samples.append(_copy.deepcopy(s))
        logger.info(
            "Eval %d: %d unique tasks × %d attempts = %d total",
            rollout_id,
            len(unique_tasks),
            n_per_prompt,
            len(task_samples),
        )
        _maybe_set_dynamic_timeout(None, config, rollout_id)
    else:
        _maybe_restore_scheduler_state(data_source)

        _task_iterator.advance_rollout_count()
        _task_iterator.reset_launch_budget()
        unique_tasks = _task_iterator.next_batch(data_source, batch_size)
        adaptive_groups = _task_iterator.build_adaptive_groups(unique_tasks, n_per_prompt)
        _task_iterator.log_schedule_summary(adaptive_groups)
        _maybe_set_dynamic_timeout(adaptive_groups, config, rollout_id)

        task_samples = []
        for _, copies, _ in adaptive_groups:
            task_samples.extend(copies)
        logger.info(
            "Rollout %d: %d unique tasks, %d total trajectories (adaptive)",
            rollout_id,
            len(unique_tasks),
            len(task_samples),
        )

    _ensure_task_sets_registered(unique_tasks, env_client)

    attempt_prefix = f"eval_r{rollout_id:05d}" if evaluation else f"r{rollout_id:05d}"

    if evaluation:
        max_retries = int(os.environ.get("MAX_EVAL_RETRIES", "4"))
        loop = asyncio.new_event_loop()
        try:
            all_step_samples = loop.run_until_complete(
                _run_eval_pooled(
                    task_samples,
                    config,
                    env_client,
                    args,
                    attempt_prefix=attempt_prefix,
                    traj_root=traj_root,
                    rollout_id=rollout_id,
                    max_retries=max_retries,
                )
            )
        finally:
            with contextlib.suppress(Exception):
                loop.run_until_complete(oss_io.close_async_fs())
            loop.close()
    else:
        loop = asyncio.new_event_loop()
        try:
            collected, trainable_tasks, task_gids_map, sync_task_rewards, collected_n_launched = (
                loop.run_until_complete(
                    _run_training_pooled(
                        adaptive_groups,
                        data_source,
                        config,
                        env_client,
                        args,
                        n_per_prompt=n_per_prompt,
                        traj_root=traj_root,
                        rollout_id=rollout_id,
                    )
                )
            )
        finally:
            with contextlib.suppress(Exception):
                loop.run_until_complete(oss_io.close_async_fs())
            loop.close()

        _stash_scheduler_state(data_source)

        n_total = len(collected)
        degrade_level = 0

        gid_to_task: dict[int, str] = {}
        for tid, gids in task_gids_map.items():
            for gid in gids:
                gid_to_task[gid] = tid
        sync_task_rewards = {}
        for gid, group_result in enumerate(collected):
            if group_result is None:
                continue
            tid = gid_to_task.get(gid, "")
            for r in group_result:
                if r is not None and not _is_env_failed(r):
                    reward = float(_sample_metadata(r[0]).get("trajectory_reward", 0.0) or 0.0)
                    sync_task_rewards.setdefault(tid, []).append(reward)
        trainable_tasks = {
            tid
            for tid, rewards in sync_task_rewards.items()
            if _is_task_trainable(rewards, is_ppo=is_ppo)
        }

        flat_step_samples = []
        if is_ppo:
            max_train_trajectories = int(os.environ.get("MAX_TRAIN_TRAJECTORIES") or "0")
            flat_step_samples = select_ppo_trajectories(
                collected,
                _sample_metadata,
                max_train_trajectories=max_train_trajectories,
                shuffle_fn=_random.shuffle,
            )
            logger.info(
                "Rollout %d (PPO): kept %d valid trajectories (%d trainable tasks, budget=%s traj)",
                rollout_id,
                len(flat_step_samples),
                len(trainable_tasks),
                max_train_trajectories or "inf",
            )
        else:
            if len(trainable_tasks) < MIN_TRAINABLE_TASKS:
                for tid, rewards in sync_task_rewards.items():
                    if tid not in trainable_tasks and len(rewards) >= 2 and len(set(rewards)) > 1:
                        trainable_tasks.add(tid)
                if trainable_tasks:
                    degrade_level = 1

            if not trainable_tasks:
                for tid, rewards in sync_task_rewards.items():
                    if rewards:
                        trainable_tasks.add(tid)
                if trainable_tasks:
                    degrade_level = 2

            for tid, rewards in sorted(sync_task_rewards.items()):
                if tid not in trainable_tasks:
                    logger.info(
                        "Rollout %d: task %s NOT trainable [%d valid, %d pass, distinct=%s]",
                        rollout_id,
                        tid,
                        len(rewards),
                        sum(1 for r in rewards if r > 0),
                        sorted(set(rewards)),
                    )

            if trainable_tasks:
                max_train_trajectories = int(os.environ.get("MAX_TRAIN_TRAJECTORIES") or "0")
                per_task_samples: dict[str, list] = {}
                for gid, group_result in enumerate(collected):
                    if group_result is None:
                        continue
                    tid = gid_to_task.get(gid, "")
                    if tid not in trainable_tasks:
                        continue
                    for sibling_steps in group_result:
                        if sibling_steps:
                            meta = _sample_metadata(sibling_steps[0])
                            if not meta.get("env_failed"):
                                per_task_samples.setdefault(tid, []).append(sibling_steps)

                def _task_pass_rate(tid: str) -> float:
                    rs = sync_task_rewards.get(tid, [])
                    return (sum(1 for r in rs if r > 0) / len(rs)) if rs else 0.0

                ranked = sorted(
                    per_task_samples,
                    key=lambda t: (abs(_task_pass_rate(t) - 0.5), -len(per_task_samples[t])),
                )
                kept_tasks: list[str] = []

                for tid in ranked:
                    if (
                        max_train_trajectories > 0
                        and len(flat_step_samples) >= max_train_trajectories
                    ):
                        break
                    flat_step_samples.extend(per_task_samples[tid])
                    kept_tasks.append(tid)
                logger.info(
                    "Rollout %d: kept %d/%d trainable tasks by pass-rate~0.5 "
                    "(%d trajectories, budget=%s traj); pass-rates kept=%s",
                    rollout_id,
                    len(kept_tasks),
                    len(per_task_samples),
                    len(flat_step_samples),
                    max_train_trajectories or "inf",
                    [round(_task_pass_rate(t), 3) for t in kept_tasks],
                )
            else:
                degrade_level = 3
                for group_result in collected:
                    if group_result is None:
                        continue
                    for sibling_steps in group_result:
                        if sibling_steps:
                            flat_step_samples.append(sibling_steps)

        logger.info(
            "Rollout %d: %d trainable tasks, %d groups, %d step samples (degrade_level=%d)",
            rollout_id,
            len(trainable_tasks),
            n_total,
            len(flat_step_samples),
            degrade_level,
        )

        _upload_trained_manifest(
            env_client,
            config,
            args,
            rollout_id=rollout_id,
            degrade_level=degrade_level,
            num_groups=n_total,
            trainable_task_ids=trainable_tasks,
            kept_task_ids=_kept_task_ids(flat_step_samples),
            task_rewards=sync_task_rewards,
            num_trained_trajectories=len(flat_step_samples),
        )

        n_mat = 0
        if os.environ.get("LAZY_MULTIMODAL") == "1":
            logger.info(
                "Rollout %d: LAZY_MULTIMODAL=1 — shipping raw multimodal inputs; "
                "pixel_values built per-microbatch in train get_batch",
                rollout_id,
            )
        else:
            try:
                _gstate = _get_generate_state(args)
                _to_mat = [
                    _s
                    for _traj in flat_step_samples
                    for _s in _traj
                    if getattr(_s, "multimodal_inputs", None) is not None
                ]
                if _to_mat:
                    _mw = int(os.environ.get("MATERIALIZE_WORKERS") or "48")
                    with ThreadPoolExecutor(max_workers=_mw) as _ex:
                        list(
                            _ex.map(
                                lambda _s: materialize_multimodal_inputs(
                                    _s, _gstate.tokenizer, _gstate.processor
                                ),
                                _to_mat,
                            )
                        )
                    n_mat = len(_to_mat)
            except Exception:
                logger.exception("Rollout %d: multimodal materialization failed", rollout_id)
            if n_mat:
                logger.info("Rollout %d: materialized %d deferred step samples", rollout_id, n_mat)

        _cap = _task_iterator.MAX_TOTAL_PER_TASK
        _batch_counts: dict[str, int] = {}
        for _sib in flat_step_samples:
            if _sib:
                _tid = str(_sample_metadata(_sib[0]).get("task_id") or "unknown")
                _batch_counts[_tid] = _batch_counts.get(_tid, 0) + 1
        _worst = max(_batch_counts.values(), default=0)
        _violators = {t: c for t, c in _batch_counts.items() if _cap > 0 and c > _cap}
        logger.warning(
            "[CAP-VERDICT] rollout=%d build=%s cap=%d distinct_tasks=%d "
            "worst_per_task=%d verdict=%s file=%s",
            rollout_id,
            _SCHED_BUILD_TAG,
            _cap,
            len(_batch_counts),
            _worst,
            "OK" if not _violators else f"VIOLATED {_violators}",
            __file__,
        )

        output = _build_train_output(
            flat_step_samples,
            len(flat_step_samples),
            rollout_id,
        )
        _warn_if_undersized(args, output, rollout_id)
        output.metrics["rollout/trainable_tasks"] = float(len(trainable_tasks))
        output.metrics["rollout/total_groups"] = float(n_total)
        output.metrics["rollout/degrade_level"] = float(degrade_level)
        output.metrics.update(aggregate_cache_metrics(flat_step_samples))
        return output

    if evaluation:
        rewards = []
        for step_list in all_step_samples:
            for s in step_list:
                rewards.append(s.reward if s.reward is not None else 0.0)
        eval_metrics = _compute_rollout_metrics(
            all_step_samples,
            prefix="eval/gui_agent",
        )

        per_task_eval: dict[str, tuple[int, int, list[int]]] = {}
        per_task_env_fail: dict[str, int] = {}
        n_eval_pass = 0
        n_eval_total = 0
        n_env_failed = 0
        for traj in all_step_samples:
            if not traj:
                continue
            meta = _sample_metadata(traj[0])
            task_id = str(meta.get("task_id") or "unknown")
            if meta.get("env_failed"):
                n_env_failed += 1
                per_task_env_fail[task_id] = per_task_env_fail.get(task_id, 0) + 1
                continue
            n_eval_total += 1
            traj_reward = float(meta.get("trajectory_reward", 0.0) or 0.0)
            n_steps = int(meta.get("n_steps", len(traj)) or len(traj))
            passed = traj_reward > 0
            if passed:
                n_eval_pass += 1
            old_p, old_t, old_steps = per_task_eval.get(task_id, (0, 0, []))
            per_task_eval[task_id] = (old_p + int(passed), old_t + 1, old_steps + [n_steps])

        eval_acc = n_eval_pass / max(n_eval_total, 1)
        n_all = n_eval_total + n_env_failed
        eval_acc_all = n_eval_pass / max(n_all, 1)
        eval_metrics["critic/eval_traj_acc"] = eval_acc
        eval_metrics["critic/eval_traj_acc_all"] = eval_acc_all
        eval_metrics["critic/eval_traj_pass"] = float(n_eval_pass)
        eval_metrics["critic/eval_traj_total"] = float(n_eval_total)
        eval_metrics["critic/eval_traj_total_all"] = float(n_all)
        eval_metrics["critic/eval_env_failed"] = float(n_env_failed)
        for task_id, (passes, total, _) in per_task_eval.items():
            eval_metrics[f"critic/eval_per_task/{task_id}/acc"] = passes / max(total, 1)

        all_steps = [s for _, _, steps in per_task_eval.values() for s in steps]
        avg_steps_all = sum(all_steps) / max(len(all_steps), 1)
        all_task_ids = sorted(set(list(per_task_eval.keys()) + list(per_task_env_fail.keys())))
        col_task = max(len("Task"), *(len(t) for t in all_task_ids)) + 2 if all_task_ids else 20
        header = f"{'Task':<{col_task}} {'Acc':>10} {'AccVal':>8} {'AvgSteps':>10} {'EnvFail':>8}"
        sep = "-" * len(header)
        lines = [
            f"Eval {rollout_id} results:",
            sep,
            header,
            sep,
        ]
        for task_id in all_task_ids:
            passes, total, steps = per_task_eval.get(task_id, (0, 0, []))
            avg_s = sum(steps) / max(len(steps), 1)
            acc_str = f"{passes}/{total}" if total > 0 else "-"
            task_acc = passes / max(total, 1) if total > 0 else 0.0
            ef = per_task_env_fail.get(task_id, 0)
            ef_str = str(ef) if ef > 0 else ""
            lines.append(
                f"{task_id:<{col_task}} {acc_str:>10} acc={task_acc:.2f} {avg_s:>10.1f} {ef_str:>8}"
            )
        lines.append(sep)
        total_acc_str = f"{n_eval_pass}/{n_eval_total}"
        ef_total_str = str(n_env_failed) if n_env_failed > 0 else ""
        lines.append(
            f"{'TOTAL':<{col_task}} {total_acc_str:>10} acc={eval_acc:.4f} {avg_steps_all:>10.1f} {ef_total_str:>8}"
        )
        lines.append(sep)
        if n_env_failed > 0:
            lines.append(f"⚠ {n_env_failed} trajectories excluded (env setup failed)")
        lines.append(f"[Eval {rollout_id}] acc={eval_acc:.4f} ({n_eval_pass}/{n_eval_total})")
        logger.info("\n".join(lines))

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

    raise RuntimeError("generate_rollout: unexpected fall-through in training path")


class _MobileAsyncRolloutWorker:
    def __init__(
        self,
        args: Any,
        data_source: Any,
        config: SlimeAdapterConfig,
        env_client: Any,
        max_concurrent_groups: int,
        traj_root: str | None = None,
    ) -> None:
        self.args = args
        self.data_source = data_source
        self.config = config
        self.env_client = env_client
        self.max_concurrent_groups = max_concurrent_groups
        self.traj_root = traj_root
        self.running = True
        self._paused = threading.Event()
        self._pause_ack = threading.Event()
        self.output_queue: queue.Queue = queue.Queue(maxsize=64)
        self.worker_thread: threading.Thread | None = None
        self._group_counter = 0

        self.current_rollout_id = 0

    def _pull_next_groups(self, n: int) -> list[tuple[str, list[Sample], int]]:

        n_per_prompt = max(1, getattr(self.args, "n_samples_per_prompt", 8))
        unique_tasks = _task_iterator.next_batch(self.data_source, n)
        if not unique_tasks:
            return []
        return _task_iterator.build_adaptive_groups(unique_tasks, n_per_prompt)

    async def _continuous_loop(self) -> None:

        pool_size = int(
            os.environ.get(
                "ROLLOUT_THREAD_POOL_SIZE",
                str(max(64, self.config.num_sessions)),
            )
        )
        loop = asyncio.get_running_loop()
        loop.set_default_executor(ThreadPoolExecutor(max_workers=pool_size))

        sem = asyncio.Semaphore(max(1, self.config.num_sessions))

        logger.info(
            "[AsyncWorker] started (pooled mode); num_sessions=%d, thread_pool=%d",
            self.config.num_sessions,
            pool_size,
        )
        _log_proc_mem("worker_start")

        _task_iterator.reset_launch_budget()

        group_results: list[list[list[Any] | None] | None] = []
        group_task_ids: list[str] = []
        group_total: list[int] = []
        group_completed: list[int] = []
        group_finalized: list[bool] = []

        pending_tasks: set[asyncio.Task] = set()

        def _add_group(task_id: str, copies: list[Any]) -> int:
            gid = len(group_results)
            group_results.append([None] * len(copies))
            group_task_ids.append(task_id)
            group_total.append(len(copies))
            group_completed.append(0)
            group_finalized.append(False)
            return gid

        async def _run_one(gid: int, sid: int, sample: Any) -> None:

            if self._paused.is_set() or not self.running:
                group_completed[gid] += 1
                _maybe_finalize(gid)
                return
            try:
                async with sem:
                    if self._paused.is_set() or not self.running:
                        group_completed[gid] += 1
                        _maybe_finalize(gid)
                        return
                    rid = self.current_rollout_id
                    result = await _run_single_task_mobile(
                        sample,
                        self.config,
                        self.env_client,
                        self.args,
                        concurrency_sem=asyncio.Semaphore(999),
                        evaluation=False,
                        attempt_id=f"r{rid:05d}_g{gid:05d}_s{sid:02d}",
                        traj_root=self.traj_root,
                        rollout_id=rid,
                    )
            except Exception:
                logger.exception("_run_one failed: gid=%d sid=%d", gid, sid)
                result = []

            group_results[gid][sid] = result
            group_completed[gid] += 1
            _maybe_finalize(gid)

        def _maybe_finalize(gid: int) -> None:

            if group_finalized[gid]:
                return
            if group_completed[gid] < group_total[gid]:
                return
            group_finalized[gid] = True
            _finalize_group(gid)

        def _finalize_group(gid: int) -> None:
            slot = group_results[gid]
            if slot is None:
                logger.warning(
                    "[AsyncWorker] _finalize_group(%d): slot already released, skipping",
                    gid,
                )
                return
            results = [r for r in slot if r is not None]
            task_id = group_task_ids[gid]
            n_launched = group_total[gid]
            n_pass = sum(
                1
                for r in results
                if not _is_env_failed(r)
                and float(_sample_metadata(r[0]).get("trajectory_reward", 0.0) or 0.0) > 0
            )
            n_valid = sum(1 for r in results if not _is_env_failed(r))
            if n_valid >= 1:
                _task_iterator.update_history(task_id, n_pass, n_valid)

            trainable = _is_trainable_group(results, n_launched=n_launched)
            logger.info(
                "[AsyncWorker] group %d (%s) done: %d pass / %d valid / %d total, trainable=%s",
                gid,
                task_id,
                n_pass,
                n_valid,
                n_launched,
                trainable,
            )

            try:
                self.output_queue.put((gid, results, n_launched), timeout=10)
            except queue.Full:
                logger.warning("[AsyncWorker] queue full, dropping group %d", gid)
            finally:
                group_results[gid] = None
                group_task_ids[gid] = ""

            del results

            _log_proc_mem(
                "worker_finalize",
                gid=gid,
                groups_alive=len(group_results),
                groups_held=sum(1 for g in group_results if g is not None),
                queue_size=self.output_queue.qsize(),
                siblings=n_launched,
            )

        n_groups_launched = 0
        target_pending = self.config.num_sessions

        while self.running:
            if self._paused.is_set():
                if pending_tasks:
                    logger.info("[AsyncWorker] pausing: cancelling %d tasks...", len(pending_tasks))
                    for t in pending_tasks:
                        t.cancel()
                    await asyncio.gather(*pending_tasks, return_exceptions=True)
                    pending_tasks.clear()

                for gid in range(len(group_results)):
                    if not group_finalized[gid]:
                        group_finalized[gid] = True
                        _finalize_group(gid)
                logger.info("[AsyncWorker] paused.")
                self._pause_ack.set()
                while self._paused.is_set() and self.running:
                    await asyncio.sleep(0.5)
                self._pause_ack.clear()

                group_results.clear()
                group_task_ids.clear()
                group_total.clear()
                group_completed.clear()
                group_finalized.clear()
                n_groups_launched = 0

                _task_iterator.reset_launch_budget()
                logger.info("[AsyncWorker] resumed.")
                continue

            n_pending = len(pending_tasks)
            if n_pending < target_pending and self.running:
                n_pull = max(
                    1,
                    (target_pending - n_pending)
                    // max(1, getattr(self.args, "n_samples_per_prompt", 8)),
                )
                n_pull = min(n_pull, 10)
                adaptive_groups = await asyncio.to_thread(self._pull_next_groups, n_pull)
                if adaptive_groups:
                    _task_iterator.log_schedule_summary(adaptive_groups)
                    for task_id, copies, _n in adaptive_groups:
                        gid = _add_group(task_id, copies)
                        n_groups_launched += 1
                        for sid, sample in enumerate(copies):
                            t = asyncio.create_task(_run_one(gid, sid, sample))
                            pending_tasks.add(t)
                            t.add_done_callback(pending_tasks.discard)

            if pending_tasks:
                done, _ = await asyncio.wait(
                    pending_tasks,
                    timeout=1.0,
                    return_when=asyncio.FIRST_COMPLETED,
                )

            else:
                await asyncio.sleep(1.0)

        if pending_tasks:
            logger.info("[AsyncWorker] draining %d tasks on exit", len(pending_tasks))
            for t in pending_tasks:
                t.cancel()
            await asyncio.gather(*pending_tasks, return_exceptions=True)
        for gid in range(len(group_results)):
            if not group_finalized[gid]:
                group_finalized[gid] = True
                _finalize_group(gid)

    def _thread_main(self) -> None:
        async def _run_and_close() -> None:
            try:
                await self._continuous_loop()
            finally:
                await oss_io.close_async_fs()

        try:
            asyncio.run(_run_and_close())
        except Exception:
            logger.exception("[AsyncWorker] thread crashed")

    def start(self) -> None:
        if self.worker_thread is None or not self.worker_thread.is_alive():
            self.worker_thread = threading.Thread(
                target=self._thread_main,
                daemon=True,
                name="MaiAsyncRolloutWorker",
            )
            self.worker_thread.start()

    def pause(self, timeout: float = 120.0) -> None:
        if self._paused.is_set():
            return
        logger.info("[AsyncWorker] requesting pause...")
        self._pause_ack.clear()
        self._paused.set()
        if not self._pause_ack.wait(timeout=timeout):
            logger.warning("[AsyncWorker] pause ack timed out after %.0fs", timeout)
        else:
            logger.info("[AsyncWorker] pause confirmed.")

    def resume(self) -> None:
        if not self._paused.is_set():
            return
        logger.info("[AsyncWorker] resuming...")
        self._paused.clear()

    def stop(self) -> None:
        self.running = False
        self._paused.clear()
        if self.worker_thread and self.worker_thread.is_alive():
            self.worker_thread.join(timeout=10)


def _count_trainable_groups(
    collected: list[tuple[int, list[list[Sample]], int]],
) -> int:

    return sum(1 for _, gr, nl in collected if _is_trainable_group(gr, n_launched=nl))


MAX_EXTRA_ROUNDS = int(os.environ.get("MAX_EXTRA_ROUNDS", "3"))

MAX_GROUPS_PER_ROLLOUT = int(os.environ.get("MAX_GROUPS_PER_ROLLOUT", "200"))
ROLLOUT_TIMEOUT_S = float(os.environ.get("ROLLOUT_TIMEOUT_S") or "1800")


DYNAMIC_TIMEOUT = os.environ.get("DYNAMIC_TIMEOUT", "0") == "1"
TIMEOUT_TRAJ_PER_HOUR = float(os.environ.get("TIMEOUT_TRAJ_PER_HOUR") or "1000")
_DYN_TIMEOUT_S: float | None = None


def _maybe_set_dynamic_timeout(adaptive_groups, config, rollout_id) -> None:

    global _DYN_TIMEOUT_S
    if not DYNAMIC_TIMEOUT or not adaptive_groups:
        _DYN_TIMEOUT_S = None
        return
    from collections import Counter

    by_env: Counter = Counter()
    for _tid, copies, n in adaptive_groups:
        if copies:
            et = _sample_metadata(copies[0]).get("env_type") or config.env_type
            by_env[et] += n
    total = sum(by_env.values())
    if total <= 0:
        _DYN_TIMEOUT_S = None
        return
    _DYN_TIMEOUT_S = total / TIMEOUT_TRAJ_PER_HOUR * 3600.0
    logger.info(
        "Rollout %d: DYNAMIC_TIMEOUT -> %.0fs (%.2fh) from total dispatched traj=%d "
        "(by_env=%s, %g traj/h)",
        rollout_id,
        _DYN_TIMEOUT_S,
        _DYN_TIMEOUT_S / 3600,
        total,
        dict(by_env),
        TIMEOUT_TRAJ_PER_HOUR,
    )


MIN_TRAINABLE_TASKS = int(os.environ.get("MIN_TRAINABLE_TASKS", "5"))


MIN_SAMPLES_PER_TASK = int(os.environ.get("MIN_SAMPLES_PER_TASK") or "6")


def _is_task_trainable(rewards: list[float], is_ppo: bool = False) -> bool:

    if is_ppo:
        return is_task_trainable_ppo(rewards)
    return len(rewards) >= MIN_SAMPLES_PER_TASK and len(set(rewards)) > 1


def _warn_if_undersized(
    args: Any,
    output: RolloutFnTrainOutput,
    rollout_id: int,
) -> None:

    gbs = int(getattr(args, "global_batch_size", 0) or 0)
    n_step_samples = int(output.metrics.get("rollout/num_step_samples", 0))
    n_traj = int(output.metrics.get("rollout/num_real_trajectories", 0))
    if gbs > 0 and 0 < n_step_samples < gbs:
        logger.warning(
            "Rollout %d: %d step samples from %d trajectories < "
            "global_batch_size %d — SLIME will trim to 0 and raise 'Not enough "
            "samples'. Raise ROLLOUT_BATCH_SIZE (more trajectories) or lower "
            "GLOBAL_BATCH_SIZE.",
            rollout_id,
            n_step_samples,
            n_traj,
            gbs,
        )


def _is_trainable_group(
    group_result: list[list[Any]],
    n_launched: int | None = None,
) -> bool:

    rewards: list[float] = []
    for sibling_steps in group_result:
        if not sibling_steps:
            continue
        meta = _sample_metadata(sibling_steps[0])
        if meta.get("env_failed"):
            continue
        rewards.append(float(meta.get("trajectory_reward", 0.0) or 0.0))
    if n_launched is not None:
        min_valid = max(3, n_launched * 2 // 3)
    else:
        min_valid = 3
    if len(rewards) < min_valid:
        return False
    return len(set(rewards)) > 1


def generate_rollout_async(
    args: Namespace,
    rollout_id: int,
    data_source: Any,
    evaluation: bool = False,
) -> RolloutFnTrainOutput | RolloutFnEvalOutput:

    if WorkerOptions.from_env().nodes != 0:
        raise ValueError("Distributed trajectory workers require synchronous pooled rollout")

    if evaluation:
        worker_ref = _mobile_async_worker
        if worker_ref is not None:
            logger.info("Eval %d: pausing async worker to free sessions...", rollout_id)
            worker_ref.pause(timeout=180.0)
        try:
            return generate_rollout(args, rollout_id, data_source, evaluation=True)
        finally:
            if worker_ref is not None:
                logger.info("Eval %d: resuming async worker.", rollout_id)
                worker_ref.resume()

    config = _get_config()
    env_client = _get_env_client(config)
    worker = _get_mobile_async_worker(args, data_source, config, env_client)

    worker.current_rollout_id = rollout_id

    _maybe_restore_scheduler_state(data_source)
    _task_iterator.advance_rollout_count()

    is_ppo = getattr(args, "advantage_estimator", "grpo") == "ppo"

    _log_proc_mem(
        "gen_async_enter",
        rollout_id=rollout_id,
        queue_size=worker.output_queue.qsize(),
    )

    collected: list[tuple[int, list[list[Sample]], int]] = []
    task_rewards: dict[str, list[float]] = {}
    trainable_tasks: set[str] = set()
    start = time.time()
    last_progress = start

    while True:
        elapsed = time.time() - start

        if len(trainable_tasks) >= MIN_TRAINABLE_TASKS:
            logger.info(
                "Rollout %d (async-mobile): %d trainable tasks (need %d), "
                "total %d groups, proceeding",
                rollout_id,
                len(trainable_tasks),
                MIN_TRAINABLE_TASKS,
                len(collected),
            )
            break

        if len(collected) >= MAX_GROUPS_PER_ROLLOUT or elapsed >= ROLLOUT_TIMEOUT_S:
            logger.warning(
                "Rollout %d (async-mobile): budget exhausted "
                "(groups=%d/%d, elapsed=%.0fs/%.0fs), "
                "%d/%d trainable tasks — stopping",
                rollout_id,
                len(collected),
                MAX_GROUPS_PER_ROLLOUT,
                elapsed,
                ROLLOUT_TIMEOUT_S,
                len(trainable_tasks),
                MIN_TRAINABLE_TASKS,
            )
            break

        if len(collected) % 10 == 0 and len(collected) > 0:
            logger.info(
                "Rollout %d (async-mobile): collecting... "
                "%d groups, %d/%d trainable tasks, elapsed %.0fs",
                rollout_id,
                len(collected),
                len(trainable_tasks),
                MIN_TRAINABLE_TASKS,
                elapsed,
            )

        try:
            item = worker.output_queue.get(timeout=30)
        except queue.Empty:
            since_last = time.time() - last_progress
            logger.warning(
                "Rollout %d (async-mobile): no group in %.0fs; collected=%d, trainable=%d/%d",
                rollout_id,
                since_last,
                len(collected),
                len(trainable_tasks),
                MIN_TRAINABLE_TASKS,
            )
            continue

        last_progress = time.time()
        collected.append(item)
        _gid, group_result, _nl = item

        task_id = "unknown"
        for sibling_steps in group_result:
            if not sibling_steps:
                continue
            meta = _sample_metadata(sibling_steps[0])
            task_id = str(meta.get("task_id") or "unknown")
            if not meta.get("env_failed"):
                reward = float(meta.get("trajectory_reward", 0.0) or 0.0)
                task_rewards.setdefault(task_id, []).append(reward)

        if task_id not in trainable_tasks:
            all_rewards = task_rewards.get(task_id, [])
            if _is_task_trainable(all_rewards, is_ppo=is_ppo):
                trainable_tasks.add(task_id)
                logger.info(
                    "Rollout %d (async-mobile): task %s trainable "
                    "[%d valid, %d pass], total: %d/%d",
                    rollout_id,
                    task_id,
                    len(all_rewards),
                    sum(1 for r in all_rewards if r > 0),
                    len(trainable_tasks),
                    MIN_TRAINABLE_TASKS,
                )

    while not worker.output_queue.empty():
        try:
            item = worker.output_queue.get_nowait()
            collected.append(item)

            for sibling_steps in item[1]:
                if not sibling_steps:
                    continue
                meta = _sample_metadata(sibling_steps[0])
                tid = str(meta.get("task_id") or "unknown")
                if not meta.get("env_failed"):
                    reward = float(meta.get("trajectory_reward", 0.0) or 0.0)
                    task_rewards.setdefault(tid, []).append(reward)
            if tid not in trainable_tasks and _is_task_trainable(
                task_rewards.get(tid, []), is_ppo=is_ppo
            ):
                trainable_tasks.add(tid)
        except queue.Empty:
            break

    _log_proc_mem(
        "gen_async_after_collect",
        rollout_id=rollout_id,
        collected=len(collected),
        queue_size=worker.output_queue.qsize(),
        trainable_tasks=len(trainable_tasks),
    )

    n_total_collected = len(collected)
    degrade_level = 0

    if len(trainable_tasks) < MIN_TRAINABLE_TASKS:
        for tid, rewards in task_rewards.items():
            if tid not in trainable_tasks and len(rewards) >= 2 and len(set(rewards)) > 1:
                trainable_tasks.add(tid)
        if trainable_tasks:
            degrade_level = 1
            logger.warning(
                "Rollout %d (async-mobile): degraded to min_valid=2, recovered %d trainable tasks",
                rollout_id,
                len(trainable_tasks),
            )

    if not trainable_tasks:
        for tid, rewards in task_rewards.items():
            if rewards:
                trainable_tasks.add(tid)
        if trainable_tasks:
            degrade_level = 2
            logger.warning(
                "Rollout %d (async-mobile): degraded to ANY valid data, recovered %d tasks",
                rollout_id,
                len(trainable_tasks),
            )

    if not trainable_tasks:
        degrade_level = 3
        logger.warning(
            "Rollout %d (async-mobile): 0 valid trajectories in %d groups! "
            "Including all data as last resort.",
            rollout_id,
            n_total_collected,
        )

    flat_step_samples: list[list[Sample]] = []
    task_count = 0
    if degrade_level < 3:
        for _gid, group_result, _nl in collected:
            for sibling_steps in group_result:
                if not sibling_steps:
                    continue
                meta = _sample_metadata(sibling_steps[0])
                tid = str(meta.get("task_id") or "unknown")
                if tid in trainable_tasks and not meta.get("env_failed"):
                    task_count += 1
                    flat_step_samples.append(sibling_steps)
    else:
        for _gid, group_result, _nl in collected:
            for sibling_steps in group_result:
                if sibling_steps:
                    task_count += 1
                    flat_step_samples.append(sibling_steps)

    logger.info(
        "Rollout %d (async-mobile): %d trainable tasks, %d step samples "
        "from %d groups (degrade_level=%d)",
        rollout_id,
        len(trainable_tasks),
        len(flat_step_samples),
        n_total_collected,
        degrade_level,
    )

    _upload_trained_manifest(
        env_client,
        config,
        args,
        rollout_id=rollout_id,
        degrade_level=degrade_level,
        num_groups=n_total_collected,
        trainable_task_ids=trainable_tasks,
        kept_task_ids=_kept_task_ids(flat_step_samples),
        task_rewards=task_rewards,
        num_trained_trajectories=len(flat_step_samples),
    )

    _log_proc_mem(
        "gen_async_after_flatten",
        rollout_id=rollout_id,
        flat_step_samples=len(flat_step_samples),
        collected=n_total_collected,
        degrade_level=degrade_level,
    )

    output = _build_train_output(
        flat_step_samples,
        task_count,
        rollout_id,
        label="async-mobile",
        elapsed=time.time() - start,
    )
    _warn_if_undersized(args, output, rollout_id)
    output.metrics["rollout/trainable_tasks"] = float(len(trainable_tasks))
    output.metrics["rollout/total_groups_collected"] = float(n_total_collected)
    output.metrics["rollout/degrade_level"] = float(degrade_level)
    output.metrics.update(aggregate_cache_metrics(flat_step_samples))
    _log_proc_mem(
        "gen_async_exit",
        rollout_id=rollout_id,
        queue_size=worker.output_queue.qsize(),
    )

    _stash_scheduler_state(data_source)
    return output


_mobile_async_worker_lock = threading.Lock()
_mobile_async_worker: _MobileAsyncRolloutWorker | None = None


def _get_mobile_async_worker(
    args: Namespace,
    data_source: Any,
    config: SlimeAdapterConfig,
    env_client: EnvClient,
) -> _MobileAsyncRolloutWorker:

    global _mobile_async_worker
    with _mobile_async_worker_lock:
        alive = (
            _mobile_async_worker is not None
            and _mobile_async_worker.worker_thread is not None
            and _mobile_async_worker.worker_thread.is_alive()
        )
        if not alive:
            n_samples = max(1, getattr(args, "n_samples_per_prompt", 8))
            max_groups = max(1, config.num_sessions // n_samples)
            traj_root = _get_traj_root(config)
            logger.info(
                "Spawning mobile-stepwise async rollout worker "
                "(sessions=%d, n_samples_per_prompt=%d, max_groups=%d, "
                "traj_root=%s)",
                config.num_sessions,
                n_samples,
                max_groups,
                traj_root,
            )
            _mobile_async_worker = _MobileAsyncRolloutWorker(
                args=args,
                data_source=data_source,
                config=config,
                env_client=env_client,
                max_concurrent_groups=max_groups,
                traj_root=traj_root,
            )
            _mobile_async_worker.start()
        return _mobile_async_worker


def _stop_mobile_async_worker() -> None:
    global _mobile_async_worker
    with _mobile_async_worker_lock:
        if _mobile_async_worker is not None:
            _mobile_async_worker.stop()
            _mobile_async_worker = None


atexit.register(_stop_mobile_async_worker)
