from __future__ import annotations

import asyncio
import atexit
import logging
import os
import queue
import threading
import time
from argparse import Namespace
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from slime.rollout.base_types import RolloutFnEvalOutput, RolloutFnTrainOutput
from slime.utils.types import Sample

from mai_gym.io import oss_io

from . import rollout_mobile_stepwise as _sw
from .config import SlimeAdapterConfig
from .helpers.rollout_metrics import _build_train_output, _sample_metadata
from .rollout import _get_config, _get_env_client

logger = logging.getLogger(__name__)


MAX_WEIGHT_STALENESS = int(os.environ.get("MAX_WEIGHT_STALENESS", "4") or "4")
BUDGET_REFRESH_S = float(os.environ.get("BUDGET_REFRESH_S", "60") or "60")
FULLY_ASYNC_QUEUE_SIZE = int(os.environ.get("FULLY_ASYNC_QUEUE_SIZE", "2000") or "2000")
HARVEST_TIMEOUT_S = float(os.environ.get("ROLLOUT_TIMEOUT_S", "2400") or "2400")


MAX_CONCURRENT_GROUPS = int(os.environ.get("MAX_CONCURRENT_GROUPS", "8") or "8")


class _FullyAsyncStepwiseWorker:
    def __init__(
        self,
        args: Namespace,
        data_source: Any,
        config: SlimeAdapterConfig,
        env_client: Any,
        traj_root: str | None = None,
    ) -> None:
        self.args = args
        self.data_source = data_source
        self.config = config
        self.env_client = env_client
        self.traj_root = traj_root
        self.running = True
        self.output_queue: queue.Queue = queue.Queue(maxsize=FULLY_ASYNC_QUEUE_SIZE)
        self.worker_thread: threading.Thread | None = None

        self.current_version: int = 0
        self._total_groups = 0

    def _pull_next_groups(self, n: int) -> list[tuple[str, list[Sample], int]]:
        n_per_prompt = max(1, getattr(self.args, "n_samples_per_prompt", 8))
        unique_tasks = _sw._task_iterator.next_batch(self.data_source, n)
        if not unique_tasks:
            return []
        return _sw._task_iterator.build_adaptive_groups(unique_tasks, n_per_prompt)

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
            "[FullyAsyncWorker] started; num_sessions=%d thread_pool=%d "
            "max_staleness=%d budget_refresh=%.0fs queue=%d max_concurrent_groups=%d",
            self.config.num_sessions,
            pool_size,
            MAX_WEIGHT_STALENESS,
            BUDGET_REFRESH_S,
            FULLY_ASYNC_QUEUE_SIZE,
            MAX_CONCURRENT_GROUPS,
        )
        _sw._task_iterator.reset_launch_budget()
        last_budget_refresh = time.time()

        group_results: list[list[list[Any] | None] | None] = []
        group_task_ids: list[str] = []
        group_total: list[int] = []
        group_completed: list[int] = []
        group_finalized: list[bool] = []

        group_launch_version: list[int] = []
        pending_tasks: set[asyncio.Task] = set()
        inflight = [0]

        def _add_group(task_id: str, copies: list[Any]) -> int:
            gid = len(group_results)
            group_results.append([None] * len(copies))
            group_task_ids.append(task_id)
            group_total.append(len(copies))
            group_completed.append(0)
            group_finalized.append(False)
            group_launch_version.append(self.current_version)
            inflight[0] += 1
            return gid

        def _finalize_group(gid: int) -> None:
            slot = group_results[gid]
            if slot is None:
                return
            results = [r for r in slot if r is not None]
            task_id = group_task_ids[gid]
            n_launched = group_total[gid]
            n_pass = sum(
                1
                for r in results
                if not _sw._is_env_failed(r)
                and float(_sample_metadata(r[0]).get("trajectory_reward", 0.0) or 0.0) > 0
            )
            n_valid = sum(1 for r in results if not _sw._is_env_failed(r))
            if n_valid >= 1:
                _sw._task_iterator.update_history(task_id, n_pass, n_valid)

            group_version = group_launch_version[gid]
            try:
                self.output_queue.put((gid, results, n_launched, group_version), timeout=10)
            except queue.Full:
                logger.warning("[FullyAsyncWorker] queue full, dropping group %d", gid)
            finally:
                group_results[gid] = None
                group_task_ids[gid] = ""
                inflight[0] -= 1
            del results

        def _maybe_finalize(gid: int) -> None:
            if group_finalized[gid] or group_completed[gid] < group_total[gid]:
                return
            group_finalized[gid] = True
            _finalize_group(gid)

        async def _run_one(gid: int, sid: int, sample: Any) -> None:
            if not self.running:
                group_completed[gid] += 1
                _maybe_finalize(gid)
                return
            try:
                async with sem:
                    result = await _sw._run_single_task_mobile(
                        sample,
                        self.config,
                        self.env_client,
                        self.args,
                        concurrency_sem=asyncio.Semaphore(999),
                        evaluation=False,
                        attempt_id=f"g{gid:05d}_s{sid:02d}",
                        traj_root=self.traj_root,
                        rollout_id=gid,
                        epoch_getter=lambda: self.current_version,
                    )
            except Exception:
                logger.exception("[FullyAsyncWorker] _run_one failed gid=%d sid=%d", gid, sid)
                result = []
            group_results[gid][sid] = result
            group_completed[gid] += 1
            _maybe_finalize(gid)

        target_pending = self.config.num_sessions
        while self.running:
            now = time.time()
            if now - last_budget_refresh >= BUDGET_REFRESH_S:
                _sw._task_iterator.reset_launch_budget()
                last_budget_refresh = now

            if (
                inflight[0] < MAX_CONCURRENT_GROUPS
                and len(pending_tasks) < target_pending
                and self.running
            ):
                n_pull = min(MAX_CONCURRENT_GROUPS - inflight[0], 10)
                adaptive_groups = await asyncio.to_thread(self._pull_next_groups, n_pull)
                if adaptive_groups:
                    _sw._task_iterator.log_schedule_summary(adaptive_groups)
                    for task_id, copies, _n in adaptive_groups:
                        gid = _add_group(task_id, copies)
                        self._total_groups += 1
                        for sid, sample in enumerate(copies):
                            t = asyncio.create_task(_run_one(gid, sid, sample))
                            pending_tasks.add(t)
                            t.add_done_callback(pending_tasks.discard)

            if pending_tasks:
                await asyncio.wait(
                    pending_tasks,
                    timeout=1.0,
                    return_when=asyncio.FIRST_COMPLETED,
                )
            else:
                await asyncio.sleep(1.0)

        for t in pending_tasks:
            t.cancel()
        if pending_tasks:
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
            logger.exception("[FullyAsyncWorker] thread crashed")

    def start(self) -> None:
        if self.worker_thread is None or not self.worker_thread.is_alive():
            self.worker_thread = threading.Thread(
                target=self._thread_main,
                daemon=True,
                name="FullyAsyncStepwiseWorker",
            )
            self.worker_thread.start()

    def stop(self) -> None:
        self.running = False
        if self.worker_thread and self.worker_thread.is_alive():
            self.worker_thread.join(timeout=10)

    def drain_completed(self) -> list[tuple[int, list[list[Any]], int, int]]:
        out = []
        while True:
            try:
                out.append(self.output_queue.get_nowait())
            except queue.Empty:
                break
        return out


_worker_lock = threading.Lock()
_fully_async_worker: _FullyAsyncStepwiseWorker | None = None


def _get_worker(
    args: Namespace,
    data_source: Any,
    config: SlimeAdapterConfig,
    env_client: Any,
) -> _FullyAsyncStepwiseWorker:
    global _fully_async_worker
    with _worker_lock:
        alive = (
            _fully_async_worker is not None
            and _fully_async_worker.worker_thread is not None
            and _fully_async_worker.worker_thread.is_alive()
        )
        if not alive:
            traj_root = _sw._get_traj_root(config)
            logger.info("Spawning fully-async stepwise worker (sessions=%d)", config.num_sessions)
            _fully_async_worker = _FullyAsyncStepwiseWorker(
                args=args,
                data_source=data_source,
                config=config,
                env_client=env_client,
                traj_root=traj_root,
            )
            _fully_async_worker.start()
        return _fully_async_worker


def _stop_worker() -> None:
    global _fully_async_worker
    with _worker_lock:
        if _fully_async_worker is not None:
            _fully_async_worker.stop()
            _fully_async_worker = None


atexit.register(_stop_worker)


def generate_rollout_fully_async(
    args: Namespace,
    rollout_id: int,
    data_source: Any,
    evaluation: bool = False,
) -> RolloutFnTrainOutput | RolloutFnEvalOutput:

    if evaluation:
        return _sw.generate_rollout(args, rollout_id, data_source, evaluation=True)

    config = _get_config()
    env_client = _get_env_client(config)
    worker = _get_worker(args, data_source, config, env_client)
    _sw._task_iterator.advance_rollout_count()

    worker.current_version = rollout_id

    collected: list[tuple[int, list[list[Sample]], int, int]] = []
    task_rewards: dict[str, list[float]] = {}
    trainable_tasks: set[str] = set()
    n_dropped_stale = 0
    staleness_seen: list[int] = []
    start = time.time()
    last_progress = start

    def _accumulate(item: tuple[int, list[list[Sample]], int, int]) -> None:
        nonlocal n_dropped_stale
        _gid, group_result, _nl, group_version = item

        filtered_group: list[list[Sample]] = []
        for sibling_steps in group_result:
            if not sibling_steps:
                continue

            if _sample_metadata(sibling_steps[0]).get("env_failed"):
                filtered_group.append(sibling_steps)
                continue
            kept: list[Sample] = []
            for s in sibling_steps:
                ge = _sample_metadata(s).get("gen_epoch")
                ge = int(ge) if ge is not None else int(group_version)
                st = rollout_id - ge
                staleness_seen.append(st)
                if st > MAX_WEIGHT_STALENESS:
                    n_dropped_stale += 1
                    continue
                kept.append(s)
            if kept:
                filtered_group.append(kept)
        if not filtered_group:
            return
        collected.append((_gid, filtered_group, _nl, group_version))
        task_id = "unknown"
        for sibling_steps in filtered_group:
            if not sibling_steps:
                continue
            meta = _sample_metadata(sibling_steps[0])
            task_id = str(meta.get("task_id") or "unknown")
            if not meta.get("env_failed"):
                task_rewards.setdefault(task_id, []).append(
                    float(meta.get("trajectory_reward", 0.0) or 0.0)
                )

        if task_id not in trainable_tasks and _sw._is_task_trainable(task_rewards.get(task_id, [])):
            trainable_tasks.add(task_id)

    gbs = int(getattr(args, "global_batch_size", 0) or 0)

    def _emittable_sample_count() -> int:

        n = 0
        for _gid, group_result, _nl, _gv in collected:
            for sibling_steps in group_result:
                if not sibling_steps:
                    continue
                meta = _sample_metadata(sibling_steps[0])
                tid = str(meta.get("task_id") or "unknown")
                if tid in trainable_tasks and not meta.get("env_failed"):
                    n += len(sibling_steps)
        return n

    def _valid_sample_count() -> int:

        n = 0
        for _gid, group_result, _nl, _gv in collected:
            for sibling_steps in group_result:
                if sibling_steps and not _sample_metadata(sibling_steps[0]).get("env_failed"):
                    n += len(sibling_steps)
        return n

    logger.info(
        "Rollout %d (fully-async): harvesting epoch=%d max_stale=%d need=%d trainable, "
        "global_batch_size=%d",
        rollout_id,
        rollout_id,
        MAX_WEIGHT_STALENESS,
        _sw.MIN_TRAINABLE_TASKS,
        gbs,
    )

    while True:
        elapsed = time.time() - start
        n_emit = _emittable_sample_count()
        enough_tasks = len(trainable_tasks) >= _sw.MIN_TRAINABLE_TASKS

        enough_samples = n_emit >= gbs if gbs > 0 else n_emit >= _sw.MIN_SAMPLES_PER_TASK
        if enough_tasks and enough_samples:
            logger.info(
                "Rollout %d (fully-async): %d trainable tasks (need %d), %d emittable "
                "step-samples (need %d), %d groups — proceeding",
                rollout_id,
                len(trainable_tasks),
                _sw.MIN_TRAINABLE_TASKS,
                n_emit,
                gbs,
                len(collected),
            )
            break

        n_valid = sum(len(r) for r in task_rewards.values())
        hard_cap = _sw.MAX_GROUPS_PER_ROLLOUT * 10
        can_build = _valid_sample_count() > 0
        if can_build and (elapsed >= HARVEST_TIMEOUT_S or len(collected) >= hard_cap):
            logger.warning(
                "Rollout %d (fully-async): stopping harvest under target (groups=%d "
                "emittable=%d/%d valid_traj=%d elapsed=%.0fs) — will pad to batch",
                rollout_id,
                len(collected),
                n_emit,
                gbs,
                n_valid,
                elapsed,
            )
            break
        if not can_build and elapsed >= HARVEST_TIMEOUT_S:
            logger.error(
                "Rollout %d (fully-async): %.0fs elapsed with ZERO valid trajectories "
                "(env outage?) — NOT exiting; pausing until the env recovers (worker "
                "keeps retrying). collected=%d",
                rollout_id,
                elapsed,
                len(collected),
            )

        try:
            item = worker.output_queue.get(timeout=30)
        except queue.Empty:
            logger.warning(
                "Rollout %d (fully-async): no group in %.0fs; collected=%d emittable=%d "
                "trainable=%d/%d (worker keeps generating)",
                rollout_id,
                time.time() - last_progress,
                len(collected),
                n_emit,
                len(trainable_tasks),
                _sw.MIN_TRAINABLE_TASKS,
            )
            continue
        last_progress = time.time()
        _accumulate(item)

    for item in worker.drain_completed():
        _accumulate(item)

    degrade_level = 0
    if len(trainable_tasks) < _sw.MIN_TRAINABLE_TASKS:
        for tid, rewards in task_rewards.items():
            if tid not in trainable_tasks and len(rewards) >= 2 and len(set(rewards)) > 1:
                trainable_tasks.add(tid)
        if trainable_tasks:
            degrade_level = 1
    if not trainable_tasks:
        for tid, rewards in task_rewards.items():
            if rewards:
                trainable_tasks.add(tid)
        if trainable_tasks:
            degrade_level = 2
    if not trainable_tasks:
        degrade_level = 3

    flat_step_samples: list[list[Sample]] = []
    task_count = 0
    for _gid, group_result, _nl, _gv in collected:
        for sibling_steps in group_result:
            if not sibling_steps:
                continue
            meta = _sample_metadata(sibling_steps[0])
            tid = str(meta.get("task_id") or "unknown")
            if degrade_level >= 3 or (tid in trainable_tasks and not meta.get("env_failed")):
                task_count += 1
                flat_step_samples.append(sibling_steps)

    if gbs > 0 and flat_step_samples:
        total_steps = sum(len(t) for t in flat_step_samples)
        if total_steps < gbs:
            logger.warning(
                "Rollout %d (fully-async): under-filled batch (%d step-samples < "
                "global_batch_size %d) — padding by repeating %d trajectories",
                rollout_id,
                total_steps,
                gbs,
                len(flat_step_samples),
            )
            src = list(flat_step_samples)
            i = 0
            while sum(len(t) for t in flat_step_samples) < gbs:
                flat_step_samples.append(src[i % len(src)])
                task_count += 1
                i += 1

    output = _build_train_output(
        flat_step_samples,
        task_count,
        rollout_id,
        label="fully-async",
        elapsed=time.time() - start,
    )
    _sw._warn_if_undersized(args, output, rollout_id)
    output.metrics["rollout/trainable_tasks"] = float(len(trainable_tasks))
    output.metrics["rollout/total_groups_collected"] = float(len(collected))
    output.metrics["rollout/degrade_level"] = float(degrade_level)

    output.metrics["rollout/policy_epoch"] = float(rollout_id)

    output.metrics["rollout/dropped_stale_groups"] = float(n_dropped_stale)
    output.metrics["rollout/dropped_stale_samples"] = float(n_dropped_stale)
    if staleness_seen:
        output.metrics["rollout/weight_staleness_mean"] = float(
            sum(staleness_seen) / len(staleness_seen)
        )
        output.metrics["rollout/weight_staleness_max"] = float(max(staleness_seen))
    n_emitted = sum(len(t) for t in flat_step_samples)
    n_stamped = sum(
        1 for t in flat_step_samples for s in t if _sample_metadata(s).get("gen_epoch") is not None
    )
    output.metrics["rollout/gen_epoch_coverage"] = (
        float(n_stamped / n_emitted) if n_emitted else 0.0
    )
    output.metrics["rollout/worker_queue_size"] = float(worker.output_queue.qsize())
    logger.info(
        "Rollout %d (fully-async): %d step samples, %d trainable tasks, "
        "dropped_stale_steps=%d, staleness_max=%s (per-step), gen_epoch_cov=%.2f, "
        "queue=%d, degrade=%d",
        rollout_id,
        len(flat_step_samples),
        len(trainable_tasks),
        n_dropped_stale,
        max(staleness_seen) if staleness_seen else "-",
        (n_stamped / n_emitted) if n_emitted else 0.0,
        worker.output_queue.qsize(),
        degrade_level,
    )
    return output
