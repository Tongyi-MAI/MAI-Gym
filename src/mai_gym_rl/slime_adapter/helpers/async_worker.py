from __future__ import annotations

import asyncio
import atexit
import logging
import queue
import threading
from argparse import Namespace
from typing import Any

from slime.utils.types import Sample

from mai_gym.client.env_client import EnvClient
from mai_gym.io import oss_io

from ..config import SlimeAdapterConfig

logger = logging.getLogger(__name__)

_async_worker_lock = threading.Lock()
_async_worker: _AsyncRolloutWorker | None = None


class _AsyncRolloutWorker:
    def __init__(
        self,
        args: Namespace,
        data_source: Any,
        config: SlimeAdapterConfig,
        env_client: EnvClient,
        max_concurrent_groups: int,
    ) -> None:
        self.args = args
        self.data_source = data_source
        self.config = config
        self.env_client = env_client
        self.max_concurrent_groups = max_concurrent_groups
        self.running = True
        self._paused = threading.Event()
        self._pause_ack = threading.Event()
        self.output_queue: queue.Queue = queue.Queue(maxsize=64)
        self.worker_thread: threading.Thread | None = None
        self._group_counter = 0

        self.current_rollout_id = 0

    async def _run_group(
        self,
        group_id: int,
        group: list[Sample],
        concurrency_sem: asyncio.Semaphore,
    ) -> tuple[int, list[list[Sample]]]:
        from .rollout import _ensure_task_sets_registered, _run_single_task

        await asyncio.to_thread(
            _ensure_task_sets_registered,
            group,
            self.env_client,
        )

        rid = self.current_rollout_id
        coros = [
            _run_single_task(
                s,
                self.config,
                self.env_client,
                self.args,
                concurrency_sem=concurrency_sem,
                evaluation=False,
                attempt_id=f"r{rid:05d}_g{group_id:05d}_s{i:02d}",
                rollout_id=rid,
            )
            for i, s in enumerate(group)
        ]
        result = await asyncio.gather(*coros, return_exceptions=False)
        return group_id, result

    async def _continuous_loop(self) -> None:
        sem = asyncio.Semaphore(max(1, self.config.num_sessions))
        logger.info(
            "[AsyncRolloutWorker] started; max_concurrent_groups=%d, concurrent_trajectories=%d",
            self.max_concurrent_groups,
            self.config.num_sessions,
        )
        active: set[asyncio.Task] = set()
        while self.running:
            if self._paused.is_set():
                if active:
                    logger.info(
                        "[AsyncRolloutWorker] pausing: draining %d in-flight group(s)...",
                        len(active),
                    )
                    done_results = await asyncio.gather(
                        *active,
                        return_exceptions=True,
                    )
                    for r in done_results:
                        if isinstance(r, Exception):
                            logger.exception("[AsyncRolloutWorker] group failed during drain")
                        else:
                            group_id, group_result = r
                            try:
                                self.output_queue.put((group_id, group_result), timeout=30)
                            except queue.Full:
                                logger.warning(
                                    "[AsyncRolloutWorker] output_queue full during drain, "
                                    "dropping group %d",
                                    group_id,
                                )
                    active.clear()
                logger.info("[AsyncRolloutWorker] paused.")
                self._pause_ack.set()

                while self._paused.is_set() and self.running:
                    await asyncio.sleep(0.5)
                self._pause_ack.clear()
                logger.info("[AsyncRolloutWorker] resumed.")
                continue

            done = {t for t in active if t.done()}
            for t in done:
                try:
                    group_id, group_result = t.result()
                    self.output_queue.put((group_id, group_result), timeout=30)
                except Exception:
                    logger.exception(
                        "[AsyncRolloutWorker] group task failed",
                    )
            active -= done

            while len(active) < self.max_concurrent_groups and self.running:
                try:
                    groups = await asyncio.to_thread(
                        self.data_source.get_samples,
                        1,
                    )
                except Exception:
                    logger.exception(
                        "[AsyncRolloutWorker] data_source.get_samples failed",
                    )
                    await asyncio.sleep(1)
                    break
                if not groups:
                    break
                group = groups[0]
                gid = self._group_counter
                self._group_counter += 1
                active.add(asyncio.create_task(self._run_group(gid, group, sem)))

            await asyncio.sleep(0.5)

        if active:
            logger.info(
                "[AsyncRolloutWorker] draining %d in-flight group(s) on exit",
                len(active),
            )
            await asyncio.wait(active)

    def _thread_main(self) -> None:
        async def _run_and_close() -> None:
            try:
                await self._continuous_loop()
            finally:
                await oss_io.close_async_fs()

        try:
            asyncio.run(_run_and_close())
        except Exception:
            logger.exception("[AsyncRolloutWorker] thread crashed")

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
        logger.info("[AsyncRolloutWorker] requesting pause...")
        self._pause_ack.clear()
        self._paused.set()
        if not self._pause_ack.wait(timeout=timeout):
            logger.warning(
                "[AsyncRolloutWorker] pause acknowledgment timed out after %.0fs",
                timeout,
            )
        else:
            logger.info("[AsyncRolloutWorker] pause confirmed.")

    def resume(self) -> None:

        if not self._paused.is_set():
            return
        logger.info("[AsyncRolloutWorker] resuming...")
        self._paused.clear()

    def stop(self) -> None:
        self.running = False
        self._paused.clear()
        if self.worker_thread and self.worker_thread.is_alive():
            self.worker_thread.join(timeout=10)


def _get_async_worker(
    args: Namespace,
    data_source: Any,
    config: SlimeAdapterConfig,
    env_client: EnvClient,
) -> _AsyncRolloutWorker:
    global _async_worker
    with _async_worker_lock:
        thread_alive = (
            _async_worker is not None
            and _async_worker.worker_thread is not None
            and _async_worker.worker_thread.is_alive()
        )
        if not thread_alive:
            n_samples = max(1, getattr(args, "n_samples_per_prompt", 8))
            max_groups = max(1, config.num_sessions // n_samples)
            logger.info(
                "Spawning async rollout worker (sessions=%d, "
                "n_samples_per_prompt=%d, max_groups_in_flight=%d)",
                config.num_sessions,
                n_samples,
                max_groups,
            )
            _async_worker = _AsyncRolloutWorker(
                args=args,
                data_source=data_source,
                config=config,
                env_client=env_client,
                max_concurrent_groups=max_groups,
            )
            _async_worker.start()
        return _async_worker


def _stop_async_worker() -> None:
    global _async_worker
    with _async_worker_lock:
        if _async_worker is not None:
            _async_worker.stop()
            _async_worker = None


atexit.register(_stop_async_worker)
