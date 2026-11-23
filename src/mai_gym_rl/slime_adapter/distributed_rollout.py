from __future__ import annotations

import asyncio
import contextlib
import copy
import logging
import math
import os
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)


class DistributedRolloutError(RuntimeError):
    pass


@dataclass(frozen=True)
class WorkerOptions:
    nodes: int = 0
    http_limit_per_worker: int = 64
    oss_limit_per_worker: int = 256
    cpus: float = 2
    threads: int = 8
    startup_timeout: float = 300
    shutdown_timeout: float = 120

    @classmethod
    def from_env(cls) -> WorkerOptions:
        nodes = int(os.environ.get("MAI_GYM_ROLLOUT_WORKERS") or "0")
        if nodes == 0:
            return cls()
        if os.environ.get("MAI_GYM_DISTRIBUTED_HTTP_BUDGET"):
            logger.warning(
                "MAI_GYM_DISTRIBUTED_HTTP_BUDGET is a legacy alias, now interpreted "
                "per worker/pool; use MAI_GYM_DISTRIBUTED_HTTP_LIMIT_PER_WORKER"
            )
        return cls(
            nodes=nodes,
            http_limit_per_worker=int(
                os.environ.get("MAI_GYM_DISTRIBUTED_HTTP_LIMIT_PER_WORKER")
                or os.environ.get("MAI_GYM_DISTRIBUTED_HTTP_BUDGET")
                or os.environ.get("MAI_GYM_MAX_INFLIGHT_REQ")
                or "64"
            ),
            oss_limit_per_worker=int(os.environ.get("MAI_GYM_MAX_INFLIGHT_OSS_GET") or "256"),
            cpus=float(os.environ.get("MAI_GYM_ROLLOUT_WORKER_CPUS") or "2"),
            threads=int(os.environ.get("MAI_GYM_ROLLOUT_WORKER_THREADS") or "8"),
        )

    def validate(self, config: Any, args: Any) -> None:
        if (
            self.nodes < -1
            or self.http_limit_per_worker < 1
            or self.oss_limit_per_worker < 1
            or not math.isfinite(self.cpus)
            or self.cpus <= 0
            or self.threads < 1
            or self.startup_timeout <= 0
            or self.shutdown_timeout <= 0
        ):
            raise ValueError("Invalid distributed rollout worker/resource/HTTP budget")
        if getattr(config, "env_client_shards", 1) != 1:
            raise ValueError("Distributed rollout requires env_client_shards=1")
        if getattr(args, "use_distributed_post", False):
            raise ValueError("Distributed rollout cannot nest distributed POST actors")
        if os.environ.get("MAI_GYM_FAKE_REWARD_SPLIT") == "1":
            raise ValueError("Process-local fake reward splitting is not distributed-safe")


@dataclass(frozen=True)
class BorrowedPool:
    url: str
    request_id: str
    screenshot_transport: str | None


@dataclass(frozen=True)
class WorkerSpec:
    node_id: str
    trajectories: int
    http_limits: dict[str, int]
    oss_limit: int = 1


def worker_specs(
    node_ids: list[str],
    options: WorkerOptions,
    concurrency: int,
    pools: list[str],
) -> list[WorkerSpec]:
    nodes = sorted(set(node_ids))
    if not nodes or concurrency < 0 or not pools:
        raise ValueError("Need actual rollout nodes, nonnegative concurrency and pools")
    count = len(nodes) if options.nodes == -1 else options.nodes
    count = min(count, len(nodes))
    if concurrency:
        count = min(count, concurrency)
    if count < 1 or options.http_limit_per_worker < 1:
        raise ValueError("Distributed worker count and HTTP limit must be positive")
    base, extra = divmod(concurrency, count)
    return [
        WorkerSpec(
            node,
            base + (i < extra),
            {pool: options.http_limit_per_worker for pool in sorted(set(pools))},
            options.oss_limit_per_worker,
        )
        for i, node in enumerate(nodes[:count])
    ]


async def _drain_cleanup(task: asyncio.Task) -> Any:
    while True:
        try:
            return await asyncio.shield(task)
        except asyncio.CancelledError:
            if task.done():
                return task.result()


def _borrowed_client(pool: BorrowedPool, cap: int) -> Any:
    from mai_gym.client.env_client import APIError, EnvClient

    class CheckedBorrower(EnvClient):
        cleanup_uncertain = False

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self._acquire_states = {}
            self.cleanup_failures = {}

        def _mark_cleanup_uncertain(self, operation, exc=None):
            self.cleanup_uncertain = True
            reason = operation
            if exc is not None:
                reason += f":{type(exc).__name__}"
                status = getattr(exc, "status_code", None)
                if isinstance(status, int):
                    reason += f":http_{status}"
            self.cleanup_failures[reason] = self.cleanup_failures.get(reason, 0) + 1

        async def _apost(self, path, *args, **kwargs):
            state = self._acquire_states.get(asyncio.current_task())
            if path != "/acquire" or state is None:
                return await super()._apost(path, *args, **kwargs)
            if state["cancel_requested"]:
                state["confirmed"] = not state["legacy_ambiguous"]
                raise asyncio.CancelledError
            try:
                return await super()._apost(path, *args, **kwargs)
            except Exception as exc:
                fields = (
                    exc.body if isinstance(exc, APIError) and isinstance(exc.body, dict) else {}
                )
                detail = fields.get("detail")
                no_lease = (
                    isinstance(exc, APIError)
                    and exc.status_code == 408
                    and isinstance(detail, str)
                    and detail.startswith("No session available within ")
                    and " — pool: " in detail
                )
                state["legacy_ambiguous"] |= not no_lease
                if state["cancel_requested"]:
                    state["confirmed"] = not state["legacy_ambiguous"]
                    raise asyncio.CancelledError from exc
                raise

        async def aacquire(self, *args, **kwargs):
            pending = asyncio.create_task(super().aacquire(*args, **kwargs))
            state = {
                "cancel_requested": False,
                "confirmed": False,
                "legacy_ambiguous": False,
            }
            self._acquire_states[pending] = state
            try:
                return await asyncio.shield(pending)
            except asyncio.CancelledError:
                state["cancel_requested"] = True
                try:
                    response = await _drain_cleanup(pending)
                    await _drain_cleanup(
                        asyncio.create_task(
                            self.arelease(response.session_id, response.lease_id),
                        )
                    )
                    if state["legacy_ambiguous"]:
                        self._mark_cleanup_uncertain("cancelled_acquire_prior_ambiguous_response")
                except BaseException as exc:
                    if not state["confirmed"]:
                        self._mark_cleanup_uncertain("cancelled_acquire", exc)
                        logger.exception("Cancelled acquire could not confirm lease cleanup")
                raise
            finally:
                self._acquire_states.pop(pending, None)

        async def _checked_cleanup(self, operation, name):
            pending = asyncio.create_task(operation)
            try:
                return await asyncio.shield(pending)
            except asyncio.CancelledError:
                try:
                    await _drain_cleanup(pending)
                except BaseException as exc:
                    self._mark_cleanup_uncertain(name, exc)
                    logger.exception("Cancelled session cleanup did not complete")
                raise
            except BaseException as exc:
                self._mark_cleanup_uncertain(name, exc)
                logger.exception("Session cleanup request failed")
                raise

        async def arelease(self, *args, **kwargs):
            return await self._checked_cleanup(super().arelease(*args, **kwargs), "release")

        async def adiscard(self, *args, **kwargs):
            return await self._checked_cleanup(super().adiscard(*args, **kwargs), "discard")

    return CheckedBorrower(
        host=pool.url,
        fleet_request_id=pool.request_id,
        max_inflight_requests=cap,
        max_async_connections=max(2, 2 * cap),
        screenshot_transport=pool.screenshot_transport,
    )


class TrajectoryWorker:
    def __init__(
        self,
        args: Any,
        config: Any,
        spec: WorkerSpec,
        pools: dict[str, BorrowedPool],
        options: WorkerOptions,
        trajectory_timeout: float | None,
    ) -> None:
        logging.basicConfig(
            level=logging.INFO,
            format="[%(asctime)s] %(filename)s:%(lineno)d - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
            force=True,
        )
        self.args = copy.copy(args)
        self.config = config
        self.spec = spec
        self.pools = pools
        self.options = options
        self.trajectory_timeout = trajectory_timeout
        self.clients: dict[str, Any] = {}
        self.tasks: set[asyncio.Task] = set()
        self.seen: set[str] = set()
        self.closing = False
        self.initialized = False

    async def ready(self) -> dict:
        if self.initialized:
            raise RuntimeError("Worker initialized twice")
        import ray

        actual_node = ray.get_runtime_context().get_node_id()
        if actual_node != self.spec.node_id:
            raise RuntimeError("Trajectory worker placed on an unexpected node")

        loop = asyncio.get_running_loop()
        if not isinstance(loop, asyncio.SelectorEventLoop):
            raise RuntimeError(
                "Distributed rollout requires a SelectorEventLoop; "
                "check this Ray version supports RAY_USE_UVLOOP=0"
            )
        loop.set_default_executor(ThreadPoolExecutor(max_workers=self.options.threads))
        self.sem = (
            asyncio.Semaphore(self.spec.trajectories)
            if self.spec.trajectories
            else contextlib.nullcontext()
        )
        for env_type, cap in self.spec.http_limits.items():
            pool = self.pools[env_type]
            self.clients[env_type] = _borrowed_client(pool, cap)
        await self._initialize_model()
        self.initialized = True
        return {
            "node_id": self.spec.node_id,
            "pid": os.getpid(),
            "trajectories": self.spec.trajectories,
            "http_limits": self.spec.http_limits,
            "oss_limit": self.spec.oss_limit,
            "event_loop": type(loop).__name__,
        }

    async def _initialize_model(self) -> None:
        from slime.utils.http_utils import init_http_client

        from . import rollout_mobile_stepwise as stepwise

        self.args.use_distributed_post = False
        init_http_client(self.args)
        stepwise._DYN_TIMEOUT_S = self.trajectory_timeout
        stepwise._traj_save_pool.shutdown(wait=True)
        stepwise._traj_save_pool = ThreadPoolExecutor(
            max_workers=self.options.threads,
            thread_name_prefix="traj_save",
        )
        stepwise._get_generate_state(self.args)
        self.run_trajectory = stepwise._run_single_task_mobile

    async def run(self, work_id: str, sample: Any, kwargs: dict) -> Any:
        if not self.initialized or self.closing:
            raise RuntimeError("Worker is not accepting work")
        if work_id in self.seen:
            raise RuntimeError("Duplicate trajectory dispatch")
        self.seen.add(work_id)
        task = asyncio.current_task()
        self.tasks.add(task)
        try:
            async with self.sem:
                result = await self.run_trajectory(
                    sample,
                    self.config,
                    None,
                    self.args,
                    concurrency_sem=asyncio.Semaphore(1),
                    clients_by_env=self.clients,
                    **kwargs,
                )
                self._raise_if_cleanup_uncertain()
                return result
        finally:
            self.tasks.discard(task)

    async def shutdown(self) -> None:
        self.closing = True
        tasks = list(self.tasks)
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        for client in self.clients.values():
            await client.aclose()
            client.close()
        await self._close_model()
        await asyncio.get_running_loop().shutdown_default_executor()
        self._raise_if_cleanup_uncertain()

    def _raise_if_cleanup_uncertain(self):
        failures = {
            env: getattr(client, "cleanup_failures", {})
            for env, client in self.clients.items()
            if client.cleanup_uncertain
        }
        if failures:
            raise DistributedRolloutError(f"Session cleanup was not confirmed: {failures}")

    async def _close_model(self) -> None:
        from slime.utils import http_utils

        from mai_gym.io import oss_io

        from . import rollout_mobile_stepwise as stepwise

        await asyncio.to_thread(stepwise._traj_save_pool.shutdown, wait=True)
        await oss_io.close_async_fs()
        if http_utils._http_client is not None:
            await http_utils._http_client.aclose()

    async def ping(self) -> bool:
        return True


async def _confirm_actor_dead(actor: Any, timeout: float = 10) -> None:
    import ray

    async with asyncio.timeout(timeout):
        while True:
            try:
                await actor.ping.remote()
            except ray.exceptions.RayActorError:
                return
            await asyncio.sleep(0.1)


class RayTrajectoryExecutor:
    def __init__(
        self,
        args: Any,
        config: Any,
        pools: dict[str, BorrowedPool],
        specs: list[WorkerSpec],
        options: WorkerOptions,
        trajectory_timeout: float | None,
        *,
        worker_class=TrajectoryWorker,
    ) -> None:
        self.args, self.config, self.pools = args, config, pools
        self.specs, self.options = specs, options
        self.trajectory_timeout, self.worker_class = trajectory_timeout, worker_class
        self.actors: list[Any] = []
        self.loads = [0] * len(specs)
        self.condition = asyncio.Condition()
        self.failure: DistributedRolloutError | None = None
        self.closing = False

    async def __aenter__(self) -> RayTrajectoryExecutor:
        import ray
        from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

        actor_class = ray.remote(self.worker_class)
        try:
            for spec in self.specs:
                self.actors.append(
                    actor_class.options(
                        num_cpus=self.options.cpus,
                        num_gpus=0,
                        max_restarts=0,
                        max_task_retries=0,
                        max_concurrency=spec.trajectories + 4 if spec.trajectories else 2**31 - 1,
                        runtime_env={
                            "env_vars": {
                                "MAI_GYM_MAX_INFLIGHT_OSS_GET": str(spec.oss_limit),
                                "OMP_NUM_THREADS": str(max(1, math.floor(self.options.cpus))),
                                "MKL_NUM_THREADS": str(max(1, math.floor(self.options.cpus))),
                                "OPENBLAS_NUM_THREADS": str(max(1, math.floor(self.options.cpus))),
                                "TOKENIZERS_PARALLELISM": "false",
                                "RAY_USE_UVLOOP": "0",
                            }
                        },
                        scheduling_strategy=NodeAffinitySchedulingStrategy(
                            node_id=spec.node_id,
                            soft=False,
                        ),
                    ).remote(
                        self.args,
                        self.config,
                        spec,
                        self.pools,
                        self.options,
                        self.trajectory_timeout,
                    )
                )
            identities = await asyncio.wait_for(
                asyncio.gather(*(actor.ready.remote() for actor in self.actors)),
                timeout=self.options.startup_timeout,
            )
            logger.info("[distributed-rollout] ready workers=%s", identities)
            return self
        except BaseException:
            await self.aclose()
            raise

    def raise_if_failed(self) -> None:
        if self.failure is not None:
            raise self.failure

    async def run(self, sample: Any, *, env_type: str, **kwargs: Any) -> Any:
        started = time.monotonic()
        self.raise_if_failed()
        eligible = [i for i, spec in enumerate(self.specs) if env_type in spec.http_limits]
        if not eligible:
            self.failure = DistributedRolloutError(
                f"No distributed worker has credits for {env_type}"
            )
            raise self.failure
        async with self.condition:
            while True:
                self.raise_if_failed()
                if self.closing:
                    raise DistributedRolloutError("Executor is closing")
                available = [
                    i
                    for i in eligible
                    if self.specs[i].trajectories == 0 or self.loads[i] < self.specs[i].trajectories
                ]
                if available:
                    index = min(
                        available, key=lambda i: self.loads[i] / max(1, self.specs[i].trajectories)
                    )
                    self.loads[index] += 1
                    break
                await self.condition.wait()
        dispatched = time.monotonic()
        work_id = f"{kwargs['rollout_id']}:{kwargs['evaluation']}:{kwargs['attempt_id']}"
        try:
            result = await self.actors[index].run.remote(work_id, sample, kwargs)
            logger.info(
                "[distributed-rollout] attempt=%s env=%s node=%s slot_wait=%.3fs remote_wall=%.3fs",
                work_id,
                env_type,
                self.specs[index].node_id,
                dispatched - started,
                time.monotonic() - dispatched,
            )
            return result
        except asyncio.CancelledError:
            self.closing = True
            raise
        except Exception as exc:
            self.failure = DistributedRolloutError(
                f"Worker {index} execution is uncertain; round aborted without replay"
            )
            raise self.failure from exc
        finally:
            async with self.condition:
                self.loads[index] -= 1
                self.condition.notify_all()

    async def aclose(self) -> None:
        import ray

        self.closing = True
        async with self.condition:
            self.condition.notify_all()
        if not self.actors:
            return
        actors, self.actors = self.actors, []
        cleanup_error = None
        try:
            outcomes = await asyncio.wait_for(
                asyncio.gather(*(a.shutdown.remote() for a in actors), return_exceptions=True),
                timeout=self.options.shutdown_timeout,
            )
            cleanup_error = next((x for x in outcomes if isinstance(x, BaseException)), None)
        except BaseException as exc:
            cleanup_error = exc
        finally:
            for actor in actors:
                try:
                    ray.kill(actor, no_restart=True)
                except Exception as exc:
                    cleanup_error = exc

            confirmations = await asyncio.gather(
                *(_confirm_actor_dead(actor) for actor in actors),
                return_exceptions=True,
            )
            cleanup_error = next(
                (x for x in confirmations if isinstance(x, BaseException)),
                cleanup_error,
            )
        if cleanup_error is not None:
            raise DistributedRolloutError(
                "Worker cleanup was not graceful; do not continue to training/weight update"
            ) from cleanup_error

    async def __aexit__(self, exc_type, exc, traceback) -> None:
        try:
            await self.aclose()
            self.raise_if_failed()
        except BaseException as cleanup_error:
            if exc is not None:
                raise BaseExceptionGroup(
                    "Rollout failed and cleanup was not confirmed", [exc, cleanup_error]
                ) from None
            raise


def make_executor(
    config: Any, args: Any, concurrency: int, trajectory_timeout: float | None
) -> RayTrajectoryExecutor:
    from .rollout import _get_env_client_for, _pool_spec

    options = WorkerOptions.from_env()
    options.validate(config, args)
    pools = {}
    for env_type in config.env_pools or [config.env_type]:
        client = _get_env_client_for(config, env_type)
        request_id = client.fleet_request_id
        if not request_id:
            raise ValueError(
                "Distributed rollout requires fleet reservation support; "
                "unscoped legacy ensure_pool is not supported"
            )
        pools[env_type] = BorrowedPool(
            _pool_spec(config, env_type)[0],
            request_id,
            config.screenshot_transport,
        )
    node_ids = getattr(args, "mai_gym_rollout_node_ids", None)
    if not node_ids:
        raise ValueError("Actual rollout node placement missing; use the patched SLIME launcher")
    specs = worker_specs(node_ids, options, concurrency, list(pools))
    return RayTrajectoryExecutor(args, config, pools, specs, options, trajectory_timeout)
