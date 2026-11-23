import asyncio
import logging
import time

from mai_gym.infra.launcher import Launcher, create_launcher
from mai_gym.infra.task_artifacts import (
    TaskArtifactManifestError,
    TaskArtifactSpec,
    TaskSetArtifactManifest,
)
from mai_gym.models.config import DeviceConfig, MobileGymConfig, RecoveryConfig
from mai_gym.models.session import RunningBackend

logger = logging.getLogger(__name__)

_DEVICE_COOLDOWN_SEC = 30.0


class InfraManager:
    def __init__(self, config: MobileGymConfig) -> None:
        self._devices = config.devices
        self._lock = asyncio.Lock()

        self._device_map: dict[str, DeviceConfig] = {d.name: d for d in self._devices}
        self._launchers: dict[str, Launcher] = {d.name: create_launcher(d) for d in self._devices}
        self._backends: dict[str, list[RunningBackend]] = {d.name: [] for d in self._devices}
        self._active_counts: dict[str, int] = {d.name: 0 for d in self._devices}
        self._device_down_until: dict[str, float] = {}

        self._backend_to_device: dict[str, str] = {}

        self._backend_sessions: dict[str, int] = {}

        self._pending_destructions: dict[str, tuple[str, RunningBackend]] = {}
        self._release_tasks: dict[str, asyncio.Task[None]] = {}

        self._pending_launches: dict[str, int] = {d.name: 0 for d in self._devices}

        self._task_sets: dict[str, bytes] = {}
        self._task_set_artifact_manifests: dict[str, TaskSetArtifactManifest] = {}
        self._artifact_task_to_set: dict[str, str] = {}

        self._task_set_locks: dict[str, asyncio.Lock] = {}
        self._task_set_locks_guard = asyncio.Lock()

    def all_backends(self) -> list[RunningBackend]:

        out: list[RunningBackend] = []
        for backends in self._backends.values():
            out.extend(backends)
        return out

    def registered_set_ids(self) -> list[str]:

        return list(self._task_sets)

    def task_set_bytes(self) -> dict[str, bytes]:

        return dict(self._task_sets)

    def store_task_set(self, set_id: str, package: bytes) -> None:

        self._task_sets[set_id] = package

    def store_task_set_artifact_manifest(
        self,
        set_id: str,
        manifest: TaskSetArtifactManifest | None,
    ) -> None:

        previous = self._task_set_artifact_manifests.get(set_id)
        conflicts = {
            task_name: existing_set
            for task_name in (manifest.task_names if manifest is not None else ())
            if (existing_set := self._artifact_task_to_set.get(task_name)) not in {None, set_id}
        }
        if conflicts:
            detail = ", ".join(f"{task} ({owner})" for task, owner in sorted(conflicts.items()))
            raise TaskArtifactManifestError(f"artifact task mappings already registered: {detail}")

        if previous is not None:
            for task_name in previous.task_names:
                if self._artifact_task_to_set.get(task_name) == set_id:
                    self._artifact_task_to_set.pop(task_name, None)
        if manifest is None:
            self._task_set_artifact_manifests.pop(set_id, None)
            return
        self._task_set_artifact_manifests[set_id] = manifest
        for task_name in manifest.task_names:
            self._artifact_task_to_set[task_name] = set_id

    def task_artifact_binding(
        self,
        task_name: str,
    ) -> tuple[str, bytes, tuple[TaskArtifactSpec, ...]] | None:

        set_id = self._artifact_task_to_set.get(task_name)
        if set_id is None:
            return None
        manifest = self._task_set_artifact_manifests.get(set_id)
        package = self._task_sets.get(set_id)
        if manifest is None or package is None:
            return None
        return set_id, package, manifest.artifacts_for(task_name)

    async def prepare_task_artifacts(
        self,
        backend: RunningBackend,
        task_name: str,
    ) -> list[dict[str, object]]:

        binding = self.task_artifact_binding(task_name)
        if binding is None:
            return []
        _set_id, _package, artifacts = binding
        device = self._device_map.get(backend.device_name)
        if device is None:
            raise KeyError(f"unknown device {backend.device_name!r}")
        launcher = self._launchers[backend.device_name]
        return await asyncio.to_thread(
            launcher.prepare_task_artifacts,
            device,
            backend,
            artifacts,
        )

    def unregister_task_set(self, set_id: str) -> bool:

        manifest = self._task_set_artifact_manifests.pop(set_id, None)
        if manifest is not None:
            for task_name in manifest.task_names:
                if self._artifact_task_to_set.get(task_name) == set_id:
                    self._artifact_task_to_set.pop(task_name, None)
        return self._task_sets.pop(set_id, None) is not None

    async def get_set_lock(self, set_id: str) -> asyncio.Lock:

        async with self._task_set_locks_guard:
            lock = self._task_set_locks.get(set_id)
            if lock is None:
                lock = asyncio.Lock()
                self._task_set_locks[set_id] = lock
            return lock

    def _remaining_capacity(self, device_name: str) -> int:
        device = self._device_map[device_name]
        per_backend = device.effective_sessions_per_backend

        pending_capacity = self._pending_launches[device_name] * per_backend
        return device.max_sessions - self._active_counts[device_name] - pending_capacity

    def _is_device_available(self, device_name: str) -> bool:
        deadline = self._device_down_until.get(device_name, 0)
        return time.monotonic() >= deadline

    def _mark_device_down(self, device_name: str) -> None:
        self._device_down_until[device_name] = time.monotonic() + _DEVICE_COOLDOWN_SEC
        logger.warning(
            "[InfraManager] Device %s marked down for %.0fs",
            device_name,
            _DEVICE_COOLDOWN_SEC,
        )

    async def provision(self, env_type: str, count: int) -> list[RunningBackend]:

        result: list[RunningBackend] = []

        launch_plan: list[tuple[DeviceConfig, int]] = []
        async with self._lock:
            still_needed = count
            for device in self._devices:
                if still_needed <= 0:
                    break
                if not device.accepts_env_type(env_type):
                    continue
                if not self._is_device_available(device.name):
                    continue

                remaining = self._remaining_capacity(device.name)
                needed = min(remaining, still_needed)
                if needed <= 0:
                    continue

                per_backend = device.effective_sessions_per_backend

                reused_taken = 0
                while needed > 0:
                    backend = self._find_free_backend(device, per_backend)
                    if backend is None:
                        break
                    result.append(backend)
                    self._active_counts[device.name] += 1
                    self._backend_sessions[backend.id] = (
                        self._backend_sessions.get(backend.id, 0) + 1
                    )
                    needed -= 1
                    still_needed -= 1
                    reused_taken += 1
                if reused_taken:
                    logger.info(
                        "[InfraManager] active_counts: device=%s now=%d (delta=+%d, reused)",
                        device.name,
                        self._active_counts[device.name],
                        reused_taken,
                    )

                if needed > 0:
                    backends_needed = -(-needed // per_backend)
                    launch_plan.append((device, backends_needed))
                    self._pending_launches[device.name] += backends_needed
                    still_needed -= needed

        if not launch_plan:
            return result

        logger.info(
            "[InfraManager] Parallel launch across %d device(s): %s",
            len(launch_plan),
            ", ".join(f"{d.name}×{n}" for d, n in launch_plan),
        )

        async def _launch_one(device: DeviceConfig, num: int) -> list[RunningBackend]:
            launcher = self._launchers[device.name]
            try:
                return await asyncio.to_thread(launcher.batch_launch, device, num)
            except Exception as exc:
                logger.exception(
                    "[InfraManager] Failed to launch %d on %s: %s",
                    num,
                    device.name,
                    exc,
                )
                self._mark_device_down(device.name)
                return []

        all_results = await asyncio.gather(
            *(_launch_one(dev, n) for dev, n in launch_plan),
        )

        async with self._lock:
            for (device, backends_needed), new_backends in zip(
                launch_plan,
                all_results,
                strict=True,
            ):
                self._pending_launches[device.name] -= backends_needed
                per_backend = device.effective_sessions_per_backend
                remaining_for_device = min(
                    self._remaining_capacity(device.name) + backends_needed * per_backend,
                    count - len(result),
                )
                fresh_taken = 0
                for backend in new_backends:
                    backend.metadata.update(
                        canonical_env_type=device.env_type,
                        device_type=device.device_type,
                        infra_type=device.infra_type,
                        environment_profile=device.env_config.get("profile"),
                        runtime_protocol=device.env_config.get("runtime_protocol"),
                    )
                    self._backends[device.name].append(backend)
                    self._backend_to_device[backend.id] = device.name
                    self._backend_sessions[backend.id] = 0
                    take = min(per_backend, remaining_for_device - fresh_taken)
                    if take <= 0:
                        continue
                    result.extend([backend] * take)
                    self._active_counts[device.name] += take
                    self._backend_sessions[backend.id] = take
                    fresh_taken += take
                if fresh_taken:
                    logger.info(
                        "[InfraManager] active_counts: device=%s now=%d (delta=+%d, fresh)",
                        device.name,
                        self._active_counts[device.name],
                        fresh_taken,
                    )

        return result

    def _find_free_backend(
        self,
        device: DeviceConfig,
        per_backend: int,
    ) -> RunningBackend | None:

        for backend in self._backends[device.name]:
            if self._backend_sessions.get(backend.id, 0) < per_backend:
                return backend
        return None

    async def release(self, backend_id: str) -> None:

        async with self._lock:
            pending = self._pending_destructions.get(backend_id)
            device_name = self._backend_to_device.get(backend_id)
            if (
                pending is None
                and device_name is not None
                and (self._backend_sessions.get(backend_id, 0) > 0)
            ):
                self._active_counts[device_name] -= 1
                self._backend_sessions[backend_id] -= 1
                logger.debug(
                    "[InfraManager] Released backend %s (device=%s, active=%d)",
                    backend_id,
                    device_name,
                    self._active_counts[device_name],
                )

                from mai_gym.models.config import K8sLaunchConfig

                device = self._device_map[device_name]
                if (
                    isinstance(device.launch, K8sLaunchConfig)
                    and device.launch.destroy_on_release
                    and self._backend_sessions.get(backend_id, 0) == 0
                ):
                    backend = next(
                        (b for b in self._backends[device_name] if b.id == backend_id),
                        None,
                    )
                    if backend is not None:
                        self._backends[device_name].remove(backend)
                        pending = (device_name, backend)
                        self._pending_destructions[backend_id] = pending
            if pending is None:
                return
            task = self._release_tasks.get(backend_id)
            if task is None or task.done():
                task = asyncio.create_task(self._finish_release_destruction(backend_id, pending))
                self._release_tasks[backend_id] = task
                task.add_done_callback(
                    lambda completed, bid=backend_id: self._release_finished(bid, completed)
                )

        await asyncio.shield(task)

    def _release_finished(self, backend_id: str, task: asyncio.Task[None]) -> None:
        if self._release_tasks.get(backend_id) is task:
            self._release_tasks.pop(backend_id, None)
        if not task.cancelled():
            task.exception()

    async def _finish_release_destruction(
        self,
        backend_id: str,
        pending: tuple[str, RunningBackend],
    ) -> None:
        device_name, backend = pending
        await asyncio.to_thread(self._launchers[device_name].shutdown, backend)
        async with self._lock:
            if self._pending_destructions.get(backend_id) is pending:
                self._pending_destructions.pop(backend_id, None)
                self._backend_to_device.pop(backend_id, None)
                self._backend_sessions.pop(backend_id, None)
        logger.info(
            "[InfraManager] destroy-on-release: shut down backend %s (device=%s)",
            backend_id,
            device_name,
        )

    def device_recovery(self, backend_id: str) -> RecoveryConfig | None:

        device_name = self._backend_to_device.get(backend_id)
        if device_name is None:
            return None
        return self._device_map[device_name].recovery

    def recycle_on_release(self, backend_id: str) -> bool:

        device_name = self._backend_to_device.get(backend_id)
        if device_name is None:
            return False
        from mai_gym.models.config import K8sLaunchConfig

        launch = self._device_map[device_name].launch
        return isinstance(launch, K8sLaunchConfig) and launch.acquisition == "pool"

    def relaunch_backend(self, backend_id: str) -> bool:

        device_name = self._backend_to_device.get(backend_id)
        if device_name is None:
            logger.warning("[InfraManager] relaunch: unknown backend %s", backend_id)
            return False
        backend = next((b for b in self._backends[device_name] if b.id == backend_id), None)
        if backend is None:
            logger.warning("[InfraManager] relaunch: backend %s not in pool", backend_id)
            return False
        launcher = self._launchers[device_name]
        device = self._device_map[device_name]
        try:
            return launcher.relaunch(device, backend)
        except Exception as exc:
            logger.exception("[InfraManager] relaunch of %s failed: %s", backend_id, exc)
            return False

    def active_counts_snapshot(self) -> dict[str, int]:

        return dict(self._active_counts)

    async def resize_sandbox_pool(self, device_name: str, replicas: int) -> str:

        device = self._device_map.get(device_name)
        if device is None:
            raise KeyError(f"unknown device {device_name!r}")
        launcher = self._launchers[device_name]
        return await asyncio.to_thread(launcher.resize_pool, device, replicas)

    async def sandbox_pool_status(self, device_name: str) -> dict[str, object]:

        device = self._device_map.get(device_name)
        if device is None:
            raise KeyError(f"unknown device {device_name!r}")
        launcher = self._launchers[device_name]
        return await asyncio.to_thread(launcher.pool_status, device)

    async def shutdown_all(self) -> None:

        for backend_id in tuple(self._pending_destructions):
            try:
                await self.release(backend_id)
            except Exception:
                logger.exception("[InfraManager] pending destruction failed for %s", backend_id)

        for launcher in self._launchers.values():
            retry_cleanup = getattr(launcher, "retry_failed_cleanup", None)
            if retry_cleanup is not None:
                await asyncio.to_thread(retry_cleanup)
        async with self._lock:
            for device_name, backends in self._backends.items():
                launcher = self._launchers[device_name]
                for backend in backends:
                    try:
                        await asyncio.to_thread(
                            launcher.shutdown_for_manager_restart,
                            backend,
                        )
                    except Exception as exc:
                        logger.warning(
                            "[InfraManager] shutdown failed for %s: %s",
                            backend.id,
                            exc,
                        )
                backends.clear()
            self._active_counts = {d.name: 0 for d in self._devices}
            self._backend_sessions.clear()
