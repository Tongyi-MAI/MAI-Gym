import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from mai_gym.infra.launcher._base import Launcher
from mai_gym.infra.launcher._helpers import (
    _MW_REUSE_HEALTH_TIMEOUT,
    _discover_device_id,
    _make_backend_id,
    allocate_4port_slot,
    respect_launch_interval,
)
from mai_gym.models.config import DeviceConfig, DockerLaunchConfig
from mai_gym.models.session import RunningBackend

logger = logging.getLogger(__name__)


class MobileWorldDockerProfile(Launcher):
    def __init__(self) -> None:
        self._containers: dict[str, str] = {}
        self._reused: set[str] = set()
        self._next_index: dict[str, int] = {}
        self._claimed_ports: dict[str, set[int]] = {}
        self._last_launch_at: dict[str, float] = {}

        self._bound: dict[str, set[int]] = {}

        self._lock = threading.Lock()

    def _bind_port(self, device_name: str, backend_port: int, *, context: str) -> None:

        bound = self._bound.setdefault(device_name, set())
        if backend_port in bound:
            raise RuntimeError(
                f"[MobileWorldDockerProfile] refusing to bind backend port "
                f"{backend_port} on {device_name} to a second backend — it is "
                f"already bound ({context}); two sessions would otherwise share "
                f"one emulator (cross-session action contamination)"
            )
        bound.add(backend_port)

    def _respect_launch_interval(self, device_name: str, interval: int) -> None:

        respect_launch_interval(
            self._last_launch_at,
            device_name,
            interval,
            on_wait=lambda wait: logger.info(
                "[MobileWorldDockerProfile] Staggering launch for %s — sleeping %.1fs",
                device_name,
                wait,
            ),
        )

    def _allocate_slot(
        self,
        device: DeviceConfig,
        exclude_slots: frozenset[int] = frozenset(),
    ) -> tuple[int, int, int, int, int]:

        from mai_gym.envs.mobileworld.docker import (
            find_next_container_index,
            is_port_available,
        )

        assert isinstance(device.launch, DockerLaunchConfig)
        launch = device.launch

        prefix = f"{launch.name_prefix}_{device.name}"
        return allocate_4port_slot(
            launch,
            self._next_index,
            self._claimed_ports,
            device.name,
            is_port_available,
            exclude_slots=exclude_slots,
            seed_index=lambda: find_next_container_index(prefix=prefix),
        )

    def _find_reusable_containers(
        self, device: DeviceConfig
    ) -> list[tuple[int, int, int, int, int, str]]:

        from mai_gym.envs.mobileworld.docker import (
            list_containers,
            wait_for_container_ready,
        )

        assert isinstance(device.launch, DockerLaunchConfig)
        launch = device.launch
        prefix = f"{launch.name_prefix}_{device.name}_"
        listed = list_containers(
            image_filter=launch.image,
            name_prefix=prefix,
            include_all=False,
        )

        candidates: list[tuple[int, int, int, int, int, str]] = []
        for info in listed:
            try:
                slot = int(info.name.removeprefix(prefix))
            except ValueError:
                logger.debug(
                    "[MobileWorldDockerProfile] Ignoring %s — non-numeric suffix",
                    info.name,
                )
                continue
            backend = launch.port_range_start + slot
            if info.backend_port != backend:
                logger.debug(
                    "[MobileWorldDockerProfile] Ignoring %s — backend_port=%s "
                    "doesn't match expected %d (config drift?)",
                    info.name,
                    info.backend_port,
                    backend,
                )
                continue
            candidates.append(
                (
                    slot,
                    backend,
                    launch.viewer_port_range_start + slot,
                    launch.vnc_port_range_start + slot,
                    launch.adb_port_range_start + slot,
                    info.name,
                )
            )
        if not candidates:
            return []

        def _ready(entry: tuple[int, int, int, int, int, str]) -> bool:
            backend_port, name = entry[1], entry[5]
            if wait_for_container_ready(backend_port, timeout=_MW_REUSE_HEALTH_TIMEOUT):
                return True
            logger.warning(
                "[MobileWorldDockerProfile] Skipping %s — not healthy "
                "within %ds.  Container left running for inspection.",
                name,
                _MW_REUSE_HEALTH_TIMEOUT,
            )
            return False

        with ThreadPoolExecutor(max_workers=min(32, len(candidates))) as ex:
            flags = list(ex.map(_ready, candidates))
        out = [entry for entry, ok in zip(candidates, flags, strict=True) if ok]
        out.sort(key=lambda t: t[0])
        return out

    def launch(self, device: DeviceConfig) -> RunningBackend:
        backends = self.batch_launch(device, 1)
        if not backends:
            raise RuntimeError(f"MobileWorld launch failed for {device.name}")
        return backends[0]

    def batch_launch(self, device: DeviceConfig, count: int) -> list[RunningBackend]:

        from mai_gym.envs.mobileworld.docker import (
            ContainerConfig,
            launch_container,
            wait_for_container_ready,
        )

        assert isinstance(device.launch, DockerLaunchConfig)
        launch = device.launch

        env_file_path = Path(launch.env_file) if launch.env_file else None
        if env_file_path is not None and not env_file_path.is_file():
            raise RuntimeError(
                f"env_file not found: {env_file_path} "
                f"(device {device.name} requires MobileWorld .env)"
            )

        backends: list[RunningBackend] = []

        reusable = self._find_reusable_containers(device)

        reuse_plan: list[tuple[int, int, str, str]] = []
        with self._lock:
            claimed = self._claimed_ports.setdefault(device.name, set())
            bound_names = set(self._containers.values())
            available = [
                entry
                for entry in reusable
                if entry[1] not in claimed and entry[5] not in bound_names
            ]
            claim = available[:count]
            excluded_slots = frozenset(slot for slot, *_ in claim)
            fresh_count = count - len(claim)

            for slot, backend_port, viewer_port, vnc_port, adb_port, name in claim:
                self._bind_port(
                    device.name,
                    backend_port,
                    context=f"reused container {name}, slot {slot}",
                )
                backend_id = _make_backend_id()
                self._containers[backend_id] = name
                self._reused.add(backend_id)

                self._next_index[device.name] = max(
                    self._next_index.get(device.name, 0),
                    slot + 1,
                )
                claimed.update((backend_port, viewer_port, vnc_port, adb_port))
                reuse_plan.append((slot, backend_port, name, backend_id))

            slots: list[tuple[int, int, int, int, int]] = [
                self._allocate_slot(device, excluded_slots) for _ in range(fresh_count)
            ]

        for slot, backend_port, name, backend_id in reuse_plan:
            url = f"http://localhost:{backend_port}"
            device_id = _discover_device_id(url) or "emulator-5554"
            backends.append(
                RunningBackend(
                    id=backend_id,
                    url=url,
                    device_name=device.name,
                    env_type=device.type,
                    device_id=device_id,
                )
            )
            logger.info(
                "[MobileWorldDockerProfile] Reusing %s on backend=%d (slot=%d) → %s",
                name,
                backend_port,
                slot,
                backend_id,
            )

        logger.info(
            "[MobileWorldDockerProfile] %s: reusing %d existing container(s), launching %d new",
            device.name,
            len(reuse_plan),
            fresh_count,
        )

        if fresh_count == 0:
            return backends
        logger.info(
            "[MobileWorldDockerProfile] Bulk-launching %d container(s) for %s "
            "(interval=%ds, ready_timeout=%ds)",
            fresh_count,
            device.name,
            launch.launch_interval,
            launch.ready_timeout,
        )

        pending: list[tuple[int, int, str]] = []
        for i, (slot, backend_port, viewer_port, vnc_port, adb_port) in enumerate(slots):
            container_name = f"{launch.name_prefix}_{device.name}_{slot}"
            config = ContainerConfig(
                name=container_name,
                backend_port=backend_port,
                viewer_port=viewer_port,
                vnc_port=vnc_port,
                adb_port=adb_port,
                image=launch.image,
                dev_mode=False,
                enable_vnc=False,
                env_file_path=env_file_path,
                dev_src_path=None,
            )
            logger.info(
                "[MobileWorldDockerProfile]  (%d/%d) launching %s on backend=%d",
                i + 1,
                fresh_count,
                container_name,
                backend_port,
            )
            result = launch_container(config, detach=True, wait_ready=False)
            if not result.success:
                logger.error(
                    "[MobileWorldDockerProfile] %s failed to launch: %s",
                    container_name,
                    result.error_message,
                )
                continue
            pending.append((slot, backend_port, container_name))
            if launch.launch_interval > 0 and i < fresh_count - 1:
                time.sleep(launch.launch_interval)

        logger.info(
            "[MobileWorldDockerProfile] Waiting for %d container(s) to be ready...",
            len(pending),
        )
        ready_flags: dict[str, bool] = {}

        def _wait(backend_port: int, name: str) -> None:
            ready_flags[name] = wait_for_container_ready(
                backend_port,
                timeout=launch.ready_timeout,
            )

        threads = [
            threading.Thread(target=_wait, args=(bp, name), daemon=True)
            for (_, bp, name) in pending
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        for _slot, backend_port, container_name in pending:
            if not ready_flags.get(container_name, False):
                logger.warning(
                    "[MobileWorldDockerProfile] %s did not become ready in %ds",
                    container_name,
                    launch.ready_timeout,
                )
                continue
            url = f"http://localhost:{backend_port}"
            device_id = _discover_device_id(url) or "emulator-5554"
            backend_id = _make_backend_id()
            with self._lock:
                self._bind_port(
                    device.name,
                    backend_port,
                    context=f"fresh container {container_name}",
                )
                self._containers[backend_id] = container_name
            backends.append(
                RunningBackend(
                    id=backend_id,
                    url=url,
                    device_name=device.name,
                    env_type=device.type,
                    device_id=device_id,
                )
            )
            logger.info(
                "[MobileWorldDockerProfile] Backend %s ready at %s (device=%s)",
                backend_id,
                url,
                device_id,
            )
        return backends

    def shutdown(self, backend: RunningBackend) -> None:

        try:
            port: int | None = int(backend.url.rsplit(":", 1)[1])
        except (ValueError, IndexError):
            port = None
        with self._lock:
            container = self._containers.pop(backend.id, None)
            self._reused.discard(backend.id)
            if port is not None:
                self._bound.get(backend.device_name, set()).discard(port)
                self._claimed_ports.get(backend.device_name, set()).discard(port)
        if container is None:
            return
        logger.info(
            "[MobileWorldDockerProfile] Leaving container %s running for reuse "
            "(backend=%s).  Run `docker rm -f %s` to clean up manually.",
            container,
            backend.id,
            container,
        )
