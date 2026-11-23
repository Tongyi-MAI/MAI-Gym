import logging
import threading
from concurrent.futures import ThreadPoolExecutor
from urllib.parse import urlparse

import httpx

from mai_gym.infra.launcher._base import Launcher
from mai_gym.infra.launcher._helpers import (
    _HEALTH_INTERVAL,
    _MW_REUSE_HEALTH_TIMEOUT,
    _discover_device_id,
    _make_backend_id,
    _wait_for_health,
    allocate_4port_slot,
    respect_launch_interval,
)
from mai_gym.models.config import AgentConnection, DeviceConfig, DockerLaunchConfig
from mai_gym.models.session import RunningBackend

logger = logging.getLogger(__name__)


def _route(host: str, port: int, agent: AgentConnection) -> str:

    if agent.proxy is not None:
        return f"http://{host}:{agent.proxy.port}/p/{port}"
    return f"http://{host}:{port}"


def _backend_port_from_url(url: str) -> int:

    parsed = urlparse(url)
    if parsed.path and parsed.path.startswith("/p/"):
        chunk = parsed.path.removeprefix("/p/").split("/", 1)[0]
        try:
            return int(chunk)
        except ValueError as exc:
            raise ValueError(f"cannot parse backend port from {url!r}") from exc
    if parsed.port is None:
        raise ValueError(f"cannot parse backend port from {url!r}")
    return parsed.port


class MobileAgentLauncher(Launcher):
    def __init__(self) -> None:
        self._containers: dict[str, tuple[str, str]] = {}
        self._next_index: dict[str, int] = {}
        self._claimed_ports: dict[str, set[int]] = {}
        self._last_launch_at: dict[str, float] = {}

        self._bound: dict[str, set[int]] = {}

        self._lock = threading.Lock()

    def _bind_port(self, device_name: str, backend_port: int, *, context: str) -> None:

        bound = self._bound.setdefault(device_name, set())
        if backend_port in bound:
            raise RuntimeError(
                f"[AgentLauncher] refusing to bind backend port {backend_port} "
                f"on {device_name} to a second backend — it is already bound "
                f"({context}); two sessions would otherwise share one emulator "
                f"(cross-session action contamination)"
            )
        bound.add(backend_port)

    def _agent_url(self, device: DeviceConfig) -> str:
        agent = device.agent
        assert agent is not None
        return _route(agent.host, agent.port, agent)

    def _is_port_free(self, agent_url: str, port: int) -> bool:
        try:
            r = httpx.get(f"{agent_url}/port-check/{port}", timeout=5)
            return r.json().get("free", False)
        except Exception as exc:
            logger.warning("[AgentLauncher] port-check failed for port %d: %s", port, exc)
            return True

    def _allocate_slot(
        self,
        device: DeviceConfig,
        exclude_slots: frozenset[int] = frozenset(),
    ) -> tuple[int, int, int, int, int]:

        assert isinstance(device.launch, DockerLaunchConfig)
        agent_url = self._agent_url(device)
        return allocate_4port_slot(
            device.launch,
            self._next_index,
            self._claimed_ports,
            device.name,
            lambda p: self._is_port_free(agent_url, p),
            exclude_slots=exclude_slots,
            location_desc=f" on agent {agent_url}",
        )

    def _find_reusable_containers(
        self,
        device: DeviceConfig,
    ) -> list[tuple[int, int, int, int, int, str]]:

        assert isinstance(device.launch, DockerLaunchConfig)
        launch = device.launch
        agent = device.agent
        assert agent is not None
        agent_url = self._agent_url(device)
        agent_host = agent.host
        prefix = f"{launch.name_prefix}_{device.name}_"
        try:
            r = httpx.get(f"{agent_url}/containers", timeout=10)
            all_containers = r.json().get("containers", [])
        except Exception:
            return []

        candidates: list[tuple[int, int, int, int, int, str]] = []
        for info in all_containers:
            name = info.get("name", "")
            if not name.startswith(prefix):
                continue
            try:
                slot = int(name.removeprefix(prefix))
            except ValueError:
                continue
            candidates.append(
                (
                    slot,
                    launch.port_range_start + slot,
                    launch.viewer_port_range_start + slot,
                    launch.vnc_port_range_start + slot,
                    launch.adb_port_range_start + slot,
                    name,
                )
            )
        if not candidates:
            return []

        def _healthy(entry: tuple[int, int, int, int, int, str]) -> bool:
            backend_port = entry[1]
            try:
                resp = httpx.get(
                    f"{_route(agent_host, backend_port, agent)}/health",
                    timeout=_MW_REUSE_HEALTH_TIMEOUT,
                )
                return resp.status_code == 200
            except Exception:
                return False

        with ThreadPoolExecutor(max_workers=min(32, len(candidates))) as ex:
            flags = list(ex.map(_healthy, candidates))
        out = [entry for entry, ok in zip(candidates, flags, strict=True) if ok]
        out.sort(key=lambda t: t[0])
        return out

    def _respect_launch_interval(self, device_name: str, interval: int) -> None:
        respect_launch_interval(self._last_launch_at, device_name, interval)

    def launch(self, device: DeviceConfig) -> RunningBackend:
        backends = self.batch_launch(device, 1)
        if not backends:
            raise RuntimeError(f"MobileWorld agent launch failed for {device.name}")
        return backends[0]

    def batch_launch(self, device: DeviceConfig, count: int) -> list[RunningBackend]:
        assert isinstance(device.launch, DockerLaunchConfig)
        launch = device.launch
        agent = device.agent
        assert agent is not None
        agent_url = self._agent_url(device)
        agent_host = agent.host

        backends: list[RunningBackend] = []

        reusable = self._find_reusable_containers(device)

        reuse_plan: list[tuple[int, int, str, str]] = []
        with self._lock:
            claimed = self._claimed_ports.setdefault(device.name, set())
            bound_names = {n for (_, n) in self._containers.values()}
            available = [
                entry
                for entry in reusable
                if entry[1] not in claimed and entry[5] not in bound_names
            ]
            claim = available[:count]
            excluded_slots = frozenset(slot for slot, *_ in claim)
            fresh_count = count - len(claim)

            for slot, bp, vp, vncp, adbp, name in claim:
                self._bind_port(device.name, bp, context=f"reused container {name}, slot {slot}")
                backend_id = _make_backend_id()
                self._containers[backend_id] = (agent_url, name)
                self._next_index[device.name] = max(
                    self._next_index.get(device.name, 0),
                    slot + 1,
                )
                claimed.update((bp, vp, vncp, adbp))
                reuse_plan.append((slot, bp, name, backend_id))

            slots = [self._allocate_slot(device, excluded_slots) for _ in range(fresh_count)]

        for slot, bp, name, backend_id in reuse_plan:
            url = _route(agent_host, bp, agent)
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
            logger.info("[AgentLauncher] Reusing %s (slot=%d, port=%d)", name, slot, bp)

        logger.info(
            "[AgentLauncher] %s: reusing %d container(s), launching %d new",
            device.name,
            len(reuse_plan),
            fresh_count,
        )

        if fresh_count == 0:
            return backends

        pending: list[tuple[int, int, str]] = []

        for i, (slot, bp, vp, vncp, adbp) in enumerate(slots):
            container_name = f"{launch.name_prefix}_{device.name}_{slot}"
            self._respect_launch_interval(device.name, launch.launch_interval)
            payload: dict[str, object] = {
                "image": launch.image,
                "name": container_name,
                "backend_port": bp,
                "viewer_port": vp,
                "vnc_port": vncp,
                "adb_port": adbp,
                "env_file": launch.env_file,
                "http_proxy": launch.http_proxy,
                "dev_src_path": launch.dev_src_path,
                "remove_existing": True,
            }
            logger.info(
                "[AgentLauncher] (%d/%d) launching %s on port %d via %s",
                i + 1,
                fresh_count,
                container_name,
                bp,
                agent_url,
            )
            try:
                resp = httpx.post(
                    f"{agent_url}/containers",
                    json=payload,
                    timeout=120,
                )
                if resp.status_code >= 400:
                    logger.error(
                        "[AgentLauncher] %s launch failed: %s",
                        container_name,
                        resp.text,
                    )
                    continue
            except Exception as exc:
                logger.error("[AgentLauncher] %s launch request failed: %s", container_name, exc)
                continue
            pending.append((slot, bp, container_name))

        ready_flags: dict[str, bool] = {}

        def _wait(backend_port: int, name: str) -> None:
            url = _route(agent_host, backend_port, agent)
            retries = max(1, int(launch.ready_timeout / _HEALTH_INTERVAL))
            ready_flags[name] = _wait_for_health(url, retries=retries)

        threads = [
            threading.Thread(target=_wait, args=(bp, name), daemon=True)
            for (_, bp, name) in pending
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        for _slot, bp, container_name in pending:
            if not ready_flags.get(container_name, False):
                logger.warning(
                    "[AgentLauncher] %s not ready after %ds",
                    container_name,
                    launch.ready_timeout,
                )
                continue
            url = _route(agent_host, bp, agent)
            device_id = _discover_device_id(url) or "emulator-5554"
            backend_id = _make_backend_id()
            with self._lock:
                self._bind_port(device.name, bp, context=f"fresh container {container_name}")
                self._containers[backend_id] = (agent_url, container_name)
            backends.append(
                RunningBackend(
                    id=backend_id,
                    url=url,
                    device_name=device.name,
                    env_type=device.type,
                    device_id=device_id,
                )
            )
            logger.info("[AgentLauncher] Backend %s ready at %s", backend_id, url)

        return backends

    def relaunch(self, device: DeviceConfig, backend: RunningBackend) -> bool:

        assert isinstance(device.launch, DockerLaunchConfig)
        launch = device.launch
        info = self._containers.get(backend.id)
        if info is None:
            logger.warning(
                "[AgentLauncher] relaunch: no container record for backend %s",
                backend.id,
            )
            return False
        agent_url, container_name = info
        try:
            bp = _backend_port_from_url(backend.url)
        except ValueError:
            logger.warning("[AgentLauncher] relaunch: cannot parse port from %s", backend.url)
            return False
        slot = bp - launch.port_range_start
        payload: dict[str, object] = {
            "image": launch.image,
            "name": container_name,
            "backend_port": bp,
            "viewer_port": launch.viewer_port_range_start + slot,
            "vnc_port": launch.vnc_port_range_start + slot,
            "adb_port": launch.adb_port_range_start + slot,
            "env_file": launch.env_file,
            "http_proxy": launch.http_proxy,
            "dev_src_path": launch.dev_src_path,
            "remove_existing": True,
        }
        logger.info(
            "[AgentLauncher] relaunching %s on port %d via %s",
            container_name,
            bp,
            agent_url,
        )
        self._respect_launch_interval(device.name, launch.launch_interval)
        try:
            resp = httpx.post(f"{agent_url}/containers", json=payload, timeout=120)
            if resp.status_code >= 400:
                logger.error(
                    "[AgentLauncher] relaunch of %s failed: %s",
                    container_name,
                    resp.text,
                )
                return False
        except Exception as exc:
            logger.error(
                "[AgentLauncher] relaunch request for %s failed: %s",
                container_name,
                exc,
            )
            return False
        retries = max(1, int(launch.ready_timeout / _HEALTH_INTERVAL))
        ok = _wait_for_health(backend.url, retries=retries)
        if ok:
            logger.info("[AgentLauncher] relaunch of %s ready", container_name)
        else:
            logger.warning(
                "[AgentLauncher] %s not ready after relaunch (%ds)",
                container_name,
                launch.ready_timeout,
            )
        return ok

    def shutdown(self, backend: RunningBackend) -> None:

        try:
            port: int | None = _backend_port_from_url(backend.url)
        except ValueError:
            port = None
        with self._lock:
            info = self._containers.pop(backend.id, None)
            if port is not None:
                self._bound.get(backend.device_name, set()).discard(port)
                self._claimed_ports.get(backend.device_name, set()).discard(port)
        if info is None:
            return
        _, container_name = info
        logger.info(
            "[AgentLauncher] Leaving container %s running for reuse (backend=%s)",
            container_name,
            backend.id,
        )
