import logging
import threading
import time
from typing import Any

import httpx

from mai_gym.infra.launcher._base import Launcher
from mai_gym.infra.launcher._helpers import (
    _OSW_FRESH_WAVE_SIZE,
    _OSW_LAUNCH_STAGGER,
    _OSW_REUSE_HEALTH_TIMEOUT,
    _OSW_SERVER_PORT_INSIDE,
    _make_backend_id,
)
from mai_gym.models.config import DeviceConfig, DockerLaunchConfig
from mai_gym.models.session import RunningBackend

logger = logging.getLogger(__name__)


class OSWorldDockerProfile(Launcher):
    def __init__(self) -> None:

        self._backends_meta: dict[str, tuple[Any, str | None, str]] = {}
        self._reused: set[str] = set()

    @staticmethod
    def _host_port_or(
        ports_attrs: dict[str, Any],
        internal: str,
        default: int,
    ) -> int:

        mapping = ports_attrs.get(internal)
        try:
            return int(mapping[0]["HostPort"])
        except (KeyError, ValueError, IndexError, TypeError):
            return default

    def _list_reusable_containers(
        self,
        image: str,
    ) -> list[tuple[Any, int]]:

        try:
            import docker as docker_sdk
        except ImportError as exc:
            raise RuntimeError(
                "OSWorldDockerProfile requires the 'docker' SDK. Run 'uv sync --extra osworld'."
            ) from exc

        docker_api: Any = docker_sdk
        client = docker_api.from_env()
        out: list[tuple[Any, int]] = []
        for container in client.containers.list():
            tags = container.image.tags or []

            if image not in tags and not any(t.split(":", 1)[0] == image for t in tags):
                continue
            ports = (container.attrs.get("NetworkSettings") or {}).get("Ports") or {}
            mapping = ports.get(_OSW_SERVER_PORT_INSIDE)
            if not mapping:
                continue
            try:
                host_port = int(mapping[0]["HostPort"])
            except (KeyError, ValueError, IndexError):
                continue
            try:
                resp = httpx.get(
                    f"http://localhost:{host_port}/screenshot",
                    timeout=_OSW_REUSE_HEALTH_TIMEOUT,
                )
                if resp.status_code != 200:
                    continue
            except Exception:
                continue
            out.append((container, host_port))
        return out

    def launch(self, device: DeviceConfig) -> RunningBackend:
        backends = self.batch_launch(device, 1)
        if not backends:
            raise RuntimeError(f"OSWorld launch failed for {device.name}")
        return backends[0]

    def batch_launch(self, device: DeviceConfig, count: int) -> list[RunningBackend]:

        try:
            from desktop_env.desktop_env import DesktopEnv
        except ImportError as exc:
            raise RuntimeError(
                "OSWorldDockerProfile requires the 'osworld' extra and the "
                "desktop_env package. Run 'uv sync --extra osworld' then "
                "'uv pip install -e /opt/OSWorld' before "
                "serving a device with type=osworld."
            ) from exc

        assert isinstance(device.launch, DockerLaunchConfig)
        launch = device.launch
        if not launch.path_to_vm:
            raise RuntimeError(
                f"Device {device.name} (type=osworld) requires "
                f"launch.path_to_vm — the host path of the QCOW2 image."
            )

        backends: list[RunningBackend] = []

        if launch.reuse_existing:
            already_attached = {meta[2] for meta in self._backends_meta.values()}
            candidates = [
                (c, p)
                for (c, p) in self._list_reusable_containers(launch.image)
                if c.id not in already_attached
            ]
            claim = candidates[:count]
        else:
            claim = []
        fresh_count = count - len(claim)

        logger.info(
            "[OSWorldDockerProfile] %s: reusing %d existing container(s), launching %d new",
            device.name,
            len(claim),
            fresh_count,
        )

        for container, host_port in claim:
            backend_id = _make_backend_id()
            url = f"http://localhost:{host_port}"
            self._backends_meta[backend_id] = (None, None, container.id)
            self._reused.add(backend_id)

            ports_attrs = (container.attrs.get("NetworkSettings") or {}).get("Ports") or {}
            metadata = {
                "container_name": container.name,
                "chromium_port": self._host_port_or(ports_attrs, "9222/tcp", 9222),
                "vlc_port": self._host_port_or(ports_attrs, "8080/tcp", 8080),
                "vnc_port": self._host_port_or(ports_attrs, "8006/tcp", 8006),
            }
            backends.append(
                RunningBackend(
                    id=backend_id,
                    url=url,
                    device_name=device.name,
                    env_type=device.type,
                    device_id=None,
                    metadata=metadata,
                )
            )
            logger.info(
                "[OSWorldDockerProfile] Reusing container %s on port %d → backend %s",
                container.short_id,
                host_port,
                backend_id,
            )

        if fresh_count == 0:
            return backends

        for wave_start in range(0, fresh_count, _OSW_FRESH_WAVE_SIZE):
            wave_end = min(wave_start + _OSW_FRESH_WAVE_SIZE, fresh_count)
            wave_size = wave_end - wave_start
            logger.info(
                "[OSWorldDockerProfile] Launching wave %d-%d/%d (size=%d)",
                wave_start + 1,
                wave_end,
                fresh_count,
                wave_size,
            )
            wave_results: list[tuple[Any, int, str] | None] = [None] * wave_size
            wave_errors: dict[int, BaseException] = {}

            def _launch_one(
                local_idx: int,
                _results: list = wave_results,
                _errors: dict = wave_errors,
            ) -> None:
                try:
                    de = DesktopEnv(
                        provider_name="docker",
                        path_to_vm=launch.path_to_vm,
                        action_space="pyautogui",
                        headless=True,
                        os_type="Ubuntu",
                        require_a11y_tree=False,
                    )
                    container_id = (
                        de.provider.container.id if de.provider.container is not None else ""
                    )
                    _results[local_idx] = (de, de.server_port, container_id)
                except Exception as exc:
                    _errors[local_idx] = exc

            threads = [
                threading.Thread(target=_launch_one, args=(i,), daemon=True)
                for i in range(wave_size)
            ]
            for i, t in enumerate(threads):
                t.start()
                if i < wave_size - 1:
                    time.sleep(_OSW_LAUNCH_STAGGER)
            for t in threads:
                t.join()

            for local_idx, result in enumerate(wave_results):
                global_idx = wave_start + local_idx
                if result is None:
                    logger.error(
                        "[OSWorldDockerProfile] Fresh launch %d/%d failed: %s",
                        global_idx + 1,
                        fresh_count,
                        wave_errors.get(local_idx),
                    )
                    continue
                de, server_port, container_id = result
                backend_id = _make_backend_id()
                url = f"http://localhost:{server_port}"
                self._backends_meta[backend_id] = (
                    de,
                    launch.path_to_vm,
                    container_id,
                )

                metadata = {
                    "desktop_env": de,
                    "container_name": (
                        de.provider.container.name if de.provider.container is not None else ""
                    ),
                    "chromium_port": int(de.chromium_port),
                    "vlc_port": int(de.vlc_port),
                    "vnc_port": int(de.vnc_port),
                }
                backends.append(
                    RunningBackend(
                        id=backend_id,
                        url=url,
                        device_name=device.name,
                        env_type=device.type,
                        device_id=None,
                        metadata=metadata,
                    )
                )
                logger.info(
                    "[OSWorldDockerProfile] Backend %s ready at %s "
                    "(container=%s, chrome=%d, vlc=%d, vnc=%d)",
                    backend_id,
                    url,
                    container_id[:12] if container_id else "?",
                    metadata["chromium_port"],
                    metadata["vlc_port"],
                    metadata["vnc_port"],
                )

        return backends

    def shutdown(self, backend: RunningBackend) -> None:
        meta = self._backends_meta.pop(backend.id, None)
        was_reused = backend.id in self._reused
        self._reused.discard(backend.id)
        if meta is None:
            return
        _, _, container_id = meta

        suffix = "(reused)" if was_reused else "(fresh)"
        logger.info(
            "[OSWorldDockerProfile] Leaving container %s running for reuse %s (backend=%s).",
            container_id[:12] if container_id else "?",
            suffix,
            backend.id,
        )
