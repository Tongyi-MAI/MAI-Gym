import logging
from typing import Any

from mai_gym.infra.launcher._base import Launcher
from mai_gym.infra.launcher._helpers import _make_backend_id
from mai_gym.models.config import DeviceConfig
from mai_gym.models.session import RunningBackend

logger = logging.getLogger(__name__)


class AttachLauncher(Launcher):
    def __init__(self, backends: list[Any]) -> None:

        self._available: list[Any] = list(backends)
        self._claimed: dict[str, Any] = {}

    def launch(self, device: DeviceConfig) -> RunningBackend:
        if not self._available:
            raise RuntimeError(
                f"AttachLauncher for device {device.name!r} exhausted: "
                f"all {len(self._claimed)} configured backend(s) already claimed."
            )
        ab = self._available.pop(0)
        backend_id = _make_backend_id()
        self._claimed[backend_id] = ab
        logger.info(
            "[AttachLauncher] Attached backend %s -> %s (device=%s, container=%s)",
            backend_id,
            ab.url,
            device.name,
            ab.container_name,
        )
        return RunningBackend(
            id=backend_id,
            url=ab.url,
            device_name=device.name,
            env_type=device.type,
            device_id=getattr(ab, "device_id", None),
            metadata={
                "container_name": ab.container_name,
                "chromium_port": ab.chromium_port,
                "vlc_port": ab.vlc_port,
                "vnc_port": ab.vnc_port,
            },
        )

    def shutdown(self, backend: RunningBackend) -> None:
        ab = self._claimed.pop(backend.id, None)
        if ab is None:
            return

        self._available.append(ab)
        logger.info(
            "[AttachLauncher] Released backend %s -> %s (no shutdown)",
            backend.id,
            ab.url,
        )
