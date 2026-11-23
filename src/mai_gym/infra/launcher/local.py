import logging
import subprocess

from mai_gym.infra.launcher._base import Launcher, _terminate_process
from mai_gym.infra.launcher._helpers import (
    _discover_device_id,
    _make_backend_id,
    _wait_for_health,
)
from mai_gym.models.config import CommandLaunchConfig, DeviceConfig
from mai_gym.models.session import RunningBackend

logger = logging.getLogger(__name__)


class LocalLauncher(Launcher):
    def __init__(self) -> None:
        self._processes: dict[str, subprocess.Popen[bytes]] = {}

    def launch(self, device: DeviceConfig) -> RunningBackend:
        assert isinstance(device.launch, CommandLaunchConfig)
        port = device.launch.port
        cmd = device.launch.command.format(port=port)
        logger.info("[LocalLauncher] Starting: %s", cmd)
        proc = subprocess.Popen(
            cmd, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )
        if proc.poll() is not None:
            raise RuntimeError(f"Process exited immediately: {cmd}")

        url = f"http://localhost:{port}"
        if not _wait_for_health(url):
            proc.terminate()
            raise RuntimeError(f"Backend at {url} failed health check")

        device_id = _discover_device_id(url)
        backend_id = _make_backend_id()
        self._processes[backend_id] = proc
        logger.info(
            "[LocalLauncher] Backend %s ready at %s (device=%s)",
            backend_id,
            url,
            device_id,
        )
        return RunningBackend(
            id=backend_id,
            url=url,
            device_name=device.name,
            env_type=device.type,
            device_id=device_id,
        )

    def shutdown(self, backend: RunningBackend) -> None:
        proc = self._processes.pop(backend.id, None)
        if proc is not None:
            _terminate_process(proc, "LocalLauncher", backend.id)
