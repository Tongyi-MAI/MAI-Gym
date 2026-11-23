import logging
import subprocess

from mai_gym.infra.launcher._base import Launcher, _terminate_process
from mai_gym.infra.launcher._helpers import (
    _discover_device_id,
    _make_backend_id,
    _ssh_base_cmd,
    _wait_for_health,
)
from mai_gym.models.config import CommandLaunchConfig, DeviceConfig
from mai_gym.models.session import RunningBackend

logger = logging.getLogger(__name__)


class SshCommandLauncher(Launcher):
    def __init__(self) -> None:
        self._procs: dict[str, subprocess.Popen[bytes]] = {}

    def launch(self, device: DeviceConfig) -> RunningBackend:
        assert isinstance(device.launch, CommandLaunchConfig)
        ssh_conn = device.ssh
        assert ssh_conn is not None
        port = device.launch.port
        remote_cmd = device.launch.command.format(port=port)
        ssh_cmd = _ssh_base_cmd(ssh_conn.host, ssh_conn.user, ssh_conn.key)
        ssh_cmd.append(remote_cmd)
        logger.info("[SshCommandLauncher] Starting on %s: %s", ssh_conn.host, remote_cmd)
        proc = subprocess.Popen(ssh_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        url = f"http://{ssh_conn.host}:{port}"
        if not _wait_for_health(url):
            proc.terminate()
            raise RuntimeError(f"Backend at {url} failed health check")

        device_id = _discover_device_id(url)
        backend_id = _make_backend_id()
        self._procs[backend_id] = proc
        logger.info("[SshCommandLauncher] Backend %s ready at %s", backend_id, url)
        return RunningBackend(
            id=backend_id,
            url=url,
            device_name=device.name,
            env_type=device.type,
            device_id=device_id,
        )

    def shutdown(self, backend: RunningBackend) -> None:
        proc = self._procs.pop(backend.id, None)
        if proc is not None:
            _terminate_process(proc, "SshCommandLauncher", backend.id)
