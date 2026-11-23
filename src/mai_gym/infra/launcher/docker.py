import contextlib
import logging
import shlex
import subprocess
import threading
import uuid

from mai_gym.infra.launcher._base import Launcher
from mai_gym.infra.launcher._helpers import (
    _HEALTH_INTERVAL,
    _PORT_SCAN_LIMIT,
    _discover_device_id,
    _is_port_free_local,
    _is_port_free_remote,
    _make_backend_id,
    _ssh_base_cmd,
    _wait_for_health,
)
from mai_gym.models.config import DeviceConfig, DockerLaunchConfig
from mai_gym.models.session import RunningBackend

logger = logging.getLogger(__name__)


class DockerLauncher(Launcher):
    def __init__(self) -> None:
        self._owner = uuid.uuid4().hex
        self._containers: dict[str, tuple[DeviceConfig, str, int]] = {}
        self._claimed_ports: dict[str, set[int]] = {}
        self._lock = threading.Lock()

    def _run(
        self, device: DeviceConfig, args: list[str], *, timeout: float = 120
    ) -> subprocess.CompletedProcess[str]:
        command = args
        if device.ssh is not None:
            ssh = device.ssh
            command = [*_ssh_base_cmd(ssh.host, ssh.user, ssh.key), shlex.join(args)]
        return subprocess.run(command, capture_output=True, text=True, timeout=timeout)

    def _allocate_port(self, device: DeviceConfig) -> int:
        assert isinstance(device.launch, DockerLaunchConfig)
        with self._lock:
            claimed = self._claimed_ports.setdefault(device.name, set())
            start = device.launch.port_range_start
            for port in range(start, start + _PORT_SCAN_LIMIT):
                if port in claimed:
                    continue
                ssh = device.ssh
                free = (
                    _is_port_free_remote(ssh.host, ssh.user, ssh.key, port)
                    if ssh is not None
                    else _is_port_free_local(port)
                )
                if free:
                    claimed.add(port)
                    return port
        raise RuntimeError(f"No free service port for {device.name!r}")

    def start_owned_container(
        self, device: DeviceConfig, port: int, *, extra_options: tuple[str, ...] = ()
    ) -> str:
        assert isinstance(device.launch, DockerLaunchConfig)
        launch = device.launch
        name = f"mai-gym-{self._owner[:12]}-{port}"
        args = [
            "docker",
            "run",
            "-d",
            "--rm",
            "--name",
            name,
            "--label",
            f"mai-gym.owner={self._owner}",
            "-p",
            f"{port}:{launch.container_port}",
        ]
        if launch.privileged:
            args.append("--privileged")
        if launch.shm_size:
            args += ["--shm-size", launch.shm_size]
        if launch.env_file:
            args += ["--env-file", launch.env_file]
        for key, value in (launch.env_vars or {}).items():
            args += ["-e", f"{key}={value}"]
        for volume in launch.volumes or []:
            args += ["-v", volume]
        args += list(extra_options)
        args += [launch.image, *launch.command]
        try:
            result = self._run(device, args)
            if result.returncode:
                raise RuntimeError(
                    f"Docker service launch failed for {device.name} (exit {result.returncode})"
                )
        except BaseException:
            self._cleanup(device, name)
            raise
        return name

    def launch(self, device: DeviceConfig) -> RunningBackend:
        assert isinstance(device.launch, DockerLaunchConfig)
        launch = device.launch
        port = self._allocate_port(device)
        name = f"mai-gym-{self._owner[:12]}-{port}"
        try:
            self.start_owned_container(device, port)
            host = device.ssh.host if device.ssh else "localhost"
            url = f"http://{host}:{port}"
            if not _wait_for_health(
                url, retries=max(1, int(launch.health_timeout / _HEALTH_INTERVAL))
            ):
                raise RuntimeError(
                    f"Service {name} did not become ready within {launch.health_timeout}s"
                )
            backend_id = _make_backend_id()
            self._containers[backend_id] = (device, name, port)
            return RunningBackend(
                backend_id,
                url,
                device.name,
                device.type,
                device_id=_discover_device_id(url),
                metadata={"container_name": name, "owner": self._owner},
            )
        except BaseException:
            self._cleanup(device, name)
            with self._lock:
                self._claimed_ports[device.name].discard(port)
            raise

    def stop_owned_container(self, device: DeviceConfig, name: str) -> None:
        self._cleanup(device, name)

    def _cleanup(self, device: DeviceConfig, name: str) -> None:
        if not name.startswith(f"mai-gym-{self._owner[:12]}-"):
            raise RuntimeError("refusing to remove a container owned by another launcher")
        with contextlib.suppress(Exception):
            self._run(device, ["docker", "rm", "-f", name], timeout=30)

    def shutdown(self, backend: RunningBackend) -> None:
        resource = self._containers.pop(backend.id, None)
        if resource is None:
            return
        device, name, port = resource
        self._cleanup(device, name)
        with self._lock:
            self._claimed_ports[device.name].discard(port)
