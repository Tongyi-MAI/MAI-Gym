from __future__ import annotations

import abc
import logging
import subprocess
from typing import TYPE_CHECKING

from mai_gym.models.config import DeviceConfig
from mai_gym.models.session import RunningBackend

if TYPE_CHECKING:
    from mai_gym.infra.task_artifacts import TaskArtifactSpec

logger = logging.getLogger(__name__)


class Launcher(abc.ABC):
    @abc.abstractmethod
    def launch(self, device: DeviceConfig) -> RunningBackend: ...

    @abc.abstractmethod
    def shutdown(self, backend: RunningBackend) -> None: ...

    def shutdown_for_manager_restart(self, backend: RunningBackend) -> None:

        self.shutdown(backend)

    def batch_launch(self, device: DeviceConfig, count: int) -> list[RunningBackend]:

        out: list[RunningBackend] = []
        for _ in range(count):
            out.append(self.launch(device))
        return out

    def relaunch(self, device: DeviceConfig, backend: RunningBackend) -> bool:

        logger.warning("[Launcher] relaunch not supported for %s", backend.device_name)
        return False

    def resize_pool(self, device: DeviceConfig, replicas: int) -> str:

        raise NotImplementedError(f"{type(self).__name__} does not support live pool resizing")

    def pool_status(self, device: DeviceConfig) -> dict[str, object]:

        raise NotImplementedError(f"{type(self).__name__} does not expose warm pool status")

    def prepare_task_artifacts(
        self,
        device: DeviceConfig,
        backend: RunningBackend,
        artifacts: tuple[TaskArtifactSpec, ...],
    ) -> list[dict[str, object]]:

        return []


def _terminate_process(proc: subprocess.Popen[bytes], label: str, backend_id: str) -> None:

    logger.info("[%s] Terminating backend %s", label, backend_id)
    proc.terminate()
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        proc.kill()
