from typing import Any

from mai_gym.envs.shared.mobile import execute_mobile_decision
from mai_gym.infra.backend import BackendClient


class MobileWorldBackend(BackendClient):
    async def execute_step(
        self, action: dict[str, Any], display_id: int | None, action_id: str | None
    ) -> dict[str, Any]:
        return await execute_mobile_decision(action, lambda child: self.do_step(child, display_id))

    async def init_device(self) -> None:
        await super().init_device()
        if self.backend.device_id is None:
            response = await self._call("get", "/devices", raise_for_status=False)
            if response.status_code == 404:
                response = await self._call("get", "/health")
            else:
                response.raise_for_status()
            data = response.json()
            statuses = data.get("device_status") or {}
            devices = data.get("devices") or [device for device, ready in statuses.items() if ready]
            devices = [device for device in devices if isinstance(device, str) and device]
            if len(devices) != 1:
                raise ValueError(
                    "MobileWorld requires one ready Android device, or an explicit connection.attach.backends[].device_id"
                )
            self.backend.device_id = devices[0]
