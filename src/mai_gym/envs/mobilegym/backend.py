from typing import Any

from mai_gym.envs.shared.backend import PooledBackend
from mai_gym.envs.shared.mobile import execute_mobile_decision


class MobileGymBackend(PooledBackend):
    async def execute_step(
        self, action: dict[str, Any], display_id: int | None, action_id: str | None
    ) -> dict[str, Any]:
        return await execute_mobile_decision(
            action,
            lambda child: self.do_step(child, display_id),
            terminal=lambda child: self.record_episode_terminal(child, display_id),
        )

    async def record_episode_terminal(self, action: dict[str, Any], display_id: int | None) -> None:
        response = await self._call(
            "post",
            "/episode/terminal",
            json={"action": action, "display_id": display_id},
            raise_for_status=False,
        )
        if response.status_code == 404:
            answer = action.get("answer", action.get("text"))
            if answer is not None:
                await self.do_step({"action_type": "answer", "text": str(answer)}, display_id)
        else:
            response.raise_for_status()
