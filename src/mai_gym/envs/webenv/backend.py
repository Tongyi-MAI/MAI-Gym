from typing import Any

from mai_gym.envs.shared.backend import BrowserBackend
from mai_gym.envs.shared.mobile import execute_mobile_decision


class WebEnvBackend(BrowserBackend):
    async def execute_step(
        self, action: dict[str, Any], display_id: int | None, action_id: str | None
    ) -> dict[str, Any]:
        return await execute_mobile_decision(
            action,
            lambda child: self.do_step(child, display_id),
            terminal=lambda child: self.record_episode_terminal(child, display_id),
        )

    async def record_episode_terminal(self, action: dict[str, Any], display_id: int | None) -> None:
        answer = action.get("answer", action.get("text"))
        if answer is not None:
            await self.do_step({"action_type": "answer", "text": answer}, display_id)
