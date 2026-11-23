import logging
from typing import Any

from mai_gym.envs.mobileworld.strategy import MobileWorldStrategy
from mai_gym.infra.backend import Backend
from mai_gym.models.session import SessionState

logger = logging.getLogger(__name__)


class WebEnvStrategy(MobileWorldStrategy):
    async def on_reset(
        self,
        client: Backend,
        state: SessionState,
        **kwargs: Any,
    ) -> str | None:
        task_id = kwargs.get("task_id")
        if not task_id:
            logger.warning("[WebEnvStrategy] on_reset called with empty task_id")
            return None
        display_id = state.context_id
        await client.task_init(task_id, display_id=display_id)
        goal = await client.task_goal(task_id, display_id=display_id)
        return goal or None

    async def on_evaluate(
        self,
        client: Backend,
        state: SessionState,
        **kwargs: Any,
    ) -> tuple[float | None, str | None]:
        task_id = kwargs.get("task_id")
        if not task_id:
            return None, None
        display_id = state.context_id
        result = await client.task_eval(task_id, display_id=display_id)
        if result is not None:
            return result
        return None, None

    async def on_tear_down(
        self,
        client: Backend,
        state: SessionState,
        **kwargs: Any,
    ) -> None:
        task_id = kwargs.get("task_id")
        if task_id:
            await client.task_tear_down(
                task_id,
                display_id=state.context_id,
            )


def create_strategy(*, requested_type: str, **settings: Any) -> WebEnvStrategy:
    return WebEnvStrategy()
