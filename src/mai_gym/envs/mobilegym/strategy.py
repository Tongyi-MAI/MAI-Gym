import logging
from typing import Any

from mai_gym.infra.backend import Backend
from mai_gym.models.session import SessionState

logger = logging.getLogger(__name__)


class MobileGymStrategy:
    async def on_reset(
        self,
        client: Backend,
        state: SessionState,
        **kwargs: Any,
    ) -> str | None:
        task_id = kwargs.get("task_id")
        if not task_id:
            return None
        await client.task_init(task_id, display_id=state.context_id)
        goal = await client.task_goal(task_id, display_id=state.context_id)
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
        result = await client.task_eval(task_id, display_id=state.context_id)
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
            await client.task_tear_down(task_id, display_id=state.context_id)


def create_strategy(*, requested_type: str, **settings: Any) -> MobileGymStrategy:
    return MobileGymStrategy()
