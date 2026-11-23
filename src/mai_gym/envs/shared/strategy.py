import logging
from typing import Protocol

from mai_gym.infra.backend import Backend
from mai_gym.models.session import SessionState, SessionTask

logger = logging.getLogger(__name__)


class EnvStrategy(Protocol):
    async def on_reset(
        self,
        client: Backend,
        state: SessionState,
        *,
        task_id: str | None,
        goal: str | None,
        target_apps: list[str] | None,
        seed_task_id: str | None,
        task_phase: str | None = None,
    ) -> str | None: ...

    async def on_attach_task(
        self,
        client: Backend,
        state: SessionState,
        *,
        task: SessionTask,
        package: bytes | None = None,
        world_path: str | None = None,
    ) -> None: ...

    async def on_detach_task(
        self,
        client: Backend,
        state: SessionState,
        *,
        task: SessionTask,
    ) -> None: ...

    async def on_evaluate(
        self,
        client: Backend,
        state: SessionState,
        *,
        task_id: str | None,
        category: str,
    ) -> tuple[float | None, str | None]: ...

    async def on_tear_down(
        self,
        client: Backend,
        state: SessionState,
        *,
        task_id: str | None,
        category: str,
    ) -> None: ...
