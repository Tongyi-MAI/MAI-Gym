from typing import Any, cast

from mai_gym.infra.backend import Backend, BackendClient
from mai_gym.models.session import SessionState, SessionTask


class NativeStrategy:
    async def on_reset(self, client: Backend, state: SessionState, **kwargs: Any) -> str:
        task_id = kwargs.get("task_id")
        if not task_id:
            raise ValueError("A native benchmark episode requires a manifest task ID")
        backend = cast(BackendClient, client)
        await backend.task_init(task_id, display_id=state.context_id)
        goal = await backend.task_goal(task_id, display_id=state.context_id)
        if not goal:
            raise ValueError(f"Native initialization returned no instruction for {task_id}")
        return goal

    async def on_evaluate(
        self, client: Backend, state: SessionState, **kwargs: Any
    ) -> tuple[float | None, str | None]:
        result = await cast(BackendClient, client).task_eval(
            kwargs["task_id"], display_id=state.context_id, strict=True
        )
        if result is None:
            raise RuntimeError("Native evaluator returned no score; the episode remains ungraded")
        return result

    async def on_tear_down(self, client: Backend, state: SessionState, **kwargs: Any) -> None:
        if kwargs.get("task_id"):
            await cast(BackendClient, client).task_tear_down(
                kwargs["task_id"], display_id=state.context_id
            )

    async def on_attach_task(
        self,
        client: Backend,
        state: SessionState,
        *,
        task: SessionTask,
        package: bytes | None = None,
        world_path: str | None = None,
    ) -> None:
        raise ValueError("Native benchmarks load only their pinned task manifests")

    async def on_detach_task(
        self, client: Backend, state: SessionState, *, task: SessionTask
    ) -> None:
        return None
