import logging
from typing import Any

from mai_gym.infra.backend import Backend
from mai_gym.models.operation_error import OperationError, bounded_detail
from mai_gym.models.session import SessionState, SessionTask

logger = logging.getLogger(__name__)


class MobileWorldStrategy:
    async def on_reset(
        self,
        client: Backend,
        state: SessionState,
        **kwargs: Any,
    ) -> str | None:
        task_id = kwargs.get("task_id")
        if not task_id:
            logger.warning(
                "[MobileWorldStrategy] on_reset called with empty task_id "
                "(kwargs keys=%s) — no goal will be resolved",
                sorted(kwargs.keys()),
            )
            return None
        await client.task_init(task_id)
        mw_goal = await client.task_goal(task_id)
        if not mw_goal:
            logger.warning(
                "[MobileWorldStrategy] task_goal empty for task_id=%s",
                task_id,
            )
        return mw_goal or None

    async def on_evaluate(
        self,
        client: Backend,
        state: SessionState,
        **kwargs: Any,
    ) -> tuple[float | None, str | None]:
        task_id = kwargs.get("task_id")
        if not task_id:
            return None, None
        if state.session_task is not None:
            result = await client.task_eval(task_id, strict=True)
        else:
            result = await client.task_eval(task_id)
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
            await client.task_tear_down(task_id)

    async def on_attach_task(
        self,
        client: Backend,
        state: SessionState,
        *,
        task: SessionTask,
        package: bytes | None = None,
        world_path: str | None = None,
    ) -> None:

        if not task.set_id or package is None:
            raise ValueError("mobileworld session task requires set_id + package (zip bytes)")
        required = {name for name in (task.task_class, task.golden_class) if name}
        if not required:
            raise OperationError(
                "mobileworld session task requires at least one expected task class",
                status_code=422,
                error_detail={
                    "code": "task_registration_error",
                    "phase": "attach",
                    "stage": "registry_import",
                    "failure_kind": "task",
                    "recovery": "repair_task",
                    "required_classes": [],
                    "registered_classes": [],
                    "missing_classes": [],
                },
            )
        result = await client.register_task_set(task.set_id, package)
        result = result if isinstance(result, dict) else {}
        registered = result.get("registered") or []
        if not isinstance(registered, (list, tuple)):
            registered = []
        registered = {name for name in registered if isinstance(name, str)}
        missing = required - registered
        errors = result.get("errors") or {}
        if missing or errors:
            stage = "registry_import"
            if (
                isinstance(errors, dict)
                and errors
                and all(
                    isinstance(error, str) and error.startswith("task_construct:")
                    for error in errors.values()
                )
            ):
                stage = "task_construct"
            raise OperationError(
                f"Task registration failed; missing classes: {sorted(missing)}; "
                f"loading errors: {bounded_detail(errors)}",
                status_code=422,
                error_detail={
                    "code": "task_registration_error",
                    "phase": "attach",
                    "stage": stage,
                    "failure_kind": "task",
                    "recovery": "repair_task",
                    "required_classes": sorted(required),
                    "registered_classes": sorted(registered),
                    "missing_classes": sorted(missing),
                    "upstream_status": 200,
                    "upstream_detail": {"errors": bounded_detail(errors)},
                },
            )

    async def on_detach_task(
        self,
        client: Backend,
        state: SessionState,
        *,
        task: SessionTask,
    ) -> None:
        if not task.set_id:
            return
        try:
            await client.unregister_task_set(task.set_id)
        except Exception as exc:
            logger.warning(
                "[MobileWorldStrategy] detach unregister set_id=%s failed "
                "(pod will be recycled anyway): %s",
                task.set_id,
                exc,
            )


def create_strategy(*, requested_type: str, **settings: Any) -> MobileWorldStrategy:
    return MobileWorldStrategy()
