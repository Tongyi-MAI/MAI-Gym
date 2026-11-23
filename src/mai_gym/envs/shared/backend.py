import logging
from typing import Any

import httpx

from mai_gym.infra.backend import BackendClient, BackendUnavailableError
from mai_gym.models.evaluation import EvaluationScore
from mai_gym.models.session import RunningBackend

logger = logging.getLogger(__name__)


class PooledBackend(BackendClient):
    def __init__(self, backend: RunningBackend) -> None:
        super().__init__(backend)
        self._owned_contexts: set[int] = set()

    async def acquire_context(self) -> int:
        response = await self._call(
            "post", "/display/acquire", params={"device": self.backend.device_id}
        )
        data = response.json()
        context_id = data.get("logical_id")
        if context_id is None:
            context_id = data["sf_display_id"]
        if not isinstance(context_id, int):
            raise ValueError("Worker returned an invalid context ID")
        self._owned_contexts.add(context_id)
        return context_id

    async def release_context(self, context_id: int) -> None:
        params: dict[str, str | int] = {"display_id": context_id}
        if self.backend.device_id is not None:
            params["device"] = self.backend.device_id
        response = await self._call(
            "post",
            "/display/release",
            params=params,
            raise_for_status=False,
        )
        if response.status_code != 409:
            response.raise_for_status()
        self._owned_contexts.discard(context_id)

    async def close(self) -> None:
        try:
            for context_id in tuple(self._owned_contexts):
                try:
                    await self.release_context(context_id)
                except Exception:
                    logger.warning(
                        "Worker context %s could not be released", context_id, exc_info=True
                    )
        finally:
            self._owned_contexts.clear()
            await super().close()


class BrowserBackend(PooledBackend):
    async def prepare_apps(self, context_id: int, apps: list[str]) -> None:
        for app in apps:
            try:
                response = await self._call(
                    "post",
                    "/display/claim_app",
                    raise_for_status=False,
                    params={
                        "device": self.backend.device_id,
                        "display_id": context_id,
                        "package_name": app,
                    },
                )
                if response.status_code != 409:
                    response.raise_for_status()
            except (httpx.HTTPError, BackendUnavailableError) as exc:
                logger.warning("Browser app %s could not be prepared: %s", app, exc)


class NativeBackend(PooledBackend):
    async def record_episode_terminal(self, action: dict[str, Any], display_id: int | None) -> None:
        await self._call(
            "post", "/episode/terminal", json={"action": action, "display_id": display_id}
        )

    async def execute_step(
        self, action: dict[str, Any], display_id: int | None, action_id: str | None
    ) -> dict[str, Any]:
        response = await self._call(
            "post",
            "/step",
            json={"action": action, "display_id": display_id, "action_id": action_id},
        )
        result: dict[str, Any] = response.json()
        return result

    async def task_eval(
        self, task_name: str, display_id: int | None = None, *, strict: bool = False
    ) -> tuple[float, str] | None:
        result = await self.code_request(
            "get",
            "/task/eval",
            timeout=900,
            params={"task_name": task_name, "display_id": display_id},
        )
        if result.get("score") is None:
            raise RuntimeError("Native evaluator returned no score")
        return EvaluationScore(float(result["score"]), result.get("success")), str(
            result.get("reason", "")
        )
