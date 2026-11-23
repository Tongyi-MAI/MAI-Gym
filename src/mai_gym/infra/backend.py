import contextlib
import json
import logging
import time
from typing import Any, Protocol, runtime_checkable

import httpx

from mai_gym.models.session import RunningBackend

logger = logging.getLogger(__name__)

_HEALTH_CHECK_TIMEOUT = 5.0
_BACKEND_CALL_TIMEOUT = 30.0
_SCREENSHOT_TIMEOUT = 60.0
_BACKEND_COOLDOWN_SEC = 10.0

_REGISTER_TIMEOUT = 180.0


_CONNECT_TIMEOUT = 5.0


def _timeout(total: float) -> httpx.Timeout:

    return httpx.Timeout(total, connect=min(_CONNECT_TIMEOUT, total))


_KEEPALIVE_EXPIRY = 3.0

_BACKEND_LIMITS = httpx.Limits(
    max_connections=8,
    max_keepalive_connections=2,
    keepalive_expiry=_KEEPALIVE_EXPIRY,
)

_BACKEND_ERRORS = (
    httpx.ConnectError,
    httpx.ReadError,
    httpx.WriteError,
    httpx.ConnectTimeout,
    httpx.ReadTimeout,
    httpx.PoolTimeout,
    httpx.RemoteProtocolError,
    ConnectionError,
    OSError,
)


_STALE_CONNECTION_ERRORS = (
    httpx.RemoteProtocolError,
    httpx.ReadError,
    httpx.WriteError,
)


class BackendUnavailableError(Exception):
    def __init__(self, backend: RunningBackend, retry_in: float) -> None:
        super().__init__(f"Backend {backend.device_name} is unavailable (retry in {retry_in:.0f}s)")
        self.backend = backend
        self.retry_in = retry_in


class BackendCallError(Exception):
    def __init__(self, backend: RunningBackend, cause: Exception) -> None:
        super().__init__(
            f"Backend {backend.device_name} connection lost: {type(cause).__name__}: {cause}"
        )
        self.backend = backend
        self.cause = cause


class UnscoredTaskEvaluationError(Exception):
    def __init__(self, result: dict[str, Any]) -> None:
        self.result = result
        self.status = result["status"]
        super().__init__(f"Task evaluation unscored: {self.status} ({result.get('verifier', '')})")


@runtime_checkable
class Backend(Protocol):
    backend: RunningBackend

    async def health_check(self, *, timeout: float | None = None) -> bool: ...
    async def init_device(self) -> None: ...
    async def screenshot(self, display_id: int | None) -> str: ...
    async def do_step(
        self, action: dict[str, Any], display_id: int | None
    ) -> dict[str, Any] | None: ...
    async def evaluate(self, display_id: int | None, task_id: str | None) -> float | None: ...
    async def list_tasks(self) -> list[dict[str, Any]]: ...
    async def close(self) -> None: ...


class BackendClient:
    def __init__(self, backend: RunningBackend) -> None:
        self.backend = backend

        self._client: httpx.AsyncClient | None = None
        self._down_until: float = 0.0

    def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(
                timeout=_timeout(_BACKEND_CALL_TIMEOUT),
                limits=_BACKEND_LIMITS,
            )
        return self._client

    async def close(self) -> None:
        await self._reset_client()

    def _build_url(self, path: str) -> str:
        return self.backend.url.rstrip("/") + path

    def _mark_down(self) -> None:
        self._down_until = time.monotonic() + _BACKEND_COOLDOWN_SEC
        logger.warning(
            "[BackendClient] Backend %s marked down for %.0fs",
            self.backend.device_name,
            _BACKEND_COOLDOWN_SEC,
        )

    def is_available(self) -> bool:
        return time.monotonic() >= self._down_until

    def _ensure_available(self) -> None:
        if not self.is_available():
            retry_in = self._down_until - time.monotonic()
            raise BackendUnavailableError(self.backend, retry_in)

    async def _reset_client(self) -> None:
        if self._client is not None:
            with contextlib.suppress(Exception):
                await self._client.aclose()
        self._client = None

    async def _request_once(
        self,
        method: str,
        url: str,
        *,
        raise_for_status: bool,
        timeout: float | None,
        kwargs: dict[str, Any],
    ) -> httpx.Response:

        if timeout is not None:
            kwargs["timeout"] = _timeout(timeout)
        r = await self._get_client().request(method, url, **kwargs)
        if raise_for_status:
            r.raise_for_status()
        return r

    async def code_request(self, method: str, path: str, *, timeout: float, **kwargs) -> dict:

        import asyncio

        async with asyncio.timeout(timeout):
            response = await self._request_once(
                method,
                self._build_url(path),
                raise_for_status=True,
                timeout=timeout,
                kwargs=kwargs,
            )
            return response.json()

    async def _call(
        self,
        method: str,
        path: str,
        *,
        raise_for_status: bool = True,
        timeout: float | None = None,
        **kwargs: Any,
    ) -> httpx.Response:
        self._ensure_available()
        method = method.upper()
        url = self._build_url(path)
        try:
            return await self._request_once(
                method,
                url,
                raise_for_status=raise_for_status,
                timeout=timeout,
                kwargs=kwargs,
            )
        except _STALE_CONNECTION_ERRORS as exc:
            if path in {"/task/init", "/task/register"}:
                self._mark_down()
                await self._reset_client()
                raise BackendCallError(self.backend, exc) from exc
            logger.debug(
                "[BackendClient] %s: stale connection on %s %s (%s); reconnecting",
                self.backend.device_name,
                method,
                path,
                type(exc).__name__,
            )
            await self._reset_client()
        except _BACKEND_ERRORS as exc:
            self._mark_down()
            await self._reset_client()
            raise BackendCallError(self.backend, exc) from exc

        try:
            return await self._request_once(
                method,
                url,
                raise_for_status=raise_for_status,
                timeout=timeout,
                kwargs=kwargs,
            )
        except _BACKEND_ERRORS as exc:
            self._mark_down()
            await self._reset_client()
            raise BackendCallError(self.backend, exc) from exc

    async def health_check(self, *, timeout: float | None = None) -> bool:

        effective = timeout if timeout is not None else _HEALTH_CHECK_TIMEOUT
        try:
            r = await self._get_client().get(
                self._build_url("/health"),
                timeout=_timeout(effective),
            )
            ok = r.status_code == 200 and r.json().get("ok", True)
            if ok:
                self._down_until = 0.0
            return ok
        except _BACKEND_ERRORS as exc:
            logger.warning(
                "[BackendClient] Health check failed for %s: %s",
                self.backend.device_name,
                exc,
            )
            self._mark_down()
            await self._reset_client()
            return False
        except Exception as exc:
            logger.warning(
                "[BackendClient] Health check failed for %s: %s",
                self.backend.device_name,
                exc,
            )
            return False

    async def init_device(self) -> None:
        payload: dict[str, Any] = {}
        if self.backend.device_id:
            payload["device"] = self.backend.device_id
        await self._call("post", "/init", json=payload)

    async def screenshot(self, display_id: int | None) -> str:
        params: dict[str, Any] = {
            "device": self.backend.device_id,
            "return_b64": "true",
        }
        if display_id is not None:
            params["display_id"] = display_id
        r = await self._call(
            "get",
            "/screenshot",
            params=params,
            timeout=_SCREENSHOT_TIMEOUT,
        )
        return r.json().get("b64_png", "")

    async def device_state(self) -> dict[str, Any]:

        params: dict[str, Any] = {}
        if self.backend.device_id:
            params["device"] = self.backend.device_id
        try:
            r = await self._call("get", "/state", params=params, timeout=10.0)
            out = r.json()
            return out if isinstance(out, dict) else {}
        except Exception:
            return {}

    async def do_step(
        self, action: dict[str, Any], display_id: int | None
    ) -> dict[str, Any] | None:
        body: dict[str, Any] = {"action": action}

        if display_id is not None:
            body["display_id"] = display_id

        if self.backend.device_id:
            body["device"] = self.backend.device_id
        resp = await self._call("post", "/step", json=body)

        try:
            return resp.json()
        except Exception:
            return None

    async def go_home(self, display_id: int | None) -> None:
        await self.do_step({"action_type": "navigate_home"}, display_id)

    async def task_init(
        self,
        task_name: str,
        display_id: int | None = None,
    ) -> None:

        body: dict[str, Any] = {
            "task_name": task_name,
            "req_device": self.backend.device_id,
        }
        if display_id is not None:
            body["display_id"] = display_id
        await self._call("post", "/task/init", json=body, timeout=300.0)

    async def task_goal(
        self,
        task_name: str,
        display_id: int | None = None,
    ) -> str | None:

        params: dict[str, Any] = {"task_name": task_name}
        if display_id is not None:
            params["display_id"] = display_id
        r = await self._call(
            "get",
            "/task/goal",
            params=params,
            raise_for_status=False,
        )
        if r.status_code != 200:
            logger.warning(
                "[BackendClient] task_goal(%s) non-200: status=%d body=%s",
                task_name,
                r.status_code,
                r.text[:200],
            )
            return None
        data = r.json()
        if not isinstance(data, str) or not data.strip():
            logger.warning(
                "[BackendClient] task_goal(%s) empty/non-string goal: %r",
                task_name,
                data,
            )
            return None
        return data

    async def task_eval(
        self,
        task_name: str,
        display_id: int | None = None,
        *,
        strict: bool = False,
    ) -> tuple[float, str] | None:

        try:
            params: dict[str, Any] = {"task_name": task_name}
            if display_id is not None:
                params["display_id"] = display_id
            r = await self._call(
                "get",
                "/task/eval",
                params=params,
                raise_for_status=strict,
                json={"task_name": task_name, "req_device": self.backend.device_id},
            )
            if r.status_code != 200:
                logger.warning(
                    "[BackendClient] task_eval %s non-200: %d body=%s",
                    task_name,
                    r.status_code,
                    r.text[:500],
                )
                return None
            data = r.json()
            if data.get("score") is None and isinstance(data.get("reason"), str):
                with contextlib.suppress(ValueError, TypeError):
                    reason = json.loads(data["reason"])
                    if (
                        isinstance(reason, dict)
                        and reason.get("evaluation_contract") == "railway_delivery_v120"
                        and reason.get("status") in {"insufficient_evidence", "error"}
                    ):
                        raise UnscoredTaskEvaluationError(reason)
            from mai_gym.models.evaluation import EvaluationScore

            if data.get("score") is None:
                return None
            return EvaluationScore(float(data.get("score", 0.0)), data.get("success")), data.get(
                "reason", ""
            )
        except UnscoredTaskEvaluationError:
            raise
        except Exception as exc:
            if strict:
                raise
            logger.warning("[BackendClient] task_eval %s exception: %s", task_name, exc)
            return None

    async def task_tear_down(
        self,
        task_name: str,
        display_id: int | None = None,
    ) -> None:

        try:
            body: dict[str, Any] = {
                "task_name": task_name,
                "req_device": self.backend.device_id,
            }
            if display_id is not None:
                body["display_id"] = display_id
            await self._call(
                "post",
                "/task/tear_down",
                json=body,
                raise_for_status=False,
            )
        except Exception as exc:
            logger.warning("[BackendClient] task_tear_down failed: %s", exc)

    async def evaluate(self, display_id: int | None, task_id: str | None) -> float | None:

        try:
            r = await self._call(
                "post",
                "/evaluate",
                raise_for_status=False,
                params={
                    "device": self.backend.device_id,
                    "display_id": display_id,
                },
                json={"task_id": task_id},
            )
            if r.status_code == 200:
                from mai_gym.models.evaluation import EvaluationScore

                data = r.json()
                reward = data.get("reward")
                return (
                    EvaluationScore(float(reward), data.get("success"))
                    if reward is not None
                    else None
                )
            return None
        except Exception:
            return None

    async def list_tasks(self) -> list[dict[str, Any]]:

        try:
            r = await self._call("get", "/task/list", raise_for_status=False)
            if r.status_code == 200:
                data = r.json()
                if isinstance(data, list):
                    return [
                        {
                            **t,
                            "task_id": t.get("name"),
                            "tags": t.get("tags", []),
                            "apps": t.get("apps", []),
                        }
                        for t in data
                        if t.get("name")
                    ]
                if isinstance(data, dict):
                    return data.get("tasks", [])
        except Exception:
            pass

        try:
            r = await self._call("get", "/tasks", raise_for_status=False)
            if r.status_code == 200:
                return r.json().get("tasks", [])
        except Exception:
            pass
        return []

    async def register_task_set(
        self, set_id: str, package: bytes, filename: str = "package.zip"
    ) -> dict[str, Any]:

        r = await self._call(
            "post",
            "/task/register",
            data={"set_id": set_id},
            files={"package": (filename, package, "application/zip")},
            timeout=_REGISTER_TIMEOUT,
        )
        return r.json()

    async def unregister_task_set(self, set_id: str) -> dict[str, Any]:

        r = await self._call("delete", f"/task/register/{set_id}", timeout=_REGISTER_TIMEOUT)
        return r.json()

    async def list_task_sets(self) -> list[dict[str, Any]]:

        try:
            r = await self._call("get", "/task/sets", raise_for_status=False)
            if r.status_code == 200:
                data = r.json()
                if isinstance(data, list):
                    return data
        except Exception:
            pass
        return []
