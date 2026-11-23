import asyncio
import contextlib
import json
import logging
import os
import random
import re
import threading
import time
import uuid
from collections.abc import AsyncIterator, Iterator
from contextlib import asynccontextmanager, contextmanager
from typing import Any, Literal, cast
from urllib.parse import urlsplit, urlunsplit

import httpx

from mai_gym.models.responses import (
    AcquireResponse,
    BatchSessionResponse,
    EvaluateResponse,
    ListSessionsResponse,
    ListTasksResponse,
    StepResponse,
)

logger = logging.getLogger(__name__)

_TRANSIENT_HTTP_STATUS: frozenset[int] = frozenset({408, 429, 500, 502, 503, 504})


_ASYNC_KEEPALIVE_EXPIRY = 3.0


_GET_NETWORK_MAX_ATTEMPTS = 3


_STEP_BUSY_MAX_ATTEMPTS = 4


_STEP_NETWORK_MAX_RETRIES = 3


_GATEWAY_HTTP_STATUS = (502, 503, 504)


_RESET_BUSY_MAX_ATTEMPTS = 14
_RESET_NETWORK_MAX_RETRIES = 3


_SUBMIT_TIMEOUT = 30.0
_POLL_TIMEOUT = 30.0


_RESET_POLL_MAX_POLLS = 240
_STEP_POLL_MAX_POLLS = 180


_POLL_INTERVAL_BASE = 0.5
_POLL_INTERVAL_CAP = 5.0


_STEP_POLL_INTERVAL_CAP = float(os.environ.get("MAI_GYM_STEP_POLL_CAP", "1.5"))


_POLL_NETWORK_MAX_RETRIES = 8


_RESUBMIT_MAX_ATTEMPTS = 3


def _resp_diag(r: "httpx.Response") -> str:

    try:
        body = (r.text or "")[:600].replace("\n", " ")
        return (
            f" [status={r.status_code} server={r.headers.get('server')!r} "
            f"via={r.headers.get('via')!r} body={body!r}]"
        )
    except Exception:
        return ""


def _looks_like_json(text: str) -> bool:

    stripped = text.lstrip()
    return bool(stripped) and stripped[0] in "{["


def safe_error_data(value: Any) -> Any:

    if isinstance(value, dict):
        return {
            str(key): safe_error_data(item)
            for key, item in list(value.items())[:64]
            if not re.search(
                r"lease|token|authorization|cookie|password|secret|credential|"
                r"package_b64|initial_setup|golden_patch|reward|request_body|headers",
                str(key),
                re.I,
            )
        }
    if isinstance(value, (list, tuple)):
        return [safe_error_data(item) for item in value[:64]]
    if isinstance(value, str):
        value = re.sub(
            r"(?i)(bearer\s+|(?:lease_id|token|password|secret|package_b64)"
            r"[\s'\"]*[:=][\s'\"]*)[^\s,'\"}]+",
            r"\1[redacted]",
            value,
        )
        value = re.sub(r"https?://[^\s/@]+:[^\s/@]+@", "http://[redacted]@", value)
        return value[:3000]
    return value


class APIError(Exception):
    def __init__(
        self,
        status_code: int,
        method: str,
        url: str,
        detail: str = "",
        *,
        body: Any = None,
    ) -> None:
        self.raw_body = detail
        if body is None and detail:
            try:
                body = json.loads(detail)
            except (ValueError, TypeError):
                body = detail
        self.body = body
        fields = body if isinstance(body, dict) else {}
        nested = fields.get("detail")
        if isinstance(nested, dict):
            fields = nested
        self.error_detail = fields.get("error_detail")
        if not isinstance(self.error_detail, dict):
            self.error_detail = None
        self.method = method
        parts = urlsplit(url)
        self.url = urlunsplit((parts.scheme, parts.netloc.rsplit("@", 1)[-1], parts.path, "", ""))
        msg = f"HTTP {status_code} for {method} {self.url}"
        summary = safe_error_data(body if body is not None else detail)
        if summary:
            msg += f" | {summary if isinstance(summary, str) else json.dumps(summary)}"[:3500]
        super().__init__(msg)
        self.status_code = status_code


class PollTimeoutError(Exception):
    pass


class EnvTransportError(RuntimeError):
    pass


class EnvClient:
    def __init__(
        self,
        host: str,
        timeout: float = 600.0,
        max_async_connections: int = 256,
        max_inflight_requests: int | None = None,
        screenshot_transport: Literal["inline", "oss"] | None = None,
        retry_attempts: int | None = None,
        fleet_request_id: str | None = None,
        env_type: str | None = None,
    ) -> None:
        if not host.startswith("http"):
            raise ValueError(f"Unsupported host: {host}")
        if retry_attempts is not None and retry_attempts < 1:
            raise ValueError(f"retry_attempts must be >= 1, got {retry_attempts}")
        self.base_url = host.rstrip("/")
        self.timeout = timeout
        self._env_type = env_type

        self._retry_attempts = retry_attempts
        self._max_async_conns = max(2, max_async_connections)
        requested_transport: str | None = screenshot_transport
        if requested_transport is None:
            requested_transport = os.environ.get("MAI_GYM_SCREENSHOT_TRANSPORT", "").strip() or None
        if requested_transport not in (None, "inline", "oss"):
            raise ValueError(
                "screenshot_transport must be 'inline', 'oss', or None; "
                f"got {requested_transport!r}"
            )

        self._screenshot_transport = cast(
            Literal["inline", "oss"] | None,
            requested_transport,
        )

        if max_inflight_requests is None:
            env_val = os.environ.get("MAI_GYM_MAX_INFLIGHT_REQ")
            if env_val and env_val.strip():
                max_inflight_requests = int(env_val)
        self._max_inflight: int | None = (
            max(1, max_inflight_requests) if max_inflight_requests is not None else None
        )

        self._client = httpx.Client(
            timeout=timeout,
            limits=httpx.Limits(
                max_connections=128,
                max_keepalive_connections=64,
                keepalive_expiry=60.0,
            ),
        )

        self._aclient: httpx.AsyncClient | None = None
        self._aclient_loop: asyncio.AbstractEventLoop | None = None

        self._inflight_sem: asyncio.Semaphore | None = None
        self._inflight_sem_loop: asyncio.AbstractEventLoop | None = None
        self._inflight_throttle_logged = False

        self._lease_cache: dict[str, str] = {}

        self._async_ops: bool | None = None

        self._fleet_request_id = fleet_request_id
        self._fleet_holder_id: str | None = None
        self._fleet_idempotency_key: str | None = None
        self._owns_fleet_request = False
        self._fleet_heartbeat_interval = 30.0
        self._fleet_heartbeat_stop = threading.Event()
        self._fleet_heartbeat_thread: threading.Thread | None = None
        self._fleet_lifecycle_lock = threading.Lock()
        logger.info(
            "[EnvClient] Initialized with base URL: %s "
            "(max_inflight_requests=%s, screenshot_transport=%s)",
            self.base_url,
            "unlimited" if self._max_inflight is None else self._max_inflight,
            self._screenshot_transport or "unspecified (inline)",
        )

    def close(self) -> None:

        self._release_fleet_request()
        self._client.close()

    async def aclose(self) -> None:

        if self._owns_fleet_request:
            await asyncio.to_thread(self._release_fleet_request)
        if self._aclient is not None:
            with contextlib.suppress(Exception):
                await self._aclient.aclose()
            self._aclient = None
            self._aclient_loop = None

    def _sleep_backoff(self, attempt: int, base: float = 1.0, cap: float = 30.0) -> None:
        exp = base * (2 ** max(0, attempt - 1))
        time.sleep(min(cap, exp) * random.uniform(0.7, 1.3))

    _NETWORK_ERRORS = (
        httpx.ConnectError,
        httpx.ReadTimeout,
        httpx.WriteError,
        httpx.RemoteProtocolError,
        httpx.ReadError,
        httpx.ConnectTimeout,
        httpx.WriteTimeout,
        httpx.PoolTimeout,
    )

    def _post(
        self, path: str, payload: dict[str, Any] | None = None, timeout: float | None = None
    ) -> httpx.Response:
        r = self._client.post(
            f"{self.base_url}{path}", json=payload or {}, timeout=timeout or self.timeout
        )
        if r.status_code >= 400:
            raise APIError(r.status_code, "POST", f"{self.base_url}{path}", r.text)
        if path.startswith(("/release/", "/discard/")):
            self._validate_cleanup_response(r, path)
        return r

    def _get(self, path: str, timeout: float | None = None) -> httpx.Response:
        url = f"{self.base_url}{path}"
        for attempt in range(1, _GET_NETWORK_MAX_ATTEMPTS + 1):
            try:
                r = self._client.get(url, timeout=timeout or self.timeout)
                if r.status_code >= 400:
                    raise APIError(r.status_code, "GET", url, r.text)
                return r
            except self._NETWORK_ERRORS as exc:
                if attempt >= _GET_NETWORK_MAX_ATTEMPTS:
                    raise
                logger.warning(
                    "[EnvClient] GET retry %d/%d %s | url=%s",
                    attempt,
                    _GET_NETWORK_MAX_ATTEMPTS - 1,
                    type(exc).__name__,
                    url,
                )
                self._sleep_backoff(attempt)
        raise AssertionError("unreachable")

    def _delete(self, path: str, timeout: float | None = None) -> httpx.Response:
        r = self._client.delete(f"{self.base_url}{path}", timeout=timeout or self.timeout)
        if r.status_code >= 400:
            raise APIError(r.status_code, "DELETE", f"{self.base_url}{path}", r.text)
        return r

    def _post_raw(
        self,
        path: str,
        *,
        params: dict[str, Any],
        content: bytes,
        timeout: float | None = None,
    ) -> httpx.Response:

        r = self._client.post(
            f"{self.base_url}{path}",
            params=params,
            content=content,
            timeout=timeout or self.timeout,
        )
        if r.status_code >= 400:
            raise APIError(r.status_code, "POST", f"{self.base_url}{path}", r.text)
        return r

    def _retry_post(
        self,
        path: str,
        payload: dict[str, Any],
        *,
        label: str,
        retry_on_transient_http: bool = False,
        max_attempts: int = 8,
    ) -> httpx.Response:

        if self._retry_attempts is not None:
            max_attempts = self._retry_attempts
        last_exc: Exception | None = None
        for attempt in range(1, max_attempts + 1):
            try:
                return self._post(path, payload)
            except APIError as e:
                last_exc = e
                is_retryable = (
                    retry_on_transient_http
                    and e.status_code in _TRANSIENT_HTTP_STATUS
                    and attempt < max_attempts
                )
                if is_retryable:
                    logger.warning(
                        "[EnvClient] %s retry %d/%d HTTP %d | url=%s%s",
                        label,
                        attempt,
                        max_attempts,
                        e.status_code,
                        self.base_url,
                        path,
                    )
                    self._sleep_backoff(attempt)
                    continue

                logger.error(
                    "[EnvClient] %s failed HTTP %d after %d attempts | url=%s%s | response: %s",
                    label,
                    e.status_code,
                    attempt,
                    self.base_url,
                    path,
                    e,
                    exc_info=True,
                )
                raise
            except self._NETWORK_ERRORS as e:
                last_exc = e
                if attempt >= max_attempts:
                    break
                logger.warning(
                    "[EnvClient] %s retry %d/%d %s | url=%s%s",
                    label,
                    attempt,
                    max_attempts,
                    type(e).__name__,
                    self.base_url,
                    path,
                )
                self._sleep_backoff(attempt)
        msg = (
            f"[EnvClient] {label} failed after {max_attempts} attempts "
            f"| url={self.base_url}{path} | last error: {last_exc}"
        )
        raise EnvTransportError(msg) from last_exc

    def _get_aclient(self) -> httpx.AsyncClient:
        loop = asyncio.get_running_loop()
        if self._aclient is None or self._aclient_loop is not loop:
            self._aclient = httpx.AsyncClient(
                timeout=self.timeout,
                limits=httpx.Limits(
                    max_connections=self._max_async_conns,
                    max_keepalive_connections=min(self._max_async_conns, 64),
                    keepalive_expiry=_ASYNC_KEEPALIVE_EXPIRY,
                ),
            )
            self._aclient_loop = loop
        return self._aclient

    def _get_inflight_sem(self) -> asyncio.Semaphore | None:

        if self._max_inflight is None:
            return None
        loop = asyncio.get_running_loop()
        if self._inflight_sem is None or self._inflight_sem_loop is not loop:
            self._inflight_sem = asyncio.Semaphore(self._max_inflight)
            self._inflight_sem_loop = loop
        return self._inflight_sem

    async def _reset_aclient(self) -> None:

        old = self._aclient
        self._aclient = None
        self._aclient_loop = None
        if old is None:
            return
        try:
            await old.aclose()
        except Exception as exc:
            logger.debug(
                "[EnvClient] _reset_aclient: aclose raised %s — proceeding (ref already dropped)",
                type(exc).__name__,
            )

    async def _asleep_backoff(self, attempt: int, base: float = 1.0, cap: float = 30.0) -> None:
        exp = base * (2 ** max(0, attempt - 1))
        await asyncio.sleep(min(cap, exp) * random.uniform(0.7, 1.3))

    async def _apost(
        self,
        path: str,
        payload: dict[str, Any] | None = None,
        timeout: float | None = None,
    ) -> httpx.Response:
        url = f"{self.base_url}{path}"
        eff_timeout = timeout or self.timeout
        sem = self._get_inflight_sem()

        if sem is not None and sem.locked() and not self._inflight_throttle_logged:
            self._inflight_throttle_logged = True
            logger.info(
                "[EnvClient] in-flight gate engaged (cap=%d) — "
                "client throttling new requests to protect the proxy",
                self._max_inflight,
            )
        cm = sem if sem is not None else contextlib.nullcontext()
        t0 = time.monotonic()
        try:
            async with cm:
                r = await self._get_aclient().post(
                    url,
                    json=payload or {},
                    timeout=eff_timeout,
                )
        except httpx.HTTPError as exc:
            logger.warning(
                "[EnvClient] POST %s NETFAIL after %.2fs (timeout=%s) %s: %r",
                url,
                time.monotonic() - t0,
                eff_timeout,
                type(exc).__name__,
                str(exc),
            )
            raise
        if r.status_code >= 400:
            if not _looks_like_json(r.text):
                logger.warning("[EnvClient] proxy non-JSON error on POST %s%s", path, _resp_diag(r))
            raise APIError(r.status_code, "POST", url, r.text)
        if path.startswith(("/release/", "/discard/")):
            self._validate_cleanup_response(r, path)
        return r

    async def _apost_raw(
        self,
        path: str,
        *,
        params: dict[str, Any],
        content: bytes,
        timeout: float | None = None,
    ) -> httpx.Response:

        url = f"{self.base_url}{path}"
        eff_timeout = timeout or self.timeout
        sem = self._get_inflight_sem()
        cm = sem if sem is not None else contextlib.nullcontext()
        t0 = time.monotonic()
        try:
            async with cm:
                r = await self._get_aclient().post(
                    url,
                    params=params,
                    content=content,
                    timeout=eff_timeout,
                )
        except httpx.HTTPError as exc:
            logger.warning(
                "[EnvClient] POST-RAW %s NETFAIL after %.2fs (timeout=%s) %s: %r",
                url,
                time.monotonic() - t0,
                eff_timeout,
                type(exc).__name__,
                str(exc),
            )
            raise
        if r.status_code >= 400:
            raise APIError(r.status_code, "POST", url, r.text)
        return r

    async def _adelete(self, path: str, timeout: float | None = None) -> httpx.Response:

        url = f"{self.base_url}{path}"
        eff_timeout = timeout or self.timeout
        sem = self._get_inflight_sem()
        cm = sem if sem is not None else contextlib.nullcontext()
        t0 = time.monotonic()
        try:
            async with cm:
                r = await self._get_aclient().delete(url, timeout=eff_timeout)
        except httpx.HTTPError as exc:
            logger.warning(
                "[EnvClient] DELETE %s NETFAIL after %.2fs (timeout=%s) %s: %r",
                url,
                time.monotonic() - t0,
                eff_timeout,
                type(exc).__name__,
                str(exc),
            )
            raise
        if r.status_code >= 400:
            raise APIError(r.status_code, "DELETE", url, r.text)
        return r

    async def _aretry_post(
        self,
        path: str,
        payload: dict[str, Any],
        *,
        label: str,
        retry_on_transient_http: bool = False,
        max_attempts: int = 8,
    ) -> httpx.Response:

        if self._retry_attempts is not None:
            max_attempts = self._retry_attempts
        last_exc: Exception | None = None
        for attempt in range(1, max_attempts + 1):
            try:
                return await self._apost(path, payload)
            except APIError as e:
                last_exc = e
                is_retryable = (
                    retry_on_transient_http
                    and e.status_code in _TRANSIENT_HTTP_STATUS
                    and attempt < max_attempts
                )
                if is_retryable:
                    logger.warning(
                        "[EnvClient] %s retry %d/%d HTTP %d",
                        label,
                        attempt,
                        max_attempts,
                        e.status_code,
                    )
                    await self._asleep_backoff(attempt)
                    continue
                raise
            except self._NETWORK_ERRORS as e:
                last_exc = e
                if attempt >= max_attempts:
                    break
                logger.warning(
                    "[EnvClient] %s retry %d/%d %s",
                    label,
                    attempt,
                    max_attempts,
                    type(e).__name__,
                )
                await self._asleep_backoff(attempt)
        msg = f"[EnvClient] {label} failed after {max_attempts} attempts"
        raise EnvTransportError(msg) from last_exc

    def req_envs(self, num_envs: int, env_type: str = "real_device") -> BatchSessionResponse:

        r = self._post(
            "/batch_start_env",
            {"num_envs": num_envs, "env_type": env_type},
            timeout=self.timeout * num_envs,
        )
        return BatchSessionResponse.model_validate_json(r.text)

    def ensure_pool(self, num_sessions: int, env_type: str) -> int:
        from mai_gym.models.environment import matches_environment

        self._env_type = env_type
        existing = self.list_sessions()
        ready = sum(
            1
            for s in existing.sessions
            if str(s.get("status", "")).lower() == "ready"
            and matches_environment(
                env_type,
                s.get("env_type") or env_type,
                infra_type=s.get("infra_type"),
                profile=s.get("environment_profile"),
            )
        )
        deficit = num_sessions - ready
        if deficit > 0:
            logger.info(
                "[EnvClient] ensure_pool: %d ready, provisioning %d more (env_type=%s)",
                ready,
                deficit,
                env_type,
            )
            started = self.req_envs(deficit, env_type=env_type)
            ready += started.total_count
        return ready

    @property
    def fleet_request_id(self) -> str | None:

        return self._fleet_request_id

    def _fleet_post(
        self,
        path: str,
        payload: dict[str, Any],
        *,
        label: str,
    ) -> httpx.Response:

        last_exc: Exception | None = None
        for attempt in range(1, 4):
            try:
                return self._post(path, payload, timeout=_POLL_TIMEOUT)
            except APIError as exc:
                last_exc = exc
                if exc.status_code not in _TRANSIENT_HTTP_STATUS or attempt >= 3:
                    raise
            except self._NETWORK_ERRORS as exc:
                last_exc = exc
                if attempt >= 3:
                    break
            logger.warning(
                "[EnvClient] %s retry %d/3 after %s",
                label,
                attempt,
                type(last_exc).__name__,
            )
            self._sleep_backoff(attempt, base=0.5, cap=2.0)
        raise RuntimeError(f"{label} failed after 3 attempts") from last_exc

    def ensure_capacity(
        self,
        num_sessions: int,
        env_type: str,
        *,
        job_id: str,
        mode: Literal["auto", "off", "required"] = "auto",
    ) -> int:

        self._env_type = env_type
        if mode == "off":
            return self.ensure_pool(num_sessions, env_type)
        if mode not in ("auto", "required"):
            raise ValueError("fleet request mode must be auto, off, or required")
        if num_sessions < 1:
            raise ValueError("num_sessions must be >= 1")

        try:
            health = self._get("/health", timeout=_POLL_TIMEOUT).json()
        except APIError as exc:
            if exc.status_code != 404:
                raise
            health = {}
        capability = (
            health.get("capabilities", {}).get("fleet_requests.v1")
            if isinstance(health, dict)
            else None
        )
        if not isinstance(capability, dict):
            if mode == "required":
                raise RuntimeError("manager does not advertise required fleet_requests.v1")
            return self.ensure_pool(num_sessions, env_type)

        with self._fleet_lifecycle_lock:
            if self._owns_fleet_request and self._fleet_request_id is not None:
                holder_id = cast(str, self._fleet_holder_id)
                idempotency_key = cast(str, self._fleet_idempotency_key)
            else:
                holder_id = uuid.uuid4().hex
                idempotency_key = uuid.uuid4().hex
        response = self._fleet_post(
            "/fleet/requests",
            {
                "job_id": job_id,
                "holder_id": holder_id,
                "idempotency_key": idempotency_key,
                "env_type": env_type,
                "capacity": num_sessions,
            },
            label="fleet reservation",
        ).json()
        request_id = str(response["request_id"])
        heartbeat_interval = float(
            response.get("heartbeat_interval_seconds")
            or capability.get("heartbeat_interval_seconds")
            or 30.0
        )
        warmup_timeout = float(
            response.get("warmup_timeout_seconds")
            or capability.get("warmup_timeout_seconds")
            or 1200.0
        )
        with self._fleet_lifecycle_lock:
            self._fleet_request_id = request_id
            self._fleet_holder_id = holder_id
            self._fleet_idempotency_key = idempotency_key
            self._fleet_heartbeat_interval = heartbeat_interval
            self._owns_fleet_request = True

        deadline = time.monotonic() + warmup_timeout
        next_heartbeat = time.monotonic() + heartbeat_interval
        try:
            while True:
                state = str(response.get("state", "warming"))
                if state == "ready":
                    self._start_fleet_heartbeat()
                    return num_sessions
                if state in ("failed", "expired", "draining"):
                    detail = response.get("last_error") or state
                    raise RuntimeError(f"fleet request {request_id} entered {state}: {detail}")
                if time.monotonic() >= deadline:
                    raise TimeoutError(
                        f"fleet request {request_id} did not warm "
                        f"{num_sessions} replicas within {warmup_timeout:.0f}s"
                    )
                poll_after = min(
                    max(float(response.get("poll_after_seconds") or 2.0), 0.1),
                    2.0,
                )
                time.sleep(poll_after)
                if time.monotonic() >= next_heartbeat:
                    self._fleet_post(
                        f"/fleet/requests/{request_id}/heartbeat",
                        {"holder_id": holder_id},
                        label="fleet heartbeat",
                    )
                    next_heartbeat = time.monotonic() + heartbeat_interval
                response = self._get(
                    f"/fleet/requests/{request_id}",
                    timeout=_POLL_TIMEOUT,
                ).json()
        except BaseException:
            self._release_fleet_request()
            raise

    def _start_fleet_heartbeat(self) -> None:
        with self._fleet_lifecycle_lock:
            if not self._owns_fleet_request or self._fleet_heartbeat_thread is not None:
                return
            self._fleet_heartbeat_stop.clear()
            thread = threading.Thread(
                target=self._fleet_heartbeat_loop,
                name="mai-gym-fleet-heartbeat",
                daemon=True,
            )
            self._fleet_heartbeat_thread = thread
            thread.start()

    def _fleet_heartbeat_loop(self) -> None:
        while not self._fleet_heartbeat_stop.wait(self._fleet_heartbeat_interval):
            with self._fleet_lifecycle_lock:
                if not self._owns_fleet_request:
                    return
                request_id = self._fleet_request_id
                holder_id = self._fleet_holder_id
            if request_id is None or holder_id is None:
                return
            try:
                self._fleet_post(
                    f"/fleet/requests/{request_id}/heartbeat",
                    {"holder_id": holder_id},
                    label="fleet heartbeat",
                )
            except Exception:
                logger.exception(
                    "[EnvClient] fleet heartbeat failed request=%s",
                    request_id,
                )

    def _release_fleet_request(self) -> None:
        with self._fleet_lifecycle_lock:
            if not self._owns_fleet_request:
                return
            request_id = self._fleet_request_id
            holder_id = self._fleet_holder_id
            self._owns_fleet_request = False
            thread = self._fleet_heartbeat_thread
            self._fleet_heartbeat_thread = None
            self._fleet_heartbeat_stop.set()
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=min(self._fleet_heartbeat_interval, 5.0))
        if request_id is not None and holder_id is not None:
            try:
                self._delete(
                    f"/fleet/requests/{request_id}/holders/{holder_id}",
                    timeout=_POLL_TIMEOUT,
                )
            except Exception:
                logger.exception(
                    "[EnvClient] fleet release failed request=%s; TTL will expire it",
                    request_id,
                )

    def list_sessions(self) -> ListSessionsResponse:

        return ListSessionsResponse.model_validate_json(self._get("/sessions").text)

    def list_tasks(self) -> ListTasksResponse:

        return ListTasksResponse.model_validate_json(self._get("/tasks").text)

    _REGISTER_TIMEOUT = 300.0

    def register_task_set(self, set_id: str, zip_path: str) -> dict[str, Any]:

        from pathlib import Path

        pkg = Path(zip_path).read_bytes()
        r = self._post_raw(
            "/register_task_set",
            params={"set_id": set_id},
            content=pkg,
            timeout=self._REGISTER_TIMEOUT,
        )
        return r.json()

    async def aregister_task_set(self, set_id: str, zip_path: str) -> dict[str, Any]:

        from pathlib import Path

        pkg = await asyncio.to_thread(Path(zip_path).read_bytes)
        r = await self._apost_raw(
            "/register_task_set",
            params={"set_id": set_id},
            content=pkg,
            timeout=self._REGISTER_TIMEOUT,
        )
        return r.json()

    def unregister_task_set(self, set_id: str) -> dict[str, Any]:

        return self._delete(
            f"/register_task_set/{set_id}",
            timeout=self._REGISTER_TIMEOUT,
        ).json()

    async def aunregister_task_set(self, set_id: str) -> dict[str, Any]:

        r = await self._adelete(
            f"/register_task_set/{set_id}",
            timeout=self._REGISTER_TIMEOUT,
        )
        return r.json()

    def list_task_sets(self) -> list[str]:

        try:
            return self._get("/task_sets").json().get("set_ids", [])
        except Exception:
            return []

    def task_set_registered(self, set_id: str) -> bool:
        return set_id in self.list_task_sets()

    _RUN_CONFIG_TIMEOUT = 60.0

    def register_run_config(
        self,
        train_id: str | None,
        config: dict[str, Any],
        *,
        model_name: str | None = None,
    ) -> dict[str, Any]:

        if not train_id:
            return {"ok": False, "skipped": True, "reason": "no train_id"}
        try:
            r = self._post(
                "/run_config",
                {"train_id": train_id, "model_name": model_name, "config": config},
                timeout=self._RUN_CONFIG_TIMEOUT,
            )
            return r.json()
        except Exception as exc:
            logger.warning("[EnvClient] register_run_config failed: %s", exc)
            return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}

    async def aregister_run_config(
        self,
        train_id: str | None,
        config: dict[str, Any],
        *,
        model_name: str | None = None,
    ) -> dict[str, Any]:

        if not train_id:
            return {"ok": False, "skipped": True, "reason": "no train_id"}
        try:
            r = await self._apost(
                "/run_config",
                {"train_id": train_id, "model_name": model_name, "config": config},
                timeout=self._RUN_CONFIG_TIMEOUT,
            )
            return r.json()
        except Exception as exc:
            logger.warning("[EnvClient] aregister_run_config failed: %s", exc)
            return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}

    def register_trained_tasks(
        self,
        train_id: str | None,
        rollout_id: int | None,
        manifest: dict[str, Any],
        *,
        evaluation: bool = False,
        model_name: str | None = None,
    ) -> dict[str, Any]:

        if not train_id:
            return {"ok": False, "skipped": True, "reason": "no train_id"}
        try:
            r = self._post(
                "/trained",
                {
                    "train_id": train_id,
                    "rollout_id": rollout_id,
                    "evaluation": evaluation,
                    "model_name": model_name,
                    "manifest": manifest,
                },
                timeout=self._RUN_CONFIG_TIMEOUT,
            )
            return r.json()
        except Exception as exc:
            logger.warning("[EnvClient] register_trained_tasks failed: %s", exc)
            return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}

    async def aregister_trained_tasks(
        self,
        train_id: str | None,
        rollout_id: int | None,
        manifest: dict[str, Any],
        *,
        evaluation: bool = False,
        model_name: str | None = None,
    ) -> dict[str, Any]:

        if not train_id:
            return {"ok": False, "skipped": True, "reason": "no train_id"}
        try:
            r = await self._apost(
                "/trained",
                {
                    "train_id": train_id,
                    "rollout_id": rollout_id,
                    "evaluation": evaluation,
                    "model_name": model_name,
                    "manifest": manifest,
                },
                timeout=self._RUN_CONFIG_TIMEOUT,
            )
            return r.json()
        except Exception as exc:
            logger.warning("[EnvClient] aregister_trained_tasks failed: %s", exc)
            return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}

    _LOG_FAILURE_TIMEOUT = 15.0

    def log_acquire_failure(
        self,
        *,
        train_id: str | None,
        task_id: str | None,
        attempt_id: str | None,
        rollout_id: int | None = None,
        evaluation: bool = False,
        model_name: str | None = None,
        env_type: str | None = None,
        reason: str | None = None,
    ) -> dict[str, Any]:

        try:
            r = self._post(
                "/log_acquire_failure",
                {
                    "train_id": train_id,
                    "task_id": task_id,
                    "attempt_id": attempt_id,
                    "rollout_id": rollout_id,
                    "evaluation": evaluation,
                    "model_name": model_name,
                    "env_type": env_type,
                    "reason": reason,
                },
                timeout=self._LOG_FAILURE_TIMEOUT,
            )
            return r.json()
        except Exception as exc:
            logger.warning("[EnvClient] log_acquire_failure failed: %s", exc)
            return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}

    async def alog_acquire_failure(
        self,
        *,
        train_id: str | None,
        task_id: str | None,
        attempt_id: str | None,
        rollout_id: int | None = None,
        evaluation: bool = False,
        model_name: str | None = None,
        env_type: str | None = None,
        reason: str | None = None,
    ) -> dict[str, Any]:

        try:
            r = await self._apost(
                "/log_acquire_failure",
                {
                    "train_id": train_id,
                    "task_id": task_id,
                    "attempt_id": attempt_id,
                    "rollout_id": rollout_id,
                    "evaluation": evaluation,
                    "model_name": model_name,
                    "env_type": env_type,
                    "reason": reason,
                },
                timeout=self._LOG_FAILURE_TIMEOUT,
            )
            return r.json()
        except Exception as exc:
            logger.warning("[EnvClient] alog_acquire_failure failed: %s", exc)
            return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}

    def reset(
        self,
        session_id: str,
        category: str,
        task_id: str | None = None,
        goal: str | None = None,
        target_apps: list[str] | None = None,
        seed_task_id: str | None = None,
        attempt_id: str | None = None,
        train_id: str | None = None,
        rollout_id: int | None = None,
        evaluation: bool = False,
        model_name: str | None = None,
        task_phase: str | None = None,
    ) -> StepResponse:

        r = self._retry_post(
            f"/reset/{session_id}",
            _reset_payload(
                category,
                task_id,
                goal,
                target_apps,
                seed_task_id,
                attempt_id,
                train_id,
                rollout_id=rollout_id,
                evaluation=evaluation,
                lease_id=self._lease_cache.get(session_id),
                model_name=model_name,
                screenshot_transport=self._screenshot_transport,
                task_phase=task_phase,
            ),
            label="reset",
            retry_on_transient_http=True,
        )
        return StepResponse.model_validate_json(r.text)

    def attach_task(
        self,
        session_id: str,
        *,
        env_type: str,
        set_id: str | None = None,
        task_class: str | None = None,
        golden_class: str | None = None,
        package_b64: str | None = None,
        initial_setup: str | None = None,
        golden_patch: str | None = None,
        reward: str | None = None,
        world_oss_uri: str | None = None,
        world_sha256: str | None = None,
        world_extract_to: str | None = None,
    ) -> dict[str, Any]:

        body: dict[str, Any] = {
            "env_type": env_type,
            "lease_id": self._lease_cache.get(session_id),
            "set_id": set_id,
            "task_class": task_class,
            "golden_class": golden_class,
            "package_b64": package_b64,
            "initial_setup": initial_setup,
            "golden_patch": golden_patch,
            "reward": reward,
            "world_oss_uri": world_oss_uri,
            "world_sha256": world_sha256,
            "world_extract_to": world_extract_to,
        }
        r = self._retry_post(
            f"/session_task/{session_id}",
            {k: v for k, v in body.items() if v is not None},
            label="attach_task",
            retry_on_transient_http=True,
        )
        return r.json()

    def step(self, session_id: str, action: dict[str, Any]) -> StepResponse:

        return StepResponse.model_validate_json(
            self._post(
                f"/step/{session_id}",
                _step_payload(
                    action,
                    self._lease_cache.get(session_id),
                    screenshot_transport=self._screenshot_transport,
                ),
            ).text
        )

    def eval(
        self,
        session_id: str,
        category: str,
        task_id: str | None = None,
        trajectory_messages: str | None = None,
        use_judge: bool = False,
        evaluation_mode: Literal["built_in", "llm_judge", "both"] = "built_in",
        require_score: bool = False,
    ) -> EvaluateResponse:

        r = self._retry_post(
            f"/evaluate/{session_id}",
            {
                "category": category,
                "task_id": task_id,
                "trajectory_messages": trajectory_messages,
                "use_judge": use_judge,
                "evaluation_mode": evaluation_mode,
                **({"require_score": True} if require_score else {}),
            },
            label="eval",
        )
        return EvaluateResponse.model_validate_json(r.text)

    def tear_down(
        self, session_id: str, category: str, task_id: str | None = None
    ) -> dict[str, Any]:

        return self._post(
            f"/tear_down/{session_id}", {"category": category, "task_id": task_id}
        ).json()

    def holds(self, session_id: str) -> bool:

        return session_id in self._lease_cache

    _ACQUIRE_MAX_ATTEMPTS = 22

    def acquire(
        self,
        target_apps: list[str] | None = None,
        timeout_s: float = 30.0,
    ) -> AcquireResponse:

        payload: dict[str, Any] = {
            "target_apps": target_apps,
            "timeout_s": timeout_s,
        }
        if self._env_type is not None:
            payload["env_type"] = self._env_type
        if self._fleet_request_id is not None:
            payload["fleet_request_id"] = self._fleet_request_id
        r = self._retry_post(
            "/acquire",
            payload,
            label="acquire",
            retry_on_transient_http=True,
            max_attempts=self._ACQUIRE_MAX_ATTEMPTS,
        )
        resp = AcquireResponse.model_validate_json(r.text)
        self._lease_cache[resp.session_id] = resp.lease_id
        return resp

    def release(self, session_id: str, lease_id: str | None = None) -> dict[str, Any]:

        lid = self._take_lease(session_id, lease_id)
        result = self._retry_post(
            f"/release/{session_id}",
            {"lease_id": lid},
            label="release",
            retry_on_transient_http=True,
            max_attempts=3,
        ).json()
        self._confirm_cleanup(session_id, lid, result)
        return result

    def discard(
        self,
        session_id: str,
        lease_id: str | None = None,
        reason: str | None = None,
    ) -> dict[str, Any]:

        lid = self._take_lease(session_id, lease_id)
        result = self._retry_post(
            f"/discard/{session_id}",
            {"lease_id": lid, "reason": reason},
            label="discard",
            retry_on_transient_http=True,
            max_attempts=3,
        ).json()
        self._confirm_cleanup(session_id, lid, result)
        return result

    def _take_lease(self, session_id: str, lease_id: str | None) -> str:
        lid = lease_id or self._lease_cache.get(session_id)
        if lid is None:
            raise ValueError(f"No cached lease_id for {session_id}; pass lease_id explicitly")

        return lid

    def _confirm_cleanup(self, session_id: str, lease_id: str, result: Any) -> None:
        if not isinstance(result, dict) or result.get("ok") is not True:
            raise EnvTransportError("Cleanup response did not confirm ok=true")
        if self._lease_cache.get(session_id) == lease_id:
            self._lease_cache.pop(session_id, None)

    def _validate_cleanup_response(self, response: httpx.Response, path: str) -> None:
        try:
            valid = 200 <= response.status_code < 300 and isinstance(response.json(), dict)
        except ValueError:
            valid = False
        if not valid:
            raise APIError(502, "POST", f"{self.base_url}{path}", "Invalid cleanup response")

    @contextmanager
    def session(
        self,
        target_apps: list[str] | None = None,
        timeout_s: float = 30.0,
        *,
        failure_log: dict[str, Any] | None = None,
    ) -> Iterator[str]:

        try:
            resp = self.acquire(target_apps, timeout_s)
        except BaseException as exc:
            if failure_log is not None:
                reason = f"acquire failed: {type(exc).__name__}: {exc}"
                try:
                    self.log_acquire_failure(reason=reason, **failure_log)
                except Exception as log_exc:
                    logger.warning(
                        "[EnvClient] session() side-channel log raised "
                        "(ignored to preserve original acquire exception): %s",
                        log_exc,
                    )
            raise
        sid = resp.session_id
        failed = False
        try:
            yield sid
        except (KeyboardInterrupt, GeneratorExit):
            raise
        except BaseException:
            failed = True
            raise
        finally:
            if sid in self._lease_cache:
                try:
                    if failed:
                        self.discard(sid, reason="exception in session() block")
                    else:
                        self.release(sid)
                except Exception as exc:
                    logger.warning(
                        "[EnvClient] session() cleanup failed for %s: %s",
                        sid,
                        exc,
                    )

    def log_messages(
        self,
        session_id: str,
        step: int,
        payload: dict[str, Any],
    ) -> dict[str, Any]:

        return self._post(
            f"/debug/log_messages/{session_id}",
            {"step": step, "payload": payload},
        ).json()

    def supports_verifier_errors(self) -> bool:

        try:
            response = self._client.get(f"{self.base_url}/health", timeout=_POLL_TIMEOUT)
            if response.status_code == 404:
                return False
            if response.status_code >= 400:
                raise APIError(
                    response.status_code, "GET", f"{self.base_url}/health", response.text
                )
            capability = response.json().get("capabilities", {}).get("verifier_errors.v1")
        except ValueError:
            return False
        version = capability.get("version") if isinstance(capability, dict) else capability
        return isinstance(version, int) and version >= 1

    async def areset(
        self,
        session_id: str,
        category: str,
        task_id: str | None = None,
        goal: str | None = None,
        target_apps: list[str] | None = None,
        seed_task_id: str | None = None,
        attempt_id: str | None = None,
        train_id: str | None = None,
        rollout_id: int | None = None,
        evaluation: bool = False,
        model_name: str | None = None,
        task_phase: str | None = None,
        action_id: str | None = None,
        strict_operations: bool = False,
        resume_operation: bool = False,
        operation_timeout: float | None = None,
    ) -> StepResponse:

        kwargs = dict(
            category=category,
            task_id=task_id,
            goal=goal,
            target_apps=target_apps,
            seed_task_id=seed_task_id,
            attempt_id=attempt_id,
            train_id=train_id,
            rollout_id=rollout_id,
            evaluation=evaluation,
            model_name=model_name,
            task_phase=task_phase,
        )
        if resume_operation and action_id is None:
            raise ValueError("resume_operation requires a persisted action_id")
        if operation_timeout is not None and operation_timeout <= 0:
            raise ValueError("operation_timeout must be positive")
        action_id = action_id or uuid.uuid4().hex
        if self._async_ops is not False:
            try:
                async with asyncio.timeout(operation_timeout):
                    resp = await self._areset_async(
                        session_id,
                        action_id,
                        strict_operations=strict_operations,
                        resume_operation=resume_operation,
                        **kwargs,
                    )
                self._async_ops = True
                return resp
            except TimeoutError as exc:
                raise PollTimeoutError(
                    f"reset operation deadline exceeded (action_id={action_id}); "
                    "discard this session before retrying"
                ) from exc
            except APIError as e:
                if (
                    e.status_code == 404
                    and self._async_ops is None
                    and not strict_operations
                    and not resume_operation
                ):
                    self._async_ops = False
                    logger.info(
                        "[EnvClient] manager lacks async submit/poll routes "
                        "(404 on /areset) — using sync /reset fallback",
                    )
                else:
                    raise
        if strict_operations or resume_operation:
            raise APIError(
                409,
                "POST",
                f"{self.base_url}/areset/{session_id}",
                body={
                    "error": "async_operation_unavailable",
                    "error_detail": {
                        "code": "async_operation_unavailable",
                        "phase": "reset",
                        "failure_kind": "environment",
                        "recovery": "fresh_session",
                        "operation_id": action_id,
                    },
                },
            )
        return await self._areset_sync(session_id, action_id=action_id, **kwargs)

    async def _areset_async(
        self,
        session_id: str,
        action_id: str,
        **kwargs: Any,
    ) -> StepResponse:

        payload = _reset_payload(
            kwargs["category"],
            kwargs["task_id"],
            kwargs["goal"],
            kwargs["target_apps"],
            kwargs["seed_task_id"],
            kwargs["attempt_id"],
            kwargs["train_id"],
            rollout_id=kwargs["rollout_id"],
            evaluation=kwargs["evaluation"],
            lease_id=self._lease_cache.get(session_id),
            model_name=kwargs["model_name"],
            action_id=action_id,
            screenshot_transport=self._screenshot_transport,
            task_phase=kwargs.get("task_phase"),
        )
        return await self._submit_poll(
            kind="reset",
            session_id=session_id,
            action_id=action_id,
            submit_path=f"/areset/{session_id}",
            readiness_path=f"/areset_readiness/{session_id}",
            submit_payload=payload,
            max_polls=_RESET_POLL_MAX_POLLS,
            strict_operations=kwargs.get("strict_operations", False),
            resume_operation=kwargs.get("resume_operation", False),
        )

    async def astep(
        self,
        session_id: str,
        action: dict[str, Any],
        *,
        action_id: str | None = None,
    ) -> StepResponse:

        action_id = action_id or uuid.uuid4().hex
        if self._async_ops is not False:
            try:
                resp = await self._astep_async(session_id, action, action_id)
                self._async_ops = True
                return resp
            except APIError as e:
                if e.status_code == 404 and self._async_ops is None:
                    self._async_ops = False
                    logger.info(
                        "[EnvClient] manager lacks async submit/poll routes "
                        "(404 on /astep) — using sync /step fallback",
                    )
                else:
                    raise
        return await self._astep_sync(session_id, action, action_id=action_id)

    def keepalive_session(self, session_id: str) -> bool:
        lease_id = self._lease_cache.get(session_id)
        if lease_id is None:
            return False
        try:
            self._post(
                f"/astep_readiness/{session_id}",
                {"action_id": "lease-keepalive", "lease_id": lease_id},
                timeout=_POLL_TIMEOUT,
            )
            return self._lease_cache.get(session_id) == lease_id
        except Exception as exc:
            logger.warning("[EnvClient] lease keepalive failed sid=%s: %s", session_id, exc)
            return False

    async def _astep_async(
        self,
        session_id: str,
        action: dict[str, Any],
        action_id: str,
    ) -> StepResponse:

        payload = _step_payload(
            action,
            self._lease_cache.get(session_id),
            action_id,
            screenshot_transport=self._screenshot_transport,
        )
        return await self._submit_poll(
            kind="step",
            session_id=session_id,
            action_id=action_id,
            submit_path=f"/astep/{session_id}",
            readiness_path=f"/astep_readiness/{session_id}",
            submit_payload=payload,
            max_polls=_STEP_POLL_MAX_POLLS,
            poll_cap=_STEP_POLL_INTERVAL_CAP,
        )

    _SUBMIT_NETWORK_MAX_RETRIES = 3

    async def _submit_with_retry(
        self,
        *,
        kind: str,
        session_id: str,
        action_id: str,
        submit_path: str,
        submit_payload: dict[str, Any],
    ) -> None:

        last_exc: Exception | None = None
        for attempt in range(1, self._SUBMIT_NETWORK_MAX_RETRIES + 1):
            try:
                await self._apost(
                    submit_path,
                    submit_payload,
                    timeout=_SUBMIT_TIMEOUT,
                )
                if attempt > 1:
                    logger.info(
                        "[EnvClient] %s submit recovered on attempt %d sid=%s (action_id=%s)",
                        kind,
                        attempt,
                        session_id,
                        action_id,
                    )
                return
            except self._NETWORK_ERRORS as exc:
                last_exc = exc
                if attempt >= self._SUBMIT_NETWORK_MAX_RETRIES:
                    break
                logger.warning(
                    "[EnvClient] %s submit transport-fault sid=%s attempt "
                    "%d/%d %s — retrying (action_id=%s)",
                    kind,
                    session_id,
                    attempt,
                    self._SUBMIT_NETWORK_MAX_RETRIES,
                    type(exc).__name__,
                    action_id,
                )
                await self._asleep_backoff(
                    attempt,
                    base=_POLL_INTERVAL_BASE,
                    cap=_POLL_INTERVAL_CAP,
                )
        raise PollTimeoutError(
            f"{kind} submit for {session_id} gave up after "
            f"{self._SUBMIT_NETWORK_MAX_RETRIES} transport faults "
            f"(action_id={action_id}, last={type(last_exc).__name__})"
        ) from last_exc

    async def _submit_poll(
        self,
        *,
        kind: str,
        session_id: str,
        action_id: str,
        submit_path: str,
        readiness_path: str,
        submit_payload: dict[str, Any],
        max_polls: int,
        poll_cap: float = _POLL_INTERVAL_CAP,
        strict_operations: bool = False,
        resume_operation: bool = False,
    ) -> StepResponse:

        readiness_body = {"action_id": action_id, "lease_id": self._lease_cache.get(session_id)}

        if not resume_operation:
            if strict_operations:
                try:
                    await self._apost(submit_path, submit_payload, timeout=_SUBMIT_TIMEOUT)
                except self._NETWORK_ERRORS:
                    pass
                except APIError as exc:
                    if exc.status_code not in _GATEWAY_HTTP_STATUS or exc.error_detail:
                        raise
            else:
                await self._submit_with_retry(
                    kind=kind,
                    session_id=session_id,
                    action_id=action_id,
                    submit_path=submit_path,
                    submit_payload=submit_payload,
                )
        resubmits = 0
        poll_faults = 0
        for poll in range(1, max_polls + 1):
            try:
                r = await self._apost(
                    readiness_path,
                    readiness_body,
                    timeout=_POLL_TIMEOUT,
                )
            except APIError as e:
                if e.status_code in _GATEWAY_HTTP_STATUS and not e.error_detail:
                    poll_faults += 1
                    if poll_faults > _POLL_NETWORK_MAX_RETRIES:
                        raise PollTimeoutError(
                            f"{kind} poll for {session_id} gave up after "
                            f"{poll_faults} consecutive gateway-{e.status_code}s "
                            f"(action_id={action_id})"
                        ) from e
                    await self._asleep_backoff(
                        poll_faults,
                        base=_POLL_INTERVAL_BASE,
                        cap=poll_cap,
                    )
                    continue
                raise
            except self._NETWORK_ERRORS as e:
                poll_faults += 1
                if poll_faults > _POLL_NETWORK_MAX_RETRIES:
                    raise PollTimeoutError(
                        f"{kind} poll for {session_id} gave up after "
                        f"{poll_faults} consecutive {type(e).__name__}s "
                        f"(action_id={action_id})"
                    ) from e
                await self._asleep_backoff(
                    poll_faults,
                    base=_POLL_INTERVAL_BASE,
                    cap=poll_cap,
                )
                continue

            if not _looks_like_json(r.text):
                poll_faults += 1
                logger.warning(
                    "[EnvClient] %s poll %d non-JSON readiness body sid=%s%s",
                    kind,
                    poll,
                    session_id,
                    _resp_diag(r),
                )
                if poll_faults > _POLL_NETWORK_MAX_RETRIES:
                    raise PollTimeoutError(
                        f"{kind} poll for {session_id} gave up after "
                        f"{poll_faults} consecutive non-JSON readiness bodies "
                        f"(action_id={action_id})"
                    )
                await self._asleep_backoff(
                    poll_faults,
                    base=_POLL_INTERVAL_BASE,
                    cap=poll_cap,
                )
                continue
            body = r.json()
            poll_faults = 0
            if body.get("ready") is True:
                return StepResponse.model_validate_json(r.text)
            if body.get("pending"):
                await self._asleep_backoff(
                    poll,
                    base=_POLL_INTERVAL_BASE,
                    cap=poll_cap,
                )
                continue
            if body.get("unknown"):
                if strict_operations or resume_operation:
                    raise APIError(
                        409,
                        "POST",
                        f"{self.base_url}{readiness_path}",
                        body={
                            "error": "operation_unknown",
                            "error_detail": {
                                "code": "operation_unknown",
                                "phase": kind,
                                "failure_kind": "environment",
                                "recovery": "fresh_session",
                                "operation_id": action_id,
                            },
                        },
                    )

                resubmits += 1
                if resubmits > _RESUBMIT_MAX_ATTEMPTS:
                    raise PollTimeoutError(
                        f"{kind} for {session_id} unknown after "
                        f"{resubmits} re-submits (action_id={action_id})"
                    )
                logger.warning(
                    "[EnvClient] %s op unknown sid=%s — reset pool + "
                    "re-submit %d/%d (action_id=%s)",
                    kind,
                    session_id,
                    resubmits,
                    _RESUBMIT_MAX_ATTEMPTS,
                    action_id,
                )
                await self._reset_aclient()
                await self._submit_with_retry(
                    kind=kind,
                    session_id=session_id,
                    action_id=action_id,
                    submit_path=submit_path,
                    submit_payload=submit_payload,
                )
                await self._asleep_backoff(
                    1,
                    base=_POLL_INTERVAL_BASE,
                    cap=poll_cap,
                )
                continue
            if "error" in body:
                error_detail = body.get("error_detail")
                recovery = (error_detail or {}).get("recovery")
                if recovery == "poll_same_operation":
                    await self._asleep_backoff(poll, base=_POLL_INTERVAL_BASE, cap=poll_cap)
                    continue

                if kind == "reset" or error_detail or strict_operations:
                    status = body.get("status_code") or (error_detail or {}).get(
                        "upstream_status",
                        502,
                    )
                    raise APIError(
                        status if isinstance(status, int) else 502,
                        "POST",
                        f"{self.base_url}{readiness_path}",
                        json.dumps(body),
                    )
                if body.get("retryable"):
                    resubmits += 1
                    if resubmits > _RESUBMIT_MAX_ATTEMPTS:
                        raise APIError(
                            502,
                            "POST",
                            f"{self.base_url}{submit_path}",
                            f"{kind} retryable error persisted after "
                            f"{resubmits} re-submits: {body.get('error')}",
                        )
                    logger.warning(
                        "[EnvClient] %s op retryable error sid=%s — re-submit "
                        "%d/%d (action_id=%s): %s",
                        kind,
                        session_id,
                        resubmits,
                        _RESUBMIT_MAX_ATTEMPTS,
                        action_id,
                        body.get("error"),
                    )
                    await self._submit_with_retry(
                        kind=kind,
                        session_id=session_id,
                        action_id=action_id,
                        submit_path=submit_path,
                        submit_payload=submit_payload,
                    )
                    await self._asleep_backoff(
                        resubmits,
                        base=_POLL_INTERVAL_BASE,
                        cap=poll_cap,
                    )
                    continue

                raise APIError(
                    502,
                    "POST",
                    f"{self.base_url}{readiness_path}",
                    f"{kind} op failed (non-retryable): {body.get('error')}",
                )

            logger.warning(
                "[EnvClient] %s poll %d unrecognised readiness body sid=%s: %s",
                kind,
                poll,
                session_id,
                body,
            )
            await self._asleep_backoff(
                poll,
                base=_POLL_INTERVAL_BASE,
                cap=poll_cap,
            )
        raise PollTimeoutError(
            f"{kind} for {session_id} not ready after {max_polls} polls "
            f"(action_id={action_id}); op still running detached server-side"
        )

    async def _areset_sync(
        self,
        session_id: str,
        category: str,
        task_id: str | None = None,
        goal: str | None = None,
        target_apps: list[str] | None = None,
        seed_task_id: str | None = None,
        attempt_id: str | None = None,
        train_id: str | None = None,
        rollout_id: int | None = None,
        evaluation: bool = False,
        model_name: str | None = None,
        action_id: str | None = None,
        task_phase: str | None = None,
    ) -> StepResponse:

        if action_id is None:
            action_id = uuid.uuid4().hex
        payload = _reset_payload(
            category,
            task_id,
            goal,
            target_apps,
            seed_task_id,
            attempt_id,
            train_id,
            rollout_id=rollout_id,
            evaluation=evaluation,
            lease_id=self._lease_cache.get(session_id),
            model_name=model_name,
            action_id=action_id,
            screenshot_transport=self._screenshot_transport,
            task_phase=task_phase,
        )
        busy_attempt = 0
        net_retries = 0
        while True:
            try:
                r = await self._apost(f"/reset/{session_id}", payload)
            except APIError as e:
                if e.status_code == 409 and "session_busy" in str(e):
                    busy_attempt += 1
                    if busy_attempt >= _RESET_BUSY_MAX_ATTEMPTS:
                        raise
                    logger.warning(
                        "[EnvClient] reset retry %d/%d session_busy sid=%s "
                        "(polling in-flight reset; action_id=%s replays server-side)",
                        busy_attempt,
                        _RESET_BUSY_MAX_ATTEMPTS,
                        session_id,
                        action_id,
                    )
                    await self._asleep_backoff(busy_attempt, base=2.0, cap=15.0)
                    continue
                if e.status_code in _GATEWAY_HTTP_STATUS:
                    net_retries += 1
                    if net_retries > _RESET_NETWORK_MAX_RETRIES:
                        raise
                    logger.warning(
                        "[EnvClient] reset retry %d/%d gateway-%d sid=%s "
                        "(reconnect; action_id=%s replays server-side)",
                        net_retries,
                        _RESET_NETWORK_MAX_RETRIES,
                        e.status_code,
                        session_id,
                        action_id,
                    )
                    await self._asleep_backoff(net_retries, base=1.0, cap=8.0)
                    continue
                raise
            except self._NETWORK_ERRORS as e:
                net_retries += 1
                if net_retries > _RESET_NETWORK_MAX_RETRIES:
                    raise
                logger.warning(
                    "[EnvClient] reset retry %d/%d %s sid=%s (reconnect; "
                    "action_id=%s replays server-side)",
                    net_retries,
                    _RESET_NETWORK_MAX_RETRIES,
                    type(e).__name__,
                    session_id,
                    action_id,
                )
                await self._asleep_backoff(net_retries, base=1.0, cap=8.0)
                continue

            if not _looks_like_json(r.text) and net_retries < _RESET_NETWORK_MAX_RETRIES:
                net_retries += 1
                logger.warning(
                    "[EnvClient] reset retry %d/%d non-JSON /reset body "
                    "(gateway error page?) sid=%s%s",
                    net_retries,
                    _RESET_NETWORK_MAX_RETRIES,
                    session_id,
                    _resp_diag(r),
                )
                await self._asleep_backoff(net_retries, base=1.0, cap=8.0)
                continue
            return StepResponse.model_validate_json(r.text)

    async def _astep_sync(
        self,
        session_id: str,
        action: dict[str, Any],
        action_id: str | None = None,
    ) -> StepResponse:

        if action_id is None:
            action_id = uuid.uuid4().hex
        payload = _step_payload(
            action,
            self._lease_cache.get(session_id),
            action_id,
            screenshot_transport=self._screenshot_transport,
        )
        busy_attempt = 0
        net_retries = 0
        while True:
            try:
                r = await self._apost(f"/step/{session_id}", payload)
            except APIError as e:
                if e.status_code == 409 and "session_busy" in str(e):
                    busy_attempt += 1
                    if busy_attempt >= _STEP_BUSY_MAX_ATTEMPTS:
                        raise
                    logger.warning(
                        "[EnvClient] step retry %d/%d session_busy sid=%s",
                        busy_attempt,
                        _STEP_BUSY_MAX_ATTEMPTS,
                        session_id,
                    )
                    await self._asleep_backoff(busy_attempt, base=0.5, cap=4.0)
                    continue
                if e.status_code in _GATEWAY_HTTP_STATUS:
                    net_retries += 1
                    if net_retries > _STEP_NETWORK_MAX_RETRIES:
                        raise
                    logger.warning(
                        "[EnvClient] step retry %d/%d gateway-%d sid=%s "
                        "(reconnect; action_id=%s dedupes server-side)",
                        net_retries,
                        _STEP_NETWORK_MAX_RETRIES,
                        e.status_code,
                        session_id,
                        action_id,
                    )
                    await self._asleep_backoff(net_retries, base=0.5, cap=4.0)
                    continue
                raise
            except self._NETWORK_ERRORS as e:
                net_retries += 1
                if net_retries > _STEP_NETWORK_MAX_RETRIES:
                    raise
                logger.warning(
                    "[EnvClient] step retry %d/%d %s sid=%s (reconnect; "
                    "action_id=%s dedupes server-side)",
                    net_retries,
                    _STEP_NETWORK_MAX_RETRIES,
                    type(e).__name__,
                    session_id,
                    action_id,
                )
                await self._asleep_backoff(net_retries, base=0.5, cap=4.0)
                continue

            if not _looks_like_json(r.text) and net_retries < _STEP_NETWORK_MAX_RETRIES:
                net_retries += 1
                logger.warning(
                    "[EnvClient] step retry %d/%d non-JSON /step body "
                    "(gateway error page?) sid=%s (reconnect; action_id=%s "
                    "dedupes server-side)",
                    net_retries,
                    _STEP_NETWORK_MAX_RETRIES,
                    session_id,
                    action_id,
                )
                await self._asleep_backoff(net_retries, base=0.5, cap=4.0)
                continue
            return StepResponse.model_validate_json(r.text)

    async def aeval(
        self,
        session_id: str,
        category: str,
        task_id: str | None = None,
        trajectory_messages: str | None = None,
        use_judge: bool = False,
        evaluation_mode: Literal["built_in", "llm_judge", "both"] = "built_in",
        require_score: bool = False,
    ) -> EvaluateResponse:

        r = await self._aretry_post(
            f"/evaluate/{session_id}",
            {
                "category": category,
                "task_id": task_id,
                "trajectory_messages": trajectory_messages,
                "use_judge": use_judge,
                "evaluation_mode": evaluation_mode,
                **({"require_score": True} if require_score else {}),
            },
            label="eval",
        )
        return EvaluateResponse.model_validate_json(r.text)

    async def atear_down(
        self, session_id: str, category: str, task_id: str | None = None
    ) -> dict[str, Any]:

        r = await self._apost(
            f"/tear_down/{session_id}",
            {"category": category, "task_id": task_id},
        )
        return r.json()

    async def aacquire(
        self,
        target_apps: list[str] | None = None,
        timeout_s: float = 30.0,
    ) -> AcquireResponse:

        payload: dict[str, Any] = {
            "target_apps": target_apps,
            "timeout_s": timeout_s,
        }
        if self._env_type is not None:
            payload["env_type"] = self._env_type
        if self._fleet_request_id is not None:
            payload["fleet_request_id"] = self._fleet_request_id
        r = await self._aretry_post(
            "/acquire",
            payload,
            label="acquire",
            retry_on_transient_http=True,
            max_attempts=self._ACQUIRE_MAX_ATTEMPTS,
        )
        resp = AcquireResponse.model_validate_json(r.text)
        self._lease_cache[resp.session_id] = resp.lease_id
        return resp

    async def arelease(self, session_id: str, lease_id: str | None = None) -> dict[str, Any]:

        lid = self._take_lease(session_id, lease_id)
        r = await self._aretry_post(
            f"/release/{session_id}",
            {"lease_id": lid},
            label="release",
            retry_on_transient_http=True,
            max_attempts=3,
        )
        result = r.json()
        self._confirm_cleanup(session_id, lid, result)
        return result

    async def adiscard(
        self,
        session_id: str,
        lease_id: str | None = None,
        reason: str | None = None,
    ) -> dict[str, Any]:

        lid = self._take_lease(session_id, lease_id)
        r = await self._aretry_post(
            f"/discard/{session_id}",
            {"lease_id": lid, "reason": reason},
            label="discard",
            retry_on_transient_http=True,
            max_attempts=3,
        )
        result = r.json()
        self._confirm_cleanup(session_id, lid, result)
        return result

    async def alog_messages(
        self,
        session_id: str,
        step: int,
        payload: dict[str, Any],
    ) -> dict[str, Any]:

        r = await self._apost(
            f"/debug/log_messages/{session_id}",
            {"step": step, "payload": payload},
        )
        return r.json()

    @asynccontextmanager
    async def asession(
        self,
        target_apps: list[str] | None = None,
        timeout_s: float = 30.0,
        *,
        failure_log: dict[str, Any] | None = None,
        release_on: tuple[type[BaseException], ...] = (),
    ) -> AsyncIterator[str]:

        try:
            resp = await self.aacquire(target_apps, timeout_s)
        except BaseException as exc:
            if failure_log is not None:
                reason = f"acquire failed: {type(exc).__name__}: {exc}"
                try:
                    await self.alog_acquire_failure(
                        reason=reason,
                        **failure_log,
                    )
                except Exception as log_exc:
                    logger.warning(
                        "[EnvClient] asession() side-channel log raised "
                        "(ignored to preserve original acquire exception): %s",
                        log_exc,
                    )
            raise
        sid = resp.session_id
        failed = False

        fail_reason = "exception in asession() block"
        try:
            yield sid
        except (asyncio.CancelledError, KeyboardInterrupt, GeneratorExit):
            raise
        except BaseException as exc:
            if not isinstance(exc, release_on):
                failed = True
                fail_reason = f"exception in asession() block: {type(exc).__name__}: {exc}"
            raise
        finally:
            if sid in self._lease_cache:
                try:
                    if failed:
                        await self.adiscard(sid, None, fail_reason)
                    else:
                        await self.arelease(sid)
                except Exception as exc:
                    logger.warning(
                        "[EnvClient] asession() cleanup failed for %s: %s",
                        sid,
                        exc,
                    )


def _step_payload(
    action: dict[str, Any],
    lease_id: str | None,
    action_id: str | None = None,
    screenshot_transport: Literal["inline", "oss"] | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {"action": action}

    if lease_id is not None:
        payload["lease_id"] = lease_id

    if action_id is not None:
        payload["action_id"] = action_id
    if screenshot_transport is not None:
        payload["screenshot_transport"] = screenshot_transport
    return payload


def _reset_payload(
    category: str,
    task_id: str | None,
    goal: str | None,
    target_apps: list[str] | None,
    seed_task_id: str | None,
    attempt_id: str | None,
    train_id: str | None = None,
    rollout_id: int | None = None,
    evaluation: bool = False,
    lease_id: str | None = None,
    model_name: str | None = None,
    action_id: str | None = None,
    screenshot_transport: Literal["inline", "oss"] | None = None,
    task_phase: str | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {"category": category, "task_id": task_id}
    if goal is not None:
        payload["goal"] = goal
    if target_apps is not None:
        payload["target_apps"] = target_apps
    if seed_task_id is not None:
        payload["seed_task_id"] = seed_task_id
    if attempt_id is not None:
        payload["attempt_id"] = attempt_id

    if train_id is not None:
        payload["train_id"] = train_id

    if rollout_id is not None:
        payload["rollout_id"] = rollout_id
    if evaluation:
        payload["evaluation"] = evaluation

    if lease_id is not None:
        payload["lease_id"] = lease_id

    if model_name is not None:
        payload["model_name"] = model_name

    if action_id is not None:
        payload["action_id"] = action_id
    if screenshot_transport is not None:
        payload["screenshot_transport"] = screenshot_transport

    if task_phase is not None:
        payload["task_phase"] = task_phase
    return payload
