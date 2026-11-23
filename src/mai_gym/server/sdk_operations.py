from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import time
from dataclasses import dataclass, field
from typing import Any

import httpx

from mai_gym.models.operation_error import OperationError
from mai_gym.models.sdk import MAX_RESULT_BYTES, RESULT_CHUNK_BYTES, SDKOperationRequest
from mai_gym.models.session import SessionStatus


def sdk_error(message: str, status: int = 409) -> OperationError:
    return OperationError(
        message,
        status_code=status,
        error_detail={
            "code": "sdk_operation_error",
            "phase": "sdk",
            "retryable": False,
            "message": message,
        },
    )


@dataclass
class SDKOperation:
    operation_id: str
    fingerprint: str
    kind: str
    task: asyncio.Task | None = None
    status: str = "pending"
    data: bytes | None = None
    sha256: str | None = None
    error: str | None = None


@dataclass
class SDKSession:
    lease_id: str
    epoch: int
    dirty: bool = True
    worker_id: str | None = None
    runtime: dict = field(default_factory=dict)
    reset: Any = None
    operations: dict[str, SDKOperation] = field(default_factory=dict)
    current: SDKOperation | None = None
    cells: set[str] = field(default_factory=set)
    opened: bool = False


class SDKOperations:
    def __init__(
        self, controller, *, lifecycle_timeout: float = 600, reconcile_interval: float = 0.25
    ):
        self.controller = controller
        self.lifecycle_timeout = lifecycle_timeout
        self.reconcile_interval = reconcile_interval
        self.tasks: set[asyncio.Task] = set()

    def _authorize(self, session_id, lease_id):
        state = self.controller._sessions.get_session(session_id)
        if (
            not lease_id
            or state.lease_id != lease_id
            or state.lease_expires_at is None
            or state.lease_expires_at <= time.monotonic()
        ):
            raise sdk_error("SDK operation requires the current lease", 403)
        self.controller._scheduler.extend_lease(session_id)
        return state

    def _current(self, state, sdk, op):
        return (
            state.sdk_session is sdk
            and state.lease_id == sdk.lease_id
            and state.epoch == sdk.epoch
            and sdk.current is op
        )

    def _check_current(self, state, sdk, op):
        if not self._current(state, sdk, op):
            raise sdk_error("SDK lease or backend changed during operation")

    def guard(self, session_id):
        state = self.controller._sessions.get_session(session_id)
        if state.sdk_session is not None and state.sdk_session.dirty:
            raise sdk_error("Session is owned by the SDK lifecycle")

    def submit(self, session_id: str, request: SDKOperationRequest) -> dict:
        state = self._authorize(session_id, request.lease_id)
        if state.backend.env_type not in {"real_device", "mobileworld", "mobileworld_k8s"}:
            raise sdk_error("SDK operations require MobileWorld", 400)
        if any(
            other.session_id != session_id and other.backend.id == state.backend.id
            for other in self.controller._sessions.list_sessions()
        ):
            raise sdk_error("SDK requires one manager session per MobileWorld backend", 400)
        sdk = state.sdk_session
        if sdk is None or sdk.lease_id != request.lease_id:
            if sdk is not None and sdk.dirty:
                raise sdk_error("Previous SDK worker has not been retired")
            if request.kind != "reset" or state.status != SessionStatus.READY:
                raise sdk_error("Start SDK lifecycle with reset on a ready lease")
            sdk = SDKSession(request.lease_id, state.epoch)
            state.sdk_session = sdk
        digest = hashlib.sha256(request.model_dump_json().encode()).hexdigest()
        previous = sdk.operations.get(request.operation_id)
        if previous:
            if previous.fingerprint != digest:
                raise sdk_error("Operation ID reused with different arguments")
            return self.status(session_id, request.lease_id, request.operation_id)
        if sdk.epoch != state.epoch or not sdk.dirty:
            raise sdk_error("SDK lifecycle has ended or been invalidated")
        if len(sdk.operations) >= 512 or (len(sdk.operations) >= 511 and request.kind != "cancel"):
            raise sdk_error("SDK operation budget exhausted")
        active = sdk.current and sdk.current.status == "pending"
        if active and request.kind != "cancel":
            raise sdk_error("An SDK operation is still running")
        if request.kind == "reset" and sdk.operations:
            raise sdk_error("Only one reset is permitted per SDK lease")
        if request.kind == "open" and (sdk.opened or sdk.reset is None):
            raise sdk_error("Open requires a completed reset and no previous worker")
        if request.kind in {"execute", "finish"} and not sdk.worker_id:
            raise sdk_error("No SDK worker is open")
        if request.kind == "execute" and request.cell_id in sdk.cells:
            raise sdk_error("Cell ID has already been used; query the original operation")
        if active:
            sdk.current.task.cancel()
        if sdk.current is not None:
            sdk.current.data = None
            if sdk.current.status == "completed":
                sdk.current.status = "expired"
        if state.status == SessionStatus.READY:
            self.controller._sessions.transition(session_id, SessionStatus.BUSY)
        elif state.status != SessionStatus.BUSY or not active:
            raise sdk_error("Session is not ready for an SDK operation")
        op = SDKOperation(request.operation_id, digest, request.kind)
        sdk.operations[request.operation_id] = sdk.current = op
        if request.kind == "execute":
            sdk.cells.add(request.cell_id)
        op.task = asyncio.create_task(self._run(state, sdk, op, request))
        self.tasks.add(op.task)
        op.task.add_done_callback(lambda task: self._forget_task(op, task))
        return {"operation_id": op.operation_id, "status": "pending"}

    def _forget_task(self, op, task):
        self.tasks.discard(task)
        op.task = None

    def status(self, session_id, lease_id, operation_id):
        state = self._authorize(session_id, lease_id)
        sdk = state.sdk_session
        if sdk is None or sdk.lease_id != lease_id or operation_id not in sdk.operations:
            raise sdk_error("Unknown SDK operation; invalidate the attempt, never replay", 404)
        op = sdk.operations[operation_id]
        result = {"operation_id": operation_id, "status": op.status}
        if op.error:
            result["error"] = op.error
        if op.data is not None:
            result.update(
                bytes=len(op.data),
                sha256=op.sha256,
                chunks=(len(op.data) + RESULT_CHUNK_BYTES - 1) // RESULT_CHUNK_BYTES,
            )
        return result

    def result(self, session_id, lease_id, operation_id, chunk):
        self.status(session_id, lease_id, operation_id)
        op = self.controller._sessions.get_session(session_id).sdk_session.operations[operation_id]
        if op.data is None or chunk * RESULT_CHUNK_BYTES >= len(op.data):
            raise sdk_error("SDK result chunk unavailable", 404)
        data = op.data[chunk * RESULT_CHUNK_BYTES : (chunk + 1) * RESULT_CHUNK_BYTES]
        return {
            "operation_id": operation_id,
            "chunk": chunk,
            "data": base64.b64encode(data).decode("ascii"),
        }

    async def _run(self, state, sdk, op, request):
        try:
            async with asyncio.timeout(self.lifecycle_timeout):
                result = await self._dispatch(state, sdk, op, request)
                if not self._current(state, sdk, op):
                    raise sdk_error("SDK lease or backend changed during operation")
                data = await asyncio.to_thread(
                    json.dumps, result, ensure_ascii=False, separators=(",", ":"), allow_nan=False
                )
                data = data.encode("utf-8")
                if not self._current(state, sdk, op):
                    raise sdk_error("SDK ownership changed while encoding result")
                if len(data) > MAX_RESULT_BYTES:
                    raise sdk_error("SDK result exceeds transfer budget; attempt invalid")
                op.data, op.sha256 = data, hashlib.sha256(data).hexdigest()
                op.status = "completed"
                if state.status == SessionStatus.BUSY:
                    self.controller._sessions.transition(state.session_id, SessionStatus.READY)
                if request.kind in {"finish", "cancel"}:
                    sdk.dirty = False
        except BaseException as exc:
            op.error = f"{type(exc).__name__}: {exc}"[:2000]
            op.status = "failed"
            if self._current(state, sdk, op):
                if state.status != SessionStatus.INVALID:
                    self.controller._sessions.transition(state.session_id, SessionStatus.INVALID)
                state.force_relaunch = True
            if isinstance(exc, asyncio.CancelledError):
                raise

    async def _dispatch(self, state, sdk, op, request):
        self._check_current(state, sdk, op)
        controller = self.controller
        sid = state.session_id
        backend = controller._get_or_create_backend(state.backend)
        if request.kind == "reset":
            args = request.reset.model_dump(exclude={"lease_id", "action_id"})
            result = await controller._run_reset(
                sid, **args, action_id=request.operation_id, already_busy=True
            )
            self._check_current(state, sdk, op)
            sdk.reset = request.reset
            return result
        if request.kind == "open":
            device = request.device or state.backend.device_id
            if not device:
                raise sdk_error("Backend has no device ID; configure sdk_device explicitly")
            sdk.opened = True
            observation = await backend.code_request(
                "POST",
                "/code/session",
                timeout=60,
                json={
                    "task_name": state.current_task,
                    "device": device,
                    "coordinate_space": request.coordinate_space,
                },
            )
            self._check_current(state, sdk, op)
            sdk.worker_id = observation["session_id"]
            sdk.runtime = observation["runtime"]
            if (
                sdk.runtime.get("sdk_api_version") != "0.2"
                or sdk.runtime.get("protocol") != "mai-ui-code-v1"
                or not isinstance(sdk.worker_id, str)
                or not sdk.worker_id
            ):
                raise sdk_error("SDK training requires a MobileWorld API 0.2 worker")
            budget = float(sdk.runtime["cell_timeout_seconds"])
            if not 0 < budget <= self.lifecycle_timeout - 30:
                raise sdk_error("Worker cell timeout exceeds manager operation budget")
            return observation
        if request.kind == "execute":
            timeout = float(sdk.runtime["cell_timeout_seconds"]) + 10
            deadline = time.monotonic() + timeout + 20
            payload = {
                "session_id": sdk.worker_id,
                "cell_id": request.cell_id,
                "code": request.code,
            }
            try:
                result = await backend.code_request(
                    "POST", "/code/execute", timeout=timeout, json=payload
                )
            except (httpx.HTTPError, TimeoutError, OSError):
                result = {"status": "running"}

            while result["status"] == "running" and time.monotonic() < deadline:
                await asyncio.sleep(self.reconcile_interval)
                self._check_current(state, sdk, op)
                try:
                    result = await backend.code_request(
                        "GET",
                        "/code/result",
                        timeout=10,
                        params={"session_id": sdk.worker_id, "cell_id": request.cell_id},
                    )
                except (httpx.HTTPError, TimeoutError, OSError):
                    continue
            self._check_current(state, sdk, op)
            if (
                result.get("session_id") != sdk.worker_id
                or result.get("cell_id") != request.cell_id
                or result["status"] not in {"ok", "code_error"}
            ):
                raise sdk_error("Cannot confirm a valid cell outcome; never replay")
            state.step_count += 1
            return {"session_id": sdk.worker_id, "result": result}

        if not sdk.worker_id:
            raise sdk_error("Worker identity is unknown; backend must be recycled")
        receipt = await backend.code_request(
            "DELETE", "/code/session", timeout=30, params={"session_id": sdk.worker_id}
        )
        self._check_current(state, sdk, op)
        if (
            not receipt.get("process_exited")
            or not receipt.get("control_drained")
            or receipt.get("invalidated")
            or (receipt.get("stop_receipt") or {}).get("state") != "stopped"
        ):
            raise sdk_error("SDK worker did not confirm a clean stop and drained controls")
        sdk.worker_id = None
        result = {"cancelled": request.kind == "cancel", "close": receipt}
        if request.kind == "finish":
            result["evaluation"] = await controller.evaluate(
                sid, sdk.reset.category, sdk.reset.task_id, None, _sdk=True
            )
        self._check_current(state, sdk, op)
        await controller.tear_down(sid, sdk.reset.category, sdk.reset.task_id, _sdk=True)
        return result
