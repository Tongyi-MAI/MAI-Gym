from __future__ import annotations

import hashlib
from typing import Any

import httpx

_SECRET_KEYS = {"authorization", "token", "lease_id", "package_b64", "password", "api_key"}


def bounded_detail(value: Any, *, depth: int = 0, _budget: list[int] | None = None) -> Any:

    budget = [12000] if _budget is None else _budget
    if depth > 5 or budget[0] <= 0:
        return "[truncated]"
    if isinstance(value, dict):
        result = {}
        for key, item in list(value.items())[:32]:
            if str(key).lower() in _SECRET_KEYS:
                continue
            if budget[0] <= 0:
                break
            original = str(key)
            name = original
            if len(name) > 100:
                suffix = hashlib.sha256(original.encode()).hexdigest()[:16]
                name = f"{original[:83]}#{suffix}"

            collision = 0
            while name in result:
                collision += 1
                suffix = hashlib.sha256(f"{original}:{collision}".encode()).hexdigest()[:16]
                name = f"{original[:83]}#{suffix}"
            budget[0] -= len(name) + 8
            result[name] = bounded_detail(item, depth=depth + 1, _budget=budget)
        return result
    if isinstance(value, (list, tuple)):
        result = []
        for item in value[:64]:
            if budget[0] <= 0:
                break
            result.append(bounded_detail(item, depth=depth + 1, _budget=budget))
        return result
    if value is None or isinstance(value, (int, float, bool)):
        budget[0] -= 20
        return value
    result = str(value)[: min(2000, max(0, budget[0]))]
    budget[0] -= len(result) + 8
    return result


class OperationError(RuntimeError):
    def __init__(self, message: str, *, error_detail: dict[str, Any], status_code: int = 500):
        super().__init__(message[:2000])
        self.error_detail = bounded_detail(error_detail)
        self.status_code = status_code


def operation_error_detail(
    exc: BaseException,
    phase: str,
    *,
    operation_id: str | None = None,
) -> dict[str, Any]:

    if isinstance(exc, OperationError):
        detail = dict(exc.error_detail)
    else:
        detail: dict[str, Any] = {
            "code": "operation_failed",
            "phase": phase,
            "failure_kind": "unknown",
            "recovery": "fresh_session",
        }
        cause = exc
        seen: set[int] = set()
        while id(cause) not in seen:
            seen.add(id(cause))
            if isinstance(cause, httpx.HTTPStatusError):
                response = cause.response
                try:
                    body = response.json()
                except Exception:
                    body = response.text[:2000]
                detail.update(
                    upstream_status=response.status_code,
                    upstream_detail=bounded_detail(body),
                )
                payload = body
                if isinstance(payload, dict):
                    payload = payload.get("error_detail", payload.get("detail", payload))
                if isinstance(payload, dict):
                    for key in (
                        "code",
                        "stage",
                        "failure_kind",
                        "recovery",
                        "message",
                        "missing_classes",
                        "registered_classes",
                        "required_classes",
                    ):
                        if key in payload:
                            detail[key] = bounded_detail(payload[key])
                detail["code"] = detail.get("code") or "backend_error"
                break
            if isinstance(cause, (httpx.TransportError, TimeoutError)) or type(cause).__name__ in {
                "BackendCallError",
                "BackendUnavailableError",
                "CancelledError",
            }:
                detail.update(code="backend_unavailable", failure_kind="environment")
            next_cause = getattr(cause, "cause", None) or cause.__cause__
            if not isinstance(next_cause, BaseException):
                break
            cause = next_cause
        if detail["failure_kind"] not in {"task", "environment", "unknown"}:
            detail["failure_kind"] = "unknown"

        detail["recovery"] = "repair_task" if detail["failure_kind"] == "task" else "fresh_session"
    detail["phase"] = phase
    if operation_id is not None:
        detail["operation_id"] = operation_id

    ordered = {
        key: detail[key]
        for key in (
            "code",
            "phase",
            "stage",
            "failure_kind",
            "recovery",
            "operation_id",
            "upstream_status",
            "missing_classes",
            "registered_classes",
            "required_classes",
            "message",
        )
        if key in detail
    }
    if "upstream_detail" in detail:
        ordered["upstream_detail"] = detail["upstream_detail"]
    return bounded_detail(ordered)
