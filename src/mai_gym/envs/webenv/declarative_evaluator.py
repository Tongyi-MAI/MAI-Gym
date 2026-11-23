from __future__ import annotations

import re
from typing import Any

from playwright.async_api import Page


class _Missing:
    def __bool__(self) -> bool:
        return False


MISSING = _Missing()
_NUMBER_RE = re.compile(r"-?\d+(?:\.\d+)?")
_PATH_TOKEN_RE = re.compile(
    r"""
      \[\s*(?P<idx>-?\d+|last|\*)\s*\]
    | \[\s*\?(?P<key>[^=\]]+)=(?P<val>[^\]]*)\]
    | \.?(?P<name>[^.\[\]]+)
    """,
    re.X,
)


def _parse_path(path: str) -> list[tuple[str, Any]]:
    steps: list[tuple[str, Any]] = []
    pos = 0
    while pos < len(path.strip()):
        match = _PATH_TOKEN_RE.match(path.strip(), pos)
        if not match:
            raise ValueError(f"bad path {path!r} at offset {pos}")
        pos = match.end()
        if match.group("idx") is not None:
            token = match.group("idx")
            steps.append(("star", None) if token == "*" else ("index", token))
        elif match.group("key") is not None:
            steps.append(("filter", (match.group("key").strip(), match.group("val").strip())))
        else:
            steps.append(("name", match.group("name")))
    return steps


def _coerce_equal(actual: Any, literal: str) -> bool:
    if actual is MISSING:
        return False
    if isinstance(actual, bool):
        return str(actual).lower() == literal.lower()
    if isinstance(actual, (int, float)):
        try:
            return float(actual) == float(literal)
        except ValueError:
            return False
    return str(actual) == literal


def _resolve_one(current: Any, kind: str, arg: Any) -> Any:
    if kind == "name":
        if isinstance(current, dict):
            return current.get(arg, MISSING)
        return MISSING
    if kind == "index":
        if not isinstance(current, (list, tuple)):
            return MISSING
        if arg in ("last", "-1"):
            return current[-1] if current else MISSING
        index = int(arg)
        return current[index] if -len(current) <= index < len(current) else MISSING
    if kind == "filter":
        key, value = arg
        if not isinstance(current, list):
            return MISSING
        return [item for item in current if _coerce_equal(_resolve(item, key), value)]
    raise AssertionError(kind)


def _resolve(obj: Any, path: str) -> Any:
    if path in ("", "$"):
        return obj
    current = obj
    fanned = False
    for kind, arg in _parse_path(path):
        if kind == "star":
            if not isinstance(current, list):
                return MISSING
            fanned = True
            continue
        if fanned:
            current = [
                value
                for value in (_resolve_one(item, kind, arg) for item in current)
                if value is not MISSING
            ]
        else:
            current = _resolve_one(current, kind, arg)
        if current is MISSING:
            return MISSING
    return current


def _as_list(value: Any) -> list[Any]:
    if value is MISSING or value is None:
        return []
    return list(value) if isinstance(value, (list, tuple)) else [value]


def _number(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        match = _NUMBER_RE.search(value)
        if match:
            return float(match.group())
    return None


def _project(items: Any, paths: list[str] | None) -> list[Any]:
    projected: list[Any] = []
    for item in _as_list(items):
        if paths:
            row = []
            for path in paths:
                value = _resolve(item, path)
                row.append(None if value is MISSING else value)
            projected.append(tuple(row))
        else:
            projected.append(tuple(item) if isinstance(item, list) else item)
    return projected


def _freeze(value: Any) -> Any:
    if isinstance(value, list):
        return tuple(_freeze(item) for item in value)
    if isinstance(value, dict):
        return tuple(sorted((key, _freeze(item)) for key, item in value.items()))
    return value


def _apply_criterion(criterion: dict[str, Any], actual: Any, context: dict[str, Any]) -> bool:
    op = criterion["op"]
    expected = criterion.get("value")

    if op == "exists":
        return actual is not MISSING
    if op == "absent":
        return actual is MISSING
    if actual is MISSING and op not in ("ne", "not_in", "not_contains", "none_where"):
        return False

    if op == "eq":
        if isinstance(actual, str) and isinstance(expected, str):
            return actual.strip() == expected.strip()
        return bool(actual == expected)
    if op == "ne":
        return bool(actual != expected)
    if op in ("lt", "lte", "gt", "gte"):
        left, right = _number(actual), _number(expected)
        if left is None or right is None:
            return False
        return {"lt": left < right, "lte": left <= right, "gt": left > right, "gte": left >= right}[
            op
        ]
    if op == "approx":
        left, right = _number(actual), _number(expected)
        return (
            left is not None
            and right is not None
            and abs(left - right) <= float(criterion.get("tol", 1e-6))
        )
    if op == "in":
        return actual in _as_list(expected)
    if op == "not_in":
        return actual not in _as_list(expected)
    if op == "contains":
        return str(expected) in actual if isinstance(actual, str) else expected in _as_list(actual)
    if op == "not_contains":
        return (
            str(expected) not in actual
            if isinstance(actual, str)
            else expected not in _as_list(actual)
        )
    if op == "regex":
        return bool(re.search(str(expected), str(actual)))
    if op in ("len_eq", "len_gte", "len_lte"):
        length = len(actual) if hasattr(actual, "__len__") else 0
        expected_length = int(criterion["value"])
        return {
            "len_eq": length == expected_length,
            "len_gte": length >= expected_length,
            "len_lte": length <= expected_length,
        }[op]
    if op == "set_eq":
        return {_freeze(item) for item in _project(actual, criterion.get("project"))} == {
            _freeze(item) for item in _project(expected, criterion.get("project"))
        }
    if op == "multiset_eq":
        actual_items = sorted(
            repr(_freeze(item)) for item in _project(actual, criterion.get("project"))
        )
        expected_items = sorted(
            repr(_freeze(item)) for item in _project(expected, criterion.get("project"))
        )
        return actual_items == expected_items
    if op.endswith("_where"):
        where = criterion.get("where") or {}
        hits = [item for item in _as_list(actual) if _evaluate_one(where, item, context)["passed"]]
        if op == "count_where":
            return len(hits) == int(criterion["value"])
        if op == "any_where":
            return bool(hits)
        if op == "all_where":
            return len(hits) == len(_as_list(actual))
        if op == "none_where":
            return not hits
    raise ValueError(f"unimplemented op: {op}")


def _evaluate_one(criterion: dict[str, Any], root: Any, context: dict[str, Any]) -> dict[str, Any]:
    source = criterion.get("source", "state")
    actual = (
        context.get("answer", MISSING)
        if source == "answer"
        else _resolve(root, criterion.get("path", ""))
    )
    if source == "answer" and actual is None:
        actual = MISSING
    if "value_path" in criterion:
        expected = _resolve(context["state"], criterion["value_path"])
        criterion = {**criterion, "value": expected}
    if criterion.get("extract") in {"single_number", "numbers"} and actual is not MISSING:
        numbers = [float(n) for n in _NUMBER_RE.findall(str(actual))]
        if criterion["extract"] == "numbers":
            actual = numbers
        else:
            actual = numbers[0] if len(numbers) == 1 else MISSING
    if criterion.get("extract") == "number" and actual is not MISSING:
        actual = _number(actual)
        if actual is None:
            actual = MISSING
    try:
        passed = criterion.get("value") is not MISSING and _apply_criterion(
            criterion, actual, context
        )
    except Exception as exc:
        passed = False
        actual = f"<verifier error: {exc}>"
    return {
        "name": criterion.get("name", "unnamed criterion"),
        "passed": bool(passed),
        "critical": bool(criterion.get("critical", True)),
        "weight": float(criterion.get("weight", 1.0)),
        "actual": None if actual is MISSING else actual,
        "expected": None if criterion.get("value") is MISSING else criterion.get("value"),
    }


def evaluate_declarative_task(
    task: dict[str, Any],
    state: dict[str, Any],
    actions: list[dict[str, Any]],
    accepted_events: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:

    meta = state.get("meta") or {}
    events = accepted_events or []
    trajectory = {
        "accepted_events": events,
        "accepted_actions": [event["action"] for event in events],
        "accepted_ids": [event["action"]["id"] for event in events if event["action"].get("id")],
        "steps": int(meta.get("steps", 0)),
        "n_actions": len(actions),
        "actions": actions,
        "ids": [action.get("id") for action in actions],
        "errors": [],
        "truncated": False,
    }
    context = {"state": state, "trajectory": trajectory, "answer": meta.get("answer")}
    results = []
    for criterion in task.get("goal", []):
        source = criterion.get("source", "state")
        results.append(_evaluate_one(criterion, context.get(source, {}), context))
    total_weight = sum(result["weight"] for result in results) or 1.0
    score = sum(result["weight"] for result in results if result["passed"]) / total_weight
    failed = [result for result in results if not result["passed"]]
    return {
        "score": round(score, 4),
        "success": all(result["passed"] for result in results if result["critical"]),
        "passed": len(results) - len(failed),
        "total": len(results),
        "failed": failed,
    }


async def run_declarative_evaluator(page: Page, task: dict[str, Any]) -> dict[str, Any]:
    payload = await page.evaluate(
        """() => ({
            env_id: window.__ENV__.ENV_ID,
            state: window.__ENV__.getState(),
            actions: window.__ENV__.trajectory(),
            events: window.maiGymAcceptedEvents,
        })"""
    )
    if payload.get("env_id") and task.get("env_id") and payload["env_id"] != task["env_id"]:
        raise ValueError(f"Expected {task['env_id']}, found {payload['env_id']}")
    return evaluate_declarative_task(
        task, payload["state"], payload["actions"], payload.get("events")
    )
