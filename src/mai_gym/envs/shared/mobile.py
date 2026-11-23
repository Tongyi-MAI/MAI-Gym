from collections.abc import Awaitable, Callable
from typing import Any


async def execute_mobile_decision(
    action: dict[str, Any],
    send: Callable[[dict[str, Any]], Awaitable[Any]],
    *,
    terminal: Callable[[dict[str, Any]], Awaitable[None]] | None = None,
) -> dict[str, Any]:
    if action.get("action_type") == "action_sequence":
        results = []
        for child in action["actions"]:
            result = await execute_mobile_decision(child, send, terminal=terminal)
            results.append(result)
            if result.get("done"):
                return {"done": True, "results": results}
        return {"done": False, "results": results}
    kind = action.get("action_type")
    if kind in {"finished", "terminate"}:
        if terminal is not None:
            await terminal(action)
        return {"done": True}
    result = await send(action)
    native_done = bool(result.get("done", False)) if isinstance(result, dict) else False
    return {"done": kind == "answer" or native_done, "result": result}
