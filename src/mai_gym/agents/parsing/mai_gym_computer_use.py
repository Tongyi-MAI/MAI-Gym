from __future__ import annotations

import ast
import json
import re
from dataclasses import dataclass
from typing import Any

_THINK_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL | re.IGNORECASE)
_THINKING_RE = re.compile(r"<thinking>(.*?)</thinking>", re.DOTALL | re.IGNORECASE)
_BASH_RE = re.compile(r"<bash\s*>(.*?)</bash\s*>", re.DOTALL | re.IGNORECASE)
_ACTION_BLOCK_RE = re.compile(
    r"<bash\s*>.*?</bash\s*>|<tool_call[^>]*>.*?</tool_call>",
    re.DOTALL | re.IGNORECASE,
)
_TOOL_BODY_RE = re.compile(
    r"^<tool_call[^>]*>(.*?)</tool_call>$",
    re.DOTALL | re.IGNORECASE,
)
_TOOL_CALL_BEFORE_BASH_RE = re.compile(
    r"<tool_call[^>]*>\s*(?=<bash\s*>)",
    re.IGNORECASE,
)
_TOOL_CALL_AFTER_BASH_RE = re.compile(
    r"(</bash\s*>)\s*</tool_call>",
    re.IGNORECASE,
)

_SAFE_PYAUTOGUI_FUNCS = frozenset(
    {
        "click",
        "leftClick",
        "rightClick",
        "middleClick",
        "doubleClick",
        "tripleClick",
        "moveTo",
        "dragTo",
        "typewrite",
        "write",
        "press",
        "hotkey",
        "keyDown",
        "keyUp",
        "scroll",
        "hscroll",
        "vscroll",
        "mouseDown",
        "mouseUp",
    }
)
_XY_FUNCS = frozenset(
    {
        "click",
        "leftClick",
        "rightClick",
        "middleClick",
        "doubleClick",
        "tripleClick",
        "moveTo",
        "dragTo",
        "mouseDown",
        "mouseUp",
    }
)
_SCROLL_FUNCS = frozenset({"scroll", "hscroll", "vscroll"})
_CANONICAL_FUNCS = {
    name.lower(): name for name in (_SAFE_PYAUTOGUI_FUNCS | _SCROLL_FUNCS)
}


@dataclass(frozen=True)
class ActionParseResult:
    action_dict: dict[str, Any]
    history_action: str
    reasoning: str
    format_reward: float
    is_done: bool
    error: str | None = None

    @property
    def is_valid(self) -> bool:
        return self.error is None and self.action_dict.get("action_type") != "FAIL"


def _literal_value(node: ast.expr) -> bool:
    try:
        value = ast.literal_eval(node)
    except (SyntaxError, ValueError, TypeError):
        return False
    if value is None or isinstance(value, (str, int, float, bool)):
        return True
    return isinstance(value, (list, tuple)) and all(
        item is None or isinstance(item, (str, int, float, bool)) for item in value
    )


def validate_pyautogui_code(code: str) -> tuple[bool, str]:
    try:
        tree = ast.parse(code.strip(), mode="exec")
    except (SyntaxError, ValueError) as exc:
        return False, f"invalid Python call syntax: {exc}"
    if not tree.body:
        return False, "empty action"
    for statement in tree.body:
        if not isinstance(statement, ast.Expr) or not isinstance(statement.value, ast.Call):
            return False, "only direct function calls are allowed"
        call = statement.value
        if not isinstance(call.func, ast.Attribute) or not isinstance(call.func.value, ast.Name):
            return False, "only pyautogui function calls are allowed"
        if call.func.value.id != "pyautogui" or call.func.attr not in _SAFE_PYAUTOGUI_FUNCS:
            return False, f"unsupported GUI call: {call.func.value.id}.{call.func.attr}"
        if any(keyword.arg is None for keyword in call.keywords):
            return False, "expanded keyword arguments are not allowed"
        if not all(_literal_value(value) for value in call.args):
            return False, "GUI arguments must be literal values"
        if not all(_literal_value(keyword.value) for keyword in call.keywords):
            return False, "GUI arguments must be literal values"
    return True, ""


def _numeric_value(node: ast.expr | None) -> float | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return float(node.value)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        inner = _numeric_value(node.operand)
        return -inner if inner is not None else None
    return None


def project_coordinates(
    code: str,
    screen_width: int,
    screen_height: int,
    range_scale: float | None,
) -> str:
    if range_scale is None:
        return code
    try:
        tree = ast.parse(code)
    except (SyntaxError, ValueError):
        return code

    line_starts = [0]
    for line in code.splitlines(keepends=True):
        line_starts.append(line_starts[-1] + len(line))

    def span(node: ast.expr) -> tuple[int, int]:
        return (
            line_starts[node.lineno - 1] + node.col_offset,
            line_starts[node.end_lineno - 1] + node.end_col_offset,
        )

    edits: list[tuple[int, int, str]] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
            continue
        name = node.func.attr
        canonical = _CANONICAL_FUNCS.get(name.lower())
        if canonical not in _XY_FUNCS | _SCROLL_FUNCS:
            continue
        x_node: ast.expr | None = None
        y_node: ast.expr | None = None
        if canonical in _XY_FUNCS:
            if len(node.args) > 0:
                x_node = node.args[0]
            if len(node.args) > 1:
                y_node = node.args[1]
        else:
            if len(node.args) > 1:
                x_node = node.args[1]
            if len(node.args) > 2:
                y_node = node.args[2]
        for keyword in node.keywords:
            if keyword.arg == "x":
                x_node = keyword.value
            elif keyword.arg == "y":
                y_node = keyword.value
        x = _numeric_value(x_node)
        y = _numeric_value(y_node)
        if x is not None and x_node is not None:
            projected_x = round(x / range_scale * screen_width) if range_scale > 0 else round(x)
            edits.append((*span(x_node), str(int(projected_x))))
        if y is not None and y_node is not None:
            projected_y = round(y / range_scale * screen_height) if range_scale > 0 else round(y)
            edits.append((*span(y_node), str(int(projected_y))))
    for start, end, replacement in sorted(edits, reverse=True):
        code = code[:start] + replacement + code[end:]
    return code


def _call_string_argument(code: str, argument_name: str) -> str | None:
    try:
        tree = ast.parse(code)
    except (SyntaxError, ValueError):
        return None
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        for keyword in node.keywords:
            if keyword.arg == argument_name and isinstance(keyword.value, ast.Constant):
                return keyword.value.value if isinstance(keyword.value.value, str) else None
    return None


def _parse_json_tool_call(code: str) -> dict[str, Any] | None:
    if not code.lstrip().startswith("{"):
        return None
    try:
        payload = json.loads(code)
    except json.JSONDecodeError:
        return None
    if not isinstance(payload, dict):
        return None
    name = str(payload.get("name") or payload.get("tool") or "")
    parameters = payload.get("parameters", payload.get("arguments", {}))
    if not isinstance(parameters, dict):
        parameters = {}
    merged = {**payload, **parameters}
    if name in {"computer.wait", "wait"}:
        return {"action_type": "wait"}
    if name in {"computer.terminate", "terminate"}:
        status = "success" if str(merged.get("status", "success")).lower() == "success" else "failure"
        action: dict[str, Any] = {"action_type": "finished", "status": status}
        if isinstance(merged.get("answer"), str):
            action["answer"] = merged["answer"]
        return action
    if name in {"computer.ask_user", "ask_user"}:
        return {"action_type": "ask_user", "text": str(merged.get("question", ""))}
    pyautogui_name = name.removeprefix("pyautogui.")
    if pyautogui_name in _SAFE_PYAUTOGUI_FUNCS:
        args = merged.get("args", [])
        if not isinstance(args, list):
            args = []
        ignored = {"name", "tool", "parameters", "arguments", "args"}
        kwargs = {key: value for key, value in merged.items() if key not in ignored}
        rendered = [repr(value) for value in args]
        rendered.extend(f"{key}={value!r}" for key, value in kwargs.items())
        return {
            "action_type": "pyautogui_code",
            "code": f"pyautogui.{pyautogui_name}({', '.join(rendered)})",
        }
    return None


def _parse_tool_block(
    code: str,
    *,
    screen_size: tuple[int, int],
    range_scale: float | None,
) -> dict[str, Any] | None:
    json_action = _parse_json_tool_call(code)
    if json_action is not None:
        if json_action.get("action_type") == "pyautogui_code":
            json_action["code"] = project_coordinates(
                json_action["code"], *screen_size, range_scale
            )
        return json_action
    lowered = code.lower()
    if "computer.bash" in lowered:
        return {"action_type": "FAIL", "error": "legacy_computer_bash_is_disabled"}
    if "computer.wait" in lowered:
        return {"action_type": "wait"}
    if "computer.terminate" in lowered:
        status = _call_string_argument(code, "status")
        action: dict[str, Any] = {
            "action_type": "finished",
            "status": "success" if status == "success" else "failure",
        }
        answer = _call_string_argument(code, "answer")
        if answer:
            action["answer"] = answer
        return action
    if "computer.ask_user" in lowered:
        return {
            "action_type": "ask_user",
            "text": _call_string_argument(code, "question") or "",
        }
    valid, _reason = validate_pyautogui_code(code)
    if not valid:
        return None
    return {
        "action_type": "pyautogui_code",
        "code": project_coordinates(code, *screen_size, range_scale),
    }


def _contains_action(action: dict[str, Any], action_type: str) -> bool:
    if action.get("action_type") == action_type:
        return True
    return action.get("action_type") == "action_sequence" and any(
        isinstance(child, dict) and _contains_action(child, action_type)
        for child in action.get("actions", [])
    )


def parse_action_response(
    text: str,
    *,
    screen_size: tuple[int, int] = (1920, 1080),
    range_scale: float | None = 1000,
    enable_ask_user: bool = False,
    finish_reason: str = "",
) -> ActionParseResult:
    original = str(text or "")
    lowered = original.lower()
    if "</think>" in lowered and "<think>" not in lowered.split("</think>", 1)[0]:
        original = "<think>" + original
    elif finish_reason == "length" and "</think>" not in lowered:
        return ActionParseResult(
            action_dict={"action_type": "FAIL"},
            history_action="",
            reasoning=original,
            format_reward=0.0,
            is_done=True,
            error="reasoning_truncated",
        )

    reasoning_parts = [match.group(1).strip() for match in _THINK_RE.finditer(original)]
    reasoning_parts.extend(match.group(1).strip() for match in _THINKING_RE.finditer(original))
    action_content = _THINK_RE.sub("", original)
    action_content = _THINKING_RE.sub("", action_content)
    action_content = _TOOL_CALL_BEFORE_BASH_RE.sub("", action_content)
    action_content = _TOOL_CALL_AFTER_BASH_RE.sub(r"\1", action_content)

    actions: list[dict[str, Any]] = []
    history_blocks: list[str] = []
    error: str | None = None
    for match in _ACTION_BLOCK_RE.finditer(action_content):
        block = match.group(0).strip()
        bash = _BASH_RE.fullmatch(block)
        if bash:
            actions.append({"action_type": "execute_bash", "command": bash.group(1).strip()})
            history_blocks.append(block)
            continue
        tool = _TOOL_BODY_RE.match(block)
        if tool is None:
            continue
        action = _parse_tool_block(
            tool.group(1).strip(),
            screen_size=screen_size,
            range_scale=range_scale,
        )
        if action is None:
            continue
        if action.get("action_type") == "FAIL":
            error = str(action.get("error", "invalid_action"))
            actions = []
            break
        actions.append(action)
        history_blocks.append(block)

    if not actions:
        return ActionParseResult(
            action_dict={"action_type": "FAIL"},
            history_action="\n".join(history_blocks),
            reasoning="\n".join(part for part in reasoning_parts if part),
            format_reward=0.0,
            is_done=True,
            error=error or "no_executable_action",
        )
    action_dict = actions[0] if len(actions) == 1 else {
        "action_type": "action_sequence",
        "actions": actions,
    }
    if not enable_ask_user and _contains_action(action_dict, "ask_user"):
        return ActionParseResult(
            action_dict={"action_type": "FAIL"},
            history_action="\n".join(history_blocks),
            reasoning="\n".join(reasoning_parts),
            format_reward=0.0,
            is_done=True,
            error="ask_user_is_disabled",
        )
    return ActionParseResult(
        action_dict=action_dict,
        history_action="\n".join(history_blocks),
        reasoning="\n".join(part for part in reasoning_parts if part),
        format_reward=1.0,
        is_done=_contains_action(action_dict, "finished"),
    )


__all__ = [
    "ActionParseResult",
    "parse_action_response",
    "project_coordinates",
    "validate_pyautogui_code",
]
