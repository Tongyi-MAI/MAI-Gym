from __future__ import annotations

import ast
import json
import re
from typing import Any

try:
    from json_repair import repair_json
except ImportError:
    repair_json = None


_XY_POS_FUNCS = frozenset(
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


_CANON_XY = {f.lower(): f for f in (_XY_POS_FUNCS | _SCROLL_FUNCS)}


_COORD_PATTERNS = [
    re.compile(r"\bx\s*=\s*([\d.]+)\s*,\s*y\s*=\s*([\d.]+)"),
    re.compile(r"\(\s*([\d.]+)\s*,\s*([\d.]+)\s*\)"),
]


def _coordinate_projection(
    x: float,
    y: float,
    screen_width: int,
    screen_height: int,
    range_scale: float,
) -> tuple[int, int]:

    if range_scale and range_scale > 0:
        return (
            int(round(x / range_scale * screen_width)),
            int(round(y / range_scale * screen_height)),
        )
    if x <= 1.0 and y <= 1.0:
        return int(round(x * screen_width)), int(round(y * screen_height))
    return int(round(x)), int(round(y))


def _func_name(func: ast.expr) -> str | None:

    if isinstance(func, ast.Attribute):
        return func.attr
    if isinstance(func, ast.Name):
        return func.id
    return None


def _num_value(node: ast.expr | None) -> float | None:

    if node is None:
        return None
    if (
        isinstance(node, ast.Constant)
        and isinstance(node.value, (int, float))
        and not isinstance(node.value, bool)
    ):
        return float(node.value)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        inner = _num_value(node.operand)
        return -inner if inner is not None else None
    return None


def _regex_project(
    code: str,
    screen_width: int,
    screen_height: int,
    range_scale: float,
) -> str:

    def repl(match: re.Match, named: bool) -> str:
        try:
            x, y = float(match.group(1)), float(match.group(2))
        except (ValueError, TypeError):
            return match.group(0)
        nx, ny = _coordinate_projection(x, y, screen_width, screen_height, range_scale)
        return f"x={nx}, y={ny}" if named else f"({nx}, {ny})"

    code = _COORD_PATTERNS[0].sub(lambda m: repl(m, named=True), code)
    code = _COORD_PATTERNS[1].sub(lambda m: repl(m, named=False), code)
    return code


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
        repaired = re.sub(
            r"\bx(\s*=\s*-?[\d.]+\s*,\s*)(-?[\d.]+)(\s*[,)])",
            r"x\1y=\2\3",
            code,
        )
        repaired = re.sub(
            r"\by(\s*=\s*-?[\d.]+\s*,\s*)(-?[\d.]+)(\s*[,)])",
            r"y\1x=\2\3",
            repaired,
        )
        try:
            tree = ast.parse(repaired)
            code = repaired
        except (SyntaxError, ValueError):
            return _regex_project(code, screen_width, screen_height, range_scale)

    line_starts = [0]
    for line in code.splitlines(keepends=True):
        line_starts.append(line_starts[-1] + len(line))

    def span(node: ast.expr) -> tuple[int, int]:
        start = line_starts[node.lineno - 1] + node.col_offset
        end = line_starts[node.end_lineno - 1] + node.end_col_offset
        return start, end

    edits: list[tuple[int, int, str]] = []

    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        fn = _func_name(node.func)
        if fn is None:
            continue

        canon = _CANON_XY.get(fn.lower())
        if canon is None:
            continue

        x_node: ast.expr | None = None
        y_node: ast.expr | None = None
        if canon in _XY_POS_FUNCS:
            if len(node.args) >= 1:
                x_node = node.args[0]
            if len(node.args) >= 2:
                y_node = node.args[1]
        else:
            if len(node.args) >= 2:
                x_node = node.args[1]
            if len(node.args) >= 3:
                y_node = node.args[2]

        if fn != canon:
            fend = line_starts[node.func.end_lineno - 1] + node.func.end_col_offset
            edits.append((fend - len(fn), fend, canon))

        for kw in node.keywords:
            if kw.arg == "x":
                x_node = kw.value
            elif kw.arg == "y":
                y_node = kw.value

        xv = _num_value(x_node)
        yv = _num_value(y_node)

        if xv is not None and yv is not None:
            nx, ny = _coordinate_projection(
                xv,
                yv,
                screen_width,
                screen_height,
                range_scale,
            )
            edits.append((*span(x_node), str(nx)))
            edits.append((*span(y_node), str(ny)))
        elif range_scale and range_scale > 0:
            if xv is not None:
                edits.append((*span(x_node), str(int(round(xv / range_scale * screen_width)))))
            if yv is not None:
                edits.append((*span(y_node), str(int(round(yv / range_scale * screen_height)))))

    for start, end, text in sorted(edits, key=lambda e: e[0], reverse=True):
        code = code[:start] + text + code[end:]
    return code


def parse_terminate_status(code: str) -> str:

    try:
        tree = ast.parse(code)
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call) or _func_name(node.func) != "terminate":
                continue
            val: object = None
            for kw in node.keywords:
                if kw.arg == "status" and isinstance(kw.value, ast.Constant):
                    val = kw.value.value
            if val is None and node.args and isinstance(node.args[0], ast.Constant):
                val = node.args[0].value
            if isinstance(val, str):
                return "success" if val.strip().lower() == "success" else "failure"
            return "failure"
    except (SyntaxError, ValueError):
        pass

    m = re.search(r"""status\s*=\s*['"]\s*([a-zA-Z]+)""", code)
    if not m:
        m = re.search(r"""terminate\s*\(\s*['"]\s*([a-zA-Z]+)""", code)
    if m:
        return "success" if m.group(1).strip().lower() == "success" else "failure"
    return "failure"


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


def _literal_tool_value(value: Any) -> bool:
    if value is None or isinstance(value, (str, int, float, bool)):
        return True
    return isinstance(value, (list, tuple)) and all(_literal_tool_value(item) for item in value)


def _json_pyautogui_call(name: str, merged: dict[str, Any]) -> str | None:
    ignored = {"name", "tool", "type", "parameters", "arguments"}
    arguments = {key: value for key, value in merged.items() if key not in ignored}

    for key in ("coordinates", "position"):
        pair = arguments.pop(key, None)
        if isinstance(pair, (list, tuple)) and len(pair) == 2:
            arguments.setdefault("x", pair[0])
            arguments.setdefault("y", pair[1])
    x_value = arguments.get("x")
    if (
        name in _XY_POS_FUNCS
        and "y" not in arguments
        and isinstance(x_value, (list, tuple))
        and len(x_value) == 2
    ):
        arguments["x"], arguments["y"] = x_value

    positional = arguments.pop("args", [])
    if not isinstance(positional, (list, tuple)):
        return None
    if not all(_literal_tool_value(value) for value in positional):
        return None
    if not all(
        isinstance(key, str) and _literal_tool_value(value) for key, value in arguments.items()
    ):
        return None
    rendered = [repr(value) for value in positional]
    rendered.extend(f"{key}={value!r}" for key, value in arguments.items())
    return f"pyautogui.{name}({', '.join(rendered)})"


def parse_json_tool_call(code: str) -> dict[str, Any] | None:

    text = code.strip()
    if not (text.startswith("{") and text.endswith("}")):
        return None

    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        try:
            payload = ast.literal_eval(text)
        except (SyntaxError, ValueError):
            if repair_json is not None:
                try:
                    payload = json.loads(repair_json(text))
                except (json.JSONDecodeError, TypeError, ValueError):
                    return _parse_loose_json_style_bash(text)
            else:
                return _parse_loose_json_style_bash(text)

    if not isinstance(payload, dict):
        return None

    name = str(payload.get("name") or payload.get("tool") or payload.get("type") or "")
    name = name.strip()
    params: Any = payload.get("parameters", payload.get("arguments", {}))
    if isinstance(params, str):
        try:
            params = json.loads(params)
        except json.JSONDecodeError:
            try:
                params = ast.literal_eval(params)
            except (SyntaxError, ValueError):
                params = {}
    if not isinstance(params, dict):
        params = {}

    merged = {**payload, **params}
    if name in {"computer.wait", "wait"}:
        return {"action_type": "wait"}

    if name in {"computer.bash", "bash"}:
        command = merged.get("command")
        if isinstance(command, str) and command:
            return {"action_type": "execute_bash", "command": command}
        return None

    if name in {"computer.terminate", "terminate", "finished", "answer"}:
        status = str(merged.get("status") or "success").strip().lower()
        action: dict[str, Any] = {
            "action_type": "finished",
            "status": "success" if status == "success" else "failure",
        }
        answer = merged.get("answer", merged.get("content"))
        if isinstance(answer, str) and answer:
            action["answer"] = answer
        return action

    pyautogui_name = name.removeprefix("pyautogui.")
    if pyautogui_name in _SAFE_PYAUTOGUI_FUNCS:
        call = _json_pyautogui_call(pyautogui_name, merged)
        if call is not None:
            return {"action_type": "pyautogui_code", "code": call}
    return None


def _parse_loose_json_style_bash(text: str) -> dict[str, Any] | None:

    if not re.search(r"""["'](?:name|tool|type)["']\s*:\s*["'](?:computer\.)?bash["']""", text):
        return None

    m = re.search(r"""["']command["']\s*:\s*["']""", text)
    if not m:
        return None

    quote = text[m.end() - 1]
    command_start = m.end()
    command_end = _find_loose_json_command_end(text, command_start, quote)
    command = text[command_start:command_end]
    if not command:
        return None
    return {"action_type": "execute_bash", "command": _unescape_loose_json_string(command)}


def _find_loose_json_command_end(text: str, start: int, quote: str) -> int:

    closing_patterns = (
        quote + "}}}",
        quote + "}}",
        quote + "\n}}}",
        quote + "\n}}",
    )
    candidates = [text.rfind(pattern) for pattern in closing_patterns]
    best = max(candidates)
    if best >= start:
        return best

    escaped = False
    for idx in range(start, len(text)):
        ch = text[idx]
        if escaped:
            escaped = False
            continue
        if ch == "\\":
            escaped = True
            continue
        if ch == quote:
            return idx
    return len(text)


def _unescape_loose_json_string(value: str) -> str:

    try:
        return json.loads(f'"{value}"')
    except json.JSONDecodeError:
        return (
            value.replace(r"\\", "\\")
            .replace(r"\"", '"')
            .replace(r"\'", "'")
            .replace(r"\n", "\n")
            .replace(r"\t", "\t")
            .replace(r"\r", "\r")
        )
