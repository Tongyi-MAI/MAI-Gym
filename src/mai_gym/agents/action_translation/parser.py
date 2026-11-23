from __future__ import annotations

import ast
import logging
from dataclasses import dataclass

from mai_gym.agents.action_translation.canonical import Canonical, CanonicalKind, noop

logger = logging.getLogger(__name__)


@dataclass
class PointerState:
    position: tuple[int, int] | None = None


def parse_pyautogui(
    code: str,
    *,
    pointer: PointerState | None = None,
    preserve_clicks: bool = False,
    preserve_moves: bool = False,
) -> list[Canonical]:

    module = ast.parse(code)
    statements = [stmt for stmt in module.body if isinstance(stmt, ast.Expr)]
    canonicals: list[Canonical] = []
    skip_next = False
    for i, stmt in enumerate(statements):
        if skip_next:
            skip_next = False
            continue

        call = stmt.value if isinstance(stmt.value, ast.Call) else None
        if call is None:
            canonicals.append(noop(ast.unparse(stmt)))
            continue

        if preserve_clicks and _is_call(call, "pyautogui", ("click", "tripleClick")):
            count = (
                3
                if _is_call(call, "pyautogui", "tripleClick")
                else _int_kw(call, "clicks", default=1)
            )
            xy = _xy(call)
            if count is not None and count > 2 and xy is not None:
                if count > 100:
                    raise ValueError("A single multi-click action exceeds 100 clicks")
                button = _string_kw(call, "button", default="left")
                kind: CanonicalKind = (
                    "right_click" if button.lower() in ("right", "secondary") else "click"
                )
                canonicals.extend(
                    Canonical(kind=kind, x=xy[0], y=xy[1], raw=ast.unparse(call))
                    for _ in range(count)
                )
                if pointer is not None:
                    pointer.position = xy
                continue

        next_expression = statements[i + 1].value if i + 1 < len(statements) else None
        next_call = next_expression if isinstance(next_expression, ast.Call) else None
        if (
            _is_call(call, "pyautogui", "moveTo")
            and next_call is not None
            and _is_call(next_call, "pyautogui", "dragTo")
        ):
            move_xy = _xy(call)
            drag_xy = _xy(next_call)
            if move_xy is not None and drag_xy is not None:
                canonicals.append(
                    Canonical(
                        kind="drag",
                        x=move_xy[0],
                        y=move_xy[1],
                        end_x=drag_xy[0],
                        end_y=drag_xy[1],
                        button=_string_kw(next_call, "button", default="left"),
                        raw=f"{ast.unparse(call)}; {ast.unparse(next_call)}",
                    )
                )
                skip_next = True
                if pointer is not None:
                    pointer.position = drag_xy
                continue

        if (pointer is not None or preserve_moves) and _is_call(call, "pyautogui", "moveTo"):
            xy = _xy(call)
            if xy is not None:
                if pointer is not None:
                    pointer.position = xy
                canonicals.append(
                    Canonical(
                        kind="move" if preserve_moves else "wait",
                        x=xy[0],
                        y=xy[1],
                        raw=ast.unparse(call),
                    )
                )
                continue
        if pointer is not None and _is_call(call, "pyautogui", "dragTo"):
            end = _xy(call)
            if end is not None and pointer.position is not None:
                start = pointer.position
                canonicals.append(
                    Canonical(
                        kind="drag",
                        x=start[0],
                        y=start[1],
                        end_x=end[0],
                        end_y=end[1],
                        raw=ast.unparse(call),
                    )
                )
                pointer.position = end
                continue
        result = _parse_call(call)
        if pointer is not None and result is not None and result.origin is not None:
            pointer.position = result.origin
        canonicals.append(result if result is not None else noop(ast.unparse(call)))

    return canonicals


def _parse_call(call: ast.Call) -> Canonical | None:
    raw = ast.unparse(call)

    if _is_call(call, "pyautogui", "click"):
        xy = _xy(call)
        if xy is None:
            return None
        button = _string_kw(call, "button", default="left")
        clicks = _int_kw(call, "clicks", default=1)
        if clicks is None:
            return None
        if clicks >= 2:
            return Canonical(kind="double_click", x=xy[0], y=xy[1], raw=raw)
        if button.lower() in ("right", "secondary"):
            return Canonical(kind="right_click", x=xy[0], y=xy[1], raw=raw)
        return Canonical(kind="click", x=xy[0], y=xy[1], button=button, raw=raw)

    if _is_call(call, "pyautogui", "doubleClick"):
        xy = _xy(call)
        if xy is None:
            return None
        return Canonical(kind="double_click", x=xy[0], y=xy[1], raw=raw)

    if _is_call(call, "pyautogui", "rightClick"):
        xy = _xy(call)
        if xy is None:
            return None
        return Canonical(kind="right_click", x=xy[0], y=xy[1], raw=raw)

    if _is_call(call, "pyautogui", ("write", "typewrite")):
        text = _first_string(call)
        if text is None:
            return None
        return Canonical(kind="type", text=text, raw=raw)

    if _is_call(call, "pyautogui", "press"):
        first = call.args[0] if call.args else None
        if first is None:
            return None
        if isinstance(first, ast.Constant) and isinstance(first.value, str):
            return Canonical(kind="key", key=first.value, raw=raw)
        if isinstance(first, (ast.List, ast.Tuple)):
            keys = [
                e.value
                for e in first.elts
                if isinstance(e, ast.Constant) and isinstance(e.value, str)
            ]
            if keys:
                return Canonical(kind="key", key=keys[0], keys=tuple(keys), raw=raw)
        return None

    if _is_call(call, "pyautogui", "hotkey"):
        hotkeys = tuple(
            a.value for a in call.args if isinstance(a, ast.Constant) and isinstance(a.value, str)
        )
        if not hotkeys:
            return None
        return Canonical(kind="hotkey", keys=hotkeys, raw=raw)

    if _is_call(call, "pyautogui", "scroll"):
        if not call.args:
            return None
        clicks_node = call.args[0]
        if not (
            isinstance(clicks_node, ast.Constant) and isinstance(clicks_node.value, (int, float))
        ):
            value = _signed_int(clicks_node)
            if value is None:
                return None
        else:
            value = int(clicks_node.value)
        xy = _xy(call, min_args=3) if len(call.args) >= 3 else None
        return Canonical(
            kind="scroll",
            scroll_clicks=value,
            x=xy[0] if xy else None,
            y=xy[1] if xy else None,
            raw=raw,
        )

    if _is_call(call, "pyautogui", "moveTo"):
        return None

    if _is_call(call, "pyautogui", "dragTo"):
        return None

    if _is_call(call, "time", "sleep"):
        if not call.args:
            return Canonical(kind="wait", duration=0.0, raw=raw)
        node = call.args[0]
        seconds = 0.0
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            seconds = float(node.value)
        return Canonical(kind="wait", duration=seconds, raw=raw)

    return None


def _is_call(call: ast.Call, module: str, attr: str | tuple[str, ...]) -> bool:

    func = call.func
    if not isinstance(func, ast.Attribute):
        return False
    if not isinstance(func.value, ast.Name) or func.value.id != module:
        return False
    if isinstance(attr, tuple):
        return func.attr in attr
    return func.attr == attr


def _xy(call: ast.Call, *, min_args: int = 2) -> tuple[int, int] | None:

    x: int | None = None
    y: int | None = None
    if len(call.args) >= min_args:
        offset = min_args - 2
        x = _signed_int(call.args[offset])
        y = _signed_int(call.args[offset + 1])
    if x is None:
        x = _int_kw(call, "x", default=None)
    if y is None:
        y = _int_kw(call, "y", default=None)
    if x is None or y is None:
        return None
    return x, y


def _signed_int(node: ast.AST | None) -> int | None:

    if node is None:
        return None
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return int(node.value)
    if (
        isinstance(node, ast.UnaryOp)
        and isinstance(node.op, ast.USub)
        and isinstance(node.operand, ast.Constant)
        and isinstance(node.operand.value, (int, float))
    ):
        return -int(node.operand.value)
    return None


def _first_string(call: ast.Call) -> str | None:
    if not call.args:
        for kw in call.keywords:
            if (
                kw.arg == "text"
                and isinstance(kw.value, ast.Constant)
                and isinstance(kw.value.value, str)
            ):
                return kw.value.value
        return None
    first = call.args[0]
    if isinstance(first, ast.Constant) and isinstance(first.value, str):
        return first.value
    return None


def _string_kw(call: ast.Call, name: str, *, default: str) -> str:
    for kw in call.keywords:
        if (
            kw.arg == name
            and isinstance(kw.value, ast.Constant)
            and isinstance(kw.value.value, str)
        ):
            return kw.value.value
    return default


def _int_kw(call: ast.Call, name: str, *, default: int | None) -> int | None:
    for kw in call.keywords:
        if kw.arg == name:
            value = _signed_int(kw.value)
            if value is not None:
                return value
    return default


__all__ = ["parse_pyautogui"]
