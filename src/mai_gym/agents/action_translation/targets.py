from __future__ import annotations

import logging
from typing import Any

from mai_gym.agents.action_translation.canonical import Canonical

logger = logging.getLogger(__name__)


_KEY_TO_MOBILE: dict[str, dict[str, Any]] = {
    "enter": {"action_type": "keyboard_enter"},
    "return": {"action_type": "keyboard_enter"},
    "escape": {"action_type": "navigate_back"},
    "esc": {"action_type": "navigate_back"},
    "home": {"action_type": "navigate_home"},
    "back": {"action_type": "navigate_back"},
    "backspace": {"action_type": "type", "text": "\b"},
    "space": {"action_type": "type", "text": " "},
    "tab": {"action_type": "type", "text": "\t"},
}


def _norm(value: int, axis_size: int) -> float:

    if axis_size <= 0:
        return 0.0
    return max(0.0, min(1.0, value / float(axis_size)))


def to_mobile(
    c: Canonical,
    screen: tuple[int, int],
    *,
    pixel_coords: bool = False,
) -> dict[str, Any] | None:

    w, h = screen

    def x_coord(value: int) -> int | float:
        return int(value) if pixel_coords else _norm(value, w)

    def y_coord(value: int) -> int | float:
        return int(value) if pixel_coords else _norm(value, h)

    if c.kind == "click":
        if c.x is None or c.y is None:
            return None
        return {"action_type": "click", "x": x_coord(c.x), "y": y_coord(c.y)}

    if c.kind == "double_click":
        if c.x is None or c.y is None:
            return None
        return {
            "action_type": "double_tap",
            "x": x_coord(c.x),
            "y": y_coord(c.y),
        }

    if c.kind == "right_click":
        if c.x is None or c.y is None:
            return None
        return {
            "action_type": "long_press",
            "x": x_coord(c.x),
            "y": y_coord(c.y),
        }

    if c.kind == "drag":
        if c.x is None or c.y is None or c.end_x is None or c.end_y is None:
            return None
        return {
            "action_type": "drag",
            "start_x": x_coord(c.x),
            "start_y": y_coord(c.y),
            "end_x": x_coord(c.end_x),
            "end_y": y_coord(c.end_y),
        }

    if c.kind == "type":
        if c.text is None:
            return None
        return {
            "action_type": "input_text" if pixel_coords else "type",
            "text": c.text,
        }

    if c.kind == "key" and c.key is not None:
        mapping = _KEY_TO_MOBILE.get(c.key.lower())
        if mapping is None:
            return None
        action = dict(mapping)
        if pixel_coords and action.get("action_type") == "type":
            action["action_type"] = "input_text"
        return action

    if c.kind == "hotkey":
        return None

    if c.kind == "scroll":
        direction = "down" if c.scroll_clicks > 0 else "up"
        action = {
            "action_type": "swipe",
            "direction": direction,
        }
        if c.x is not None and c.y is not None:
            action["x"] = x_coord(c.x)
            action["y"] = y_coord(c.y)
        return action

    if c.kind == "wait":
        return {"action_type": "wait"}

    return None


def to_web(
    c: Canonical, screen: tuple[int, int], *, extended: bool = False
) -> dict[str, Any] | None:

    del screen

    if extended and c.kind == "move" and c.x is not None and c.y is not None:
        return {"action_type": "move", "x": c.x, "y": c.y}

    if extended and c.kind in {"key", "hotkey"}:
        names = {
            "ctrl": "Control",
            "control": "Control",
            "alt": "Alt",
            "option": "Alt",
            "shift": "Shift",
            "cmd": "Meta",
            "command": "Meta",
            "win": "Meta",
            "super": "Meta",
            "superleft": "Meta",
            "superright": "Meta",
            "enter": "Enter",
            "return": "Enter",
            "escape": "Escape",
            "esc": "Escape",
            "tab": "Tab",
            "backspace": "Backspace",
            "delete": "Delete",
            "home": "Home",
            "end": "End",
            "up": "ArrowUp",
            "down": "ArrowDown",
            "left": "ArrowLeft",
            "right": "ArrowRight",
            "space": "Space",
            "pageup": "PageUp",
            "pagedown": "PageDown",
        }
        keys = c.keys or ((c.key,) if c.key else ())
        actions = [
            {
                "action_type": "key",
                "key": names.get(
                    key.lower(),
                    key.upper()
                    if key.lower().startswith("f") and key[1:].isdigit() and 1 <= int(key[1:]) <= 24
                    else key,
                ),
            }
            for key in keys
        ]
        if not actions:
            return None
        if c.kind == "hotkey":
            return {"action_type": "key", "key": "+".join(action["key"] for action in actions)}
        return (
            actions[0]
            if len(actions) == 1
            else {"action_type": "action_sequence", "actions": actions}
        )

    if c.kind == "click":
        if c.x is None or c.y is None:
            return None
        return {"action_type": "click", "x": int(c.x), "y": int(c.y)}

    if c.kind == "double_click":
        if c.x is None or c.y is None:
            return None
        return {"action_type": "double_tap", "x": int(c.x), "y": int(c.y)}

    if c.kind == "right_click":
        if c.x is None or c.y is None:
            return None
        return {"action_type": "long_press", "x": int(c.x), "y": int(c.y)}

    if c.kind == "drag":
        if c.x is None or c.y is None or c.end_x is None or c.end_y is None:
            return None
        return {
            "action_type": "drag",
            "start_x": int(c.x),
            "start_y": int(c.y),
            "end_x": int(c.end_x),
            "end_y": int(c.end_y),
        }

    if c.kind == "type":
        if c.text is None:
            return None
        return {"action_type": "type", "text": c.text}

    if c.kind == "key" and c.key is not None:
        mapping = _KEY_TO_MOBILE.get(c.key.lower())
        return dict(mapping) if mapping is not None else None

    if c.kind == "hotkey":
        return None

    if c.kind == "scroll":
        direction = (
            ("up" if c.scroll_clicks > 0 else "down")
            if extended
            else ("down" if c.scroll_clicks > 0 else "up")
        )
        action: dict[str, Any] = {
            "action_type": "scroll",
            "direction": direction,
        }
        if c.x is not None and c.y is not None:
            action["x"] = int(c.x)
            action["y"] = int(c.y)
        if extended:
            action["pixels"] = abs(c.scroll_clicks) * 100
        return action

    if c.kind == "wait":
        return {"action_type": "wait"}

    return None


__all__ = ["to_mobile", "to_web"]
