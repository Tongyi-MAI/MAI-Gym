import json
from typing import Any


def action_text(action: dict[str, Any]) -> str:
    kind = action["action_type"]
    if kind == "move":
        return f"HOVER coords({action['x']}, {action['y']}) -> Move pointer over the location | SUCCESS"
    if kind in {"click", "double_tap", "long_press"}:
        target = f"coords({action['x']}, {action['y']})"
        description = {
            "click": "primary click",
            "double_tap": "double click",
            "long_press": "right click",
        }[kind]
        return f"{target} -> CLICK -> {description} | SUCCESS"
    if kind in {"type", "input_text"}:
        return f":focus -> TYPE -> {json.dumps(action['text'], ensure_ascii=False)} | SUCCESS"
    if kind in {"key", "keyboard_enter"}:
        return f"page -> PRESS_KEY -> {action.get('key', 'Enter')} | SUCCESS"
    if kind == "scroll":
        return f"page -> SCROLL -> {action.get('direction', 'down')}, {action.get('pixels', 600)} pixels | SUCCESS"
    if kind in {"open_url", "goto"}:
        return "page -> NAVIGATE -> Direct URL navigation | SUCCESS"
    if kind == "navigate_back":
        return "page -> GO_BACK -> Browser back | SUCCESS"
    if kind == "wait":
        return f"page -> WAIT -> {action.get('duration', 1)} seconds | SUCCESS"
    if kind == "drag":
        return f"coords({action['start_x']}, {action['start_y']}) -> DRAG -> coords({action['end_x']}, {action['end_y']}) | SUCCESS"
    raise ValueError(f"Action has no native OnlineMind2Web export: {kind}")
