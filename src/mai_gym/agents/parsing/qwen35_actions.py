from __future__ import annotations

import contextlib
import json
import re
from typing import Any

from mai_gym.agents.action_types import (
    ANSWER,
    ASK_USER,
    CLICK,
    DRAG,
    FINISHED,
    INPUT_TEXT,
    KEYBOARD_ENTER,
    LONG_PRESS,
    NAVIGATE_BACK,
    NAVIGATE_HOME,
    OPEN_APP,
    UNKNOWN,
    WAIT,
)

SCALE_FACTOR = 1000


def parse_qwen35_response(text: str) -> dict[str, Any]:

    text = text.strip()

    action_text = None
    action_match = re.search(r"Action:\s*(.+?)(?:\n|<tool_call>)", text, re.DOTALL)
    if action_match:
        action_text = action_match.group(1).strip()

    tool_call_match = re.search(
        r"<tool_call>\s*<function=(\w+)>(.*?)</function>\s*</tool_call>",
        text,
        re.DOTALL,
    )
    if not tool_call_match:
        raise ValueError("No <tool_call> block found in response")

    function_name = tool_call_match.group(1)
    params_text = tool_call_match.group(2)

    params: dict[str, Any] = {}
    for param_match in re.finditer(r"<parameter=(\w+)>(.*?)</parameter>", params_text, re.DOTALL):
        param_name = param_match.group(1)
        param_value = param_match.group(2).strip()
        with contextlib.suppress(json.JSONDecodeError, ValueError):
            param_value = json.loads(param_value)
        params[param_name] = param_value

    return {
        "action_text": action_text,
        "function_name": function_name,
        "parameters": params,
    }


def _normalize_coordinate(coord: list) -> list[float]:

    if len(coord) == 2:
        return [coord[0] / SCALE_FACTOR, coord[1] / SCALE_FACTOR]
    if len(coord) == 4:
        x1, y1, x2, y2 = coord
        return [(x1 + x2) / 2 / SCALE_FACTOR, (y1 + y2) / 2 / SCALE_FACTOR]
    raise ValueError(f"Expected 2 or 4 coordinate values, got {len(coord)}")


def _coord_to_pixel(coord: list[float], image_width: int, image_height: int) -> tuple[int, int]:
    norm = _normalize_coordinate(coord)
    return round(norm[0] * image_width), round(norm[1] * image_height)


def parsed_response_to_env_action(
    parsed: dict[str, Any], image_height: int, image_width: int
) -> dict:

    params = parsed["parameters"]
    action = params.get("action", "")

    if action in ("click", "long_press"):
        x, y = _coord_to_pixel(params["coordinate"], image_width, image_height)
        action_type = CLICK if action == "click" else LONG_PRESS
        return {"action_type": action_type, "x": x, "y": y}

    if action == "swipe":
        sx, sy = _coord_to_pixel(params["coordinate"], image_width, image_height)
        ex, ey = _coord_to_pixel(params["coordinate2"], image_width, image_height)
        return {
            "action_type": DRAG,
            "start_x": sx,
            "start_y": sy,
            "end_x": ex,
            "end_y": ey,
        }

    if action == "type":
        return {"action_type": INPUT_TEXT, "text": params.get("text", "")}

    if action in ("open", "open_app"):
        return {"action_type": OPEN_APP, "app_name": params.get("text", "")}

    if action == "system_button":
        button = params.get("button", "")
        button_map = {
            "Back": NAVIGATE_BACK,
            "Home": NAVIGATE_HOME,
            "Enter": KEYBOARD_ENTER,
        }
        if button in button_map:
            return {"action_type": button_map[button]}
        return {"action_type": UNKNOWN, "text": f"system_button:{button}"}

    if action == "key":
        key_text = params.get("text", "")
        if key_text.lower() == "enter":
            return {"action_type": KEYBOARD_ENTER}
        return {"action_type": UNKNOWN, "text": f"key:{key_text}"}

    if action == "wait":
        return {"action_type": WAIT}

    if action == "terminate":
        return {"action_type": FINISHED, "text": params.get("status", "success")}

    if action == "answer":
        return {"action_type": ANSWER, "text": params.get("text", "")}

    if action == "ask_user":
        return {"action_type": ASK_USER, "text": params.get("text", "")}

    raise ValueError(f"Unknown action: {action}")
