from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

_THINK_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL)
_TOOL_CALL_RE = re.compile(r"<tool_call>(.*?)</tool_call>", re.DOTALL)

_VALID_ACTIONS = frozenset(
    {
        "click",
        "double_click",
        "long_press",
        "type",
        "open",
        "drag",
        "system_button",
        "wait",
        "ask_user",
        "terminate",
        "answer",
    }
)
_TERMINAL_ACTIONS = frozenset({"terminate", "answer"})


@dataclass(frozen=True)
class ActionParseResult:
    raw_text: str
    action_dict: dict[str, Any]
    format_reward: float
    is_done: bool
    error: str | None = None

    @property
    def is_valid(self) -> bool:
        action_type = self.action_dict.get("action_type")
        return self.error is None and action_type not in (None, "", "unknown")


def _format_reward(text: str) -> float:
    try:
        has_think = bool(_THINK_RE.search(text)) or "</think>" in text
        tool_match = _TOOL_CALL_RE.search(text)
        if not (has_think and tool_match):
            return 0.0
        body = json.loads(tool_match.group(1).strip())
        arguments = body.get("arguments", {}) if isinstance(body, dict) else {}
        action_type = arguments.get("action_type") if isinstance(arguments, dict) else None
        return 1.0 if isinstance(action_type, str) and action_type in _VALID_ACTIONS else 0.0
    except (json.JSONDecodeError, TypeError, AttributeError):
        return 0.0


def _scaled_point(
    coordinate: Any,
    width: int | None,
    height: int | None,
) -> tuple[int, int]:
    if width is None or height is None:
        raise ValueError("coordinate actions require original image width and height")
    if not isinstance(coordinate, (list, tuple)) or len(coordinate) != 2:
        raise ValueError(f"invalid coordinate: {coordinate!r}")
    return int(float(coordinate[0]) / 999 * width), int(float(coordinate[1]) / 999 * height)


def convert_action(
    arguments: dict[str, Any],
    width: int | None,
    height: int | None,
) -> dict[str, Any]:
    action = arguments.get("action_type", arguments.get("action", ""))

    if action in ("click", "long_press", "double_click"):
        x, y = _scaled_point(arguments.get("coordinate", [0, 0]), width, height)
        return {
            "action_type": "double_tap" if action == "double_click" else action,
            "x": x,
            "y": y,
        }
    if action == "drag":
        start_x, start_y = _scaled_point(arguments.get("start_coordinate", [0, 0]), width, height)
        end_x, end_y = _scaled_point(arguments.get("end_coordinate", [0, 0]), width, height)
        return {
            "action_type": "drag",
            "start_x": start_x,
            "start_y": start_y,
            "end_x": end_x,
            "end_y": end_y,
        }
    if action == "type":
        return {"action_type": "input_text", "text": arguments.get("text", "")}
    if action == "open":
        return {"action_type": "open_app", "app_name": arguments.get("text", "")}
    if action == "system_button":
        button_map = {
            "back": "navigate_back",
            "home": "navigate_home",
            "enter": "keyboard_enter",
            "menu": "navigate_home",
        }
        button = str(arguments.get("button", "")).lower()
        return {"action_type": button_map.get(button, "unknown")}
    if action == "terminate":
        answer = arguments.get("answer")
        if answer is not None:
            return {"action_type": "answer", "text": str(answer)}
        return {"action_type": "terminate", "status": arguments.get("status", "success")}
    if action == "answer":
        answer = arguments.get("answer")
        if answer is None:
            answer = arguments.get("text", "")
        return {
            "action_type": "answer",
            "text": str(answer),
        }
    if action == "wait":
        result: dict[str, Any] = {"action_type": "wait"}
        if "waiting_time" in arguments:
            result["waiting_time"] = arguments["waiting_time"]
        return result
    if action == "ask_user":
        return {"action_type": "ask_user", "text": arguments.get("text", "")}
    return {"action_type": "unknown", "text": f"Unknown action: {action}"}


def parse_action_response(
    text: str,
    *,
    original_image_width: int | None = None,
    original_image_height: int | None = None,
) -> ActionParseResult:
    """Parse one MAI-UI response into a deployment-agnostic action dictionary."""

    try:
        tool_match = _TOOL_CALL_RE.search(text.strip())
        if tool_match is None:
            raise ValueError("No <tool_call> block found")
        body = json.loads(tool_match.group(1).strip())
        if not isinstance(body, dict):
            raise ValueError("tool call must be a JSON object")
        arguments = body.get("arguments", {})
        if not isinstance(arguments, dict):
            raise ValueError("tool call arguments must be a JSON object")
        action_dict = convert_action(
            arguments,
            original_image_width,
            original_image_height,
        )
        action_type = action_dict.get("action_type")
        if action_type == "unknown":
            raise ValueError(str(action_dict.get("text", "unknown action")))
        return ActionParseResult(
            raw_text=text,
            action_dict=action_dict,
            format_reward=_format_reward(text),
            is_done=action_type in _TERMINAL_ACTIONS,
        )
    except (ValueError, TypeError, KeyError, IndexError, json.JSONDecodeError) as exc:
        return ActionParseResult(
            raw_text=text,
            action_dict={"action_type": "unknown", "error": str(exc)},
            format_reward=0.0,
            is_done=True,
            error=str(exc),
        )


class MAIGymMobileUseActionParser:
    """Small object interface for deployments that prefer an injectable parser."""

    def parse(
        self,
        text: str,
        *,
        original_image_width: int | None = None,
        original_image_height: int | None = None,
    ) -> ActionParseResult:
        return parse_action_response(
            text,
            original_image_width=original_image_width,
            original_image_height=original_image_height,
        )


__all__ = [
    "ActionParseResult",
    "MAIGymMobileUseActionParser",
    "convert_action",
    "parse_action_response",
]
