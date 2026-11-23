from __future__ import annotations

import asyncio
import json
import re
from abc import ABC, abstractmethod
from typing import Any

from mai_gym.agents.action_types import (
    ANSWER as MW_ANSWER,
)
from mai_gym.agents.action_types import (
    ASK_USER as MW_ASK_USER,
)
from mai_gym.agents.action_types import (
    CLICK as MW_CLICK,
)
from mai_gym.agents.action_types import (
    DOUBLE_TAP as MW_DOUBLE_TAP,
)
from mai_gym.agents.action_types import (
    DRAG as MW_DRAG,
)
from mai_gym.agents.action_types import (
    FINISHED as MW_FINISHED,
)
from mai_gym.agents.action_types import (
    INPUT_TEXT as MW_INPUT_TEXT,
)
from mai_gym.agents.action_types import (
    KEYBOARD_ENTER as MW_KEYBOARD_ENTER,
)
from mai_gym.agents.action_types import (
    LONG_PRESS as MW_LONG_PRESS,
)
from mai_gym.agents.action_types import (
    NAVIGATE_BACK as MW_NAVIGATE_BACK,
)
from mai_gym.agents.action_types import (
    NAVIGATE_HOME as MW_NAVIGATE_HOME,
)
from mai_gym.agents.action_types import (
    OPEN_APP as MW_OPEN_APP,
)
from mai_gym.agents.action_types import (
    SCROLL as MW_SCROLL,
)
from mai_gym.agents.action_types import (
    UNKNOWN as MW_UNKNOWN,
)
from mai_gym.agents.action_types import (
    WAIT as MW_WAIT,
)
from mai_gym.agents.parsing.mai_ui_tags import parse_tagged_text
from mai_gym.agents.parsing.mw_action_translation import args_to_action_dict

from ._types import FunctionCall

__all__ = [
    "FunctionCall",
    "MAIUIToolParser",
    "MW_ANSWER",
    "MW_ASK_USER",
    "MW_CLICK",
    "MW_DOUBLE_TAP",
    "MW_DRAG",
    "MW_FINISHED",
    "MW_INPUT_TEXT",
    "MW_KEYBOARD_ENTER",
    "MW_LONG_PRESS",
    "MW_NAVIGATE_BACK",
    "MW_NAVIGATE_HOME",
    "MW_OPEN_APP",
    "MW_SCROLL",
    "MW_UNKNOWN",
    "MW_WAIT",
    "MobileToolParser",
    "ToolParser",
    "parse_tagged_text",
]


CLICK = "click"
LONG_PRESS = "long_press"
SWIPE = "swipe"
DRAG = "drag"
INPUT_TEXT = "type"
OPEN_APP = "open_app"
NAVIGATE_BACK = "navigate_back"
NAVIGATE_HOME = "navigate_home"
KEYBOARD_ENTER = "keyboard_enter"
WAIT = "wait"
FINISHED = "status"
ANSWER = "answer"
UNKNOWN = "unknown"


class ToolParser(ABC):
    _registry: dict[str, type[ToolParser]] = {}

    def __init__(self, tokenizer: Any) -> None:
        self.tokenizer = tokenizer

    @abstractmethod
    async def extract_tool_calls(
        self,
        responses_ids: list[int],
    ) -> tuple[dict[str, Any], list[FunctionCall]]:
        raise NotImplementedError

    @classmethod
    def get_tool_parser(cls, name: str, tokenizer: Any, **kwargs: Any) -> ToolParser:
        if name not in cls._registry:
            raise ValueError(f"Unknown tool parser: {name}")
        kwargs = {k: v for k, v in kwargs.items() if v is not None}
        return cls._registry[name](tokenizer, **kwargs)

    @classmethod
    def register(cls, name: str):
        def decorator(subclass: type[ToolParser]) -> type[ToolParser]:
            cls._registry[name] = subclass
            return subclass

        return decorator


@ToolParser.register("mai_gym_mobile_use")
class MAIGymToolParser(ToolParser):
    def __init__(
        self,
        tokenizer: Any,
        original_image_width: int | None = None,
        original_image_height: int | None = None,
    ) -> None:
        super().__init__(tokenizer)
        self.width = original_image_width
        self.height = original_image_height

    async def extract_tool_calls(
        self, responses_ids: list[int] | str
    ) -> tuple[dict[str, Any], list[FunctionCall]]:
        from mai_gym.agents.parsing.mai_gym_computer_use import (
            parse_action_response as parse_computer,
        )
        from mai_gym.agents.parsing.mai_gym_mobile_use import (
            parse_action_response as parse_mobile,
        )

        text = await asyncio.to_thread(
            self.tokenizer.decode, responses_ids, skip_special_tokens=True
        )
        if isinstance(self, MAIGymComputerToolParser):
            parsed = parse_computer(text, screen_size=(self.width or 1920, self.height or 1080))
            tool_name = "computer_use"
        else:
            parsed = parse_mobile(
                text, original_image_width=self.width, original_image_height=self.height
            )
            tool_name = "mobile_use"
        info = {
            "raw_text": text,
            "text": text,
            "action_dict": parsed.action_dict,
            "format_reward": parsed.format_reward,
            "is_done": parsed.is_done,
        }
        calls = []
        if parsed.is_valid:
            calls.append(
                FunctionCall(
                    name=tool_name,
                    arguments=json.dumps(parsed.action_dict, ensure_ascii=False),
                )
            )
        return info, calls


@ToolParser.register("mai_gym_computer_use")
class MAIGymComputerToolParser(MAIGymToolParser):
    pass


@ToolParser.register("mobile")
class MobileToolParser(ToolParser):
    def __init__(
        self,
        tokenizer: Any,
        is_relative_coordinates: bool = True,
        original_image_height: int = 2400,
        original_image_width: int = 1080,
        curr_image_height: int = 2400,
        curr_image_width: int = 1080,
    ) -> None:
        super().__init__(tokenizer)
        if is_relative_coordinates:
            self.height_factor = original_image_height / 999
            self.width_factor = original_image_width / 999
        else:
            self.height_factor = original_image_height / curr_image_height
            self.width_factor = original_image_width / curr_image_width

    @staticmethod
    def _normalize_double_braces(text: str) -> str:

        def _fix(m: re.Match) -> str:
            raw = m.group(1)
            s = raw.strip()
            if not s.startswith("{{"):
                return m.group(0)
            for candidate in (s[1:-1], s[1:]):
                try:
                    json.loads(candidate)
                    return "<tool_call>" + raw.replace(s, candidate) + "</tool_call>"
                except (ValueError, json.JSONDecodeError):
                    continue
            return m.group(0)

        return re.sub(r"<tool_call>(.*?)</tool_call>", _fix, text, flags=re.DOTALL)

    async def extract_tool_calls(
        self,
        responses_ids: list[int],
    ) -> tuple[dict[str, Any], list[FunctionCall]]:
        loop = asyncio.get_running_loop()
        text = await loop.run_in_executor(
            None,
            lambda: self.tokenizer.decode(responses_ids, skip_special_tokens=True),
        )
        text = self._normalize_double_braces(text)

        info: dict[str, Any] = {
            "raw_text": text,
            "text": text,
            "format_reward": 0.0,
            "is_done": True,
            "action_dict": {},
        }

        try:
            results = parse_tagged_text(text)
        except Exception as e:
            info["action_dict"] = {"action_type": "invalid", "error": str(e)}
            return info, []

        info["text"] = text
        thinking = results.get("thinking")
        tool_call = results["tool_call"]
        if not thinking or not tool_call:
            info["action_dict"] = {"action_type": "invalid", "error": "Parse failed"}
            return info, []

        format_reward = self._calculate_format_reward(text)

        if not isinstance(tool_call, dict):
            info["action_dict"] = {"action_type": "invalid", "error": "Invalid tool_call"}
            return info, []

        arguments = tool_call.get("arguments", {})
        if not isinstance(arguments, dict):
            info["action_dict"] = {"action_type": "invalid", "error": "Invalid arguments"}
            return info, []

        try:
            action_dict = self._convert_to_action(arguments)
        except Exception as e:
            info["action_dict"] = {"action_type": "invalid", "error": str(e)}
            return info, []

        is_done = action_dict.get("action_type") in (FINISHED, ANSWER)
        arguments_str = json.dumps(action_dict, ensure_ascii=False, sort_keys=True)

        info["action_dict"] = action_dict
        info["format_reward"] = format_reward
        info["is_done"] = is_done
        return info, [FunctionCall(name="mobile_use", arguments=arguments_str)]

    @staticmethod
    def _calculate_format_reward(text: str) -> float:
        try:
            has_think = bool(re.search(r"<thinking>.*?</thinking>", text, re.DOTALL))
            has_tc = bool(re.search(r"<tool_call>.*?</tool_call>", text, re.DOTALL))
            if not (has_think and has_tc):
                return 0.0

            tc_match = re.search(r"<tool_call>(.*?)</tool_call>", text, re.DOTALL)
            if not tc_match:
                return 0.0
            data = json.loads(tc_match.group(1).strip())
            if data.get("name") != "mobile_use" or "arguments" not in data:
                return 0.0
            args = data["arguments"]
            if not isinstance(args, dict) or "action" not in args:
                return 0.0
            valid = {
                "click",
                "swipe",
                "type",
                "long_press",
                "system_button",
                "key",
                "open",
                "wait",
                "terminate",
                "answer",
                "drag",
            }
            return 1.0 if args["action"] in valid else 0.0
        except (json.JSONDecodeError, KeyError, IndexError):
            return 0.0

    def _convert_to_action(self, arguments: dict) -> dict:

        action = arguments.get("action", "")

        if action in ("click", "long_press"):
            coord = arguments.get("coordinate", [])
            if not coord or len(coord) < 2:
                return {"action_type": UNKNOWN, "text": f"Invalid coordinate for {action}"}
            x = int(float(coord[0]) * self.width_factor)
            y = int(float(coord[1]) * self.height_factor)
            at = CLICK if action == "click" else LONG_PRESS
            return {"action_type": at, "x": x, "y": y}

        if action == "swipe":
            direction = arguments.get("direction", "up")
            coord = arguments.get("coordinate")
            d: dict[str, Any] = {"action_type": SWIPE, "direction": direction}
            if coord and len(coord) >= 2:
                d["x"] = int(float(coord[0]) * self.width_factor)
                d["y"] = int(float(coord[1]) * self.height_factor)
            return d

        if action == "drag":
            sc = arguments.get("start_coordinate", [0, 0])
            ec = arguments.get("end_coordinate", [0, 0])
            return {
                "action_type": DRAG,
                "start_x": int(float(sc[0]) * self.width_factor),
                "start_y": int(float(sc[1]) * self.height_factor),
                "end_x": int(float(ec[0]) * self.width_factor),
                "end_y": int(float(ec[1]) * self.height_factor),
            }

        if action == "type":
            return {"action_type": INPUT_TEXT, "text": arguments.get("text", "")}

        if action == "open":
            return {"action_type": OPEN_APP, "app_name": arguments.get("text", "")}

        if action == "system_button":
            button = arguments.get("button", "").lower()
            button_map = {
                "back": NAVIGATE_BACK,
                "home": NAVIGATE_HOME,
                "menu": NAVIGATE_HOME,
                "enter": KEYBOARD_ENTER,
            }
            at = button_map.get(button)
            if at:
                return {"action_type": at}
            return {"action_type": UNKNOWN, "text": f"Unknown button: {button}"}

        if action == "terminate":
            result = {"action_type": "terminate", "status": arguments.get("status", "success")}
            answer = arguments.get("answer")
            if answer is not None:
                result["answer"] = str(answer)
            return result

        if action == "answer":
            return {"action_type": ANSWER, "text": str(arguments.get("text", ""))}

        if action == "wait":
            return {"action_type": WAIT}

        return {"action_type": UNKNOWN, "text": f"Unknown action: {action}"}


@ToolParser.register("mai_ui")
class MAIUIToolParser(ToolParser):
    def __init__(
        self,
        tokenizer: Any,
        original_image_width: int | None = None,
        original_image_height: int | None = None,
    ) -> None:
        super().__init__(tokenizer)
        self.original_image_width = original_image_width
        self.original_image_height = original_image_height

    async def extract_tool_calls(
        self,
        responses_ids: list[int],
    ) -> tuple[dict[str, Any], list[FunctionCall]]:
        loop = asyncio.get_running_loop()
        text = await loop.run_in_executor(
            None,
            lambda: self.tokenizer.decode(responses_ids, skip_special_tokens=True),
        )
        text = MobileToolParser._normalize_double_braces(text)

        info: dict[str, Any] = {
            "raw_text": text,
            "text": text,
            "format_reward": 0.0,
            "is_done": True,
            "action_dict": {},
        }

        try:
            results = parse_tagged_text(text)
        except Exception as e:
            info["action_dict"] = {"action_type": MW_UNKNOWN, "error": str(e)}
            return info, []

        thinking = results.get("thinking")
        tool_call = results.get("tool_call")

        info["text"] = text
        if not thinking or not tool_call or not isinstance(tool_call, dict):
            info["action_dict"] = {"action_type": MW_UNKNOWN, "error": "Parse failed"}
            return info, []

        arguments = tool_call.get("arguments", {})
        if not isinstance(arguments, dict):
            info["action_dict"] = {"action_type": MW_UNKNOWN, "error": "Invalid arguments"}
            return info, []

        format_reward = self._calculate_format_reward(text)
        try:
            action_dict = self._convert_to_action(arguments)
        except Exception as e:
            info["action_dict"] = {"action_type": MW_UNKNOWN, "error": str(e)}
            return info, []

        is_done = action_dict.get("action_type") in (MW_FINISHED, MW_ANSWER)
        arguments_str = json.dumps(action_dict, ensure_ascii=False, sort_keys=True)

        info["action_dict"] = action_dict
        info["format_reward"] = format_reward
        info["is_done"] = is_done
        return info, [FunctionCall(name="mobile_use", arguments=arguments_str)]

    def _convert_to_action(self, arguments: dict[str, Any]) -> dict[str, Any]:
        if self.original_image_width is None or self.original_image_height is None:
            raise ValueError("MAIUIToolParser needs original_image_width/height at call time")
        return args_to_action_dict(
            arguments,
            self.original_image_width,
            self.original_image_height,
        )

    @staticmethod
    def _calculate_format_reward(text: str) -> float:
        try:
            has_think = bool(re.search(r"<thinking>.*?</thinking>", text, re.DOTALL))
            has_tc = bool(re.search(r"<tool_call>.*?</tool_call>", text, re.DOTALL))
            if not (has_think and has_tc):
                return 0.0
            tc_match = re.search(r"<tool_call>(.*?)</tool_call>", text, re.DOTALL)
            if not tc_match:
                return 0.0
            data = json.loads(tc_match.group(1).strip())
            if data.get("name") != "mobile_use" or "arguments" not in data:
                return 0.0
            args = data["arguments"]
            if not isinstance(args, dict) or "action" not in args:
                return 0.0

            valid = {
                "click",
                "long_press",
                "double_click",
                "swipe",
                "drag",
                "type",
                "open",
                "system_button",
                "wait",
                "terminate",
                "answer",
                "ask_user",
            }
            return 1.0 if args["action"] in valid else 0.0
        except (json.JSONDecodeError, KeyError, IndexError):
            return 0.0
