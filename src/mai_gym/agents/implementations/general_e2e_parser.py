from __future__ import annotations

import asyncio
import contextlib
import json
import re
from typing import Any

from mai_gym.agents.action_types import (
    ANSWER as MW_ANSWER,
)
from mai_gym.agents.action_types import (
    FINISHED as MW_FINISHED,
)
from mai_gym.agents.action_types import (
    MCP as MW_MCP,
)
from mai_gym.agents.action_types import (
    UNKNOWN as MW_UNKNOWN,
)
from mai_gym.agents.parsing.qwen35_actions import (
    parse_qwen35_response,
    parsed_response_to_env_action,
)
from mai_gym.agents.tool_parser import FunctionCall, ToolParser

_VALID_ACTIONS = frozenset(
    {
        "click",
        "long_press",
        "swipe",
        "type",
        "open",
        "open_app",
        "system_button",
        "key",
        "wait",
        "terminate",
        "answer",
        "ask_user",
    }
)


@ToolParser.register("general_e2e")
@ToolParser.register("qwen35")
class GeneralE2EToolParser(ToolParser):
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

        info: dict[str, Any] = {
            "raw_text": text,
            "text": text,
            "format_reward": 0.0,
            "is_done": True,
            "action_dict": {},
        }

        try:
            parsed = parse_qwen35_response(text)
        except Exception as exc:
            info["action_dict"] = {"action_type": MW_UNKNOWN, "error": str(exc)}
            info["action_text"] = ""
            return info, []

        function_name = parsed.get("function_name", "") or ""
        params = parsed.get("parameters") or {}

        info["action_text"] = parsed.get("action_text") or ""

        format_reward = self._calculate_format_reward(text)

        try:
            action_dict = self._convert_to_action(function_name, params)
        except Exception as exc:
            info["action_dict"] = {"action_type": MW_UNKNOWN, "error": str(exc)}
            return info, []

        is_done = action_dict.get("action_type") in (MW_FINISHED, MW_ANSWER)
        arguments_str = json.dumps(action_dict, ensure_ascii=False, sort_keys=True)

        info["action_dict"] = action_dict
        info["format_reward"] = format_reward
        info["is_done"] = is_done
        return info, [FunctionCall(name=function_name or "mobile_use", arguments=arguments_str)]

    def _convert_to_action(
        self,
        function_name: str,
        params: dict[str, Any],
    ) -> dict[str, Any]:

        if function_name == "mobile_use":
            if self.original_image_width is None or self.original_image_height is None:
                raise ValueError(
                    "GeneralE2EToolParser needs original_image_width/height at call time"
                )
            return parsed_response_to_env_action(
                {"function_name": function_name, "parameters": params, "action_text": None},
                image_height=self.original_image_height,
                image_width=self.original_image_width,
            )
        return {
            "action_type": MW_MCP,
            "action_name": function_name,
            "action_json": params,
        }

    @staticmethod
    def _calculate_format_reward(text: str) -> float:

        try:
            if not re.search(r"Action:\s*", text):
                return 0.0
            tc_match = re.search(
                r"<tool_call>\s*<function=(\w+)>(.*?)</function>\s*</tool_call>",
                text,
                re.DOTALL,
            )
            if not tc_match:
                return 0.0
            params_text = tc_match.group(2)
            action_match = re.search(
                r"<parameter=action>(.*?)</parameter>",
                params_text,
                re.DOTALL,
            )
            if not action_match:
                return 0.0
            action_value = action_match.group(1).strip()
            with contextlib.suppress(json.JSONDecodeError, ValueError):
                action_value = json.loads(action_value)
            return 1.0 if action_value in _VALID_ACTIONS else 0.0
        except (json.JSONDecodeError, KeyError, IndexError):
            return 0.0
