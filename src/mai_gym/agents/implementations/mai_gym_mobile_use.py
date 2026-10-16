from __future__ import annotations

import base64
import inspect
import json
import os
import struct
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from mai_gym.agents.backends.http_chat_model import HttpChatModel
from mai_gym.agents.parsing.mai_gym_mobile_use import (
    ActionParseResult,
    parse_action_response,
)

MAX_ACTION_PARSE_RETRIES = 3
HISTORY_IMAGE_PLACEHOLDER = "<image_collapsed>"

APP_LISTS: dict[str, list[str]] = {
    "mw": [
        "桌面",
        "Contacts",
        "Settings",
        "设置",
        "Clock",
        "Maps",
        "Chrome",
        "Calendar",
        "files",
        "Gallery",
        "淘店",
        "Taodian",
        "Mattermost",
        "Mastodon",
        "Mail",
        "SMS",
        "Camera",
    ],
    "my": [
        "alipay",
        "bilibili",
        "ebay",
        "map",
        "railway12306",
        "redbook",
        "reddit",
        "spotify",
        "tencent_meeting",
        "wechat",
        "wechat_reading",
        "x",
        "answer_sheet",
        "browser",
        "calculator",
        "calculator2",
        "calendar",
        "clock",
        "compass",
        "contacts",
        "file_manager",
        "gallery",
        "notes",
        "phone",
        "settings",
        "sms",
        "theme_store",
        "weather",
    ],
}


def render_prompt_template(
    template_name: str,
    *,
    apps: list[str],
    tools: str | list[dict[str, Any]] | None = None,
) -> str:
    """Render the small checked-in prompt without requiring Jinja."""

    template_path = Path(__file__).resolve().parents[1] / "prompts" / template_name
    template = template_path.read_text(encoding="utf-8")
    template = template.replace("{% raw %}", "").replace("{% endraw %}", "")

    block_start = "{% if tools -%}"
    block_end = "{% endif -%}"
    if block_start in template:
        before, remainder = template.split(block_start, 1)
        tools_block, after = remainder.split(block_end, 1)
        if tools:
            rendered_tools = tools if isinstance(tools, str) else json.dumps(tools, ensure_ascii=False)
            template = before + tools_block + after
        else:
            rendered_tools = ""
            template = before + after
    else:
        rendered_tools = ""

    return (
        template.replace("{{ apps }}", json.dumps(apps, ensure_ascii=False))
        .replace("{{ tools }}", rendered_tools)
        .strip()
    )


def render_system_prompt(
    platform: str | None = None,
    *,
    apps: list[str] | None = None,
    tools: str | list[dict[str, Any]] | None = None,
) -> str:
    platform = platform or os.environ.get("ENV_PLATFORM", "mw")
    selected_apps = apps if apps is not None else APP_LISTS.get(platform, APP_LISTS["mw"])
    return render_prompt_template(
        "mai_gym_mobile_use_system.j2",
        apps=selected_apps,
        tools=tools,
    )


SYSTEM_PROMPT = render_system_prompt()


@dataclass
class StepRecord:
    response_token_ids: list[int]
    response_logprobs: list[float]
    response_text: str
    observation_screenshot: str
    action_dict: dict[str, Any]
    format_reward: float
    is_terminal: bool
    rendered_messages: list[dict[str, Any]] = field(default_factory=list)
    finish_reason: str = ""
    action_text: str = ""
    prompt_token_ids: list[int] = field(default_factory=list)
    multimodal_train_inputs: dict[str, Any] | None = None
    cached_tokens: int = 0
    prompt_tokens: int = 0
    gen_epoch: int | None = None


@dataclass(frozen=True)
class ImageTransformConfig:
    mode: str
    resize_width: int | None = None
    resize_height: int | None = None
    resize_longest_edge: int | None = None
    max_pixels: int | None = None
    min_pixels: int | None = None
    factor: int | None = None


class _NoOpMessagesLogger:
    enabled = False

    def ensure_dir(self) -> None:
        return None

    def write_step(self, _step_idx: int, _payload: dict[str, Any]) -> None:
        return None


def _is_image_turn(message: dict[str, Any]) -> bool:
    if message.get("role") != "user":
        return False
    content = message.get("content")
    return isinstance(content, list) and any(
        isinstance(part, dict) and part.get("type") == "image_url" for part in content
    )


def keep_recent_images(
    messages: list[dict[str, Any]],
    retention: int,
) -> list[dict[str, Any]]:
    if retention < 0:
        raise ValueError("image retention must be non-negative")
    total = sum(_is_image_turn(message) for message in messages)
    collapse_count = max(0, total - retention)
    image_ordinal = 0
    out: list[dict[str, Any]] = []
    for message in messages:
        if _is_image_turn(message):
            if image_ordinal < collapse_count:
                out.append({"role": "user", "content": HISTORY_IMAGE_PLACEHOLDER})
            else:
                out.append(message)
            image_ordinal += 1
        else:
            out.append(message)
    return out


def decode_png_size(image_b64: str) -> tuple[int, int] | None:
    try:
        raw = base64.b64decode(image_b64, validate=True)
    except (ValueError, TypeError):
        return None
    if len(raw) < 24 or raw[:8] != b"\x89PNG\r\n\x1a\n" or raw[12:16] != b"IHDR":
        return None
    return struct.unpack(">II", raw[16:24])


class MAIGymMobileUsePolicy:
    """Deployment-independent mobile-use agent.

    The model transport is injected through ``chat_model``. This module owns only
    the agent loop and prompt/history state; action conversion lives in
    ``mai_gym.agents.parsing.mai_gym_mobile_use``.
    """

    NATIVE_ACTION_SPACE = "mobile_use"

    def __init__(
        self,
        chat_model: HttpChatModel,
        sampling_params: dict[str, Any],
        *,
        tools: list[dict[str, Any]] | None = None,
        original_image_width: int | None = None,
        original_image_height: int | None = None,
        resize_width: int | None = None,
        resize_height: int | None = None,
        resize_longest_edge: int | None = None,
        max_pixels: int | None = None,
        min_pixels: int | None = None,
        image_factor: int | None = None,
        system_prompt: str | None = None,
        image_retention: int = 3,
        max_parse_retries: int = MAX_ACTION_PARSE_RETRIES,
        image_transform: Any | None = None,
        messages_log_dir: Any = None,
        remote_log_callback: Any = None,
        oss_creds: Any = None,
        epoch_getter: Any = None,
        **_: Any,
    ) -> None:
        del messages_log_dir, oss_creds
        self.chat_model = chat_model
        self.sampling_params = sampling_params
        self.tools = tools
        self.original_image_width = original_image_width
        self.original_image_height = original_image_height
        self.system_prompt = SYSTEM_PROMPT if system_prompt is None else system_prompt
        self.image_retention = image_retention
        self.max_parse_retries = max_parse_retries
        self.image_transform = image_transform
        self.remote_log_callback = remote_log_callback
        self._epoch_getter = epoch_getter

        if max_pixels is not None or min_pixels is not None or resize_longest_edge is not None:
            resize_mode = "smart_resize"
        elif resize_width is not None or resize_height is not None:
            resize_mode = "fixed"
        else:
            resize_mode = "none"
        self._resizer = ImageTransformConfig(
            mode=resize_mode,
            resize_width=resize_width,
            resize_height=resize_height,
            resize_longest_edge=resize_longest_edge,
            max_pixels=max_pixels,
            min_pixels=min_pixels,
            factor=image_factor,
        )
        self._msg_log = _NoOpMessagesLogger()
        self.timing: Any = None
        self.step_records: list[StepRecord] = []
        self._instruction = ""
        self._messages: list[dict[str, Any]] = []
        self.last_prediction: str | None = None

    def initialize(self, instruction: str) -> None:
        if instruction is None:
            raise ValueError("instruction must not be None")
        self.step_records.clear()
        self.chat_model.reset()
        self._instruction = instruction
        self._messages = []
        if self.system_prompt:
            self._messages.append({"role": "system", "content": self.system_prompt})
        if instruction:
            self._messages.append(
                {"role": "user", "content": [{"type": "text", "text": instruction}]}
            )
        self._msg_log.ensure_dir()

    def done(self) -> None:
        return None

    def _prune_messages(self) -> list[dict[str, Any]]:
        return keep_recent_images(self._messages, self.image_retention)

    async def _resize_screenshot_for_model(self, image_b64: str) -> str:
        if self.image_transform is None:
            return image_b64
        transformed = self.image_transform(image_b64)
        if inspect.isawaitable(transformed):
            transformed = await transformed
        if not isinstance(transformed, str):
            raise TypeError("image_transform must return a base64 string")
        return transformed

    async def select_action(self, observation: dict[str, Any]) -> dict[str, Any]:
        tool_call_result = observation.get("tool_call")
        ask_user_response = observation.get("ask_user_response")
        if tool_call_result is not None:
            self._messages.append(
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "Tool call result: "
                            + json.dumps(tool_call_result, ensure_ascii=False),
                        }
                    ],
                }
            )
        elif ask_user_response is not None:
            self._messages.append(
                {
                    "role": "user",
                    "content": [{"type": "text", "text": str(ask_user_response)}],
                }
            )

        screenshot_b64 = observation.get("image", "") or ""
        if not screenshot_b64:
            if self.step_records:
                screenshot_b64 = self.step_records[-1].observation_screenshot
            else:
                return {"action_type": "wait"}

        if self.original_image_width is None or self.original_image_height is None:
            size = decode_png_size(screenshot_b64)
            if size is not None:
                self.original_image_width, self.original_image_height = size
        screenshot_b64 = await self._resize_screenshot_for_model(screenshot_b64)

        if tool_call_result is None and ask_user_response is None:
            self._messages.append(
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": f"data:image/png;base64,{screenshot_b64}",
                        }
                    ],
                }
            )

        pruned = self._prune_messages()
        result: Any = None
        parsed: ActionParseResult | None = None
        for _attempt in range(self.max_parse_retries + 1):
            result = await self.chat_model.generate(
                pruned,
                self.sampling_params,
                tools=self.tools if not self.step_records else None,
                original_image_width=self.original_image_width,
                original_image_height=self.original_image_height,
                timing=self.timing,
            )
            parsed = parse_action_response(
                str(getattr(result, "content", "")),
                original_image_width=self.original_image_width,
                original_image_height=self.original_image_height,
            )
            if parsed.is_valid:
                break

        assert result is not None and parsed is not None
        if parsed.is_valid:
            action_dict = parsed.action_dict
            is_done = parsed.is_done
            format_reward = parsed.format_reward
        else:
            action_dict = {"action_type": "terminate", "status": "failure"}
            is_done = True
            format_reward = 0.0

        content = str(getattr(result, "content", ""))
        record = StepRecord(
            response_token_ids=list(getattr(result, "output_ids", [])),
            response_logprobs=list(getattr(result, "output_logprobs", [])),
            response_text=content,
            observation_screenshot=screenshot_b64,
            action_dict=action_dict,
            format_reward=format_reward,
            is_terminal=is_done,
            finish_reason=str(getattr(result, "finish_reason", "")),
            prompt_token_ids=list(getattr(result, "prompt_token_ids", [])),
            multimodal_train_inputs=getattr(result, "multimodal_train_inputs", None),
            cached_tokens=int(getattr(result, "cached_tokens", 0)),
            prompt_tokens=int(getattr(result, "prompt_tokens", 0)),
            gen_epoch=(self._epoch_getter() if self._epoch_getter else None),
        )
        self.step_records.append(record)
        self.last_prediction = content
        if parsed.is_valid:
            self._messages.append(
                {"role": "assistant", "content": content, "_output_ids": record.response_token_ids}
            )
        return action_dict


__all__ = [
    "APP_LISTS",
    "HISTORY_IMAGE_PLACEHOLDER",
    "HttpChatModel",
    "MAX_ACTION_PARSE_RETRIES",
    "MAIGymMobileUsePolicy",
    "SYSTEM_PROMPT",
    "StepRecord",
    "decode_png_size",
    "keep_recent_images",
    "render_prompt_template",
    "render_system_prompt",
]
