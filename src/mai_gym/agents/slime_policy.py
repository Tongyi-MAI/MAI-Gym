from __future__ import annotations

import asyncio
import base64
import json
import logging
from dataclasses import dataclass, field
from io import BytesIO
from typing import Any

from PIL import Image

from mai_gym.io import oss_io
from mai_gym.utils.images import to_data_url
from mai_gym.utils.messages import elide_images, hide_history_images

from ._types import ChatModel, GenerateResult
from .image_resize import ImageResizer, ResizeScreenshotMixin

logger = logging.getLogger(__name__)


class MessagesLogger:
    def __init__(self, log_dir: Any = None, oss_creds: Any = None) -> None:

        self.log_dir = log_dir
        self.oss_creds = oss_creds

    @property
    def enabled(self) -> bool:
        return self.log_dir is not None

    def ensure_dir(self) -> None:

        if not self.enabled:
            return
        try:
            p = str(self.log_dir)
            if not oss_io.is_oss_path(p):
                oss_io.sync_makedirs(p)
        except Exception as exc:
            logger.warning("[MessagesLogger] failed to create log dir: %s", exc)

    def write_step(self, step_idx: int, payload: dict[str, Any]) -> None:

        if not self.enabled:
            return
        path = oss_io.join(str(self.log_dir), f"step_{step_idx:04d}.json")
        try:
            fs = (
                oss_io.get_sync_fs(self.oss_creds)
                if oss_io.is_oss_path(path) and self.oss_creds
                else None
            )
            oss_io.sync_write_text(
                path,
                json.dumps(payload, ensure_ascii=False, indent=2),
                fs=fs,
            )
        except Exception as exc:
            logger.warning("[MessagesLogger] failed to write %s: %s", path, exc)


def _decode_png_size(b64_png: str) -> tuple[int, int] | None:

    if not b64_png:
        return None
    try:
        raw = base64.b64decode(b64_png)
        with Image.open(BytesIO(raw)) as img:
            return img.size
    except Exception as exc:
        logger.warning("[SLIMEPolicy] failed to decode screenshot size: %s", exc)
        return None


def _resize_screenshot(
    b64_png: str,
    target_width: int,
    target_height: int,
) -> str:

    raw = base64.b64decode(b64_png)
    with Image.open(BytesIO(raw)) as img:
        resized = img.resize((target_width, target_height))
        buf = BytesIO()
        resized.save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode("utf-8")


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


class SLIMEPolicy(ResizeScreenshotMixin):
    def __init__(
        self,
        chat_model: ChatModel,
        sampling_params: dict[str, Any],
        *,
        tools: list[dict] | None = None,
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
        messages_log_dir: Any = None,
        remote_log_callback: Any = None,
        oss_creds: Any = None,
    ) -> None:
        self.chat_model = chat_model
        self.sampling_params = sampling_params
        self.tools = tools

        self.original_image_width = original_image_width
        self.original_image_height = original_image_height

        self._resizer = ImageResizer.from_config(
            resize_width=resize_width,
            resize_height=resize_height,
            resize_longest_edge=resize_longest_edge,
            max_pixels=max_pixels,
            min_pixels=min_pixels,
            image_factor=image_factor,
        )
        self.system_prompt = system_prompt
        self.image_retention = image_retention

        self._msg_log = MessagesLogger(messages_log_dir, oss_creds)

        self.remote_log_callback = remote_log_callback

        self.timing: Any = None
        self.step_records: list[StepRecord] = []
        self._messages: list[dict[str, Any]] = []
        self._instruction: str = ""
        self._initialized = False
        self.last_prediction: str | None = None

    def initialize(self, instruction: str) -> None:

        self.step_records.clear()
        self.chat_model.reset()
        self._messages = []
        self._instruction = instruction or ""
        if self.system_prompt:
            self._messages.append({"role": "system", "content": self.system_prompt})
        if self._instruction:
            self._messages.append(
                {
                    "role": "user",
                    "content": [{"type": "text", "text": self._instruction}],
                }
            )
        self._initialized = True
        self._msg_log.ensure_dir()

    def done(self) -> None:
        pass

    async def select_action(self, observation: dict[str, Any]) -> dict[str, Any]:

        screenshot_b64 = observation.get("image", "")

        ask_user_response = observation.get("ask_user_response")
        tool_call_result = observation.get("tool_call")

        if screenshot_b64 and (
            self.original_image_width is None or self.original_image_height is None
        ):
            size = _decode_png_size(screenshot_b64)
            if size is not None:
                self.original_image_width, self.original_image_height = size
                logger.info(
                    "[SLIMEPolicy] captured screenshot size: %dx%d",
                    *size,
                )

        screenshot_b64 = await self._resize_screenshot_for_model(screenshot_b64)

        if tool_call_result is not None:
            user_msg = {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": f"Tool call result: {tool_call_result}",
                    }
                ],
            }
        elif ask_user_response is not None:
            user_msg = {
                "role": "user",
                "content": [{"type": "text", "text": ask_user_response}],
            }
        elif screenshot_b64:
            user_msg = {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": to_data_url(screenshot_b64),
                    }
                ],
            }
        else:
            raise ValueError("No observation provided")

        self._messages.append(user_msg)

        pruned = hide_history_images(self._messages, self.image_retention)

        result: GenerateResult = await self.chat_model.generate(
            pruned,
            self.sampling_params,
            tools=self.tools if len(self.step_records) == 0 else None,
            original_image_width=self.original_image_width,
            original_image_height=self.original_image_height,
            timing=self.timing,
        )

        info = result.info
        action_dict = info.get("action_dict", {})
        is_done = info.get("is_done", False)
        format_reward = info.get("format_reward", 0.0)

        logging_enabled = self._msg_log.enabled or self.remote_log_callback is not None

        rendered = elide_images(pruned) if logging_enabled else []

        step_record = StepRecord(
            response_token_ids=result.output_ids,
            response_logprobs=result.output_logprobs,
            response_text=result.content,
            observation_screenshot=screenshot_b64,
            action_dict=action_dict,
            format_reward=format_reward,
            is_terminal=is_done,
            rendered_messages=rendered,
            finish_reason=result.finish_reason,
        )
        self.step_records.append(step_record)

        self.last_prediction = result.content

        if logging_enabled:
            self._write_step_log(len(self.step_records) - 1, step_record)

        self._messages.append({"role": "assistant", "content": result.content})

        return action_dict

    def _write_step_log(self, step_idx: int, record: StepRecord) -> None:

        payload = {
            "step": step_idx,
            "system_prompt": self.system_prompt,
            "instruction": self._instruction,
            "rendered_messages": record.rendered_messages,
            "raw_response_text": record.response_text,
            "parsed_action_dict": record.action_dict,
            "format_reward": record.format_reward,
            "is_done": record.is_terminal,
            "finish_reason": record.finish_reason,
        }
        self._msg_log.write_step(step_idx, payload)
        maybe_async_log(self.remote_log_callback, step_idx, payload)


def _safe_remote_log(cb: Any, step_idx: int, payload: dict[str, Any]) -> None:
    try:
        cb(step_idx, payload)
    except Exception as exc:
        logger.warning("[SLIMEPolicy] remote log callback failed: %s", exc)


_BG_LOG_TASKS: set[asyncio.Task[Any]] = set()


def maybe_async_log(callback: Any, step_idx: int, payload: dict[str, Any]) -> None:

    if callback is None:
        return
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        _safe_remote_log(callback, step_idx, payload)
        return
    task = loop.create_task(asyncio.to_thread(_safe_remote_log, callback, step_idx, payload))
    _BG_LOG_TASKS.add(task)
    task.add_done_callback(_BG_LOG_TASKS.discard)
