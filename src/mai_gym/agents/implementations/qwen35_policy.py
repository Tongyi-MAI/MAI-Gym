from __future__ import annotations

import logging
from typing import Any

from mai_gym.agents._types import ChatModel, GenerateResult
from mai_gym.agents.action_space import ActionSpace
from mai_gym.agents.image_resize import ImageResizer, ResizeScreenshotMixin
from mai_gym.agents.slime_policy import (
    MessagesLogger,
    StepRecord,
    _decode_png_size,
    maybe_async_log,
)
from mai_gym.utils.images import to_data_url
from mai_gym.utils.messages import elide_images, hide_history_images

logger = logging.getLogger(__name__)


class Qwen35SLIMEPolicy(ResizeScreenshotMixin):
    NATIVE_ACTION_SPACE = ActionSpace.MOBILE_USE

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
        image_retention: int = 5,
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

        self.step_records: list[StepRecord] = []
        self._messages: list[dict[str, Any]] = []
        self._instruction: str = ""
        self.last_prediction: str | None = None

        self.timing: Any = None

    def initialize(self, instruction: str) -> None:

        self.step_records.clear()
        self.chat_model.reset()
        self._instruction = instruction or ""
        self._messages = []
        if self.system_prompt:
            self._messages.append({"role": "system", "content": self.system_prompt})
        if self._instruction:
            self._messages.append(
                {
                    "role": "user",
                    "content": [{"type": "text", "text": self._instruction}],
                }
            )
        self._msg_log.ensure_dir()

    def done(self) -> None:
        pass

    async def select_action(self, observation: dict[str, Any]) -> dict[str, Any]:

        screenshot_b64 = observation.get("image", "") or ""
        if not screenshot_b64:
            if self.step_records:
                screenshot_b64 = self.step_records[-1].observation_screenshot
                logger.info(
                    "[Qwen35SLIMEPolicy] obs lacks image — reusing last screenshot",
                )
            else:
                logger.warning(
                    "[Qwen35SLIMEPolicy] first obs has no image; emitting wait",
                )
                return {"action_type": "wait"}

        if self.original_image_width is None or self.original_image_height is None:
            size = _decode_png_size(screenshot_b64)
            if size is not None:
                self.original_image_width, self.original_image_height = size
                logger.info(
                    "[Qwen35SLIMEPolicy] captured screenshot size: %dx%d",
                    *size,
                )

        screenshot_b64 = await self._resize_screenshot_for_model(screenshot_b64)

        self._messages.append(
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": to_data_url(screenshot_b64),
                    }
                ],
            }
        )

        pruned = hide_history_images(self._messages, self.image_retention)

        step_idx = len(self.step_records)
        if step_idx < 3 and logger.isEnabledFor(logging.DEBUG):
            msg_summary = []
            for m in pruned:
                role = m["role"]
                content = m["content"]
                if isinstance(content, str):
                    msg_summary.append(f"{role}:str({len(content)})")
                elif isinstance(content, list):
                    parts = []
                    for c in content:
                        ctype = c.get("type", "?")
                        if ctype == "text":
                            parts.append(f"text({len(c.get('text', ''))})")
                        elif ctype == "image_url":
                            url = c.get("image_url", "")
                            parts.append(f"img({len(url)})")
                    msg_summary.append(f"{role}:[{','.join(parts)}]")
            logger.debug(
                "[Qwen35Policy.INFERENCE] step=%d, n_messages=%d, msgs=[%s]",
                step_idx,
                len(pruned),
                " | ".join(msg_summary),
            )

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
        action_text = info.get("action_text", "") or ""

        logging_enabled = self._msg_log.enabled or self.remote_log_callback is not None
        rendered = elide_images(pruned) if logging_enabled else []

        record = StepRecord(
            response_token_ids=result.output_ids,
            response_logprobs=result.output_logprobs,
            response_text=result.content,
            observation_screenshot=screenshot_b64,
            action_dict=action_dict,
            format_reward=format_reward,
            is_terminal=is_done,
            rendered_messages=rendered,
            finish_reason=result.finish_reason,
            action_text=action_text,
            prompt_token_ids=result.prompt_token_ids,
            multimodal_train_inputs=result.multimodal_train_inputs,
        )
        self.step_records.append(record)

        if step_idx < 3 and logger.isEnabledFor(logging.DEBUG):
            avg_lp = sum(result.output_logprobs) / max(len(result.output_logprobs), 1)
            logger.debug(
                "[Qwen35Policy.INFERENCE] step=%d: response_tokens=%d, "
                "avg_logprob=%.4f, first_3_logprobs=%s, "
                "first_5_response_ids=%s",
                step_idx,
                len(result.output_ids),
                avg_lp,
                [round(lp, 4) for lp in result.output_logprobs[:3]],
                result.output_ids[:5],
            )
        self.last_prediction = result.content

        if logging_enabled:
            self._write_step_log(len(self.step_records) - 1, record)

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
