from __future__ import annotations

import asyncio
import base64
import contextlib
import logging
import os
import struct
import time
from typing import Any

from slime.utils.http_utils import post
from slime.utils.processing_utils import encode_image_for_rollout_engine, process_vision_info

from mai_gym.agents._types import GenerateResult
from mai_gym.agents.token_native import (
    TERMINAL_TOKENS,
    VISION_TOKENS,
    control_token_ids,
    first_control_index,
    gen_header_suffix,
    splice_assistant_output_ids,
    token_ids,
)
from mai_gym.agents.tool_parser import ToolParser

logger = logging.getLogger(__name__)


def _png_data_url_size(url: str) -> tuple[int, int]:

    prefix = "data:image/png;base64,"
    if not isinstance(url, str) or not url.lower().startswith(prefix):
        raise ValueError("image is not a PNG data URL")
    payload = url.split(",", 1)[1]
    if len(payload) < 32:
        raise ValueError("PNG data URL is truncated")
    header = base64.b64decode(payload[:32], validate=True)
    if header[:8] != b"\x89PNG\r\n\x1a\n" or header[12:16] != b"IHDR":
        raise ValueError("PNG data URL has no valid IHDR header")
    width, height = struct.unpack(">II", header[16:24])
    if width <= 0 or height <= 0:
        raise ValueError("PNG has invalid dimensions")
    return height, width


def _message_png_urls(messages: list[dict[str, Any]]) -> list[str]:

    urls: list[str] = []
    for message in messages:
        content = message.get("content")
        if not isinstance(content, list):
            continue
        for part in content:
            if not isinstance(part, dict):
                continue
            media_type = part.get("type")
            if media_type in ("video", "video_url", "input_video"):
                raise ValueError("fast multimodal path does not support video")
            if media_type not in ("image", "image_url", "input_image"):
                continue
            raw = part.get("image_url")
            if isinstance(raw, dict):
                raw = raw.get("url")
            if raw is None and media_type == "image":
                raw = part.get("image")
            if not isinstance(raw, str):
                raise ValueError("fast multimodal path requires image data URLs")
            _png_data_url_size(raw)
            urls.append(raw)
    return urls


def _fast_vlm_inputs(
    text: str,
    messages: list[dict[str, Any]],
    processor: Any,
    tokenizer: Any,
) -> tuple[list[int], list[str]]:

    image_token = getattr(processor, "image_token", None)
    image_processor = getattr(processor, "image_processor", None)
    patch_counter = getattr(image_processor, "get_number_of_image_patches", None)
    merge_size = getattr(image_processor, "merge_size", None)
    if not image_token or not callable(patch_counter) or not merge_size:
        raise ValueError("processor does not expose Qwen image-token planning")

    urls = _message_png_urls(messages)
    placeholder_count = text.count(image_token)
    if placeholder_count != len(urls):
        raise ValueError(
            f"rendered image placeholders ({placeholder_count}) != images ({len(urls)})"
        )

    token_counts: list[int] = []
    for url in urls:
        height, width = _png_data_url_size(url)
        patches = int(patch_counter(height, width, {}))
        if patches <= 0 or patches % (int(merge_size) ** 2):
            raise ValueError(f"invalid Qwen image patch count: {patches}")
        token_counts.append(patches // (int(merge_size) ** 2))

    parts = text.split(image_token)
    expanded = parts[0] + "".join(
        image_token * count + suffix for count, suffix in zip(token_counts, parts[1:], strict=True)
    )
    return tokenizer.encode(expanded), urls


class CustomChatModel:
    def __init__(
        self,
        url: str,
        tokenizer: Any,
        processor: Any | None = None,
        model_type: str = "llm",
        tool_parser_name: str | None = "mobile",
        preserve_thinking: bool = False,
        token_native: bool = False,
        routing_key: str | None = None,
        reasoning_end_token: str | None = None,
        max_context_tokens: int | None = None,
    ) -> None:
        if max_context_tokens is not None and max_context_tokens < 1:
            raise ValueError("max_context_tokens must be positive")
        self.max_context_tokens = max_context_tokens
        self.url = url
        self.tokenizer = tokenizer
        self.processor = processor
        self.model_type = model_type
        self.tool_parser_name = tool_parser_name
        self.reasoning_end_id = None
        if reasoning_end_token is not None:
            if tool_parser_name is not None:
                raise ValueError("Explicit reasoning split is supported only in raw SDK mode")
            vocabulary = tokenizer.get_vocab()
            if reasoning_end_token not in vocabulary:
                raise ValueError(
                    f"Reasoning end token is absent from the tokenizer: {reasoning_end_token}"
                )
            self.reasoning_end_id = vocabulary[reasoning_end_token]

        self.token_native = token_native

        self.preserve_thinking = preserve_thinking

        self.routing_key = routing_key
        self._fast_mm_fallback_warned = False
        self._guard_ids: tuple[frozenset[int], frozenset[int], frozenset[int]] | None = None

    def _token_guards(self) -> tuple[frozenset[int], frozenset[int], frozenset[int]]:
        if self._guard_ids is None:
            terminal = set(token_ids(self.tokenizer, TERMINAL_TOKENS))
            eos = getattr(self.tokenizer, "eos_token_id", None)
            if isinstance(eos, int):
                terminal.add(eos)
            self._guard_ids = (
                control_token_ids(self.tokenizer),
                frozenset(terminal),
                token_ids(self.tokenizer, VISION_TOKENS),
            )
        return self._guard_ids

    def reset(self) -> None:
        pass

    async def generate(
        self,
        messages: list[dict[str, Any]],
        sampling_params: dict[str, Any],
        *,
        tools: list[dict] | None = None,
        original_image_width: int | None = None,
        original_image_height: int | None = None,
        timing: Any | None = None,
    ) -> GenerateResult:

        t0 = time.monotonic()
        preprocessed = await self._preprocess(messages, tools=tools)
        dt_preprocess = time.monotonic() - t0

        if self.max_context_tokens is not None:
            prompt_length = len(preprocessed["token_ids"])
            remaining = self.max_context_tokens - prompt_length - 1
            if remaining <= 0:
                raise ValueError(
                    f"Prompt uses {prompt_length} tokens; no generation budget remains "
                    f"within the {self.max_context_tokens}-token context limit"
                )
            sampling_params = dict(sampling_params)
            sampling_params["max_new_tokens"] = min(
                int(sampling_params.get("max_new_tokens", remaining)), remaining
            )

        t_request = time.monotonic()
        payload = {
            "input_ids": preprocessed["token_ids"],
            "sampling_params": sampling_params,
            "return_logprob": True,
            "image_data": preprocessed.get("image_data"),
        }
        dt_request_prepare = time.monotonic() - t_request

        t1 = time.monotonic()

        headers = {"X-SMG-Routing-Key": self.routing_key} if self.routing_key else None
        output = await post(self.url, payload, headers=headers)
        dt_http = time.monotonic() - t1

        t2 = time.monotonic()
        result = await self._postprocess(
            output,
            original_image_width=original_image_width,
            original_image_height=original_image_height,
        )
        dt_postprocess = time.monotonic() - t2

        result.prompt_token_ids = preprocessed["token_ids"]
        result.multimodal_train_inputs = preprocessed.get("multimodal_train_inputs")

        if timing is not None:
            detail = preprocessed.get("_timing_breakdown", {})
            vision_extract = float(detail.get("vision_extract", 0.0))
            image_encode = float(detail.get("image_encode", 0.0))
            img_time = vision_extract + image_encode
            timing.image_preprocess += img_time
            timing.tokenize += max(0.0, dt_preprocess - img_time)
            timing.chat_template += float(detail.get("chat_template", 0.0))
            timing.vision_extract += vision_extract
            timing.image_encode += image_encode
            timing.hf_processor += float(detail.get("hf_processor", 0.0))
            timing.fast_mm_token_plan += float(detail.get("fast_mm_token_plan", 0.0))
            timing.text_encode += float(detail.get("text_encode", 0.0))
            timing.token_native_splice += float(detail.get("token_native_splice", 0.0))
            accounted = sum(float(v) for v in detail.values())
            timing.preprocess_unattributed += max(0.0, dt_preprocess - accounted)
            timing.request_prepare += dt_request_prepare
            timing.sglang_http_post += dt_http
            timing.postprocess += dt_postprocess
            timing.n_model_calls += 1
            timing.n_images += int(preprocessed.get("_n_images", 0) or 0)

            meta = output.get("meta_info", {}) if isinstance(output, dict) else {}
            e2e = meta.get("e2e_latency")
            if e2e is None:
                e2e = meta.get("e2e_latency_s") or meta.get("spend_time")
            if e2e is not None:
                with contextlib.suppress(TypeError, ValueError):
                    timing.sglang_server_e2e += float(e2e)

            n_prompt = meta.get("prompt_tokens")
            if n_prompt is None:
                n_prompt = len(preprocessed.get("token_ids") or [])
            n_completion = meta.get("completion_tokens")
            if n_completion is None:
                n_completion = len(output.get("output_ids") or [])
            try:
                timing.sglang_prompt_tokens += int(n_prompt)
                timing.sglang_completion_tokens += int(n_completion)
            except (TypeError, ValueError):
                pass

        return result

    async def _preprocess(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict] | None = None,
    ) -> dict[str, Any]:

        loop = asyncio.get_running_loop()

        if self.token_native:
            output_ids_list: list[list[int]] | None = [
                list(m.get("_output_ids") or []) for m in messages if m.get("role") == "assistant"
            ]
            render_messages = [
                {"role": "assistant", "content": ""} if m.get("role") == "assistant" else m
                for m in messages
            ]
            preserve = False
        else:
            output_ids_list = None
            render_messages = messages
            preserve = self.preserve_thinking

        phase_times: dict[str, float] = {}

        t_phase = time.monotonic()
        text = await loop.run_in_executor(
            None,
            lambda: self.tokenizer.apply_chat_template(
                render_messages,
                tools=tools,
                add_generation_prompt=True,
                tokenize=False,
                preserve_thinking=preserve,
            ),
        )
        phase_times["chat_template"] = time.monotonic() - t_phase

        processed: dict[str, Any] = {}

        if self.model_type == "vlm" and self.processor is not None:
            use_fast_mm = (
                os.environ.get("SLIME_FAST_ROLLOUT_MM", "0") == "1"
                and os.environ.get("LAZY_MULTIMODAL") == "1"
            )
            fast_result: tuple[list[int], list[str]] | None = None
            if use_fast_mm:
                t_phase = time.monotonic()
                try:
                    fast_result = await loop.run_in_executor(
                        None,
                        lambda: _fast_vlm_inputs(
                            text,
                            render_messages,
                            self.processor,
                            self.tokenizer,
                        ),
                    )
                except Exception as exc:
                    if not self._fast_mm_fallback_warned:
                        logger.warning(
                            "Fast rollout multimodal preprocessing unavailable; "
                            "falling back to the full HF processor: %s",
                            exc,
                        )
                        self._fast_mm_fallback_warned = True
                phase_times["fast_mm_token_plan"] = time.monotonic() - t_phase

            if fast_result is not None:
                token_ids, image_inputs = fast_result
                n_imgs = len(image_inputs)
                processed["token_ids"] = token_ids
                processed["image_data"] = image_inputs or None
                processed["multimodal_train_inputs"] = None
            else:
                t_phase = time.monotonic()
                multimodal_inputs = await loop.run_in_executor(
                    None,
                    lambda: process_vision_info(render_messages, self.processor),
                )
                phase_times["vision_extract"] = time.monotonic() - t_phase
                image_inputs = multimodal_inputs.get("images")
                video_inputs = multimodal_inputs.get("videos")
                n_imgs = len(image_inputs) if image_inputs else 0
                t_phase = time.monotonic()
                if image_inputs:
                    image_inputs = await loop.run_in_executor(
                        None,
                        lambda imgs=image_inputs: [
                            encode_image_for_rollout_engine(img) for img in imgs
                        ],
                    )
                phase_times["image_encode"] = time.monotonic() - t_phase
                t_phase = time.monotonic()
                batch_feature = await loop.run_in_executor(
                    None,
                    lambda: self.processor(
                        text=text,
                        images=image_inputs,
                        videos=video_inputs,
                        return_tensors="pt",
                    ),
                )
                phase_times["hf_processor"] = time.monotonic() - t_phase
                processed["token_ids"] = batch_feature["input_ids"][0].tolist()
                processed["image_data"] = image_inputs

            if fast_result is None and os.environ.get("LAZY_MULTIMODAL") != "1":
                processed["multimodal_train_inputs"] = {
                    k: v
                    for k, v in batch_feature.items()
                    if k not in ("input_ids", "attention_mask", "mm_token_type_ids")
                } or None

            mm_keys = (
                list(processed["multimodal_train_inputs"].keys())
                if processed.get("multimodal_train_inputs")
                else "None"
            )
            logger.debug(
                "[ChatModel._preprocess] n_tokens=%d, n_images=%d, "
                "first_5_ids=%s, last_5_ids=%s, text_len=%d, mm_keys=%s",
                len(processed["token_ids"]),
                n_imgs,
                processed["token_ids"][:5],
                processed["token_ids"][-5:],
                len(text),
                mm_keys,
            )
        else:
            t_phase = time.monotonic()
            processed["token_ids"] = await loop.run_in_executor(
                None,
                lambda: self.tokenizer.encode(text),
            )
            phase_times["text_encode"] = time.monotonic() - t_phase
            n_imgs = 0

        if self.token_native and output_ids_list is not None:
            t_phase = time.monotonic()
            asst_open = self.tokenizer.encode(
                "<|im_start|>assistant\n",
                add_special_tokens=False,
            )
            im_end_id = self.tokenizer.convert_tokens_to_ids("<|im_end|>")

            tids = processed["token_ids"]
            think_prefix = gen_header_suffix(tids, asst_open)
            filled = [list(think_prefix) + list(o) for o in output_ids_list]
            processed["token_ids"] = splice_assistant_output_ids(
                tids,
                filled,
                asst_open,
                im_end_id,
                forbidden=self._token_guards()[2],
            )
            phase_times["token_native_splice"] = time.monotonic() - t_phase

        processed["_timing_breakdown"] = phase_times
        processed["_n_images"] = n_imgs

        return processed

    async def _postprocess(
        self,
        output: dict[str, Any],
        *,
        original_image_width: int | None = None,
        original_image_height: int | None = None,
    ) -> GenerateResult:

        output_ids = output["output_ids"]
        meta = output["meta_info"]
        response_logprobs = [item[0] for item in meta["output_token_logprobs"]]
        finish_reason = meta["finish_reason"]["type"]
        weight_version = meta.get("weight_version")

        cached_tokens = int(meta.get("cached_tokens", 0) or 0)
        prompt_tokens = int(meta.get("prompt_tokens", 0) or 0)

        if self.tool_parser_name is None and [
            item[1] for item in meta["output_token_logprobs"]
        ] != list(output_ids):
            raise ValueError("SDK output token IDs do not align with returned log probabilities")

        control, terminal, _ = self._token_guards()
        cut = first_control_index(output_ids, control, terminal)
        if cut is not None:
            logger.warning(
                "sampled control token %d at %d/%d; truncating the response there",
                output_ids[cut],
                cut,
                len(output_ids),
            )
            output_ids = output_ids[:cut]
            response_logprobs = response_logprobs[:cut]
            finish_reason = "control_token"

        if self.tool_parser_name is None:
            content = output.get("text") if cut is None else None
            if content is None:
                content = self.tokenizer.decode(output_ids, skip_special_tokens=True)
            info, function_calls = {"text": content}, []
            if self.reasoning_end_id is not None:
                boundary = next(
                    (i for i, token in enumerate(output_ids) if token == self.reasoning_end_id),
                    len(output_ids),
                )
                info["raw_text"] = content
                info["reasoning_content"] = self.tokenizer.decode(
                    output_ids[:boundary],
                    skip_special_tokens=True,
                )
                content = self.tokenizer.decode(
                    output_ids[boundary + 1 :],
                    skip_special_tokens=True,
                )
                info["text"] = content
        else:
            tool_parser = ToolParser.get_tool_parser(
                self.tool_parser_name,
                self.tokenizer,
                original_image_width=original_image_width,
                original_image_height=original_image_height,
            )
            info, function_calls = await tool_parser.extract_tool_calls(output_ids)
            content = info.get("text", "") if isinstance(info, dict) else str(info)

        return GenerateResult(
            output_ids=output_ids,
            output_logprobs=response_logprobs,
            content=content,
            function_calls=function_calls,
            info=info if isinstance(info, dict) else {"text": str(info)},
            finish_reason=finish_reason,
            weight_version=weight_version,
            cached_tokens=cached_tokens,
            prompt_tokens=prompt_tokens,
        )
