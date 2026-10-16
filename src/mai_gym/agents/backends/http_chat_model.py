from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from typing import Any
from urllib.request import Request, urlopen


@dataclass
class GenerateResult:
    """Minimal generation result consumed by the mobile-use policies."""

    content: str
    finish_reason: str = ""
    output_ids: list[int] = field(default_factory=list)
    output_logprobs: list[float] = field(default_factory=list)
    prompt_token_ids: list[int] = field(default_factory=list)
    multimodal_train_inputs: dict[str, Any] | None = None
    cached_tokens: int = 0
    prompt_tokens: int = 0


class HttpChatModel:
    """Small OpenAI-compatible Chat Completions client.

    ``base_url`` defaults to a loopback OpenAI-compatible API and may be either
    an API root or the full ``.../chat/completions`` endpoint. The class does
    not parse actions; the policy delegates that to the standalone parser.
    """

    def __init__(
        self,
        base_url: str = "http://127.0.0.1:8000/v1",
        model: str = "default",
        *,
        api_key: str | None = None,
        timeout: float = 600.0,
        preserve_thinking: bool = False,
        enable_thinking: bool | None = None,
        max_pixels: int | None = None,
        min_pixels: int | None = None,
    ) -> None:
        base = base_url.rstrip("/")
        self.url = base if base.endswith("/chat/completions") else f"{base}/chat/completions"
        self.model = model
        self.api_key = api_key
        self.timeout = timeout
        self.preserve_thinking = preserve_thinking
        self.enable_thinking = enable_thinking
        self.max_pixels = max_pixels
        self.min_pixels = min_pixels

    def reset(self) -> None:
        return None

    async def generate(
        self,
        messages: list[dict[str, Any]],
        sampling_params: dict[str, Any],
        *,
        tools: list[dict[str, Any]] | None = None,
        original_image_width: int | None = None,
        original_image_height: int | None = None,
        timing: Any | None = None,
    ) -> GenerateResult:
        del original_image_width, original_image_height, timing

        payload: dict[str, Any] = {
            "model": self.model,
            "messages": self._to_openai_multimodal(messages),
        }
        payload.update(self._map_sampling(sampling_params))
        if tools:
            payload["tools"] = tools

        chat_template_kwargs: dict[str, Any] = {}
        if self.preserve_thinking:
            chat_template_kwargs["preserve_thinking"] = True
        if self.enable_thinking is not None:
            chat_template_kwargs["enable_thinking"] = self.enable_thinking
            payload["enable_thinking"] = self.enable_thinking
        if chat_template_kwargs:
            payload["chat_template_kwargs"] = chat_template_kwargs
        if self.max_pixels is not None:
            payload["max_pixels"] = self.max_pixels
        if self.min_pixels is not None:
            payload["min_pixels"] = self.min_pixels

        data = await asyncio.to_thread(self._post_json, payload)
        choice = (data.get("choices") or [{}])[0]
        message = choice.get("message") or {}
        content = self._response_text(message.get("content"))
        reasoning = self._first_response_text(
            message.get("reasoning_content"),
            message.get("reasoning"),
            message.get("analysis"),
            choice.get("reasoning_content"),
            choice.get("reasoning"),
            choice.get("analysis"),
            data.get("reasoning_content"),
            data.get("reasoning"),
            data.get("analysis"),
        )
        if reasoning and "<think" not in content:
            content = f"<think>{reasoning}</think>{content}"

        usage = data.get("usage") or {}
        return GenerateResult(
            content=content,
            finish_reason=str(choice.get("finish_reason") or ""),
            cached_tokens=int(usage.get("cached_tokens", 0) or 0),
            prompt_tokens=int(usage.get("prompt_tokens", 0) or 0),
        )

    def _post_json(self, payload: dict[str, Any]) -> dict[str, Any]:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        request = Request(
            self.url,
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        with urlopen(request, timeout=self.timeout) as response:
            body = response.read().decode("utf-8")
        parsed = json.loads(body)
        if not isinstance(parsed, dict):
            raise ValueError("Chat Completions response must be a JSON object")
        return parsed

    @staticmethod
    def _map_sampling(sampling_params: dict[str, Any]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        if sampling_params.get("max_new_tokens") is not None:
            result["max_tokens"] = sampling_params["max_new_tokens"]
        elif sampling_params.get("max_tokens") is not None:
            result["max_tokens"] = sampling_params["max_tokens"]
        for key in ("temperature", "top_p", "stop"):
            if sampling_params.get(key) is not None:
                result[key] = sampling_params[key]
        if sampling_params.get("no_stop_trim") is not None:
            result["no_stop_trim"] = sampling_params["no_stop_trim"]
        return result

    @staticmethod
    def _to_openai_multimodal(
        messages: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        converted: list[dict[str, Any]] = []
        for message in messages:
            message = {
                key: value for key, value in message.items() if not key.startswith("_")
            }
            content = message.get("content")
            if isinstance(content, list):
                parts: list[Any] = []
                for part in content:
                    if (
                        isinstance(part, dict)
                        and part.get("type") == "image_url"
                        and isinstance(part.get("image_url"), str)
                    ):
                        part = {**part, "image_url": {"url": part["image_url"]}}
                    parts.append(part)
                message = {**message, "content": parts}
            converted.append(message)
        return converted

    @staticmethod
    def _response_text(value: Any) -> str:
        if value is None:
            return ""
        if isinstance(value, str):
            return value
        if isinstance(value, list):
            parts: list[str] = []
            for block in value:
                if isinstance(block, str):
                    parts.append(block)
                elif isinstance(block, dict):
                    text = block.get("text", block.get("content"))
                    if text is not None:
                        parts.append(str(text))
            return "".join(parts)
        if isinstance(value, dict):
            text = value.get("text", value.get("content"))
            return "" if text is None else str(text)
        return str(value)

    @classmethod
    def _first_response_text(cls, *values: Any) -> str:
        for value in values:
            text = cls._response_text(value)
            if text.strip():
                return text
        return ""

    async def aclose(self) -> None:
        return None


__all__ = ["GenerateResult", "HttpChatModel"]
