from __future__ import annotations

import asyncio
import json
import logging
from typing import Any

import httpx

from mai_gym.agents._types import GenerateResult
from mai_gym.agents.errors import AgentAPIError
from mai_gym.agents.tool_parser import ToolParser

logger = logging.getLogger(__name__)

_RATE_LIMIT_BACKOFF_S = (20.0, 40.0, 80.0, 160.0)
_ERROR_BACKOFF_S = (20.0, 40.0, 80.0)


class _PassthroughDecoder:
    def decode(self, value: Any, skip_special_tokens: bool = True) -> str:
        return value if isinstance(value, str) else str(value)


class HttpChatModel:
    def __init__(
        self,
        url: str | None = None,
        model: str = "default",
        *,
        base_url: str | None = None,
        tool_parser_name: str = "mobile",
        api_key: str | None = None,
        timeout: float = 600.0,
        preserve_thinking: bool = False,
        enable_thinking: bool | None = None,
        max_pixels: int | None = None,
        min_pixels: int | None = None,
        client: httpx.AsyncClient | None = None,
    ) -> None:

        if url is not None and base_url is not None and url.rstrip("/") != base_url.rstrip("/"):
            raise ValueError("url and base_url must identify the same endpoint")
        base = (url or base_url or "http://127.0.0.1:8000/v1").rstrip("/")
        self.url = base if base.endswith("/chat/completions") else f"{base}/chat/completions"
        self.model = model
        self.tool_parser_name = tool_parser_name
        self.api_key = api_key
        self.timeout = timeout

        self.preserve_thinking = preserve_thinking

        self.enable_thinking = enable_thinking
        self.max_pixels = max_pixels
        self.min_pixels = min_pixels

        self._client = client or httpx.AsyncClient(timeout=timeout)
        self._owns_client = client is None

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
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": self._to_openai_multimodal(messages),
        }
        payload.update(self._map_sampling(sampling_params))
        if tools:
            payload["tools"] = tools

        ctk: dict[str, Any] = {}
        if self.preserve_thinking:
            ctk["preserve_thinking"] = True
        if self.enable_thinking is not None:
            ctk["enable_thinking"] = self.enable_thinking
            payload["enable_thinking"] = self.enable_thinking
        if ctk:
            payload["chat_template_kwargs"] = ctk
        if self.max_pixels is not None:
            payload["max_pixels"] = self.max_pixels
        if self.min_pixels is not None:
            payload["min_pixels"] = self.min_pixels

        headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
        resp = await self._post_with_backoff(payload, headers)
        resp.raise_for_status()
        data = resp.json()

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
        native_calls = message.get("tool_calls")
        if isinstance(native_calls, list) and native_calls and "<tool_call>" not in content:
            call = native_calls[0]
            function = call.get("function") if isinstance(call, dict) else None
            if isinstance(function, dict):
                arguments = function.get("arguments", {})
                if isinstance(arguments, str):
                    try:
                        arguments = json.loads(arguments)
                    except json.JSONDecodeError:
                        arguments = {"_raw_arguments": arguments}
                tool_call = json.dumps(
                    {"name": function.get("name") or "mobile_use", "arguments": arguments},
                    ensure_ascii=False,
                )
                content = f"{content}<tool_call>{tool_call}</tool_call>"
        finish_reason = choice.get("finish_reason") or ""

        parser = ToolParser.get_tool_parser(
            self.tool_parser_name,
            _PassthroughDecoder(),
            original_image_width=original_image_width,
            original_image_height=original_image_height,
        )

        info, function_calls = await parser.extract_tool_calls(content)

        return GenerateResult(
            output_ids=[],
            output_logprobs=[],
            content=content,
            function_calls=function_calls,
            info=info,
            finish_reason=finish_reason,
            cached_tokens=int((data.get("usage") or {}).get("cached_tokens", 0) or 0),
            prompt_tokens=int((data.get("usage") or {}).get("prompt_tokens", 0) or 0),
        )

    async def _post_with_backoff(
        self, payload: dict[str, Any], headers: dict[str, str]
    ) -> httpx.Response:
        retries = {"rate_limit": 0, "error": 0}
        while True:
            try:
                response = await self._client.post(self.url, json=payload, headers=headers)
            except httpx.TransportError as exc:
                kind, failure = "error", f"{type(exc).__name__}: {exc}"
            else:
                if response.status_code == 429:
                    kind = "rate_limit"
                elif response.status_code >= 500:
                    kind = "error"
                else:
                    return response
                failure = f"HTTP {response.status_code}: {response.text[:300]}"

            schedule = _RATE_LIMIT_BACKOFF_S if kind == "rate_limit" else _ERROR_BACKOFF_S
            attempt = retries[kind]
            if attempt >= len(schedule):
                raise AgentAPIError(f"LLM call failed after {attempt} {kind} retries: {failure}")
            retries[kind] += 1
            logger.warning(
                "[HttpChatModel] %s; retry %d/%d in %.0fs",
                failure,
                attempt + 1,
                len(schedule),
                schedule[attempt],
            )
            await asyncio.sleep(schedule[attempt])

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

    @staticmethod
    def _map_sampling(sp: dict[str, Any]) -> dict[str, Any]:

        out: dict[str, Any] = {}
        if sp.get("max_new_tokens") is not None:
            out["max_tokens"] = sp["max_new_tokens"]
        elif sp.get("max_tokens") is not None:
            out["max_tokens"] = sp["max_tokens"]
        for key in ("temperature", "top_p", "stop"):
            if sp.get(key) is not None:
                out[key] = sp[key]

        if sp.get("no_stop_trim") is not None:
            out["no_stop_trim"] = sp["no_stop_trim"]
        if isinstance(sp.get("extra_body"), dict):
            out.update(sp["extra_body"])
        return out

    @staticmethod
    def _to_openai_multimodal(
        messages: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:

        out: list[dict[str, Any]] = []
        for m in messages:
            m = {key: value for key, value in m.items() if not key.startswith("_")}
            content = m.get("content")
            if isinstance(content, list):
                parts = []
                for part in content:
                    if (
                        isinstance(part, dict)
                        and part.get("type") == "image_url"
                        and isinstance(part.get("image_url"), str)
                    ):
                        part = {**part, "image_url": {"url": part["image_url"]}}
                    parts.append(part)
                m = {**m, "content": parts}
            out.append(m)
        return out

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()
