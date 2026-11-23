from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from pydantic import BaseModel


class FunctionCall(BaseModel):
    arguments: str
    name: str


@dataclass
class GenerateResult:
    output_ids: list[int] = field(default_factory=list)
    output_logprobs: list[float] = field(default_factory=list)
    content: str = ""
    function_calls: list[FunctionCall] = field(default_factory=list)
    info: dict[str, Any] = field(default_factory=dict)
    finish_reason: str = ""
    weight_version: str | None = None
    prompt_token_ids: list[int] = field(default_factory=list)

    cached_tokens: int = 0
    prompt_tokens: int = 0

    multimodal_train_inputs: dict[str, Any] | None = None


@runtime_checkable
class ChatModel(Protocol):
    def reset(self) -> None: ...

    async def generate(
        self,
        messages: list[dict[str, Any]],
        sampling_params: dict[str, Any],
        *,
        tools: list[dict] | None = None,
        original_image_width: int | None = None,
        original_image_height: int | None = None,
        timing: Any | None = None,
    ) -> GenerateResult: ...
