from __future__ import annotations

import copy
import math
from dataclasses import dataclass
from typing import Any

from mai_gym.agents._types import ChatModel, GenerateResult


@dataclass(frozen=True)
class GenerationRecord:
    messages: list[dict]
    sampling_params: dict
    result: GenerateResult
    gen_epoch: int | None = None


class GenerationCaptureError(RuntimeError):
    pass


class RecordingChatModel:
    def __init__(self, model: ChatModel, *, epoch_getter=None):
        self.model = model
        self.epoch_getter = epoch_getter
        self.records: list[GenerationRecord] = []

    def reset(self):
        self.records.clear()
        self.model.reset()

    async def generate(self, messages, sampling_params, **kwargs: Any) -> GenerateResult:
        snapshot = copy.deepcopy(messages)
        sampling = copy.deepcopy(sampling_params)
        epoch = self.epoch_getter() if self.epoch_getter else None
        result = await self.model.generate(messages, sampling_params, **kwargs)
        if not result.prompt_token_ids or not result.output_ids:
            raise GenerationCaptureError("Inference did not return prompt and completion token IDs")
        if len(result.output_logprobs) != len(result.output_ids) or not all(
            math.isfinite(x) for x in result.output_logprobs
        ):
            raise GenerationCaptureError("Completion log probabilities are missing or misaligned")

        evidence = copy.copy(result)
        evidence.prompt_token_ids = list(result.prompt_token_ids)
        evidence.output_ids = list(result.output_ids)
        evidence.output_logprobs = list(result.output_logprobs)
        evidence.info = copy.deepcopy(result.info)
        self.records.append(GenerationRecord(snapshot, sampling, evidence, epoch))
        return result
