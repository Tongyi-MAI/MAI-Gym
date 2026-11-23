from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

from mai_gym.runner.rollout import LifecyclePolicy


class AgentPolicy(LifecyclePolicy, Protocol):
    step_records: list[Any]
    last_prediction: str | None


@dataclass(frozen=True)
class AgentVariant:
    name: str
    policy_cls: type
    parser_name: str
    system_prompt: str
    default_image_retention: int | None = 3
    description: str = ""

    preserve_thinking: bool = False

    stop_sequences: tuple[str, ...] | None = None
    extra: dict[str, Any] = field(default_factory=dict)
