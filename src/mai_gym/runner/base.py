from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Literal

from pydantic import BaseModel

from mai_gym.client.env_client import EnvClient


class TaskConfig(BaseModel):
    category: str
    task_id: str | None = None
    goal: str | None = None
    target_apps: list[str] | None = None
    max_steps: int = 50

    env_type: str | None = None

    seed_task_id: str | None = None

    attempt_id: str | None = None

    train_id: str | None = None

    rollout_id: int | None = None
    is_eval: bool = False

    evaluation_mode: Literal["built_in", "llm_judge", "both"] = "built_in"

    use_judge: bool = False
    require_score: bool = False


@dataclass
class Trajectory:
    task_id: str | None
    steps: list[dict[str, Any]] = field(default_factory=list)
    reward: float = 0.0
    success: bool | None = None


class TaskRunner(ABC):
    def __init__(self, env_client: EnvClient) -> None:
        self.env_client = env_client

    @abstractmethod
    async def run(self, session_id: str, task: TaskConfig) -> Trajectory: ...
