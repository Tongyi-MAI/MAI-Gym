from pathlib import Path
from typing import Any, Literal, Self

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from mai_gym.models.config import (
    DeviceConfig,
    JudgeConfig,
    ManagerConfig,
    _normalize_legacy_field,
)


class Settings(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class ModelSettings(Settings):
    checkpoint: str | None = None
    name: str | None = None
    base_url: str | None = None
    api_key: str = Field(default="env/OPENAI_API_KEY", repr=False)
    agent_variant: str = "unified"
    policy_variant: Literal["mai_ui", "qwen35"] = "mai_ui"
    tool_parser: str = "mai_gym_mobile_use"
    runtime_conf: dict[str, Any] = Field(default_factory=dict)
    unified_variant_map: dict[str, str] = Field(
        default_factory=lambda: {
            "mobile_use": "mai_gym_mobile_use_windows_wise",
            "computer_use": "mai_gym_computer_use",
        }
    )


class TaskPool(Settings):
    pool: str
    env_type: str
    manager_url: str = "http://127.0.0.1:8090"
    num_sessions: int | None = Field(default=None, gt=0)

    @model_validator(mode="before")
    @classmethod
    def _normalize_legacy_fields(cls, value: Any) -> Any:
        return _normalize_legacy_field(value, canonical="manager_url", legacy="controller_url")

    @property
    def controller_url(self) -> str:
        return self.manager_url


class AlgorithmSettings(Settings):
    name: Literal["grpo", "dapo", "gspo", "reinforce++", "ppo", "sao"] = "grpo"
    group_size: int = Field(default=8, ge=2)
    learning_rate: float = Field(default=1e-6, gt=0)
    kl_loss_coef: float = Field(default=1e-3, ge=0)
    entropy_coef: float = Field(default=0.0, ge=0)
    per_token_loss: bool = True
    reward_allocation: Literal["uniform"] = "uniform"


class SamplingSettings(Settings):
    success_threshold: float = Field(default=0.8, ge=0, le=1)
    fail_budget: int = Field(default=4, gt=0)
    mastered_budget: int = Field(default=2, gt=0)


class RolloutSettings(Settings):
    concurrency: int = Field(default=16, gt=0)
    num_sessions: int = Field(default=16, gt=0)
    temperature: float = Field(default=1.0, ge=0)
    max_turns: int = Field(default=100, gt=0)
    max_total_length: int = Field(default=65536, gt=0)
    max_response_length: int = Field(default=6144, gt=0)
    rounds: int = Field(default=2000, gt=0)
    batch_size: int = Field(default=100, gt=0)


class HistorySettings(Settings):
    window_size: int = Field(default=10, gt=0)
    overlap: int = Field(default=2, ge=0)
    image_retention: int = Field(default=10, ge=0)
    token_native_history: bool = True
    windowwise_train: bool = True

    @model_validator(mode="before")
    @classmethod
    def _normalize_legacy_fields(cls, value: Any) -> Any:
        return _normalize_legacy_field(value, canonical="window_size", legacy="window")

    @property
    def window(self) -> int:
        return self.window_size

    @model_validator(mode="after")
    def check_overlap(self) -> Self:
        if self.overlap >= self.window_size:
            raise ValueError("history.overlap must be smaller than history.window_size")
        return self


class ObservationSettings(Settings):
    screenshot_transport: Literal["inline", "oss"] = "inline"
    oss: dict[str, Any] | None = None


class Parallelism(Settings):
    tensor: int = Field(default=4, gt=0)
    pipeline: int = Field(default=2, gt=0)
    context: int = Field(default=4, gt=0)
    data: int | None = Field(default=None, gt=0)


class TrainerSettings(Settings):
    backend: Literal["slime", "megatron"] = "slime"
    parallelism: Parallelism = Field(default_factory=Parallelism)
    nodes: int = Field(default=4, gt=0)
    gpus_per_node: int = Field(default=8, gt=0)
    engine_gpus: int = Field(default=2, gt=0)
    output: str = "./runs/mai-gym-9b"
    data: str | None = None
    resume: str | None = None
    run_name: str | None = None
    ray_address: str = "auto"
    global_batch_size: int = Field(default=320, gt=0)
    save_interval: int = Field(default=1, gt=0)
    extra_args: list[str] = Field(default_factory=list)
    entrypoint: str | None = None


class BenchmarkSettings(Settings):
    env_type: str | None = None
    manager_url: str | None = None
    manager_config: str | None = None
    tasks: list[str] | None = None
    tasks_file: str | None = None
    tasks_dir: str | None = None
    task_metadata: dict[str, dict[str, Any]] = Field(default_factory=dict)
    num_envs: int | None = Field(default=None, gt=0)
    agent_variant: str | None = None

    @model_validator(mode="before")
    @classmethod
    def _normalize_legacy_fields(cls, value: Any) -> Any:
        return _normalize_legacy_field(value, canonical="manager_url", legacy="controller_url")

    @property
    def controller_url(self) -> str | None:
        return self.manager_url


class WorkloadConfig(Settings):
    model: ModelSettings = Field(default_factory=ModelSettings)
    tasks: list[TaskPool] = Field(default_factory=list)
    algorithm: AlgorithmSettings = Field(default_factory=AlgorithmSettings)
    sampling: SamplingSettings = Field(default_factory=SamplingSettings)
    rollout: RolloutSettings = Field(default_factory=RolloutSettings)
    history: HistorySettings = Field(default_factory=HistorySettings)
    observations: ObservationSettings = Field(default_factory=ObservationSettings)
    trainer: TrainerSettings = Field(default_factory=TrainerSettings)
    benchmarks: dict[str, BenchmarkSettings] = Field(default_factory=dict)
    manager: ManagerConfig = Field(default_factory=ManagerConfig)
    devices: list[DeviceConfig] = Field(default_factory=list)
    judge: JudgeConfig | None = None

    @classmethod
    def from_yaml(cls, path: str | Path) -> Self:
        data = yaml.safe_load(Path(path).read_text())
        return cls.model_validate(data)
