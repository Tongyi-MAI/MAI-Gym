from typing import Any, Literal

from pydantic import BaseModel, Field


class BatchStartEnvRequest(BaseModel):
    num_envs: int
    env_type: str = "real_device"


class FleetRequestCreate(BaseModel):
    job_id: str = Field(min_length=1, max_length=256)
    holder_id: str = Field(min_length=1, max_length=128)
    idempotency_key: str = Field(min_length=1, max_length=128)
    env_type: str = Field(min_length=1, max_length=128)
    capacity: int = Field(ge=1)


class FleetHeartbeatRequest(BaseModel):
    holder_id: str = Field(min_length=1, max_length=128)


class ResetRequest(BaseModel):
    category: str
    task_id: str | None = None
    goal: str | None = None
    target_apps: list[str] | None = None
    seed_task_id: str | None = None

    attempt_id: str | None = None

    train_id: str | None = None

    rollout_id: int | None = None
    evaluation: bool = False

    model_name: str | None = None

    screenshot_width: int | None = None
    screenshot_height: int | None = None

    lease_id: str | None = None

    action_id: str | None = None

    screenshot_transport: Literal["inline", "oss"] | None = None

    task_phase: Literal["initial", "golden"] | None = None


class SessionTaskRequest(BaseModel):
    lease_id: str | None = None
    env_type: Literal["mobileworld", "osworld"]

    set_id: str | None = None
    task_class: str | None = None
    golden_class: str | None = None
    package_b64: str | None = None

    initial_setup: str | None = None
    golden_patch: str | None = None
    reward: str | None = None

    world_oss_uri: str | None = None
    world_sha256: str | None = None
    world_extract_to: str | None = None


class RunConfigRequest(BaseModel):
    train_id: str | None = None
    model_name: str | None = None
    config: dict[str, Any] = Field(default_factory=dict)


class TrainedManifestRequest(BaseModel):
    train_id: str | None = None
    rollout_id: int | None = None
    evaluation: bool = False
    model_name: str | None = None
    manifest: dict[str, Any] = Field(default_factory=dict)


class StepRequest(BaseModel):
    action: dict[str, Any]
    screenshot_width: int | None = None
    screenshot_height: int | None = None

    lease_id: str | None = None

    action_id: str | None = None

    screenshot_transport: Literal["inline", "oss"] | None = None


class ReadinessRequest(BaseModel):
    action_id: str
    lease_id: str | None = None


class EvaluateRequest(BaseModel):
    category: str
    task_id: str | None = None
    trajectory_messages: str | None = None

    evaluation_mode: Literal["built_in", "llm_judge", "both"] = "built_in"

    use_judge: bool = False
    require_score: bool = False


class TearDownRequest(BaseModel):
    category: str
    task_id: str | None = None


class AcquireRequest(BaseModel):
    env_type: str | None = None
    target_apps: list[str] | None = None
    timeout_s: float = 30.0

    fleet_request_id: str | None = None


class ReleaseRequest(BaseModel):
    lease_id: str


class DiscardRequest(BaseModel):
    lease_id: str
    reason: str | None = None


class AcquireFailureRequest(BaseModel):
    train_id: str | None = None
    task_id: str | None = None
    attempt_id: str | None = None
    rollout_id: int | None = None
    evaluation: bool = False
    model_name: str | None = None
    env_type: str | None = None
    reason: str | None = None
