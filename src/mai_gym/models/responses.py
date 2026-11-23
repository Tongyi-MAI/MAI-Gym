from typing import Any

from pydantic import BaseModel


class SessionResponse(BaseModel):
    session_id: str
    message: str
    worker_idx: str


class BatchSessionResponse(BaseModel):
    sessions: list[SessionResponse]
    total_count: int


class StepResponse(BaseModel):
    env_idx: str
    obs_messages: list[dict[str, Any]] | None = None
    is_done: bool
    format_reward: float


class EvaluateResponse(BaseModel):
    reward: float
    success: bool | None = None
    reason: str | None = None

    built_in_reward: float | None = None
    judge_reward: float | None = None


class SessionTaskResponse(BaseModel):
    ok: bool
    kind: str
    detail: str | None = None


class ListSessionsResponse(BaseModel):
    sessions: list[dict[str, Any]]


class TaskInfo(BaseModel):
    task_id: str

    instruction: str | None = None
    description: str | None = None
    tags: list[str] | None = None
    apps: list[str] | None = None
    max_steps: int | None = None
    env_type: str | None = None
    infra_type: str | None = None
    environment_profile: str | None = None
    benchmark: str | None = None


class ListTasksResponse(BaseModel):
    tasks: list[TaskInfo]


class AcquireResponse(BaseModel):
    session_id: str
    backend_idx: str
    lease_id: str
    lease_expires_at: float
