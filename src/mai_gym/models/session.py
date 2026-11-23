import time
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any


class SessionStatus(StrEnum):
    INITIALIZING = "initializing"
    READY = "ready"
    BUSY = "busy"
    INVALID = "invalid"
    RECOVERING = "recovering"
    DESTROYING = "destroying"


@dataclass
class PendingOp:
    kind: str
    action_id: str
    task: "object | None" = None
    done: bool = False
    result: dict[str, Any] | None = None
    error: str | None = None
    retryable: bool = False
    error_detail: dict[str, Any] | None = None
    status_code: int | None = None


@dataclass
class RunningBackend:
    id: str
    url: str
    device_name: str
    env_type: str
    device_id: str | None = None

    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class SessionTask:
    kind: str

    set_id: str | None = None
    task_class: str | None = None
    golden_class: str | None = None

    initial_setup: str | None = None
    golden_patch: str | None = None
    reward: str | None = None

    world_oss_uri: str | None = None
    world_sha256: str | None = None
    world_extract_to: str | None = None


@dataclass
class SessionState:
    session_id: str
    backend: RunningBackend
    status: SessionStatus = SessionStatus.INITIALIZING
    context_id: int | None = None
    current_task: str | None = None
    task_goal: str | None = None
    step_count: int = 0
    _last_screenshot_b64: str = ""
    traj_logger: object | None = None
    traj_name: str | None = None

    traj_scored: bool = False

    last_action_id: str | None = None
    last_step_result: dict[str, Any] | None = None

    last_reset_id: str | None = None
    last_reset_result: dict[str, Any] | None = None

    pending_op: PendingOp | None = None

    verifier_reset_ops: dict[str, PendingOp] = field(default_factory=dict)

    train_id: str | None = None

    current_task_config: dict | None = None

    action_history: list[str] = field(default_factory=list)

    lease_id: str | None = None
    lease_expires_at: float | None = None

    created_at: float = field(default_factory=time.time)

    leased_since: float | None = None

    session_task: "SessionTask | None" = None

    ephemeral_dirty: bool = False

    cleanup_lease_id: str | None = None

    busy_since: float | None = None

    epoch: int = 0

    last_op_failure_at: float | None = None

    force_relaunch: bool = False

    sdk_session: Any = None
