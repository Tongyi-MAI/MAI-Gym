from __future__ import annotations

import base64
import json
import logging
import re
import time
import uuid
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from mai_gym.io import oss_io

if TYPE_CHECKING:
    from ossfs import AioOSSFileSystem

    from mai_gym.models.config import OSSConfig

logger = logging.getLogger(__name__)


def model_slug(name: str | None) -> str:

    slug = re.sub(r"[^A-Za-z0-9._-]", "_", name or "")
    return slug or "unknown_model"


def _session_fallback_slug(session_id: str | None) -> str:

    ts = time.strftime("%Y%m%d_%H%M%S")
    short_sid = (session_id or "").split("-")[0] or "unknown"
    short_uid = uuid.uuid4().hex[:6]
    return f"session_{ts}_{short_sid}_{short_uid}"


def _auto_attempt_dir(session_id: str | None) -> str:

    ts = time.strftime("%Y%m%d_%H%M%S")
    short_sid = (session_id or "").split("-")[0] or "unknown"
    short_uid = uuid.uuid4().hex[:6]
    return f"{ts}_{short_sid}_{short_uid}"


def _attempt_slug(attempt_id: str) -> str:

    slug = re.sub(r"[^A-Za-z0-9._-]", "_", attempt_id)
    return slug or "attempt"


def _seg_slug(value: str, fallback: str) -> str:

    slug = re.sub(r"[^A-Za-z0-9._-]", "_", value)
    return slug or fallback


def _rollout_seg(rollout_id: int | None, evaluation: bool) -> str:

    if rollout_id is not None:
        return f"eval_r{rollout_id:05d}" if evaluation else f"r{rollout_id:05d}"
    return "eval_r_misc" if evaluation else "r_misc"


def _traj_subpath(
    task_id: str | None,
    env_type: str | None,
    session_id: str | None,
    attempt_id: str | None = None,
    *,
    train_id: str | None = None,
    rollout_id: int | None = None,
    evaluation: bool = False,
) -> str:

    base = _seg_slug(task_id, "unknown_task") if task_id else _session_fallback_slug(session_id)
    auto = _auto_attempt_dir(session_id)
    attempt_dir = f"{auto}_{_attempt_slug(attempt_id)}" if attempt_id else auto
    if train_id:
        return (
            f"{_seg_slug(train_id, 'unknown_train')}/"
            f"{_rollout_seg(rollout_id, evaluation)}/{base}/{attempt_dir}"
        )
    return f"{base}/{attempt_dir}"


_join = oss_io.join


@dataclass
class TrajContext:
    log_file_dir: str
    log_file_name: str = "traj.json"
    screenshots_dir: str = "screenshots"
    tools: list[dict[str, Any]] = field(default_factory=list)
    oss_cfg: OSSConfig | None = None

    @property
    def traj_json_path(self) -> str:
        return _join(self.log_file_dir, self.log_file_name)

    @property
    def screenshots_path(self) -> str:
        return _join(self.log_file_dir, self.screenshots_dir)


def _fs_for(ctx: TrajContext | None) -> AioOSSFileSystem | None:

    if ctx is None or ctx.oss_cfg is None:
        return None
    if not oss_io.is_oss_path(ctx.log_file_dir):
        return None
    return oss_io.get_async_fs(ctx.oss_cfg)


async def create_traj_logger(
    log_dir: str,
    task_id: str | None,
    env_type: str | None = None,
    session_id: str | None = None,
    attempt_id: str | None = None,
    *,
    oss_cfg: OSSConfig | None = None,
    train_id: str | None = None,
    rollout_id: int | None = None,
    evaluation: bool = False,
    model: str | None = None,
) -> tuple[TrajContext | None, str | None]:

    traj_name = _traj_subpath(
        task_id,
        env_type,
        session_id,
        attempt_id=attempt_id,
        train_id=train_id,
        rollout_id=rollout_id,
        evaluation=evaluation,
    )
    log_file_dir = _join(log_dir, traj_name)

    is_oss = oss_io.is_oss_path(log_file_dir)
    if is_oss and oss_cfg is None:
        logger.warning(
            "create_traj_logger: oss:// path %r requires oss_cfg; "
            "trajectory logging disabled for this session",
            log_file_dir,
        )
        return None, None

    ctx = TrajContext(
        log_file_dir=log_file_dir,
        oss_cfg=oss_cfg if is_oss else None,
    )
    fs = _fs_for(ctx)
    try:
        await oss_io.makedirs(ctx.log_file_dir, fs=fs)
        await oss_io.makedirs(ctx.screenshots_path, fs=fs)

        await oss_io.write_text(ctx.traj_json_path, "{}", fs=fs)

        meta = {
            "train_id": train_id,
            "rollout_id": rollout_id,
            "evaluation": evaluation,
            "session_id": session_id,
            "task_id": task_id,
            "attempt_id": attempt_id,
            "model": model,
            "env_type": env_type,
            "traj_name": traj_name,
            "created_at": time.time(),
        }
        await oss_io.write_text(
            _join(ctx.log_file_dir, "meta.json"),
            json.dumps(meta, ensure_ascii=False),
            fs=fs,
        )
    except (OSError, ValueError) as exc:
        logger.warning(
            "create_traj_logger: %s failed (%s); continuing without traj log",
            log_file_dir,
            exc,
        )
        return None, None
    except Exception as exc:
        logger.warning(
            "create_traj_logger: unexpected error on %s (%s); continuing without traj log",
            log_file_dir,
            exc,
        )
        return None, None
    return ctx, traj_name


async def alog_step(
    traj_logger: TrajContext | None,
    traj_name: str,
    task_goal: str,
    step: int,
    prediction: str,
    action: dict[str, Any],
    screenshot_b64: str,
) -> None:

    if traj_logger is None:
        return
    try:
        fs = _fs_for(traj_logger)

        traj_name_for_log = traj_name.replace("/", "_") if traj_name else traj_name
        task_id = "0"

        try:
            raw = await oss_io.read_bytes(traj_logger.traj_json_path, fs=fs)
            log_data: dict[str, Any] = json.loads(raw) if raw else {}
        except (FileNotFoundError, json.JSONDecodeError):
            log_data = {}
        if task_id not in log_data:
            log_data[task_id] = {"tools": traj_logger.tools, "traj": []}
        log_data[task_id]["traj"].append(
            {
                "task_goal": task_goal,
                "step": step,
                "prediction": prediction,
                "action": action,
                "ask_user_response": None,
                "tool_call": None,
            }
        )
        await oss_io.write_bytes(
            traj_logger.traj_json_path,
            json.dumps(log_data, ensure_ascii=False, indent=4).encode("utf-8"),
            fs=fs,
        )

        png_bytes = base64.b64decode(screenshot_b64)
        screenshot_path = _join(
            traj_logger.screenshots_path,
            f"{traj_name_for_log}-{task_id}-{step}.png",
        )
        await oss_io.write_bytes(screenshot_path, png_bytes, fs=fs)
    except Exception as exc:
        logger.warning("Failed to log trajectory step: %s", exc)


async def log_score(
    traj_logger: TrajContext | None,
    reward: float,
    reason: str | None = None,
) -> None:

    if traj_logger is None:
        return
    try:
        fs = _fs_for(traj_logger)
        text = f"{reward}\n{reason}\n" if reason is not None else f"{reward}\n"
        await oss_io.write_text(
            _join(traj_logger.log_file_dir, "result.txt"),
            text,
            fs=fs,
        )
    except Exception as exc:
        logger.warning("Failed to log trajectory score: %s", exc)


_MAX_FAILURE_REASON_CHARS = 2000


async def log_failure(
    traj_logger: TrajContext | None,
    reason: str | None,
) -> None:

    if traj_logger is None:
        return
    try:
        fs = _fs_for(traj_logger)
        text = (reason or "unknown failure").strip()
        if len(text) > _MAX_FAILURE_REASON_CHARS:
            text = text[:_MAX_FAILURE_REASON_CHARS] + "…"
        await oss_io.write_text(
            _join(traj_logger.log_file_dir, "error.txt"),
            text + "\n",
            fs=fs,
        )
    except Exception as exc:
        logger.warning("Failed to log trajectory failure: %s", exc)
