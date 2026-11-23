from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from mai_gym.models.config import MobileGymConfig


SCHEMA_VERSION = 1


def build_cli_run_config(
    cfg: MobileGymConfig,
    *,
    train_id: str,
    env_type: str,
    num_envs: int,
    sampling_params: dict[str, Any],
    tasks: list[str] | None,
) -> dict[str, Any]:

    agent = cfg.agent
    judge = getattr(cfg, "judge", None)
    return {
        "schema_version": SCHEMA_VERSION,
        "source": "cli_rollout",
        "created_at": time.time(),
        "train_id": train_id,
        "model_name": agent.model_name if agent else None,
        "agent_variant": agent.type if agent else None,
        "sampling_params": dict(sampling_params),
        "pass_n": cfg.rollout.pass_n,
        "num_envs": num_envs,
        "env_type": env_type,
        "tasks": list(tasks) if tasks else None,
        "task_filter": cfg.rollout.tasks,
        "tasks_dir": cfg.rollout.tasks_dir,
        "evaluator_contract": {
            "fixed_verifier_patches_requested": cfg.rollout.use_fixed_verifier,
            "evaluation_mode": cfg.rollout.evaluation_mode,
        },
        "judge": judge.model_dump(exclude={"api_key"}) if judge else None,
        "benchmark": {
            "suite": cfg.rollout.benchmark.suite,
            "model": cfg.rollout.benchmark.model,
            "run_id": cfg.rollout.benchmark.run_id,
        },
    }


def build_slime_run_config(
    *,
    train_id: str,
    model_name: str | None,
    agent_variant: str | None,
    sampling_params: dict[str, Any],
    num_sessions: int,
    env_type: str,
    evaluation: bool,
) -> dict[str, Any]:

    return {
        "schema_version": SCHEMA_VERSION,
        "source": "slime_eval" if evaluation else "slime_train",
        "created_at": time.time(),
        "train_id": train_id,
        "model_name": model_name,
        "agent_variant": agent_variant,
        "sampling_params": dict(sampling_params),
        "num_envs": num_sessions,
        "env_type": env_type,
        "evaluation": evaluation,
    }
