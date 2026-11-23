from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from typing import Any

_SLUG_RE = re.compile(r"[^A-Za-z0-9._-]")


def task_seg_slug(task_id: str) -> str:

    return _SLUG_RE.sub("_", task_id) or "unknown_task"


def build_trained_manifest(
    *,
    rollout_id: int,
    evaluation: bool,
    advantage_estimator: str,
    degrade_level: int,
    num_groups: int,
    trainable_task_ids: Iterable[str],
    kept_task_ids: Iterable[str],
    task_rewards: Mapping[str, list[float]],
    num_trained_trajectories: int,
    min_samples_per_task: int = 8,
    created_at: float | None = None,
) -> dict[str, Any]:

    trainable = set(trainable_task_ids)
    kept = set(kept_task_ids)
    is_ppo = advantage_estimator == "ppo"

    tasks: list[dict[str, Any]] = []
    for tid in trainable | kept:
        rewards = list(task_rewards.get(tid, []))
        n_valid = len(rewards)
        n_pass = sum(1 for r in rewards if r > 0)
        pass_rate = round(n_pass / n_valid, 4) if n_valid else 0.0

        if is_ppo:
            strict = n_valid >= 1
        else:
            strict = n_valid >= min_samples_per_task and len(set(rewards)) > 1
        tasks.append(
            {
                "task_id": tid,
                "task_seg": task_seg_slug(tid),
                "pass_rate": pass_rate,
                "n_valid": n_valid,
                "n_pass": n_pass,
                "rewards": rewards,
                "kept": tid in kept,
                "degraded": not strict,
            }
        )

    tasks.sort(key=lambda t: (not t["kept"], t["task_id"]))

    return {
        "schema_version": 1,
        "train_id": None,
        "rollout_id": rollout_id,
        "evaluation": evaluation,
        "created_at": created_at,
        "advantage_estimator": advantage_estimator,
        "degrade_level": degrade_level,
        "num_groups": num_groups,
        "num_trainable_tasks": len(trainable),
        "num_kept_tasks": len(kept),
        "num_trained_trajectories": num_trained_trajectories,
        "tasks": tasks,
    }
