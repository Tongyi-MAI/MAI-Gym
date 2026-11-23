from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Any

Trajectory = list[Any]
GroupResult = list[Trajectory]


def is_task_trainable_ppo(rewards: list[float]) -> bool:

    return len(rewards) >= 1


def select_ppo_trajectories(
    group_results: Iterable[GroupResult | None],
    meta_fn: Callable[[Any], dict],
    *,
    max_train_trajectories: int = 0,
    shuffle_fn: Callable[[list], None] | None = None,
) -> list[Trajectory]:

    selected: list[Trajectory] = []
    for group_result in group_results:
        if not group_result:
            continue
        for sibling_steps in group_result:
            if sibling_steps and not meta_fn(sibling_steps[0]).get("env_failed"):
                selected.append(sibling_steps)
    if shuffle_fn is not None:
        shuffle_fn(selected)
    if max_train_trajectories > 0:
        selected = selected[:max_train_trajectories]
    return selected
