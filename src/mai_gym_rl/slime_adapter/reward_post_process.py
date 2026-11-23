from __future__ import annotations

import logging
import statistics
from typing import Any

logger = logging.getLogger(__name__)


_CENTERED_ESTIMATORS = ("grpo", "gspo", "reinforce_plus_plus_baseline")

_STD_ESTIMATORS = ("grpo", "gspo")


def _group_key(sample: Any, position: int) -> Any:

    gid = getattr(sample, "group_id", None)
    if gid is not None:
        return ("gid", gid)
    idx = getattr(sample, "index", None)
    if idx is not None:
        return ("idx", idx)
    return ("pos", position)


def _traj_key(sample: Any, position: int) -> Any:

    meta = getattr(sample, "metadata", None)
    if isinstance(meta, dict):
        uid = meta.get("traj_uid")
        if uid is not None:
            return uid
    return ("pos", position)


def grpo_normalize_by_group_id(args: Any, samples: list[Any]) -> tuple[list[float], list[float]]:

    raw_rewards: list[float] = [sample.get_reward_value(args) for sample in samples]

    estimator = getattr(args, "advantage_estimator", None)
    if estimator not in _CENTERED_ESTIMATORS or not getattr(args, "rewards_normalization", True):
        return raw_rewards, raw_rewards

    use_std = estimator in _STD_ESTIMATORS and getattr(args, "grpo_std_normalization", True)

    groups: dict[Any, list[int]] = {}
    for pos, sample in enumerate(samples):
        groups.setdefault(_group_key(sample, pos), []).append(pos)

    out: list[float] = [0.0] * len(samples)

    written = bytearray(len(samples))
    n_singleton = 0
    n_zero_var = 0
    n_traj_total = 0
    for positions in groups.values():
        trajectories: dict[Any, list[int]] = {}
        for pos in positions:
            trajectories.setdefault(_traj_key(samples[pos], pos), []).append(pos)
        n_traj_total += len(trajectories)

        traj_rewards = [
            statistics.fmean([float(raw_rewards[p]) for p in ps]) for ps in trajectories.values()
        ]
        mean = statistics.fmean(traj_rewards)
        centered = [r - mean for r in traj_rewards]
        if len(traj_rewards) < 2:
            n_singleton += 1
        else:
            if len(set(traj_rewards)) == 1:
                n_zero_var += 1
            if use_std:
                denom = statistics.stdev(centered) + 1e-6
                centered = [c / denom for c in centered]

        for ps, value in zip(trajectories.values(), centered, strict=True):
            for pos in ps:
                out[pos] = value
                written[pos] += 1

    if any(count != 1 for count in written):
        missed = [i for i, c in enumerate(written) if c != 1]
        raise RuntimeError(
            f"[grpo-norm] {len(missed)} of {len(samples)} samples were not "
            f"assigned exactly one advantage (first offenders: {missed[:5]}). "
            "The group/trajectory bucketing no longer partitions the batch; "
            "refusing to return a partly-zeroed advantage vector."
        )

    n_fallback = sum(1 for key in groups if key[0] != "gid")
    n_no_traj = sum(1 for s in samples if _traj_key(s, -1) == ("pos", -1))
    logger.info(
        "[grpo-norm] per-group normalization: %d samples / %d trajectories in "
        "%d groups (zero-variance=%d, single-traj=%d, non-group_id keys=%d, "
        "no-traj_uid samples=%d, std_norm=%s)",
        len(samples),
        n_traj_total,
        len(groups),
        n_zero_var,
        n_singleton,
        n_fallback,
        n_no_traj,
        use_std,
    )
    if n_fallback == len(groups) and len(groups) > 1:
        logger.warning(
            "[grpo-norm] NO sample carried group_id — every sample became its own "
            "group, so all advantages are 0. The rollout is not annotating "
            "group_id; training will not learn from this batch."
        )
    return raw_rewards, out
