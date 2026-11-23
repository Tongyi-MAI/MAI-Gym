from __future__ import annotations

import json
import logging
import statistics
from collections import Counter, defaultdict
from typing import Any

from slime.rollout.base_types import RolloutFnTrainOutput
from slime.utils.types import Sample

logger = logging.getLogger(__name__)


def _sample_metadata(sample: Sample) -> dict[str, Any]:

    meta = sample.metadata or {}
    if isinstance(meta, str):
        try:
            meta = json.loads(meta)
        except json.JSONDecodeError:
            meta = {}
    return meta if isinstance(meta, dict) else {}


def _percentile(values: list[float], p: float) -> float:

    if not values:
        return 0.0
    s = sorted(values)
    if len(s) == 1:
        return float(s[0])
    k = (len(s) - 1) * (p / 100.0)
    lo, hi = int(k), min(int(k) + 1, len(s) - 1)
    return float(s[lo] + (s[hi] - s[lo]) * (k - lo))


def _compute_rollout_metrics(
    trajectories: list[list[Sample]],
    *,
    prefix: str,
) -> dict[str, float]:

    metrics: dict[str, float] = {}
    if not trajectories:
        return metrics

    step_counts: list[int] = []
    response_lengths: list[int] = []
    traj_rewards: list[float] = []
    format_rewards: list[float] = []
    per_task_rewards: dict[str, list[float]] = defaultdict(list)
    per_task_steps: dict[str, list[int]] = defaultdict(list)
    task_id_counter: Counter[str] = Counter()

    for traj in trajectories:
        if not traj:
            continue
        n = len(traj)
        step_counts.append(n)
        meta_first = _sample_metadata(traj[0])
        task_id = str(meta_first.get("task_id") or "unknown")
        task_id_counter[task_id] += 1
        traj_reward = float(meta_first.get("trajectory_reward", 0.0) or 0.0)
        traj_rewards.append(traj_reward)
        per_task_rewards[task_id].append(traj_reward)
        per_task_steps[task_id].append(n)
        for s in traj:
            response_lengths.append(int(getattr(s, "response_length", 0) or 0))
            meta = _sample_metadata(s)
            fr = meta.get("format_reward")
            if fr is not None:
                format_rewards.append(float(fr))

    n_traj = len(traj_rewards)
    n_pass = sum(1 for r in traj_rewards if r > 0)

    metrics[f"{prefix}/num_trajectories"] = float(n_traj)
    metrics[f"{prefix}/num_step_samples"] = float(sum(step_counts))
    metrics[f"{prefix}/num_distinct_tasks"] = float(len(task_id_counter))

    metrics[f"{prefix}/pass_rate"] = n_pass / max(n_traj, 1)
    metrics[f"{prefix}/num_successes"] = float(n_pass)
    metrics[f"{prefix}/num_failures"] = float(n_traj - n_pass)
    metrics[f"{prefix}/mean_trajectory_reward"] = (
        statistics.fmean(traj_rewards) if traj_rewards else 0.0
    )

    if step_counts:
        metrics[f"{prefix}/steps/mean"] = statistics.fmean(step_counts)
        metrics[f"{prefix}/steps/median"] = float(statistics.median(step_counts))
        metrics[f"{prefix}/steps/min"] = float(min(step_counts))
        metrics[f"{prefix}/steps/max"] = float(max(step_counts))
        metrics[f"{prefix}/steps/p25"] = _percentile(
            [float(x) for x in step_counts],
            25,
        )
        metrics[f"{prefix}/steps/p75"] = _percentile(
            [float(x) for x in step_counts],
            75,
        )

    if response_lengths:
        rl = [float(x) for x in response_lengths]
        metrics[f"{prefix}/response_len/mean"] = statistics.fmean(rl)
        metrics[f"{prefix}/response_len/median"] = float(statistics.median(rl))
        metrics[f"{prefix}/response_len/p95"] = _percentile(rl, 95)
        metrics[f"{prefix}/response_len/max"] = float(max(rl))

    if format_rewards:
        metrics[f"{prefix}/format_reward/mean"] = statistics.fmean(format_rewards)
        metrics[f"{prefix}/format_reward/min"] = float(min(format_rewards))

    for task_id, rewards in per_task_rewards.items():
        n = len(rewards)
        passes = sum(1 for r in rewards if r > 0)
        metrics[f"{prefix}/per_task/{task_id}/pass_rate"] = passes / max(n, 1)
        metrics[f"{prefix}/per_task/{task_id}/count"] = float(n)
        steps = per_task_steps[task_id]
        if steps:
            metrics[f"{prefix}/per_task/{task_id}/steps_mean"] = statistics.fmean(
                steps,
            )

    return metrics


def annotate_training_groups(trajectories: list[list[Any]]) -> tuple[int, int]:

    tid_to_gid: dict[str, int] = {}
    n_stamped = 0
    for traj_uid, traj in enumerate(trajectories):
        for s in traj:
            tid = str(_sample_metadata(s).get("task_id") or "unknown")
            s.group_id = tid_to_gid.setdefault(tid, len(tid_to_gid))
            if isinstance(s.metadata, dict):
                s.metadata["traj_uid"] = traj_uid
                n_stamped += 1
    return len(tid_to_gid), n_stamped


def _build_train_output(
    all_step_samples: list[list[Sample]],
    task_count: int,
    rollout_id: int,
    *,
    label: str = "",
    elapsed: float | None = None,
) -> RolloutFnTrainOutput:

    output_samples: list[list[Sample]] = []
    for step_list in all_step_samples:
        if not step_list:
            continue
        cleaned = [
            s
            for s in step_list
            if not getattr(s, "remove_sample", False)
            and getattr(s, "rollout_log_probs", None) is not None
        ]
        if not cleaned:
            continue
        output_samples.append(cleaned)

    if not output_samples:
        logger.warning(
            "Rollout %d: 0 usable samples after cleaning — "
            "returning empty RolloutFnTrainOutput (training step will be skipped)",
            rollout_id,
        )
        return RolloutFnTrainOutput(
            samples=[],
            metrics={
                "rollout/num_step_samples": 0.0,
                "rollout/num_trajectories": 0.0,
                "rollout/pass_rate": 0.0,
                "rollout/empty_rollout": 1.0,
            },
        )

    _n_groups, _n_stamped = annotate_training_groups(output_samples)
    logger.info(
        "Rollout %d: annotated %d task groups / %d trajectories on %d samples",
        rollout_id,
        _n_groups,
        len(output_samples),
        _n_stamped,
    )

    n_trajectories = len(output_samples)
    total_step_reward = sum((s.reward or 0.0) for traj in output_samples for s in traj)

    metrics = _compute_rollout_metrics(output_samples, prefix="rollout")
    metrics["rollout/num_trajectories_provisioned"] = float(task_count)
    metrics["rollout/num_trajectories_dropped"] = float(task_count - n_trajectories)
    metrics["rollout/num_real_trajectories"] = float(n_trajectories)
    n_train_steps = int(metrics.get("rollout/num_step_samples", 0))
    metrics["rollout/mean_reward_per_step"] = total_step_reward / max(n_train_steps, 1)

    per_task: dict[str, tuple[int, int, list[int]]] = {}
    per_task_env_fail: dict[str, int] = {}
    n_traj_total = 0
    n_traj_pass = 0
    n_env_failed = 0
    n_step_total = 0
    n_step_format_ok = 0
    n_step_format_total = 0
    for traj in output_samples:
        if not traj:
            continue
        meta_first = _sample_metadata(traj[0])
        task_id = str(meta_first.get("task_id") or "unknown")
        if meta_first.get("env_failed"):
            n_env_failed += 1
            per_task_env_fail[task_id] = per_task_env_fail.get(task_id, 0) + 1
            continue
        n_traj_total += 1
        traj_reward = float(meta_first.get("trajectory_reward", 0.0) or 0.0)
        n_steps = int(meta_first.get("n_steps", len(traj)) or len(traj))
        passed = traj_reward > 0
        if passed:
            n_traj_pass += 1
        old_p, old_t, old_s = per_task.get(task_id, (0, 0, []))
        per_task[task_id] = (old_p + int(passed), old_t + 1, old_s + [n_steps])
        for s in traj:
            n_step_total += 1
            meta = _sample_metadata(s)
            fr = meta.get("format_reward")
            if fr is not None:
                n_step_format_total += 1
                n_step_format_ok += int(float(fr) > 0)

    traj_acc = n_traj_pass / max(n_traj_total, 1)

    metrics["critic/traj_acc"] = traj_acc
    metrics["critic/traj_pass"] = float(n_traj_pass)
    metrics["critic/traj_total"] = float(n_traj_total)
    metrics["critic/env_failed"] = float(n_env_failed)
    if n_step_format_total:
        metrics["critic/step_format_acc"] = n_step_format_ok / n_step_format_total
    metrics["critic/step_format_ok"] = float(n_step_format_ok)
    metrics["critic/step_format_total"] = float(n_step_format_total)
    metrics["critic/step_total"] = float(n_step_total)
    for task_id, (passes, total, _) in per_task.items():
        metrics[f"critic/per_task/{task_id}/acc"] = passes / max(total, 1)
        metrics[f"critic/per_task/{task_id}/pass"] = float(passes)
        metrics[f"critic/per_task/{task_id}/total"] = float(total)

    tag = f" ({label})" if label else ""
    elapsed_str = f", elapsed {elapsed:.1f}s" if elapsed is not None else ""

    all_steps = [s for _, _, steps in per_task.values() for s in steps]
    avg_steps_all = sum(all_steps) / max(len(all_steps), 1)
    all_task_ids = sorted(set(list(per_task.keys()) + list(per_task_env_fail.keys())))
    col_task = max(len("Task"), *(len(t) for t in all_task_ids)) + 2 if all_task_ids else 20
    header = f"{'Task':<{col_task}} {'Acc':>10} {'AccVal':>8} {'AvgSteps':>10} {'EnvFail':>8}"
    sep = "-" * len(header)
    format_summary = (
        f"{n_step_format_ok}/{n_step_format_total}" if n_step_format_total else "n/a (no labels)"
    )
    lines = [
        f"Rollout {rollout_id}{tag}: {n_train_steps} step samples"
        f", fmt_acc {format_summary}{elapsed_str}",
        sep,
        header,
        sep,
    ]
    for task_id in all_task_ids:
        passes, total, steps = per_task.get(task_id, (0, 0, []))
        avg_s = sum(steps) / max(len(steps), 1)
        acc_str = f"{passes}/{total}" if total > 0 else "-"
        task_acc = passes / max(total, 1) if total > 0 else 0.0
        ef = per_task_env_fail.get(task_id, 0)
        ef_str = str(ef) if ef > 0 else ""
        lines.append(
            f"{task_id:<{col_task}} {acc_str:>10} acc={task_acc:.2f} {avg_s:>10.1f} {ef_str:>8}"
        )
    lines.append(sep)
    total_acc_str = f"{n_traj_pass}/{n_traj_total}"
    ef_total_str = str(n_env_failed) if n_env_failed > 0 else ""
    lines.append(
        f"{'TOTAL':<{col_task}} {total_acc_str:>10} acc={traj_acc:.4f} {avg_steps_all:>10.1f} {ef_total_str:>8}"
    )
    lines.append(sep)
    if n_env_failed > 0:
        lines.append(f"⚠ {n_env_failed} trajectories excluded (env setup failed)")
    lines.append(f"[Rollout {rollout_id}] acc={traj_acc:.4f} ({n_traj_pass}/{n_traj_total})")
    logger.info("\n".join(lines))

    timing_sums: dict[str, float] = {}
    timing_count = 0
    for traj in output_samples:
        if not traj:
            continue
        meta = _sample_metadata(traj[0])
        t = meta.get("timing")
        if not isinstance(t, dict):
            continue
        timing_count += 1
        for k, v in t.items():
            timing_sums[k] = timing_sums.get(k, 0.0) + float(v)

    if timing_count > 0:
        for k, v in timing_sums.items():
            name = k.split("timing/", 1)[-1] if "timing/" in k else k
            metrics[f"perf/{name}/sum"] = v
            metrics[f"perf/{name}/mean"] = v / timing_count
        metrics["perf/timing_trajectories"] = float(timing_count)

        total_sum = timing_sums.get("timing/total", 1.0) or 1.0
        pct_phases = [
            "timing/concurrency_wait",
            "timing/env_session_acquire",
            "timing/env_reset",
            "timing/env_step_action_and_screenshot",
            "timing/env_eval",
            "timing/model_image_preprocess",
            "timing/model_tokenize",
            "timing/model_request_prepare",
            "timing/model_sglang_http_and_gpu",
            "timing/model_postprocess",
            "timing/overhead",
        ]
        for phase_key in pct_phases:
            phase_sum = timing_sums.get(phase_key, 0.0)
            name = phase_key.split("timing/", 1)[-1]
            metrics[f"perf/{name}/pct"] = phase_sum / total_sum * 100

        cpu_mm = timing_sums.get("timing/model_cpu_multimodal_total", 0.0)
        calls = timing_sums.get("timing/model_calls", 0.0)
        images = timing_sums.get("timing/images", 0.0)
        logger.info(
            "[perf-step] rollout=%d trajectories=%d work_s=%.1f "
            "cpu_mm=%.1f(%.1f%%) template=%.1f vision_extract=%.1f "
            "png_encode=%.1f hf_processor=%.1f fast_mm_plan=%.1f splice=%.1f misc=%.1f "
            "sglang=%.1f(%.1f%%) env_step=%.1f(%.1f%%) reset=%.1f(%.1f%%) "
            "model_calls=%.0f images=%.0f cpu_mm_per_call=%.3f "
            "hf_processor_per_call=%.3f sglang_per_call=%.3f",
            rollout_id,
            timing_count,
            total_sum,
            cpu_mm,
            cpu_mm / total_sum * 100,
            timing_sums.get("timing/model_chat_template", 0.0),
            timing_sums.get("timing/model_vision_extract", 0.0),
            timing_sums.get("timing/model_image_encode", 0.0),
            timing_sums.get("timing/model_hf_multimodal_processor", 0.0),
            timing_sums.get("timing/model_fast_mm_token_plan", 0.0),
            timing_sums.get("timing/model_token_native_splice", 0.0),
            timing_sums.get("timing/model_preprocess_unattributed", 0.0),
            timing_sums.get("timing/model_sglang_http_and_gpu", 0.0),
            timing_sums.get("timing/model_sglang_http_and_gpu", 0.0) / total_sum * 100,
            timing_sums.get("timing/env_step_action_and_screenshot", 0.0),
            timing_sums.get("timing/env_step_action_and_screenshot", 0.0) / total_sum * 100,
            timing_sums.get("timing/env_reset", 0.0),
            timing_sums.get("timing/env_reset", 0.0) / total_sum * 100,
            calls,
            images,
            cpu_mm / max(calls, 1.0),
            timing_sums.get("timing/model_hf_multimodal_processor", 0.0) / max(calls, 1.0),
            timing_sums.get("timing/model_sglang_http_and_gpu", 0.0) / max(calls, 1.0),
        )

    return RolloutFnTrainOutput(samples=output_samples, metrics=metrics)
