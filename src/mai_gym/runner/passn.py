from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field

from joblib import Parallel, delayed

from mai_gym.client.env_client import EnvClient
from mai_gym.models.config import MobileGymConfig
from mai_gym.models.evaluation import task_succeeded
from mai_gym.runner.batch import (
    RolloutRunResult,
    RolloutSetupError,
    _ensure_capacity_with_wait,
    _resolve_run_tasks,
    _run_task,
    _sampling_params,
    _scan_finished_tasks,
    run_batch_rollout,
)
from mai_gym.server.trajectory import model_slug


@dataclass
class TaskPassStats:
    task_id: str
    scores: list[float | None]
    pass_count: int
    scored_count: int
    env_failed_count: int
    pass_rate: float
    passed_at_least_once: bool
    best_score: float = 0.0
    avg_score: float = 0.0


@dataclass
class PassNReport:
    n_runs: int
    n_tasks: int
    per_run_sr: list[float]
    per_run_passed: list[int]
    per_run_scored: list[int]
    per_run_elapsed_s: list[float]
    avg_sr: float
    pass_at_n: float
    total_elapsed_s: float
    buckets: dict[int, list[str]]
    per_task: list[TaskPassStats] = field(default_factory=list)
    model_name: str | None = None
    train_id: str | None = None

    concurrent: bool = False

    retry_rounds: int = 0
    unscored_after_retry: list[str] = field(default_factory=list)
    per_run_avg_score: list[float] = field(default_factory=list)
    avg_score: float = 0.0
    score_at_n: float = 0.0
    pass_all_n: float = 0.0
    n_complete_tasks: int = 0


def _run_sr(res: RolloutRunResult) -> tuple[int, int, float, float]:
    scored = [s for s in res.scores.values() if s is not None]
    success = sum(1 for s in scored if task_succeeded(s))
    sr = success / len(scored) if scored else 0.0
    avg = sum(scored) / len(scored) if scored else 0.0
    return success, len(scored), sr, avg


def aggregate_passn(
    per_run: list[RolloutRunResult],
    *,
    model_name: str | None = None,
    train_id: str | None = None,
    total_elapsed_s: float | None = None,
    concurrent: bool = False,
) -> PassNReport:
    n_runs = len(per_run)

    task_ids: list[str] = []
    seen: set[str] = set()
    for res in per_run:
        for tid in res.scores:
            if tid not in seen:
                seen.add(tid)
                task_ids.append(tid)

    per_run_tuples = [_run_sr(res) for res in per_run]
    per_run_sr = [t[2] for t in per_run_tuples]
    per_run_passed = [t[0] for t in per_run_tuples]
    per_run_scored = [t[1] for t in per_run_tuples]
    per_run_avg_score = [t[3] for t in per_run_tuples]

    per_task: list[TaskPassStats] = []
    buckets: dict[int, list[str]] = {}
    n_tasks_scored = 0
    n_tasks_passed = 0
    best_score_total = 0.0
    for tid in task_ids:
        scores = [res.scores.get(tid) for res in per_run]
        scored = [s for s in scores if s is not None]
        pass_count = sum(1 for s in scored if task_succeeded(s))
        scored_count = len(scored)
        env_failed_count = len(scores) - scored_count
        pass_rate = pass_count / scored_count if scored_count else 0.0
        passed = pass_count > 0
        best_score = max(scored) if scored else 0.0
        avg_task_score = sum(scored) / scored_count if scored_count else 0.0
        per_task.append(
            TaskPassStats(
                task_id=tid,
                scores=scores,
                pass_count=pass_count,
                scored_count=scored_count,
                env_failed_count=env_failed_count,
                pass_rate=pass_rate,
                passed_at_least_once=passed,
                best_score=best_score,
                avg_score=avg_task_score,
            )
        )
        buckets.setdefault(pass_count, []).append(tid)
        if scored_count > 0:
            n_tasks_scored += 1
            best_score_total += best_score
            if passed:
                n_tasks_passed += 1

    avg_sr = sum(per_run_sr) / len(per_run_sr) if per_run_sr else 0.0
    pass_at_n = n_tasks_passed / n_tasks_scored if n_tasks_scored else 0.0
    avg_score = sum(per_run_avg_score) / len(per_run_avg_score) if per_run_avg_score else 0.0
    score_at_n = best_score_total / n_tasks_scored if n_tasks_scored else 0.0
    total = (
        total_elapsed_s if total_elapsed_s is not None else sum(res.elapsed_s for res in per_run)
    )

    return PassNReport(
        n_runs=n_runs,
        n_tasks=len(task_ids),
        per_run_sr=per_run_sr,
        per_run_passed=per_run_passed,
        per_run_scored=per_run_scored,
        per_run_elapsed_s=[res.elapsed_s for res in per_run],
        avg_sr=avg_sr,
        pass_at_n=pass_at_n,
        per_run_avg_score=per_run_avg_score,
        avg_score=avg_score,
        score_at_n=score_at_n,
        total_elapsed_s=total,
        buckets=buckets,
        pass_all_n=(
            sum(t.pass_count == n_runs for t in per_task if t.scored_count == n_runs)
            / sum(t.scored_count == n_runs for t in per_task)
            if n_runs and any(t.scored_count == n_runs for t in per_task)
            else 0.0
        ),
        n_complete_tasks=sum(t.scored_count == n_runs for t in per_task) if n_runs else 0,
        per_task=per_task,
        model_name=model_name,
        train_id=train_id,
        concurrent=concurrent,
    )


def _print_passn_summary(report: PassNReport) -> None:
    n = report.n_runs
    print(f"\n{'=' * 60}")
    print(f"pass@{n} SUMMARY")
    print(f"{'=' * 60}")
    mode = "concurrent" if report.concurrent else "sequential"
    print(
        f"Tasks: {report.n_tasks} | Runs: {n} | Total wall-clock "
        f"({mode}): {report.total_elapsed_s:.1f}s"
    )
    for i, sr in enumerate(report.per_run_sr):
        p, s = report.per_run_passed[i], report.per_run_scored[i]
        avg = report.per_run_avg_score[i]
        if report.concurrent:
            print(f"  Run {i + 1}: avg_score={avg:.1%}  success_rate={sr:.1%} ({p}/{s})")
        else:
            print(
                f"  Run {i + 1}: avg_score={avg:.1%}  success_rate={sr:.1%} ({p}/{s})"
                f"  ({report.per_run_elapsed_s[i]:.1f}s)"
            )
    print(f"Average score (partial credit): {report.avg_score:.1%}")
    print(f"score@{n} (partial credit): {report.score_at_n:.1%}")
    print(f"Average success rate: {report.avg_sr:.1%}")
    print(f"pass@{n}: {report.pass_at_n:.1%}")
    print(
        f"pass^{n} (all trials; {report.n_complete_tasks} complete tasks): {report.pass_all_n:.1%}"
    )
    print("Pass-count distribution:")
    for k in sorted(report.buckets, reverse=True):
        print(f"  passed {k}/{n}: {len(report.buckets[k])} task(s)")


def _render_summary_md(report: PassNReport) -> str:
    n = report.n_runs
    mode = "concurrent" if report.concurrent else "sequential"
    lines: list[str] = [
        f"# pass@{n} report",
        "",
        f"- Model: `{report.model_name}`",
        f"- Run id: `{report.train_id}`",
        f"- Tasks: {report.n_tasks}",
        f"- Runs: {n}",
        f"- Total wall-clock ({mode}): {report.total_elapsed_s:.1f}s",
        f"- **Average score (partial credit): {report.avg_score:.1%}**",
        f"- **score@{n} (partial credit): {report.score_at_n:.1%}**",
        f"- Average success rate: {report.avg_sr:.1%}",
        f"- pass@{n}: {report.pass_at_n:.1%}",
        f"- pass^{n} (all trials): {report.pass_all_n:.1%}",
        f"- Tasks with all trials scored: {report.n_complete_tasks}",
        "",
        "## Per-run",
        "",
    ]
    if report.concurrent:
        lines += ["| Run | Avg score | Success rate |", "| --- | --- | --- |"]
        for i, sr in enumerate(report.per_run_sr):
            lines.append(f"| {i + 1} | {report.per_run_avg_score[i]:.1%} | {sr:.1%} |")
    else:
        lines += [
            "| Run | Avg score | Success rate | Duration (s) |",
            "| --- | --- | --- | --- |",
        ]
        for i, sr in enumerate(report.per_run_sr):
            lines.append(
                f"| {i + 1} | {report.per_run_avg_score[i]:.1%} | {sr:.1%}"
                f" | {report.per_run_elapsed_s[i]:.1f} |"
            )
    lines += ["", "## Pass-count buckets", ""]
    for k in sorted(report.buckets, reverse=True):
        lines.append(f"### Passed {k}/{n}")
        lines.append("")
        lines += [f"- {tid}" for tid in sorted(report.buckets[k])]
        lines.append("")
    lines += [
        "## Per-task",
        "",
        f"| Task | Best score | Passed | Scores (run 1..{n}) |",
        "| --- | --- | --- | --- |",
    ]
    for t in report.per_task:
        scores_str = ", ".join("env-fail" if s is None else f"{s:.2f}" for s in t.scores)
        lines.append(
            f"| {t.task_id} | {t.best_score:.2f} | {t.pass_count}/{t.scored_count} | {scores_str} |"
        )
    lines.append("")
    return "\n".join(lines)


def write_passn_report(report: PassNReport, out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    payload = {
        "n_runs": report.n_runs,
        "n_tasks": report.n_tasks,
        "model_name": report.model_name,
        "train_id": report.train_id,
        "concurrent": report.concurrent,
        "retry_rounds": report.retry_rounds,
        "unscored_after_retry": report.unscored_after_retry,
        "n_unscored_after_retry": len(report.unscored_after_retry),
        "avg_success_rate": report.avg_sr,
        "pass_at_n": report.pass_at_n,
        "pass_all_n": report.pass_all_n,
        f"pass^{report.n_runs}": report.pass_all_n,
        "n_complete_tasks": report.n_complete_tasks,
        f"pass@{report.n_runs}": report.pass_at_n,
        "per_run_success_rate": report.per_run_sr,
        "avg_score": report.avg_score,
        "score_at_n": report.score_at_n,
        f"score@{report.n_runs}": report.score_at_n,
        "per_run_avg_score": report.per_run_avg_score,
        "per_run_elapsed_s": None if report.concurrent else report.per_run_elapsed_s,
        "total_elapsed_s": report.total_elapsed_s,
        "pass_count_distribution": {str(k): len(v) for k, v in sorted(report.buckets.items())},
        "per_task": [
            {
                "task_id": t.task_id,
                "scores": t.scores,
                "successes": [task_succeeded(s) if s is not None else None for s in t.scores],
                "pass_count": t.pass_count,
                "scored_count": t.scored_count,
                "env_failed_count": t.env_failed_count,
                "pass_rate": t.pass_rate,
                "passed_at_least_once": t.passed_at_least_once,
                "best_score": t.best_score,
                "avg_score": t.avg_score,
            }
            for t in report.per_task
        ],
    }
    with open(os.path.join(out_dir, "results.json"), "w") as f:
        json.dump(payload, f, indent=2)
    with open(os.path.join(out_dir, "summary.md"), "w") as f:
        f.write(_render_summary_md(report))


def _assemble_per_run(
    meta: list[tuple[int, int, str]],
    results: list[tuple[str, int, float | None] | None],
    finished_per_pass: list[dict[str, float]],
    pass_n: int,
) -> list[RolloutRunResult]:

    per_run: list[RolloutRunResult] = []
    scores: list[dict[str, float | None]] = [dict(finished_per_pass[p]) for p in range(pass_n)]
    n_steps: list[dict[str, int]] = [{} for _ in range(pass_n)]
    for (p, _s, _tid), r in zip(meta, results, strict=True):
        if r is None:
            continue
        task_id, steps, score = r
        scores[p][task_id] = score
        n_steps[p][task_id] = steps
    for p in range(pass_n):
        per_run.append(RolloutRunResult(scores=scores[p], n_steps=n_steps[p], elapsed_s=0.0))
    return per_run


def _accumulate_round(
    scores: list[dict[str, float | None]],
    n_steps: list[dict[str, int]],
    round_meta: list[tuple[int, int, str]],
    round_results: list[tuple[str, int, float | None] | None],
) -> list[tuple[int, int, str]]:

    for (p, _s, _tid), r in zip(round_meta, round_results, strict=True):
        if r is None:
            continue
        task_id, steps, score = r
        scores[p][task_id] = score
        n_steps[p][task_id] = steps
    return [(p, s, tid) for (p, s, tid) in round_meta if scores[p].get(tid) is None]


def _run_passn_concurrent(
    cfg: MobileGymConfig,
    *,
    server: str,
    num_envs: int,
    env_type: str,
    task_filter: str | None,
    no_resume: bool,
    tasks_dir: str | None,
    domains: set[str] | None,
    pass_n: int,
    retry_until_scored: bool = False,
    max_retry_rounds: int = 5,
) -> PassNReport:

    client = EnvClient(host=server)
    model = model_slug(cfg.agent.model_name if cfg.agent else None)
    run_train_id = cfg.rollout.train_id or f"eval_{model}"
    ready = _ensure_capacity_with_wait(
        client,
        num_envs,
        env_type,
        job_id=run_train_id,
        mode=cfg.rollout.fleet_request_mode,
    )
    print(f"Pool: {ready} READY session(s)")
    if ready <= 0:
        client.close()
        raise RolloutSetupError(
            "No sessions available — server pool empty and provisioning yielded none."
        )
    concurrency = min(num_envs, ready)

    try:
        task_list = _resolve_run_tasks(client, cfg, task_filter, env_type, tasks_dir, domains)
    except Exception:
        client.close()
        raise

    base_log_dir = cfg.manager.traj_log_dir or "./traj_logs"
    base_dir = f"{base_log_dir.rstrip('/')}/{model}"

    meta: list[tuple[int, int, str]] = []
    work: list[tuple[dict, int, int, str]] = []
    finished_per_pass: list[dict[str, float]] = []
    task_ids = [t["task_id"] for t in task_list]
    from mai_gym.models.run_config import build_cli_run_config

    client.register_run_config(
        run_train_id,
        build_cli_run_config(
            cfg,
            train_id=run_train_id,
            env_type=env_type,
            num_envs=num_envs,
            sampling_params=_sampling_params((cfg.agent.runtime_conf if cfg.agent else None) or {}),
            tasks=task_ids,
        ),
        model_name=cfg.agent.model_name if cfg.agent else None,
    )
    for p in range(pass_n):
        pass_dir = f"{base_dir}/pass{p + 1}"
        fin = {} if no_resume else _scan_finished_tasks(pass_dir, task_ids)
        finished_per_pass.append(fin)
        for s, td in enumerate(task_list):
            if td["task_id"] in fin:
                continue
            meta.append((p, s, td["task_id"]))
            work.append((td, s, p, pass_dir))

    total_grid = pass_n * len(task_list)
    resumed = sum(len(f) for f in finished_per_pass)
    print(f"Run id (dashboard group): {run_train_id}")
    if resumed:
        print(f"Resume: skipping {resumed} already-finished (pass, task) rollout(s)")
    print(
        f"Running {len(work)} of {total_grid} rollout(s) "
        f"({len(task_list)} task(s) × {pass_n} pass(es)) "
        f"concurrently at concurrency {concurrency}"
    )

    def _run_cells(cells: list[tuple[dict, int, int, str]]):
        return Parallel(n_jobs=concurrency, backend="threading")(
            delayed(_run_task)(
                td,
                idx=s,
                train_id=run_train_id,
                cfg=cfg,
                server=server,
                env_type=env_type,
                log_dir=pass_dir,
                attempt_rollout=p + retry_rounds * pass_n,
                fleet_request_id=client.fleet_request_id,
            )
            for (td, s, p, pass_dir) in cells
        )

    t0 = time.monotonic()

    retry_rounds = 0
    try:
        results0 = _run_cells(work)
        per_run0 = _assemble_per_run(meta, results0, finished_per_pass, pass_n)
        scores = [dict(r.scores) for r in per_run0]
        n_steps = [dict(r.n_steps) for r in per_run0]
        pending = [(p, s, tid) for (p, s, tid) in meta if scores[p].get(tid) is None]

        while (
            retry_until_scored
            and pending
            and (max_retry_rounds == 0 or retry_rounds < max_retry_rounds)
        ):
            retry_rounds += 1
            retry_meta: list[tuple[int, int, str]] = []
            retry_work: list[tuple[dict, int, int, str]] = []
            for p, s, tid in pending:
                retry_meta.append((p, s, tid))
                retry_work.append((task_list[s], s, p, f"{base_dir}/pass{p + 1}"))
            print(
                f"retry round {retry_rounds}"
                f"{'' if max_retry_rounds == 0 else f'/{max_retry_rounds}'}: "
                f"re-running {len(retry_work)} env-failed cell(s)"
            )
            pending = _accumulate_round(scores, n_steps, retry_meta, _run_cells(retry_work))
            scored_cells = sum(
                1 for p in range(pass_n) for tid in task_ids if scores[p].get(tid) is not None
            )
            print(
                f"  after round {retry_rounds}: {scored_cells} scored / "
                f"{len(pending)} still-env-failed cell(s)"
            )
    finally:
        client.close()

    total_elapsed = time.monotonic() - t0

    per_run = [
        RolloutRunResult(scores=scores[p], n_steps=n_steps[p], elapsed_s=0.0) for p in range(pass_n)
    ]

    unscored = [tid for tid in task_ids if all(scores[p].get(tid) is None for p in range(pass_n))]
    report = aggregate_passn(
        per_run,
        model_name=cfg.agent.model_name if cfg.agent else None,
        train_id=run_train_id,
        total_elapsed_s=total_elapsed,
        concurrent=True,
    )
    report.retry_rounds = retry_rounds
    report.unscored_after_retry = unscored
    _print_passn_summary(report)
    if retry_until_scored:
        scored_total = sum(
            1 for p in range(pass_n) for tid in task_ids if scores[p].get(tid) is not None
        )
        print(
            f"Retry: {retry_rounds} round(s) used | "
            f"{scored_total}/{total_grid} cell(s) scored | "
            f"{len(unscored)} task(s) still unscored across all passes"
            + (f": {unscored}" if unscored else "")
        )

    out_dir = f"{base_dir}/passn_report_{time.strftime('%Y%m%d_%H%M%S')}"
    write_passn_report(report, out_dir)
    print(f"\npass@{pass_n} report written to: {out_dir}")
    return report


def run_passn_rollout(
    cfg: MobileGymConfig,
    *,
    server: str,
    num_envs: int,
    env_type: str,
    task_filter: str | None,
    no_resume: bool,
    tasks_dir: str | None = None,
    domains: set[str] | None = None,
    pass_n: int = 1,
    retry_until_scored: bool = False,
    max_retry_rounds: int = 5,
) -> PassNReport:

    pass_n = max(1, pass_n)
    if pass_n == 1 and not retry_until_scored:
        res = run_batch_rollout(
            cfg,
            server=server,
            num_envs=num_envs,
            env_type=env_type,
            task_filter=task_filter,
            no_resume=no_resume,
            tasks_dir=tasks_dir,
            domains=domains,
            pass_index=0,
            pass_total=1,
        )
        return aggregate_passn(
            [res],
            model_name=cfg.agent.model_name if cfg.agent else None,
            train_id=res.train_id,
        )

    return _run_passn_concurrent(
        cfg,
        server=server,
        num_envs=num_envs,
        env_type=env_type,
        task_filter=task_filter,
        no_resume=no_resume,
        tasks_dir=tasks_dir,
        domains=domains,
        pass_n=pass_n,
        retry_until_scored=retry_until_scored,
        max_retry_rounds=max_retry_rounds,
    )
