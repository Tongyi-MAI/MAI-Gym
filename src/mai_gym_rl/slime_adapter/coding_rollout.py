from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from argparse import Namespace
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from slime.rollout.base_types import RolloutFnEvalOutput, RolloutFnTrainOutput
from slime.rollout.sglang_rollout import GenerateState
from slime.utils.http_utils import post
from slime.utils.types import Sample

from .helpers.coding_sandbox import extract_code_from_response
from .rewards.coding import compute_coding_reward

logger = logging.getLogger(__name__)

__all__ = ["generate_rollout"]


CODING_TIMEOUT_S = float(os.environ.get("CODING_TIMEOUT_S", "30"))
MAX_CONCURRENT_TESTS = int(os.environ.get("MAX_CONCURRENT_TESTS", "64"))

CODING_SYSTEM_PROMPT = os.environ.get(
    "CODING_SYSTEM_PROMPT",
    """\
You are an expert Python programmer. Solve the given coding problem \
by writing a correct and efficient Python solution.

Think through the problem step by step, then provide your solution \
inside a ```python``` code block.

Important:
- Write a complete, self-contained function as specified in the problem.
- Do NOT include test cases, input/output handling, or example usage.
- Make sure your solution handles all edge cases.""",
)


def _sample_metadata(sample: Sample) -> dict[str, Any]:
    meta = sample.metadata
    if isinstance(meta, dict):
        return meta
    if isinstance(meta, str):
        try:
            parsed = json.loads(meta)
            if isinstance(parsed, dict):
                return parsed
        except (json.JSONDecodeError, TypeError, ValueError):
            pass
    return {}


async def _generate_and_reward(
    sample: Sample,
    args: Namespace,
    state: GenerateState,
    *,
    evaluation: bool = False,
    test_sem: asyncio.Semaphore,
) -> Sample:

    meta = _sample_metadata(sample)
    task_id = meta.get("task_id", "unknown")
    test_cases = meta.get("test_cases", "")

    loop = asyncio.get_running_loop()

    prompt_text = sample.prompt if isinstance(sample.prompt, str) else str(sample.prompt)
    token_ids = await loop.run_in_executor(
        None,
        lambda: state.tokenizer.encode(prompt_text, add_special_tokens=False),
    )

    eval_temp = getattr(args, "eval_temperature", 0.0)
    sampling_params = {
        "max_new_tokens": args.rollout_max_response_len,
        "temperature": eval_temp if evaluation else args.rollout_temperature,
        "top_p": getattr(args, "rollout_top_p", 1.0),
    }

    url = f"http://{args.sglang_router_ip}:{args.sglang_router_port}/generate"

    try:
        output = await post(
            url,
            {
                "input_ids": token_ids,
                "sampling_params": sampling_params,
                "return_logprob": True,
            },
        )
    except Exception as exc:
        logger.warning("sglang error for %s: %s", task_id, exc)
        sample.status = Sample.Status.FAILED
        sample.reward = 0.0
        sample.remove_sample = True
        return sample

    output_ids = output["output_ids"]
    logprobs = [item[0] for item in output["meta_info"]["output_token_logprobs"]]
    response_text = await loop.run_in_executor(
        None,
        lambda: state.tokenizer.decode(output_ids, skip_special_tokens=True),
    )

    loop = asyncio.get_running_loop()
    reward = 0.0
    reward_info: dict[str, Any] = {}
    if test_cases:
        async with test_sem:
            reward, reward_info = await loop.run_in_executor(
                None,
                lambda: compute_coding_reward(
                    response_text,
                    test_cases,
                    timeout=CODING_TIMEOUT_S,
                ),
            )
    else:
        logger.warning("No test_cases for %s", task_id)

    solution_code = extract_code_from_response(response_text)
    format_reward = 1.0 if solution_code.strip() else 0.0

    sample.tokens = token_ids + output_ids
    sample.response = response_text
    sample.response_length = len(output_ids)
    sample.loss_mask = [1] * len(output_ids)
    sample.rollout_log_probs = logprobs
    sample.reward = reward
    sample.status = Sample.Status.COMPLETED
    sample.metadata = {
        **meta,
        "trajectory_reward": reward,
        "format_reward": format_reward,
        "n_steps": 1,
        "step_idx": 0,
        "exit_code": reward_info.get("exit_code"),
        "timed_out": reward_info.get("timed_out", False),
    }

    logger.info(
        "[CodingRollout] task=%s reward=%.1f fmt=%.1f resp_len=%d prompt_len=%d",
        task_id,
        reward,
        format_reward,
        len(output_ids),
        len(token_ids),
    )
    return sample


def _save_trajectories(results: list[Sample], rollout_id: int, evaluation: bool) -> None:

    output_dir = os.environ.get("OUTPUT_DIR", "")
    if not output_dir:
        return

    traj_dir = os.path.join(output_dir, "trajectories")
    os.makedirs(traj_dir, exist_ok=True)

    prefix = "eval" if evaluation else "train"
    path = os.path.join(traj_dir, f"{prefix}_rollout_{rollout_id:05d}.jsonl")

    try:
        with open(path, "w") as f:
            for s in results:
                meta = _sample_metadata(s)
                code = extract_code_from_response(s.response or "")
                entry = {
                    "task_id": meta.get("task_id", "unknown"),
                    "reward": s.reward,
                    "response": (s.response or "")[:8000],
                    "extracted_code": code[:4000],
                    "prompt_len": len(s.tokens or []) - (s.response_length or 0),
                    "response_len": s.response_length,
                    "exit_code": meta.get("exit_code"),
                    "timed_out": meta.get("timed_out", False),
                }
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        logger.info("[CodingRollout] Saved %d trajectories to %s", len(results), path)
    except Exception as exc:
        logger.warning("[CodingRollout] Failed to save trajectories: %s", exc)


def generate_rollout(
    args: Namespace,
    rollout_id: int,
    data_source: Any,
    evaluation: bool = False,
) -> RolloutFnTrainOutput | RolloutFnEvalOutput:

    _skip = getattr(args, "skip_initial_eval", False) or os.environ.get("SKIP_INITIAL_EVAL") == "1"
    if evaluation and rollout_id == 0 and _skip:
        logger.info("Skipping initial eval (rollout_id=0)")
        return RolloutFnEvalOutput(data={})

    if evaluation:
        batch_size = getattr(args, "rl_prompts_per_eval", 32)
    else:
        batch_size = getattr(args, "rollout_batch_size", 32)

    task_groups = data_source.get_samples(batch_size)
    task_samples = [s for group in task_groups for s in group]

    logger.info(
        "Coding rollout %d: %d samples, eval=%s",
        rollout_id,
        len(task_samples),
        evaluation,
    )
    t0 = time.time()

    pool_size = int(os.environ.get("ROLLOUT_THREAD_POOL_SIZE", "128"))

    async def _run_all() -> list[Sample]:
        loop = asyncio.get_running_loop()
        loop.set_default_executor(ThreadPoolExecutor(max_workers=pool_size))

        state = GenerateState(args)

        logger.info("[CodingRollout] tokenizer=%s", type(state.tokenizer).__name__)

        test_sem = asyncio.Semaphore(MAX_CONCURRENT_TESTS)
        return await asyncio.gather(
            *[
                _generate_and_reward(
                    s,
                    args,
                    state,
                    evaluation=evaluation,
                    test_sem=test_sem,
                )
                for s in task_samples
            ]
        )

    loop = asyncio.new_event_loop()
    try:
        results = loop.run_until_complete(_run_all())
    finally:
        loop.close()

    elapsed = time.time() - t0
    valid = [s for s in results if s.status == Sample.Status.COMPLETED]
    rewards = [s.reward for s in valid if s.reward is not None]
    n_pass = sum(1 for r in rewards if r > 0)
    mean_r = sum(rewards) / max(len(rewards), 1)

    logger.info(
        "Coding rollout %d done: %d/%d valid, %d pass (%.1f%%), mean_reward=%.4f, %.1fs",
        rollout_id,
        len(valid),
        len(results),
        n_pass,
        100 * n_pass / max(len(rewards), 1),
        mean_r,
        elapsed,
    )

    _save_trajectories(results, rollout_id, evaluation)

    if evaluation:
        return RolloutFnEvalOutput(
            data={
                "coding": {
                    "rewards": [s.reward or 0.0 for s in results],
                    "truncated": [False] * len(results),
                    "samples": results,
                    "custom_metrics": {
                        "eval/coding/mean_reward": mean_r,
                        "eval/coding/pass_rate": n_pass / max(len(rewards), 1),
                        "eval/coding/n_pass": float(n_pass),
                        "eval/coding/n_total": float(len(valid)),
                    },
                },
            },
        )

    from collections import defaultdict

    groups: dict[str, list[Sample]] = defaultdict(list)
    for s in valid:
        if not s.remove_sample:
            tid = _sample_metadata(s).get("task_id", "unknown")
            groups[tid].append(s)

    samples_grouped: list[list[Sample]] = []
    n_zero_groups = 0
    n_kept = 0
    for _tid, g in groups.items():
        group_rewards = {s.reward for s in g}
        if len(group_rewards) <= 1:
            n_zero_groups += 1
            for s in g:
                s.remove_sample = True
        else:
            n_kept += len(g)
        samples_grouped.append(g)

    return RolloutFnTrainOutput(
        samples=samples_grouped,
        metrics={
            "rollout/mean_reward": mean_r,
            "rollout/pass_rate": n_pass / max(len(rewards), 1),
            "rollout/n_pass": float(n_pass),
            "rollout/n_valid": float(len(valid)),
            "rollout/n_zero_groups": float(n_zero_groups),
            "rollout/n_kept": float(n_kept),
            "rollout/n_groups": float(len(samples_grouped)),
            "rollout/elapsed_s": elapsed,
        },
    )
