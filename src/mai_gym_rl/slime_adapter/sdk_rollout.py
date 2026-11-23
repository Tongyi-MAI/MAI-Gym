import asyncio
import logging
import time

from slime.utils.types import Sample

from mai_gym.agents.generation import RecordingChatModel
from mai_gym.client.sdk_client import SDKClient
from mai_gym.runner.sdk_rollout import SDKRolloutAgent, sdk_conversation_type
from mai_gym_rl.slime_adapter.helpers.chat_model import CustomChatModel
from mai_gym_rl.slime_adapter.helpers.rollout_metrics import _sample_metadata
from mai_gym_rl.slime_adapter.sdk_samples import build_generation_samples

logger = logging.getLogger(__name__)


async def run_sdk_task(
    sample,
    task,
    config,
    client,
    args,
    *,
    concurrency_sem,
    state_factory,
    timing,
    evaluation=False,
    epoch_getter=None,
    timeout=1200,
):

    config.validate_sdk()
    sdk_conversation_type()
    await SDKClient.require_capability(client)
    if task.use_judge or task.evaluation_mode != "built_in":
        raise ValueError("SDK training v1 requires the MobileWorld built-in evaluator")
    start = time.monotonic()
    meta = _sample_metadata(sample)
    failure_log = {
        "train_id": task.train_id,
        "task_id": task.task_id,
        "attempt_id": task.attempt_id,
        "rollout_id": task.rollout_id,
        "evaluation": evaluation,
        "env_type": task.env_type,
    }
    try:
        async with concurrency_sem:
            timing.concurrency_wait = time.monotonic() - start
            acquire_start = time.monotonic()
            async with client.asession(
                target_apps=task.target_apps, timeout_s=10, failure_log=failure_log
            ) as sid:
                timing.env_acquire = time.monotonic() - acquire_start
                state = state_factory(args)
                model = RecordingChatModel(
                    CustomChatModel(
                        url=f"http://{args.sglang_router_ip}:{args.sglang_router_port}/generate",
                        tokenizer=state.tokenizer,
                        processor=state.processor,
                        model_type="llm" if args.multimodal_keys is None else "vlm",
                        tool_parser_name=None,
                        preserve_thinking=True,
                        routing_key=sid,
                        reasoning_end_token=config.sdk_reasoning_end_token,
                        max_context_tokens=getattr(args, "rollout_max_context_len", None),
                    ),
                    epoch_getter=epoch_getter,
                )
                sampling = {
                    "max_new_tokens": args.rollout_max_response_len,
                    "temperature": getattr(args, "eval_temperature", 0.0)
                    if evaluation
                    else args.rollout_temperature,
                    "top_p": getattr(args, "rollout_top_p", 1.0),
                    "top_k": getattr(args, "rollout_top_k", -1),
                }
                agent = SDKRolloutAgent(
                    client,
                    model,
                    sampling,
                    coordinate_space=config.sdk_coordinate_space,
                    device=config.sdk_device,
                    image_retention=config.image_retention,
                    timing=timing,
                )
                async with asyncio.timeout(timeout if timeout > 0 else None):
                    trajectory = await agent.run(sid, task)
        timing.total = time.monotonic() - start
        timing.n_steps = len(model.records)
        common = {
            **meta,
            "sdk_runtime": agent.runtime,
            "attempt_id": task.attempt_id,
            "task_id": task.task_id,
            "n_steps": len(model.records),
            "trajectory_reward": trajectory.reward,
            "rollout_agent": "sdk",
            "env_failed": False,
        }
        if evaluation:
            sample.reward = trajectory.reward
            sample.status = Sample.Status.COMPLETED
            sample.metadata = {**common, "timing": timing.to_dict()}
            return [sample]
        samples = build_generation_samples(
            model.records, trajectory, config, metadata=common, processor=state.processor
        )
        if samples:
            samples[0].metadata["timing"] = timing.to_dict()
            return samples
        sample.metadata = {**common, "filtered_reason": "all_generations_exceed_max_total_length"}
    except asyncio.CancelledError:
        raise
    except Exception as exc:
        logger.exception("SDK rollout failed for task %s", task.task_id)
        sample.metadata = {
            **meta,
            "task_id": task.task_id,
            "rollout_agent": "sdk",
            "env_failed": True,
            "failure_reason": f"{type(exc).__name__}: {exc}"[:2000],
        }
    sample.status = Sample.Status.FAILED
    sample.reward = 0.0
    sample.remove_sample = True
    return [sample]
