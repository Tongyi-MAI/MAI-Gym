import copy
import math

from slime.utils.types import Sample

from mai_gym.agents.generation import GenerationCaptureError


def build_generation_samples(records, trajectory, config, *, metadata=None, processor=None):
    samples = []
    for index, record in enumerate(records):
        result = record.result
        if (
            not result.prompt_token_ids
            or not result.output_ids
            or len(result.output_ids) != len(result.output_logprobs)
            or not all(math.isfinite(x) for x in result.output_logprobs)
        ):
            raise GenerationCaptureError("SDK training evidence is missing or misaligned")
        tokens = list(result.prompt_token_ids) + list(result.output_ids)
        if config.max_total_length > 0 and len(tokens) > config.max_total_length:
            continue
        has_images = any(
            part.get("type") == "image_url"
            for message in record.messages
            if isinstance(message.get("content"), list)
            for part in message["content"]
        )
        if has_images and processor is None:
            raise GenerationCaptureError(
                "SDK emitted images but the model has no multimodal processor"
            )
        samples.append(
            Sample(
                tokens=tokens,
                response=result.content,
                response_length=len(result.output_ids),
                loss_mask=[1] * len(result.output_ids),
                rollout_log_probs=list(result.output_logprobs),
                reward=trajectory.reward,
                status=Sample.Status.TRUNCATED
                if result.finish_reason == "length"
                else Sample.Status.COMPLETED,
                multimodal_train_inputs=result.multimodal_train_inputs,
                multimodal_inputs=(
                    {"messages": copy.deepcopy(record.messages)}
                    if has_images and result.multimodal_train_inputs is None
                    else None
                ),
                weight_versions=[result.weight_version]
                if result.weight_version is not None
                else [],
                metadata={
                    **(metadata or {}),
                    "task_id": trajectory.task_id,
                    "step_idx": index,
                    "n_steps": len(records),
                    "trajectory_reward": trajectory.reward,
                    "rollout_agent": "sdk",
                    "env_failed": False,
                    "format_reward": 0.0 if trajectory.steps[index].get("format_error") else 1.0,
                    "cached_tokens": result.cached_tokens,
                    "prompt_tokens": len(result.prompt_token_ids),
                    "gen_epoch": record.gen_epoch,
                    "sampling_params": copy.deepcopy(record.sampling_params),
                },
            )
        )
    return samples
