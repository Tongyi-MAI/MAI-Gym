from __future__ import annotations

import copy
import logging
import os
from typing import Any

from slime.utils.processing_utils import process_vision_info
from slime.utils.types import Sample

from mai_gym.agents.slime_policy import StepRecord
from mai_gym.agents.window_utils import (
    WindowPackError,
    collapse_to_window,
    pack_windows,
    window_start,
)
from mai_gym.runner.base import Trajectory
from mai_gym.utils.images import to_data_url
from mai_gym.utils.messages import hide_history_images as _hide_history_images

from .config import SlimeAdapterConfig

logger = logging.getLogger(__name__)


def build_step_samples(
    trajectory: Trajectory,
    step_records: list[StepRecord],
    original_prompt: list[dict[str, Any]],
    config: SlimeAdapterConfig,
    tokenizer: Any,
    processor: Any | None = None,
    *,
    system_prompt: str | None = None,
    instruction: str | None = None,
    defer_multimodal: bool = False,
    window_size: int | None = None,
    overlap: int = 1,
) -> list[Sample]:

    n_steps = len(step_records)
    if n_steps == 0:
        return []

    final_reward = trajectory.reward

    if system_prompt is not None or instruction is not None:
        initial_messages: list[dict[str, Any]] = []
        if system_prompt:
            initial_messages.append({"role": "system", "content": system_prompt})
        if instruction:
            initial_messages.append(
                {
                    "role": "user",
                    "content": [{"type": "text", "text": instruction}],
                }
            )
    else:
        initial_messages = list(original_prompt)

    samples: list[Sample] = []
    messages_so_far: list[dict[str, Any]] = list(initial_messages)

    for k, record in enumerate(step_records):
        if record.observation_screenshot:
            user_msg: dict[str, Any] = {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": to_data_url(record.observation_screenshot),
                    }
                ],
            }
            messages_so_far.append(user_msg)

        rollout_messages = getattr(record, "rollout_messages", None)
        if rollout_messages is not None:
            pruned_messages = copy.deepcopy(rollout_messages)
        elif window_size is not None:
            pruned_messages = collapse_to_window(
                messages_so_far,
                step_idx=k,
                window_size=window_size,
                overlap=overlap,
            )
        else:
            pruned_messages = _hide_history_images(messages_so_far, config.image_retention)

        if k < 3:
            logger.info(
                "[StepBuilder.DEBUG] task=%s step=0: has_prompt_ids=%s (len=%d), "
                "has_mm_inputs=%s (type=%s, keys=%s)",
                trajectory.task_id,
                bool(record.prompt_token_ids),
                len(record.prompt_token_ids) if record.prompt_token_ids else 0,
                record.multimodal_train_inputs is not None,
                type(record.multimodal_train_inputs).__name__,
                list(record.multimodal_train_inputs.keys())
                if isinstance(record.multimodal_train_inputs, dict)
                else "N/A",
            )

        if defer_multimodal:
            if record.prompt_token_ids:
                prompt_token_ids = record.prompt_token_ids
            else:
                prompt_token_ids, _ = _tokenize_messages(pruned_messages, tokenizer, processor)
            multimodal_train_inputs = None
        elif record.prompt_token_ids and record.multimodal_train_inputs is not None:
            prompt_token_ids = record.prompt_token_ids
            multimodal_train_inputs = record.multimodal_train_inputs
            if k < 3:
                logger.info(
                    "[StepBuilder] task=%s step=%d: REUSING inference data", trajectory.task_id, k
                )
        elif record.prompt_token_ids:
            prompt_token_ids = record.prompt_token_ids
            _, multimodal_train_inputs = _tokenize_messages(
                pruned_messages,
                tokenizer,
                processor,
            )
        else:
            prompt_token_ids, multimodal_train_inputs = _tokenize_messages(
                pruned_messages,
                tokenizer,
                processor,
            )

        full_tokens = prompt_token_ids + record.response_token_ids

        if k < 3:
            msg_summary = []
            for m in pruned_messages:
                role = m["role"]
                content = m["content"]
                if isinstance(content, str):
                    msg_summary.append(f"{role}:str({len(content)})")
                elif isinstance(content, list):
                    parts = []
                    for c in content:
                        ctype = c.get("type", "?")
                        if ctype == "text":
                            parts.append(f"text({len(c.get('text', ''))})")
                        elif ctype == "image_url":
                            url = c.get("image_url", "")
                            parts.append(f"img({len(url)})")
                    msg_summary.append(f"{role}:[{','.join(parts)}]")
            logger.info(
                "[StepBuilder.TRAINING] task=%s step=%d: prompt_tokens=%d, "
                "response_tokens=%d, total=%d, n_messages=%d, msgs=[%s]",
                trajectory.task_id,
                k,
                len(prompt_token_ids),
                len(record.response_token_ids),
                len(full_tokens),
                len(pruned_messages),
                " | ".join(msg_summary),
            )
            logger.info(
                "[StepBuilder.TRAINING] task=%s step=%d: first_5_ids=%s, last_5_prompt_ids=%s",
                trajectory.task_id,
                k,
                prompt_token_ids[:5],
                prompt_token_ids[-5:],
            )

        if config.max_total_length > 0 and len(full_tokens) > config.max_total_length:
            logger.info(
                "Skipping step %d/%d for task %s: %d tokens > max_total_length %d",
                k,
                n_steps,
                trajectory.task_id,
                len(full_tokens),
                config.max_total_length,
            )

            messages_so_far.append({"role": "assistant", "content": record.response_text})
            continue

        if not record.response_token_ids:
            logger.warning(
                "Skipping step %d/%d for task %s: empty response",
                k,
                n_steps,
                trajectory.task_id,
            )
            messages_so_far.append({"role": "assistant", "content": record.response_text})
            continue

        sample = Sample(
            tokens=full_tokens,
            response=record.response_text,
            response_length=len(record.response_token_ids),
            loss_mask=[1] * len(record.response_token_ids),
            rollout_log_probs=record.response_logprobs,
            reward=final_reward,
            status=Sample.Status.COMPLETED,
            multimodal_train_inputs=multimodal_train_inputs,
            multimodal_inputs=(
                {"messages": pruned_messages}
                if (defer_multimodal and processor is not None)
                else None
            ),
            metadata={
                "task_id": trajectory.task_id,
                "step_idx": k,
                "n_steps": n_steps,
                "trajectory_reward": final_reward,
                "format_reward": record.format_reward,
                "cached_tokens": int(getattr(record, "cached_tokens", 0)),
                "prompt_tokens": int(getattr(record, "prompt_tokens", 0) or len(prompt_token_ids)),
                "gen_epoch": getattr(record, "gen_epoch", None),
            },
        )
        samples.append(sample)

        messages_so_far.append({"role": "assistant", "content": record.response_text})

    return samples


def _log_window_drift(
    task_id: str | None,
    step_records: list[StepRecord],
    tokenizer: Any,
) -> None:

    prompts = [r.prompt_token_ids for r in step_records]
    resps = [r.response_token_ids for r in step_records]
    n = len(step_records)
    pairs = prefix_ok = carried = 0
    total_tok_diff = 0
    for k in range(n - 1):
        pk, pk1, rk = prompts[k], prompts[k + 1], resps[k]
        if not pk or not pk1 or not rk:
            continue
        pairs += 1
        if pk1[: len(pk)] != pk:
            logger.info("[DRIFT] task=%s step=%d: PROMPT PREFIX BROKEN", task_id, k)
            continue
        prefix_ok += 1
        seg = pk1[len(pk) : len(pk) + len(rk)]
        if seg == rk:
            carried += 1
            continue

        m = min(len(seg), len(rk))
        d = next((i for i in range(m) if seg[i] != rk[i]), m)
        ndiff = sum(1 for a, b in zip(seg, rk, strict=False) if a != b) + abs(len(seg) - len(rk))
        total_tok_diff += ndiff
        try:
            dec_r = tokenizer.decode(rk[max(0, d - 3) : d + 6])
            dec_s = tokenizer.decode(seg[max(0, d - 3) : d + 6])
        except Exception:
            dec_r = dec_s = "<decode-failed>"
        logger.info(
            "[DRIFT] task=%s step=%d: response_len=%d first_div=%d tok_diff=%d | "
            "output_ids[%d:%d]=%s reembed=%s | decode(out)=%r decode(reembed)=%r",
            task_id,
            k,
            len(rk),
            d,
            ndiff,
            max(0, d - 3),
            d + 6,
            rk[max(0, d - 3) : d + 6],
            seg[max(0, d - 3) : d + 6],
            dec_r,
            dec_s,
        )
    logger.info(
        "[DRIFT.SUMMARY] task=%s n_steps=%d pairs=%d prefix_ok=%d response_carried=%d "
        "(%.0f%%) mismatched=%d total_tok_diff=%d",
        task_id,
        n,
        pairs,
        prefix_ok,
        carried,
        100.0 * carried / pairs if pairs else 0.0,
        prefix_ok - carried,
        total_tok_diff,
    )


def build_window_samples(
    trajectory: Trajectory,
    step_records: list[StepRecord],
    original_prompt: list[dict[str, Any]],
    config: SlimeAdapterConfig,
    tokenizer: Any,
    processor: Any | None = None,
    *,
    system_prompt: str | None = None,
    instruction: str | None = None,
    defer_multimodal: bool = False,
    window_size: int = 5,
    overlap: int = 1,
) -> list[Sample]:

    n_steps = len(step_records)
    if n_steps == 0:
        return []

    missing = [k for k, r in enumerate(step_records) if not r.prompt_token_ids]
    if missing:
        logger.warning(
            "build_window_samples: steps %s lack prompt_token_ids — cannot pack "
            "window for task %s; returning []",
            missing[:5],
            trajectory.task_id,
        )
        return []

    if os.environ.get("DEBUG_WINDOW_DRIFT"):
        _log_window_drift(trajectory.task_id, step_records, tokenizer)

    final_reward = trajectory.reward
    try:
        packed = pack_windows(
            [r.prompt_token_ids for r in step_records],
            [r.response_token_ids for r in step_records],
            [r.response_logprobs for r in step_records],
            window_size,
            overlap,
        )
    except WindowPackError as exc:
        logger.warning(
            "build_window_samples: cannot pack task %s windowwise (%s); falling back to stepwise",
            trajectory.task_id,
            exc,
        )
        return build_step_samples(
            trajectory=trajectory,
            step_records=step_records,
            original_prompt=original_prompt,
            config=config,
            tokenizer=tokenizer,
            processor=processor,
            system_prompt=system_prompt,
            instruction=instruction,
            defer_multimodal=defer_multimodal,
            window_size=window_size,
            overlap=overlap,
        )

    if os.environ.get("DEBUG_WINDOW_SPEEDUP"):
        n = len(step_records)
        sw_tokens = sum(len(r.prompt_token_ids) + len(r.response_token_ids) for r in step_records)
        sw_imgs = sum(k - window_start(k, window_size, overlap) + 1 for k in range(n))
        ww_tokens = sum(len(p.tokens) for p in packed)
        ww_imgs = sum(
            p.last_step_index - window_start(p.step_indices[0], window_size, overlap) + 1
            for p in packed
        )
        logger.info(
            "[SPEEDUP] task=%s n_steps=%d | samples sw=%d ww=%d (%.2fx) | "
            "tokens sw=%d ww=%d (%.2fx) | img_enc sw=%d ww=%d (%.2fx)",
            trajectory.task_id,
            n,
            n,
            len(packed),
            n / max(len(packed), 1),
            sw_tokens,
            ww_tokens,
            sw_tokens / max(ww_tokens, 1),
            sw_imgs,
            ww_imgs,
            sw_imgs / max(ww_imgs, 1),
        )

    if system_prompt is not None or instruction is not None:
        initial_messages: list[dict[str, Any]] = []
        if system_prompt:
            initial_messages.append({"role": "system", "content": system_prompt})
        if instruction:
            initial_messages.append(
                {
                    "role": "user",
                    "content": [{"type": "text", "text": instruction}],
                }
            )
    else:
        initial_messages = list(original_prompt)

    msgs_at_step: list[list[dict[str, Any]]] = []
    messages_so_far: list[dict[str, Any]] = list(initial_messages)
    for record in step_records:
        if record.observation_screenshot:
            messages_so_far = messages_so_far + [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": to_data_url(record.observation_screenshot),
                        }
                    ],
                }
            ]
        msgs_at_step.append(messages_so_far)
        messages_so_far = messages_so_far + [{"role": "assistant", "content": record.response_text}]

    samples: list[Sample] = []
    for pw in packed:
        if config.max_total_length > 0 and len(pw.tokens) > config.max_total_length:
            logger.info(
                "Skipping window block steps=%s for task %s: %d tokens > max_total_length %d",
                pw.step_indices,
                trajectory.task_id,
                len(pw.tokens),
                config.max_total_length,
            )
            continue

        if pw.response_length == 0:
            logger.warning(
                "Skipping window block steps=%s for task %s: empty response region",
                pw.step_indices,
                trajectory.task_id,
            )
            continue

        last = pw.last_step_index
        if defer_multimodal and processor is not None:
            rollout_messages = getattr(step_records[last], "rollout_messages", None)
            if rollout_messages is not None:
                pruned = copy.deepcopy(rollout_messages)
            else:
                pruned = collapse_to_window(
                    msgs_at_step[last],
                    step_idx=last,
                    window_size=window_size,
                    overlap=overlap,
                )
            multimodal_inputs: dict[str, Any] | None = {"messages": pruned}
            multimodal_train_inputs = None
        else:
            multimodal_inputs = None

            multimodal_train_inputs = step_records[last].multimodal_train_inputs

        samples.append(
            Sample(
                tokens=pw.tokens,
                response=" ".join(step_records[k].response_text for k in pw.step_indices),
                response_length=pw.response_length,
                loss_mask=pw.loss_mask,
                rollout_log_probs=pw.rollout_log_probs,
                reward=final_reward,
                status=Sample.Status.COMPLETED,
                multimodal_train_inputs=multimodal_train_inputs,
                multimodal_inputs=multimodal_inputs,
                metadata={
                    "task_id": trajectory.task_id,
                    "step_indices": pw.step_indices,
                    "n_steps": n_steps,
                    "trajectory_reward": final_reward,
                    "windowwise": True,
                    "cached_tokens": sum(
                        int(getattr(step_records[k], "cached_tokens", 0)) for k in pw.step_indices
                    ),
                    "prompt_tokens": sum(
                        int(
                            getattr(step_records[k], "prompt_tokens", 0)
                            or len(step_records[k].prompt_token_ids)
                        )
                        for k in pw.step_indices
                    ),
                    "gen_epoch": getattr(step_records[last], "gen_epoch", None),
                },
            )
        )

    return samples


def materialize_multimodal_inputs(sample: Sample, tokenizer: Any, processor: Any | None) -> None:

    mi = getattr(sample, "multimodal_inputs", None)
    if mi is None or sample.multimodal_train_inputs is not None or processor is None:
        return
    messages = mi.get("messages")
    if not messages:
        sample.multimodal_inputs = None
        return

    img_proc = getattr(processor, "image_processor", None)
    if img_proc is not None:
        try:
            vi = process_vision_info(messages, processor)
            images = vi.get("images")
            mti = dict(img_proc(images=images, return_tensors="pt")) if images else None
            sample.multimodal_train_inputs = mti
            sample.multimodal_inputs = None
            return
        except Exception:
            logger.warning(
                "image-only materialize failed; falling back to full processor", exc_info=True
            )
    _, multimodal_train_inputs = _tokenize_messages(messages, tokenizer, processor)
    sample.multimodal_train_inputs = multimodal_train_inputs
    sample.multimodal_inputs = None


def _tokenize_messages(
    messages: list[dict[str, Any]],
    tokenizer: Any,
    processor: Any | None,
) -> tuple[list[int], dict[str, Any] | None]:

    text = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=False,
    )

    if processor is not None:
        from slime.utils.processing_utils import encode_image_for_rollout_engine

        multimodal_inputs = process_vision_info(messages, processor)
        image_inputs = multimodal_inputs.get("images")
        video_inputs = multimodal_inputs.get("videos")

        if image_inputs:
            image_inputs = [encode_image_for_rollout_engine(img) for img in image_inputs]

        batch_feature = processor(
            text=text,
            images=image_inputs,
            videos=video_inputs,
            return_tensors="pt",
            return_mm_token_type_ids=False,
        )
        token_ids = batch_feature["input_ids"][0].tolist()
        mm_data = {
            k: v
            for k, v in batch_feature.items()
            if k not in ("input_ids", "attention_mask", "mm_token_type_ids")
        } or None
        return token_ids, mm_data

    token_ids = tokenizer.encode(text)
    return token_ids, None
