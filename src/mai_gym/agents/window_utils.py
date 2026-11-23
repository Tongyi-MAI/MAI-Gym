from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

HISTORY_IMAGE_PLACEHOLDER = "<image_collapsed>"


class WindowPackError(Exception):
    pass


def window_start(step_idx: int, window_size: int = 5, overlap: int = 1) -> int:

    stride = window_size - overlap
    if stride <= 0:
        raise ValueError(f"window_size - overlap must be > 0 (W={window_size}, OV={overlap})")
    return max(0, stride * ((step_idx - overlap) // stride))


def _is_image_turn(message: dict[str, Any]) -> bool:

    if message.get("role") != "user":
        return False
    content = message.get("content")
    if not isinstance(content, list):
        return False
    return any(isinstance(part, dict) and part.get("type") == "image_url" for part in content)


def collapse_history_images(
    messages: list[dict[str, Any]],
    retention: int,
) -> list[dict[str, Any]]:

    if retention < 0:
        raise ValueError(f"retention must be >= 0 (got {retention})")
    image_positions = [i for i, m in enumerate(messages) if _is_image_turn(m)]

    keep_from = max(0, len(image_positions) - retention) if retention > 0 else len(image_positions)
    collapse = set(image_positions[:keep_from])
    out: list[dict[str, Any]] = []
    for i, message in enumerate(messages):
        if i in collapse:
            out.append({"role": "user", "content": HISTORY_IMAGE_PLACEHOLDER})
        else:
            out.append(message)
    return out


def collapse_to_window(
    messages: list[dict[str, Any]],
    step_idx: int,
    window_size: int = 5,
    overlap: int = 1,
) -> list[dict[str, Any]]:

    start = window_start(step_idx, window_size, overlap)
    out: list[dict[str, Any]] = []
    img_ordinal = -1
    for message in messages:
        if _is_image_turn(message):
            img_ordinal += 1
            if img_ordinal < start:
                out.append({"role": "user", "content": HISTORY_IMAGE_PLACEHOLDER})
                continue
        out.append(message)
    return out


@dataclass
class PackedWindow:
    tokens: list[int]
    loss_mask: list[int]
    response_length: int
    rollout_log_probs: list[float]
    step_indices: list[int]
    last_step_index: int


def _window_blocks(n_steps: int, window_size: int, overlap: int) -> list[list[int]]:

    blocks: list[list[int]] = []
    for k in range(n_steps):
        ws = window_start(k, window_size, overlap)
        if blocks and window_start(blocks[-1][0], window_size, overlap) == ws:
            blocks[-1].append(k)
        else:
            blocks.append([k])
    return blocks


def pack_windows(
    prompt_token_ids: list[list[int]],
    response_token_ids: list[list[int]],
    response_logprobs: list[list[float]],
    window_size: int = 5,
    overlap: int = 1,
) -> list[PackedWindow]:

    n = len(prompt_token_ids)
    out: list[PackedWindow] = []
    for block in _window_blocks(n, window_size, overlap):
        opening = prompt_token_ids[block[0]]
        resp_region: list[int] = []
        loss_mask: list[int] = []
        rollout_lp: list[float] = []
        for idx, k in enumerate(block):
            out_ids = response_token_ids[k]
            resp_region += out_ids
            loss_mask += [1] * len(out_ids)
            rollout_lp += response_logprobs[k]
            if idx + 1 < len(block):
                cur_prompt = prompt_token_ids[k]
                nxt_prompt = prompt_token_ids[block[idx + 1]]
                cut = len(cur_prompt) + len(out_ids)
                if nxt_prompt[: len(cur_prompt)] != cur_prompt:
                    raise WindowPackError(
                        f"step {block[idx + 1]} prompt does not extend step {k} "
                        f"prompt (within-block prefix broken)"
                    )
                if nxt_prompt[len(cur_prompt) : cut] != out_ids:
                    raise WindowPackError(
                        f"step {block[idx + 1]} prompt does not carry step {k} "
                        f"output_ids verbatim (token-native history broken — is "
                        f"scheme D on?)"
                    )
                bridge = nxt_prompt[cut:]
                resp_region += bridge
                loss_mask += [0] * len(bridge)
                rollout_lp += [0.0] * len(bridge)
        out.append(
            PackedWindow(
                tokens=list(opening) + resp_region,
                loss_mask=loss_mask,
                response_length=len(resp_region),
                rollout_log_probs=rollout_lp,
                step_indices=list(block),
                last_step_index=block[-1],
            )
        )
    return out


def aggregate_cache_metrics(samples: list[Any]) -> dict[str, float]:

    def _flatten(items: Any):

        for it in items:
            if isinstance(it, (list, tuple)):
                yield from _flatten(it)
            else:
                yield it

    def _m(s: Any) -> dict:

        md = getattr(s, "metadata", None)
        if md is None and isinstance(s, dict):
            md = s.get("metadata")
        if isinstance(md, str):
            try:
                md = json.loads(md)
            except (ValueError, TypeError):
                md = None
        return md if isinstance(md, dict) else {}

    flat = list(_flatten(samples))
    cached = sum(int(_m(s).get("cached_tokens", 0)) for s in flat)
    prompt = sum(int(_m(s).get("prompt_tokens", 0)) for s in flat)
    return {
        "rollout/prefix_cache_hit_rate": (cached / prompt) if prompt else 0.0,
        "rollout/cached_tokens_total": float(cached),
    }
