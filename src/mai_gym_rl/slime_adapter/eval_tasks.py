from __future__ import annotations

import json
import logging
from typing import Any

logger = logging.getLogger(__name__)


_eval_tasks_cache: dict[str, list[Any]] = {}


def sample_metadata(sample: Any) -> dict[str, Any]:

    meta = getattr(sample, "metadata", None)
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


def dedup_tasks_by_id(samples: list[Any]) -> list[Any]:

    seen: dict[str, Any] = {}
    for s in samples:
        key = str(sample_metadata(s).get("task_id") or id(s))
        if key not in seen:
            seen[key] = s
    return list(seen.values())


def load_eval_tasks(args: Any) -> list[Any]:

    eval_datasets = getattr(args, "eval_datasets", None) or []
    if not eval_datasets:
        return []
    cfg = eval_datasets[0]
    cache_key = cfg.path
    cached = _eval_tasks_cache.get(cache_key)
    if cached is not None:
        return cached

    from slime.utils.data import Dataset
    from slime.utils.processing_utils import load_processor, load_tokenizer

    tokenizer = load_tokenizer(args.hf_checkpoint, trust_remote_code=True)
    processor = load_processor(args.hf_checkpoint, trust_remote_code=True)
    max_length = (
        getattr(args, "eval_max_prompt_len", None)
        or getattr(args, "rollout_max_context_len", None)
        or 16384
    )
    dataset = Dataset(
        path=cfg.path,
        tokenizer=tokenizer,
        processor=processor,
        max_length=max_length,
        prompt_key=getattr(cfg, "input_key", None) or getattr(args, "input_key", "prompt"),
        label_key=getattr(cfg, "label_key", None),
        tool_key=getattr(cfg, "tool_key", None),
        metadata_key=(
            getattr(cfg, "metadata_key", None) or getattr(args, "metadata_key", "metadata")
        ),
        multimodal_keys=getattr(args, "multimodal_keys", None),
        apply_chat_template=getattr(args, "apply_chat_template", False),
        apply_chat_template_kwargs=getattr(args, "apply_chat_template_kwargs", None),
    )
    tasks = dedup_tasks_by_id(list(dataset.samples))
    _eval_tasks_cache[cache_key] = tasks
    logger.info("Loaded %d unique eval tasks from %s", len(tasks), cfg.path)
    return tasks
