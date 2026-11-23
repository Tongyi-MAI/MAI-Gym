# Modified by MAI-Gym contributors.

import os
import sys


def qwen35_debug_enabled() -> bool:
    return os.environ.get("QWEN35_DEBUG", "").strip().lower() in ("1", "true", "yes", "on")


def _rank() -> str:
    try:
        import torch.distributed as dist

        if dist.is_available() and dist.is_initialized():
            return str(dist.get_rank())
    except Exception:
        pass
    return "?"


def qdebug(tag: str, msg: str) -> None:
    if not qwen35_debug_enabled():
        return
    print(f"[QWEN35_DEBUG][rank={_rank()}][{tag}] {msg}", file=sys.stderr, flush=True)


def tstat(t) -> str:
    try:
        import torch

        if not isinstance(t, torch.Tensor):
            return f"<not-a-tensor:{type(t).__name__}>"
        parts = [
            f"shape={tuple(t.shape)}",
            f"dtype={str(t.dtype).replace('torch.', '')}",
            f"dev={t.device}",
            f"contig={t.is_contiguous()}",
        ]
        flat = t.detach().reshape(-1).float()
        n = flat.numel()
        if n:
            parts.append(f"norm={flat.norm().item():.4f}")
            parts.append(f"nan={bool(torch.isnan(flat).any().item())}")
            parts.append(f"inf={bool(torch.isinf(flat).any().item())}")
            head = flat[: min(3, n)].tolist()
            parts.append("head=[" + ",".join(f"{x:.5g}" for x in head) + "]")
        return " ".join(parts)
    except Exception as e:
        return f"<tstat-error:{type(e).__name__}:{e}>"
