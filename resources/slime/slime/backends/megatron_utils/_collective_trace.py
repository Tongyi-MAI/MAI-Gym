# Modified by MAI-Gym contributors.

import os
import sys
import time

_TRACE_ENABLED = os.environ.get("MAI_COLLECTIVE_TRACE", "0") == "1"


def _rank():
    try:
        import torch.distributed as dist

        if dist.is_available() and dist.is_initialized():
            return dist.get_rank()
    except Exception:
        pass
    return -1


def _say(msg: str) -> None:
    pid = os.getpid()
    rank = _rank()
    t = time.monotonic()
    print(f"[MAI COLL t={t:.3f} rank={rank} pid={pid}] {msg}", file=sys.stderr, flush=True)


def apply_collective_tracer() -> bool:
    if not _TRACE_ENABLED:
        return False

    from slime.utils import reloadable_process_group as _rpg

    if getattr(_rpg.monkey_patch_torch_dist, "_mai_traced", False):
        return False

    _original_monkey_patch = _rpg.monkey_patch_torch_dist

    def _patched_monkey_patch():
        _original_monkey_patch()

        import torch.distributed as dist

        if getattr(dist.all_gather_object, "_mai_traced", False):
            return

        traced_names = [
            "all_reduce",
            "all_gather",
            "all_gather_into_tensor",
            "all_gather_object",
            "broadcast",
            "broadcast_object_list",
            "barrier",
            "reduce_scatter_tensor",
        ]
        for name in traced_names:
            fn = getattr(dist, name, None)
            if fn is None:
                continue

            def _make_wrapper(fname, inner):
                def _w(*args, **kwargs):
                    _say(f"{fname} ENTER")
                    try:
                        return inner(*args, **kwargs)
                    finally:
                        _say(f"{fname} EXIT")

                _w._mai_traced = True
                _w.__name__ = f"traced_{fname}"
                return _w

            setattr(dist, name, _make_wrapper(name, fn))

        _say("collective tracer attached")

    _patched_monkey_patch._mai_traced = True
    _rpg.monkey_patch_torch_dist = _patched_monkey_patch
    return True


def apply_param_groups_tracer() -> bool:
    if not _TRACE_ENABLED:
        return False

    try:
        from megatron.core.optimizer import __init__ as _optim_init
    except Exception:
        import megatron.core.optimizer as _optim_init

    if getattr(_optim_init._get_param_groups, "_mai_traced", False):
        return False

    _original = _optim_init._get_param_groups

    def _traced(model_chunks, config, config_overrides):
        import torch
        import torch.distributed as dist

        pid = os.getpid()
        rank = dist.get_rank() if dist.is_initialized() else -1
        ws = dist.get_world_size() if dist.is_initialized() else -1
        _say(f"_get_param_groups ENTER world={ws}")

        n_total = 0
        n_with_grad = 0
        for chunk in model_chunks:
            for name, param in chunk.named_parameters():
                n_total += 1
                if param.requires_grad:
                    n_with_grad += 1
        _say(f"_get_param_groups params total={n_total} requires_grad={n_with_grad}")

        _say(f"_get_param_groups CALLING original (which will all_gather_object on WORLD)")
        try:
            result = _original(model_chunks, config, config_overrides)
        except Exception as e:
            _say(f"_get_param_groups RAISED: {type(e).__name__}: {e}")
            raise
        _say(f"_get_param_groups DONE -> {len(result)} groups")
        return result

    _traced._mai_traced = True
    _optim_init._get_param_groups = _traced
    return True
