# Modified by MAI-Gym contributors.

import os
from typing import Any


_TRACE_ENABLED = bool(os.environ.get("MAI_GDN_TRACE"))
_TRACE_BUDGET = int(os.environ.get("MAI_GDN_TRACE_BUDGET", "2"))
_STATE: dict[str, int] = {
    "qwen3vl_forward": 0,
    "gdn_forward": 0,
    "undo_alb": 0,
}


def _rank() -> int:
    try:
        import torch.distributed as dist

        if dist.is_available() and dist.is_initialized():
            return dist.get_rank()
    except Exception:
        pass
    return 0


def _cp_info() -> tuple[int, int]:
    try:
        from megatron.core import mpu

        return mpu.get_context_parallel_world_size(), mpu.get_context_parallel_rank()
    except Exception:
        return -1, -1


def _stats(t) -> str:
    try:
        import torch

        if t is None:
            return "None"
        if not torch.is_tensor(t):
            return f"<{type(t).__name__}>"
        s = tuple(t.shape)
        dt = str(t.dtype).replace("torch.", "")
        if t.numel() == 0:
            return f"shape={s} dtype={dt} empty"
        try:
            f = t.detach().float()
            has_nan = bool(torch.isnan(f).any().item())
            has_inf = bool(torch.isinf(f).any().item())
            return (
                f"shape={s} dtype={dt} "
                f"mean={float(f.mean()):.4f} std={float(f.std()):.4f} "
                f"min={float(f.min()):.4f} max={float(f.max()):.4f} "
                f"nan={has_nan} inf={has_inf}"
            )
        except Exception as e:
            return f"shape={s} dtype={dt} stats_err={e!r}"
    except Exception as e:
        return f"<stats_err: {e!r}>"


def _emit(key: str, msg: str) -> None:
    n = _STATE[key]
    if n >= _TRACE_BUDGET:
        return
    _STATE[key] = n + 1
    cp_size, cp_rank = _cp_info()
    print(
        f"[MAI GDN TRACE rank={_rank()} cp={cp_rank}/{cp_size} key={key} n={n + 1}/{_TRACE_BUDGET}] {msg}",
        flush=True,
    )


def _print_layer_types_once(model) -> None:
    if _STATE.get("layer_types_printed"):
        return
    _STATE["layer_types_printed"] = 1
    try:
        decoder = getattr(model, "language_model", model).decoder
        layers = decoder.layers
        info = []
        for i, layer in enumerate(layers):
            sa = getattr(layer, "self_attention", None)
            cls = type(sa).__module__ + "." + type(sa).__name__ if sa is not None else "None"
            info.append((i, cls))
        print(f"[MAI GDN TRACE rank={_rank()}] LAYER TYPES ({len(info)} layers):", flush=True)
        for i, cls in info:
            print(f"[MAI GDN TRACE rank={_rank()}]   layer[{i}].self_attention = {cls}", flush=True)
    except Exception as e:
        print(f"[MAI GDN TRACE rank={_rank()}] layer types dump failed: {e!r}", flush=True)


def _patch_qwen3vl_forward() -> None:
    try:
        from megatron.bridge.models.qwen_vl.modelling_qwen3_vl import model as _vl_model
        from megatron.bridge.models.qwen_vl.modelling_qwen3_vl import utils as _vl_utils
    except Exception as e:
        print(f"[MAI GDN TRACE] cannot patch Qwen3VLModel: {e!r}", flush=True)
        return

    Qwen3VLModel = _vl_model.Qwen3VLModel
    original_forward = Qwen3VLModel.forward
    original_split_data_cp_rank = _vl_utils.split_data_cp_rank

    def traced_split_data_cp_rank(val, cp_size, seq_dim, cp_rank=None):
        before = val
        out = original_split_data_cp_rank(val, cp_size, seq_dim, cp_rank)
        _emit(
            "qwen3vl_forward",
            f"split_data_cp_rank: in {_stats(before)} -> out {_stats(out)} (seq_dim={seq_dim}, cp_size={cp_size}, cp_rank={cp_rank})",
        )
        return out

    _vl_model.split_data_cp_rank = traced_split_data_cp_rank
    _vl_utils.split_data_cp_rank = traced_split_data_cp_rank

    def traced_forward(self, input_ids, *args, **kwargs):
        cp_size, cp_rank = _cp_info()
        _print_layer_types_once(self)
        if _STATE["qwen3vl_forward"] < _TRACE_BUDGET:
            mm_keys = [
                k
                for k in ("pixel_values", "image_grid_thw", "image_input_mask", "video_input_mask")
                if k in kwargs
            ]
            mm_dbg = {k: _stats(kwargs.get(k)) for k in mm_keys}
            _emit(
                "qwen3vl_forward",
                f"Qwen3VLModel.forward entry: input_ids {_stats(input_ids)} packed_seq_params={kwargs.get('packed_seq_params')} cp={cp_size} mm={mm_dbg}",
            )
        return original_forward(self, input_ids, *args, **kwargs)

    Qwen3VLModel.forward = traced_forward
    print("[MAI GDN TRACE] patched Qwen3VLModel.forward + split_data_cp_rank", flush=True)


def _patch_gdn_forward() -> None:
    try:
        from megatron.core.ssm import gated_delta_net as _gdn_mod
    except Exception as e:
        print(f"[MAI GDN TRACE] cannot patch GatedDeltaNet: {e!r}", flush=True)
        return

    GatedDeltaNet = _gdn_mod.GatedDeltaNet
    original_forward = GatedDeltaNet.forward
    original_a2a_cp2hp = _gdn_mod.tensor_a2a_cp2hp
    original_a2a_hp2cp = _gdn_mod.tensor_a2a_hp2cp

    def traced_a2a_cp2hp(tensor, *args, **kwargs):
        out = original_a2a_cp2hp(tensor, *args, **kwargs)
        _emit(
            "gdn_forward",
            f"tensor_a2a_cp2hp: in {_stats(tensor)} -> out {_stats(out)} "
            f"undo_alb={kwargs.get('undo_attention_load_balancing', True)} "
            f"split_sections={kwargs.get('split_sections')}",
        )
        return out

    def traced_a2a_hp2cp(tensor, *args, **kwargs):
        out = original_a2a_hp2cp(tensor, *args, **kwargs)
        _emit(
            "gdn_forward",
            f"tensor_a2a_hp2cp: in {_stats(tensor)} -> out {_stats(out)} "
            f"redo_alb={kwargs.get('redo_attention_load_balancing', True)}",
        )
        return out

    _gdn_mod.tensor_a2a_cp2hp = traced_a2a_cp2hp
    _gdn_mod.tensor_a2a_hp2cp = traced_a2a_hp2cp

    def traced_forward(self, hidden_states, *args, **kwargs):
        if _STATE["gdn_forward"] < _TRACE_BUDGET:
            _emit(
                "gdn_forward",
                f"GDN.forward entry: layer={getattr(self, 'layer_number', '?')} "
                f"hidden_states {_stats(hidden_states)} "
                f"packed_seq_params={kwargs.get('packed_seq_params')} "
                f"cp_size={getattr(self, 'cp_size', '?')} tp_size={getattr(self, 'tp_size', '?')} "
                f"sp_size={getattr(self, 'sp_size', '?')}",
            )
        out = original_forward(self, hidden_states, *args, **kwargs)
        if isinstance(out, tuple) and out and out[0] is not None:
            _emit(
                "gdn_forward",
                f"GDN.forward exit: layer={getattr(self, 'layer_number', '?')} output {_stats(out[0])}",
            )
        return out

    GatedDeltaNet.forward = traced_forward
    print(
        "[MAI GDN TRACE] patched GatedDeltaNet.forward + tensor_a2a_cp2hp + tensor_a2a_hp2cp",
        flush=True,
    )


def _patch_undo_alb() -> None:
    try:
        from megatron.core.ssm import mamba_context_parallel as _mcp
    except Exception as e:
        print(f"[MAI GDN TRACE] cannot patch _undo_attention_load_balancing: {e!r}", flush=True)
        return

    original = _mcp._undo_attention_load_balancing

    def traced(input_, cp_size, packed_seq_params=None):
        out = original(input_, cp_size, packed_seq_params)
        _emit(
            "undo_alb",
            f"_undo_attention_load_balancing: in {_stats(input_)} -> out {_stats(out)} "
            f"cp_size={cp_size} packed_seq_params={packed_seq_params}",
        )
        return out

    _mcp._undo_attention_load_balancing = traced
    try:
        from megatron.core.ssm import gated_delta_net as _gdn_mod

        _gdn_mod._undo_attention_load_balancing = traced
    except Exception:
        pass
    print("[MAI GDN TRACE] patched _undo_attention_load_balancing", flush=True)


def apply_trace_patches() -> None:
    if not _TRACE_ENABLED:
        return
    print(
        f"[MAI GDN TRACE] applying patches (budget={_TRACE_BUDGET} emits per category)",
        flush=True,
    )
    _patch_qwen3vl_forward()
    _patch_gdn_forward()
    _patch_undo_alb()
