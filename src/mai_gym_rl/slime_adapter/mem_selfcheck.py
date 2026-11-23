import os


def _rank():
    try:
        import torch.distributed as dist

        if dist.is_available() and dist.is_initialized():
            return dist.get_rank()
    except Exception:
        pass
    return 0


def print_alloc_env(tag=""):

    if _rank() != 0:
        return
    import torch

    alloc_conf = os.environ.get("PYTORCH_ALLOC_CONF")
    cuda_alloc_conf = os.environ.get("PYTORCH_CUDA_ALLOC_CONF")
    try:
        backend = torch.cuda.get_allocator_backend()
    except Exception as e:
        backend = "<unavailable: %s>" % e

    conf = cuda_alloc_conf or alloc_conf or ""
    expandable_requested = "expandable_segments:true" in conf.lower()

    expandable_effective = expandable_requested and backend == "native"

    pfx = "[mai_mem_selfcheck]%s" % ((" " + tag) if tag else "")
    print("%s torch=%s" % (pfx, torch.__version__), flush=True)
    print("%s allocator_backend=%s" % (pfx, backend), flush=True)
    print("%s PYTORCH_ALLOC_CONF=%r" % (pfx, alloc_conf), flush=True)
    print("%s PYTORCH_CUDA_ALLOC_CONF=%r" % (pfx, cuda_alloc_conf), flush=True)
    print(
        "%s expandable_segments: requested=%s effective=%s"
        % (pfx, expandable_requested, expandable_effective),
        flush=True,
    )
    if not expandable_effective:
        print(
            "%s WARNING expandable_segments NOT effective -- "
            "fragmentation / high-water-mark OOM is NOT mitigated" % pfx,
            flush=True,
        )


def enable_vit_recompute():

    if os.environ.get("MAI_VIT_RECOMPUTE", "0") != "1":
        return
    try:
        from megatron.bridge.models.qwen_vl.modelling_qwen3_vl.transformer_block import (
            Qwen3VLVisionTransformerBlock,
        )
    except Exception as e:
        if _rank() == 0:
            print("[mai_vit_recompute] MAI_VIT_RECOMPUTE=1 but import failed: %s" % e, flush=True)
        return

    orig_init = Qwen3VLVisionTransformerBlock.__init__
    if getattr(orig_init, "_mai_vit_recompute_wrapped", False):
        return
    method = os.environ.get("VIT_RECOMPUTE_METHOD", "uniform")
    try:
        num_layers = int(os.environ.get("VIT_RECOMPUTE_NUM_LAYERS", "1"))
    except ValueError:
        num_layers = 1

    def __init__(self, *args, **kwargs):
        orig_init(self, *args, **kwargs)
        cfg = getattr(self, "config", None)
        if cfg is not None:
            cfg.recompute_granularity = "full"
            cfg.recompute_method = method
            cfg.recompute_num_layers = num_layers
            if _rank() == 0:
                print(
                    "[mai_vit_recompute] ViT recompute ON (full/%s/%d) on vision block"
                    % (method, num_layers),
                    flush=True,
                )

    __init__._mai_vit_recompute_wrapped = True
    Qwen3VLVisionTransformerBlock.__init__ = __init__


def enable_lm_recompute():

    if os.environ.get("MAI_LM_RECOMPUTE", "0") != "1":
        return
    try:
        from megatron.bridge.models.qwen_vl.modelling_qwen3_vl.transformer_block import (
            Qwen3VLTransformerBlock,
        )
    except Exception as e:
        if _rank() == 0:
            print("[mai_lm_recompute] MAI_LM_RECOMPUTE=1 but import failed: %s" % e, flush=True)
        return

    orig_init = Qwen3VLTransformerBlock.__init__
    if getattr(orig_init, "_mai_lm_recompute_wrapped", False):
        return
    method = os.environ.get("LM_RECOMPUTE_METHOD", "uniform")
    try:
        num_layers = int(os.environ.get("LM_RECOMPUTE_NUM_LAYERS", "1"))
    except ValueError:
        num_layers = 1

    def __init__(self, *args, **kwargs):
        orig_init(self, *args, **kwargs)
        cfg = getattr(self, "config", None)
        if cfg is not None:
            chunk = num_layers
            n_local = getattr(self, "num_layers_per_pipeline_rank", None)
            if n_local and chunk > 1 and n_local % chunk != 0:
                chunk = max(d for d in range(1, chunk + 1) if n_local % d == 0)
                print(
                    "[mai_lm_recompute] chunk %d does not divide this stage's %d "
                    "layers (bridge uniform loop has no tail clamp) -> using %d"
                    % (num_layers, n_local, chunk),
                    flush=True,
                )
            cfg.recompute_granularity = "full"
            cfg.recompute_method = method
            cfg.recompute_num_layers = chunk
            if _rank() == 0:
                print(
                    "[mai_lm_recompute] LM recompute ON (full/%s/%d) on decoder block"
                    % (method, chunk),
                    flush=True,
                )

    __init__._mai_lm_recompute_wrapped = True
    Qwen3VLTransformerBlock.__init__ = __init__


def custom_megatron_init(args):

    print_alloc_env()
    enable_vit_recompute()
    enable_lm_recompute()
