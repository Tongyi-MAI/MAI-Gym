# Modified by MAI-Gym contributors.
import logging

import torch

try:
    import deep_ep
    from torch_memory_saver import torch_memory_saver

    old_init = deep_ep.Buffer.__init__

    def new_init(self, *args, **kwargs):
        if torch_memory_saver._impl is not None:
            torch_memory_saver._impl._binary_wrapper.cdll.tms_set_interesting_region(False)
        old_init(self, *args, **kwargs)
        torch.cuda.synchronize()
        if torch_memory_saver._impl is not None:
            torch_memory_saver._impl._binary_wrapper.cdll.tms_set_interesting_region(True)

    deep_ep.Buffer.__init__ = new_init
except ImportError:
    logging.warning("deep_ep is not installed, some functionalities may be limited.")

try:
    from megatron.bridge.models.qwen_vl.modelling_qwen3_vl.text_model import (
        Qwen3VLMoETextRotaryEmbedding,
        Qwen3VLTextRotaryEmbedding,
    )

    def patch_rotary_embedding(cls):
        _original_forward = cls.forward

        def _patched_forward(self, *args, packed_seq_params=None, **kwargs):
            return _original_forward(self, *args, **kwargs)

        cls.forward = _patched_forward

    patch_rotary_embedding(Qwen3VLTextRotaryEmbedding)
    patch_rotary_embedding(Qwen3VLMoETextRotaryEmbedding)
except ImportError:
    pass

logging.getLogger("megatron").setLevel(logging.WARNING)


try:
    from ._bridge_patches import (
        apply_bridge_qwen3vl_init_patch as _apply_bridge_init_patch,
    )
    from ._bridge_patches import (
        apply_qwen35_provider_hybrid_spec_patch as _apply_qwen35_provider_patch,
    )

    _apply_bridge_init_patch()
    _apply_qwen35_provider_patch()
except Exception as _e:
    logging.warning("MAI bridge patches failed to apply: %r", _e)


try:
    from ._gdn_trace import apply_trace_patches as _apply_gdn_trace_patches

    _apply_gdn_trace_patches()
except Exception as _e:
    logging.warning("MAI GDN trace patch failed to apply: %r", _e)


try:
    from ._collective_trace import (
        apply_collective_tracer as _apply_collective_tracer,
    )
    from ._collective_trace import (
        apply_param_groups_tracer as _apply_param_groups_tracer,
    )

    _apply_collective_tracer()
    _apply_param_groups_tracer()
except Exception as _e:
    logging.warning("MAI collective trace patch failed to apply: %r", _e)
