# Modified by MAI-Gym contributors.

import copy
import logging
import os

logger = logging.getLogger(__name__)


def apply_bridge_qwen3vl_init_patch() -> bool:
    if os.environ.get("MAI_BRIDGE_PATCHES", "1") == "0":
        logger.info("MAI_BRIDGE_PATCHES=0 — skipping Qwen3VLModel.__init__ patch")
        return False

    try:
        from megatron.bridge.models.qwen_vl.modelling_qwen3_vl import model as _vl_model
    except ImportError as e:
        logger.warning("megatron.bridge Qwen3VLModel not importable, skipping patch: %r", e)
        return False

    if getattr(_vl_model.Qwen3VLModel.__init__, "_mai_patched", False):
        return False

    _original_init = _vl_model.Qwen3VLModel.__init__

    def _patched_init(
        self,
        language_transformer_config,
        language_transformer_layer_spec,
        vision_transformer_config,
        *args,
        **kwargs,
    ):
        spec = language_transformer_layer_spec
        if hasattr(spec, "layer_specs") and spec.layer_specs is not None:
            for layer_spec in spec.layer_specs:
                sub = getattr(layer_spec, "submodules", None)
                if sub is not None:
                    layer_spec.submodules = copy.deepcopy(sub)
        return _original_init(
            self,
            language_transformer_config,
            spec,
            vision_transformer_config,
            *args,
            **kwargs,
        )

    _patched_init._mai_patched = True
    _vl_model.Qwen3VLModel.__init__ = _patched_init
    logger.info(
        "Applied MAI patch: Qwen3VLModel.__init__ now preserves per-layer GDN specs "
        "(fixes Qwen3.5 + CP train log_probs garbage)"
    )
    print(
        "[MAI BRIDGE PATCH] Qwen3VLModel.__init__ wrapped — per-layer GDN specs preserved",
        flush=True,
    )
    return True


def apply_qwen35_provider_hybrid_spec_patch() -> bool:
    if os.environ.get("MAI_BRIDGE_PATCHES", "1") == "0":
        return False

    try:
        from megatron.bridge.models.qwen_vl import qwen35_vl_provider as _qwen35
        from megatron.bridge.models.qwen_vl.modelling_qwen3_vl.attention import Qwen3VLSelfAttention
        from megatron.bridge.models.qwen_vl.modelling_qwen3_vl.model import Qwen3VLModel
        from megatron.core.models.gpt.experimental_attention_variant_module_specs import (
            get_transformer_block_with_experimental_attention_variant_spec,
        )
    except ImportError as e:
        logger.warning("Cannot import qwen35 provider deps, skipping hybrid-spec patch: %r", e)
        return False

    _patch_standard_attention_specs = _qwen35._patch_standard_attention_specs

    def _make_patched_provide(original_provide, provider_class_name):
        if getattr(original_provide, "_mai_patched", False):
            return original_provide

        def _patched_provide(self, pre_process=None, post_process=None, vp_stage=None):
            language_transformer_config = self
            hf_vision_config = self.vision_config

            block_spec = get_transformer_block_with_experimental_attention_variant_spec(
                language_transformer_config,
                vp_stage=vp_stage,
            )
            _patch_standard_attention_specs(block_spec, Qwen3VLSelfAttention)

            model = Qwen3VLModel(
                language_transformer_config=language_transformer_config,
                language_transformer_layer_spec=block_spec,
                vision_transformer_config=hf_vision_config,
                pre_process=pre_process,
                post_process=post_process,
            )

            if (
                self.freeze_language_model
                or self.freeze_vision_model
                or self.freeze_vision_projection
            ):
                model.freeze(
                    freeze_language_model=self.freeze_language_model,
                    freeze_vision_model=self.freeze_vision_model,
                    freeze_vision_projection=self.freeze_vision_projection,
                )
            return model

        _patched_provide._mai_patched = True
        print(
            f"[MAI BRIDGE PATCH] {provider_class_name}.provide wrapped — "
            f"now uses experimental-variant spec builder (hybrid GDN+attention)",
            flush=True,
        )
        return _patched_provide

    _qwen35.Qwen35VLModelProvider.provide = _make_patched_provide(
        _qwen35.Qwen35VLModelProvider.provide, "Qwen35VLModelProvider"
    )
    if hasattr(_qwen35, "Qwen35VLMoEModelProvider"):
        _qwen35.Qwen35VLMoEModelProvider.provide = _make_patched_provide(
            _qwen35.Qwen35VLMoEModelProvider.provide, "Qwen35VLMoEModelProvider"
        )

    logger.info(
        "Applied MAI patch: Qwen35VLModelProvider.provide now uses experimental-variant spec builder"
    )
    return True
