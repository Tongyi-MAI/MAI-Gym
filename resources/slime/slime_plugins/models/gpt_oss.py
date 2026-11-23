# Modified by MAI-Gym contributors.

from megatron.core.models.gpt.gpt_layer_specs import get_gpt_decoder_block_spec

from slime_plugins.models.flash_dot_product_attention import FlashDotProductAttention


def _replace_core_attention_in_spec(spec, replacement_cls):
    if hasattr(spec, "layer_specs") and not hasattr(spec, "submodules"):
        for layer_spec in spec.layer_specs:
            _replace_core_attention_in_spec(layer_spec, replacement_cls)
        return
    if hasattr(spec, "submodules"):
        sub = spec.submodules
        if hasattr(sub, "core_attention"):
            sub.core_attention = replacement_cls
        if hasattr(sub, "layer_specs"):
            for layer_spec in sub.layer_specs:
                _replace_core_attention_in_spec(layer_spec, replacement_cls)
        for attr in dir(sub):
            if attr.startswith("_") or attr == "layer_specs":
                continue
            val = getattr(sub, attr)
            if hasattr(val, "submodules"):
                _replace_core_attention_in_spec(val, replacement_cls)


def get_gpt_oss_spec(args, config, vp_stage):
    kwargs = {"use_transformer_engine": True}
    if vp_stage is not None:
        kwargs["vp_stage"] = vp_stage
    transformer_layer_spec = get_gpt_decoder_block_spec(config, **kwargs)

    _replace_core_attention_in_spec(transformer_layer_spec, FlashDotProductAttention)

    from megatron.bridge.models.conversion.param_mapping import AutoMapping

    AutoMapping.register_module_type("FlashDotProductAttention", "column")

    return transformer_layer_spec
