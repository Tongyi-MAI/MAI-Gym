# Modified by MAI-Gym contributors.
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_with_transformer_engine_spec

from mbridge.core import LLMBridge, register_model


@register_model("glm4")
class GLM4Bridge(LLMBridge):

    _DIRECT_MAPPING = {
        "embedding.word_embeddings.weight": "model.embed_tokens.weight",
        "decoder.final_layernorm.weight": "model.norm.weight",
        "output_layer.weight": "lm_head.weight",
    }
    _ATTENTION_MAPPING = {
        "self_attention.linear_proj.weight": ["model.layers.{layer_number}.self_attn.o_proj.weight"],
        "self_attention.linear_qkv.layer_norm_weight": ["model.layers.{layer_number}.input_layernorm.weight"],
        "self_attention.q_layernorm.weight": ["model.layers.{layer_number}.self_attn.q_norm.weight"],
        "self_attention.k_layernorm.weight": ["model.layers.{layer_number}.self_attn.k_norm.weight"],
        "self_attention.linear_qkv.weight": [
            "model.layers.{layer_number}.self_attn.q_proj.weight",
            "model.layers.{layer_number}.self_attn.k_proj.weight",
            "model.layers.{layer_number}.self_attn.v_proj.weight",
        ],
        "self_attention.linear_qkv.bias": [
            "model.layers.{layer_number}.self_attn.q_proj.bias",
            "model.layers.{layer_number}.self_attn.k_proj.bias",
            "model.layers.{layer_number}.self_attn.v_proj.bias",
        ],
    }
    _MLP_MAPPING = {
        "mlp.linear_fc1.weight": [
            "model.layers.{layer_number}.mlp.gate_up_proj.weight",
        ],
        "mlp.linear_fc1.layer_norm_weight": ["model.layers.{layer_number}.post_attention_layernorm.weight"],
        "mlp.linear_fc2.weight": ["model.layers.{layer_number}.mlp.down_proj.weight"],
    }

    def _build_config(self):
        return self._build_base_config(
            add_qkv_bias=True,
            qk_layernorm=False,
            post_mlp_layernorm=True,
            post_self_attn_layernorm=True,
            rotary_interleaved=True,
        )

    def _get_transformer_layer_spec(self):
        transformer_layer_spec = get_gpt_layer_with_transformer_engine_spec(
            post_self_attn_layernorm=True,
            post_mlp_layernorm=True,
        )
        return transformer_layer_spec

    def _weight_name_mapping_mcore_to_hf(self, mcore_weights_name: str) -> list[str]:
        assert "_extra_state" not in mcore_weights_name, "extra_state should not be loaded"

        if mcore_weights_name in self._DIRECT_MAPPING:
            return [self._DIRECT_MAPPING[mcore_weights_name]]

        if "post_self_attn_layernorm" in mcore_weights_name:
            layer_number = mcore_weights_name.split(".")[2]
            return [f"model.layers.{layer_number}.post_self_attn_layernorm.weight"]
        elif "post_mlp_layernorm" in mcore_weights_name:
            layer_number = mcore_weights_name.split(".")[2]
            return [f"model.layers.{layer_number}.post_mlp_layernorm.weight"]
        elif "self_attention" in mcore_weights_name:
            return self._weight_name_mapping_attention(mcore_weights_name)
        elif "mlp" in mcore_weights_name:
            return self._weight_name_mapping_mlp(mcore_weights_name)
        else:
            raise NotImplementedError(f"Unsupported parameter name: {mcore_weights_name}")
