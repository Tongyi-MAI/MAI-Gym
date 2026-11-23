# Modified by MAI-Gym contributors.
import torch
from mbridge.core import register_model
from mbridge.core.safetensor_io import SafeTensorIO
from mbridge.models import DeepseekV3Bridge


@register_model("glm4_moe_lite")
class GLM4MoELiteBridge(DeepseekV3Bridge):

    def __init__(self, hf_config, **kwargs):
        if not hasattr(hf_config, "rope_theta"):
            rope_params = getattr(hf_config, "rope_parameters", None) or {}
            hf_config.rope_theta = rope_params.get("rope_theta", 1000000)
        super().__init__(hf_config, **kwargs)
        n = hf_config.num_hidden_layers
        if getattr(hf_config, "num_nextn_predict_layers", 0) and n:
            self._SHARED_STATE_DICT_MAPPING = {
                "embedding.word_embeddings.weight": [
                    "model.embed_tokens.weight",
                    f"model.layers.{n}.embed_tokens.weight",
                ],
                "output_layer.weight": [
                    "lm_head.weight",
                    f"model.layers.{n}.shared_head.head.weight",
                ],
            }

    def _get_safetensor_io(self, weights_path: str):
        return SafeTensorIO(self._get_actual_hf_path(weights_path))

    def _weight_to_hf_format(
        self, mcore_weights_name: str, mcore_weights: torch.Tensor
    ) -> tuple[list[str], list[torch.Tensor]]:
        if (
            self.config.mtp_num_layers is not None
            and self.config.mtp_num_layers >= 1
            and mcore_weights_name in self._SHARED_STATE_DICT_MAPPING
        ):
            hf_names = self._SHARED_STATE_DICT_MAPPING[mcore_weights_name]
            return hf_names, [mcore_weights] * len(hf_names)
        return super(DeepseekV3Bridge, self)._weight_to_hf_format(mcore_weights_name, mcore_weights)

    def _convert_mtp_param(self, name: str) -> list[str]:
        assert self.config.mtp_num_layers == 1, "only support one mtp layer for now"
        n = self.config.num_layers
        direct_name_mapping = {
            "mtp.layers.0.enorm.weight": f"model.layers.{n}.enorm.weight",
            "mtp.layers.0.hnorm.weight": f"model.layers.{n}.hnorm.weight",
            "mtp.layers.0.eh_proj.weight": f"model.layers.{n}.eh_proj.weight",
            "mtp.layers.0.final_layernorm.weight": f"model.layers.{n}.shared_head.norm.weight",
        }
        if name in direct_name_mapping:
            return [direct_name_mapping[name]]
        assert "mtp.layers.0.transformer_layer" in name, f"mtp not found in {name}"
        proxy_name = name.replace("mtp.layers.0.transformer_layer", f"decoder.layers.{n}")
        if "self_attention" in proxy_name or "input_layernorm.weight" in proxy_name:
            return self._weight_name_mapping_attention(proxy_name)
        elif "mlp" in proxy_name:
            return self._weight_name_mapping_mlp(proxy_name)
        else:
            raise NotImplementedError(f"Unsupported MTP parameter name: {name}")
