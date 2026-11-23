# Modified by MAI-Gym contributors.
# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

import torch
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_mtp_block_spec

from mbridge.core import register_model
from mbridge.models import Qwen2Bridge


@register_model("mimo")
class MimoBridge(Qwen2Bridge):

    def _build_config(self):
        hf_config = self.hf_config

        mtp_args = {}
        if "num_nextn_predict_layers" in hf_config:
            mtp_args["mtp_num_layers"] = hf_config.num_nextn_predict_layers

        return self._build_base_config(
            add_qkv_bias=True,
            qk_layernorm=False,
            **mtp_args,
        )

    def _get_gptmodel_args(self) -> dict:
        ret = super()._get_gptmodel_args()

        if self.config.mtp_num_layers is not None:
            transformer_layer_spec = self.config
            mtp_block_spec = get_gpt_mtp_block_spec(self.config, transformer_layer_spec, use_transformer_engine=True)
            ret["mtp_block_spec"] = mtp_block_spec

        return ret

    def _weight_name_mapping_mcore_to_hf(self, mcore_weights_name: str) -> list[str]:
        if "mtp" in mcore_weights_name:
            return self._convert_mtp_param(mcore_weights_name)

        return super()._weight_name_mapping_mcore_to_hf(mcore_weights_name)

    def _convert_mtp_param(self, name: str) -> list[str]:
        if "mtp.layers." not in name:
            raise NotImplementedError(f"Invalid MTP parameter name: {name}")

        parts = name.split(".")
        mtp_layer_idx = parts[2]

        direct_name_mapping = {
            f"mtp.layers.{mtp_layer_idx}.enorm.weight": f"model.mtp_layers.{mtp_layer_idx}.token_layernorm.weight",
            f"mtp.layers.{mtp_layer_idx}.hnorm.weight": f"model.mtp_layers.{mtp_layer_idx}.hidden_layernorm.weight",
            f"mtp.layers.{mtp_layer_idx}.eh_proj.weight": f"model.mtp_layers.{mtp_layer_idx}.input_proj.weight",
            f"mtp.layers.{mtp_layer_idx}.final_layernorm.weight": f"model.mtp_layers.{mtp_layer_idx}.final_layernorm.weight",
        }

        if name in direct_name_mapping:
            return [direct_name_mapping[name]]

        if "transformer_layer" in name:
            proxy_name = name.replace(
                f"mtp.layers.{mtp_layer_idx}.transformer_layer",
                f"decoder.layers.{mtp_layer_idx}",
            )

            if "self_attention" in proxy_name or "input_layernorm.weight" in proxy_name:
                convert_names = super()._weight_name_mapping_attention(proxy_name)
            elif "mlp" in proxy_name:
                convert_names = super()._weight_name_mapping_mlp(proxy_name)
            else:
                raise NotImplementedError(f"Unsupported transformer component in MTP: {name}")

            convert_names = [
                cn.replace(f"model.layers.{mtp_layer_idx}", f"model.mtp_layers.{mtp_layer_idx}")
                for cn in convert_names
            ]
            return convert_names
        else:
            raise NotImplementedError(f"Unsupported MTP parameter name: {name}")
        return convert_names

    def _weight_to_mcore_format(self, mcore_weights_name: str, hf_weights: list[torch.Tensor]) -> torch.Tensor:
        weight = super()._weight_to_mcore_format(mcore_weights_name, hf_weights)
        if mcore_weights_name.endswith("eh_proj.weight"):
            first_half, second_half = weight.chunk(2, dim=1)
            weight = torch.cat([second_half, first_half], dim=1)
        return weight

    def _weight_to_hf_format(
        self, mcore_weights_name: str, mcore_weights: torch.Tensor
    ) -> tuple[list[str], list[torch.Tensor]]:
        if mcore_weights_name.endswith("eh_proj.weight"):
            first_half, second_half = mcore_weights.chunk(2, dim=1)
            mcore_weights = torch.cat([second_half, first_half], dim=1)
        return super()._weight_to_hf_format(mcore_weights_name, mcore_weights)
