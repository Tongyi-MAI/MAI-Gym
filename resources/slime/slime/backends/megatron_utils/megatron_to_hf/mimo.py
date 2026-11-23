# Modified by MAI-Gym contributors.
import re
import torch
from .qwen2 import convert_qwen2_to_hf


def convert_mimo_to_hf(args, name, param):

    if "mtp" in name:
        return convert_mimo_mtp_param(args, name, param)

    return convert_qwen2_to_hf(args, name, param)


def convert_mimo_mtp_param(args, name, param):
    mtp_pattern = r"module\.module\.mtp\.layers\.(\d+)\.(.+)"
    match = re.match(mtp_pattern, name)

    if not match:
        raise ValueError(f"Invalid MTP parameter name: {name}")

    layer_idx, component = match.groups()

    direct_mappings = {
        "enorm.weight": f"model.mtp_layers.{layer_idx}.token_layernorm.weight",
        "hnorm.weight": f"model.mtp_layers.{layer_idx}.hidden_layernorm.weight",
        "eh_proj.weight": f"model.mtp_layers.{layer_idx}.input_proj.weight",
        "final_layernorm.weight": f"model.mtp_layers.{layer_idx}.final_layernorm.weight",
    }
    if component == "eh_proj.weight":
        first_half, second_half = param.chunk(2, dim=1)
        param = torch.cat([second_half, first_half], dim=1)

    if component in direct_mappings:
        return [(direct_mappings[component], param)]

    if component.startswith("transformer_layer."):
        transformer_component = component[len("transformer_layer.") :]

        proxy_name = f"module.module.decoder.layers.{layer_idx}.{transformer_component}"

        results = convert_qwen2_to_hf(args, proxy_name, param)

        converted_results = []
        for hf_name, hf_param in results:
            hf_name = hf_name.replace(f"model.layers.{layer_idx}", f"model.mtp_layers.{layer_idx}")
            converted_results.append((hf_name, hf_param))

        return converted_results

    raise ValueError(f"Unknown MTP component: {component} in {name}")
