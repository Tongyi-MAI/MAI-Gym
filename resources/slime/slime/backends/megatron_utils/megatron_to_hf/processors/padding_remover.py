# Modified by MAI-Gym contributors.
import torch

from slime.backends.megatron_utils.misc_utils import strip_param_name_prefix


def remove_padding(name: str, param: torch.Tensor, vocab_size: int) -> torch.Tensor:
    if strip_param_name_prefix(name) in {"embedding.word_embeddings.weight", "output_layer.weight"}:
        return param[:vocab_size]
    return param
