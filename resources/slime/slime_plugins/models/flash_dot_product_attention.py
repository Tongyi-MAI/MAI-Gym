# Modified by MAI-Gym contributors.

import math

import torch
from flash_attn.flash_attn_interface import flash_attn_varlen_func
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.transformer.utils import is_layer_window_attention
from megatron.core.utils import divide
from torch import Tensor

from slime_plugins.models.learnable_softmax_attention import learnable_softmax_flash_attn_varlen


class FlashDotProductAttention(MegatronModule):

    def __init__(
        self,
        config: TransformerConfig,
        layer_number: int,
        attn_mask_type: AttnMaskType,
        attention_type: str,
        attention_dropout: float | None = None,
        softmax_scale: float | None = None,
        k_channels: int | None = None,
        v_channels: int | None = None,
        cp_comm_type: str = "p2p",
        pg_collection: ProcessGroupCollection = None,
    ):
        super().__init__(config=config)
        self.config = config
        self.layer_number = max(1, layer_number)

        if softmax_scale is not None:
            self.softmax_scale = softmax_scale
        else:
            self.softmax_scale = 1.0 / math.sqrt(config.kv_channels)

        if config.apply_query_key_layer_scaling:
            self.softmax_scale /= self.layer_number

        if is_layer_window_attention(config.window_size, config.window_attn_skip_freq, layer_number):
            self.window_size = config.window_size
        else:
            self.window_size = (-1, -1)

        self.attention_dropout = config.attention_dropout if attention_dropout is None else attention_dropout

        self.current_max_attn_logits = None
        if pg_collection is None:
            from megatron.core import parallel_state

            world_size = parallel_state.get_tensor_model_parallel_world_size()
        else:
            world_size = pg_collection.tp.size()
        num_heads_per_partition = divide(config.num_attention_heads, world_size)

        if config.softmax_type == "vanilla":
            self.softmax_offset = None
        elif config.softmax_type == "off-by-one":
            self.softmax_offset = torch.zeros(
                num_heads_per_partition,
                device=torch.cuda.current_device(),
                dtype=config.params_dtype,
            )
        elif config.softmax_type == "learnable":
            self.register_parameter(
                "softmax_offset",
                torch.nn.Parameter(
                    torch.empty(
                        num_heads_per_partition,
                        device=torch.cuda.current_device(),
                        dtype=config.params_dtype,
                    )
                ),
            )
            if config.perform_initialization:
                self.softmax_offset = config.init_method(self.softmax_offset)
        else:
            raise ValueError(f"Unsupported softmax type: {config.softmax_type}")

    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        attention_mask: Tensor,
        attn_mask_type: AttnMaskType = None,
        attention_bias: Tensor = None,
        packed_seq_params: PackedSeqParams | None = None,
    ):
        assert attention_bias is None, "Attention bias not supported with FlashDotProductAttention"

        causal = (
            attn_mask_type
            in (
                AttnMaskType.causal,
                AttnMaskType.padding_causal,
                AttnMaskType.causal_bottom_right,
            )
            if attn_mask_type is not None
            else True
        )

        is_training = self.training
        dropout_p = self.attention_dropout if is_training else 0.0

        if packed_seq_params is not None and hasattr(packed_seq_params, "qkv_format"):
            qkv_format = packed_seq_params.qkv_format
        else:
            qkv_format = "sbhd"

        if qkv_format == "thd":
            cu_seqlens_q = packed_seq_params.cu_seqlens_q
            cu_seqlens_k = packed_seq_params.cu_seqlens_kv
            max_seqlen_q = packed_seq_params.max_seqlen_q
            max_seqlen_k = packed_seq_params.max_seqlen_kv
        else:
            s, b, h, d = query.shape
            query = query.permute(1, 0, 2, 3).reshape(b * s, h, d)
            key = key.permute(1, 0, 2, 3).reshape(b * s, key.size(2), d)
            value = value.permute(1, 0, 2, 3).reshape(b * s, value.size(2), d)
            cu_seqlens_q = torch.arange(0, (b + 1) * s, s, dtype=torch.int32, device=query.device)
            cu_seqlens_k = cu_seqlens_q
            max_seqlen_q = s
            max_seqlen_k = s

        if self.softmax_offset is not None:
            output = learnable_softmax_flash_attn_varlen(
                query,
                key,
                value,
                cu_seqlens_q,
                cu_seqlens_k,
                max_seqlen_q,
                max_seqlen_k,
                self.softmax_offset,
                softmax_scale=self.softmax_scale,
                causal=causal,
                window_size=self.window_size,
                dropout_p=dropout_p,
                deterministic=self.config.deterministic_mode if hasattr(self.config, "deterministic_mode") else False,
            )
        else:
            output = flash_attn_varlen_func(
                query,
                key,
                value,
                cu_seqlens_q,
                cu_seqlens_k,
                max_seqlen_q,
                max_seqlen_k,
                dropout_p=dropout_p,
                softmax_scale=self.softmax_scale,
                causal=causal,
                window_size=self.window_size,
                deterministic=self.config.deterministic_mode if hasattr(self.config, "deterministic_mode") else False,
            )

        if qkv_format != "thd":
            output = output.reshape(b, s, h, d).permute(1, 0, 2, 3)

        return output
