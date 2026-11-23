# Modified by MAI-Gym contributors.

from __future__ import annotations

import itertools
import logging
from copy import deepcopy
from dataclasses import dataclass, field

import torch
from megatron.bridge.models.conversion.mapping_registry import MegatronMappingRegistry
from megatron.bridge.models.conversion.model_bridge import MegatronModelBridge
from megatron.bridge.models.conversion.param_mapping import AutoMapping, GatedMLPMapping, QKVMapping, ReplicatedMapping
from megatron.bridge.models.qwen.qwen_provider import Qwen3MoEModelProvider
from megatron.bridge.utils.common_utils import hook_hf_module_setattr_for_tp_grad_sync
from megatron.core import parallel_state, tensor_parallel
from megatron.core.models.gpt import GPTModel as MCoreGPTModel
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_decoder_block_spec
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.transformer.module import MegatronModule

logger = logging.getLogger(__name__)


def _thd_to_bshd(packed: torch.Tensor, cu_seqlens: torch.Tensor) -> torch.Tensor:
    seqlens = cu_seqlens[1:] - cu_seqlens[:-1]
    max_seq = seqlens.max().item()
    bs = len(cu_seqlens) - 1
    out = packed.new_zeros(bs, max_seq, *packed.shape[2:])
    for i, sl in enumerate(seqlens):
        out[i, :sl] = packed[0, cu_seqlens[i] : cu_seqlens[i] + sl]
    return out


def _bshd_to_thd(unpacked: torch.Tensor, cu_seqlens: torch.Tensor) -> torch.Tensor:
    seqlens = cu_seqlens[1:] - cu_seqlens[:-1]
    total = cu_seqlens[-1].item()
    out = unpacked.new_zeros(1, total, *unpacked.shape[2:])
    for i, sl in enumerate(seqlens):
        out[0, cu_seqlens[i] : cu_seqlens[i] + sl] = unpacked[i, :sl]
    return out


def _gather_input_ids_from_cp(
    input_ids: torch.Tensor,
    cu_seqlens: torch.Tensor,
) -> torch.Tensor:
    cp_size = parallel_state.get_context_parallel_world_size()
    if cp_size <= 1:
        return input_ids

    gathered = torch.distributed.nn.all_gather(
        input_ids, group=parallel_state.get_context_parallel_group()
    )

    local_cu_seqlens = cu_seqlens // cp_size
    num_seqs = len(cu_seqlens) - 1
    whole_list = []
    for i in range(num_seqs):
        seqlen = (cu_seqlens[i + 1] - cu_seqlens[i]).item()
        chunk_size = seqlen // 2 // cp_size
        whole_list.extend(
            gathered[cp_rank][0, local_cu_seqlens[i] : local_cu_seqlens[i] + chunk_size] for cp_rank in range(cp_size)
        )
        whole_list.extend(
            [
                gathered[cp_rank][0, local_cu_seqlens[i] + chunk_size : local_cu_seqlens[i + 1]]
                for cp_rank in range(cp_size)
            ][::-1]
        )
    return torch.cat(whole_list).unsqueeze(0)


def _select_local_image_embeds(
    full_input_ids: torch.Tensor,
    cu_seqlens: torch.Tensor,
    image_token_id: int,
    image_embeds: torch.Tensor,
    cp_rank: int,
    cp_size: int,
) -> torch.Tensor:
    device = full_input_ids.device
    full_flat = full_input_ids[0]
    full_mask = full_flat == image_token_id

    T_global = full_flat.shape[0]
    rank_mask = torch.zeros(T_global, dtype=torch.bool, device=device)

    num_seqs = len(cu_seqlens) - 1
    for i in range(num_seqs):
        seq_start = cu_seqlens[i].item()
        seqlen = (cu_seqlens[i + 1] - cu_seqlens[i]).item()
        chunk_size = seqlen // (2 * cp_size)

        first_start = seq_start + cp_rank * chunk_size
        rank_mask[first_start : first_start + chunk_size] = True

        second_end = seq_start + seqlen - cp_rank * chunk_size
        rank_mask[second_end - chunk_size : second_end] = True

    local_image_mask = full_mask & rank_mask
    n_local = local_image_mask.sum().item()

    if n_local == 0:
        return image_embeds[:0]
    if n_local == image_embeds.shape[0]:
        return image_embeds

    image_cumsum = full_mask.long().cumsum(0)
    local_positions = local_image_mask.nonzero(as_tuple=True)[0]
    embed_indices = image_cumsum[local_positions] - 1
    return image_embeds[embed_indices]


class Glm4vMoeVLModel(MegatronModule):

    def __init__(
        self,
        language_transformer_config,
        language_transformer_layer_spec,
        hf_vision_config,
        parallel_output: bool = True,
        pre_process: bool = True,
        post_process: bool = True,
    ) -> None:
        super().__init__(config=language_transformer_config)

        self.pre_process = pre_process
        self.post_process = post_process
        self.image_token_id = language_transformer_config.image_token_id
        self.video_token_id = language_transformer_config.video_token_id
        self.spatial_merge_size = language_transformer_config.spatial_merge_size

        self.share_embeddings_and_output_weights = False

        self.vision_model = None
        if self.pre_process:
            from transformers.models.glm4v_moe.modeling_glm4v_moe import Glm4vMoeVisionModel

            self.vision_model = Glm4vMoeVisionModel._from_config(hf_vision_config)
            self.vision_model.requires_grad_(False)
            self.vision_model.eval()
            hook_hf_module_setattr_for_tp_grad_sync(self.vision_model)
            if torch.cuda.is_available():
                self.vision_model = self.vision_model.to("cuda")

        self.language_model = MCoreGPTModel(
            config=language_transformer_config,
            transformer_layer_spec=language_transformer_layer_spec,
            vocab_size=language_transformer_config.vocab_size,
            max_sequence_length=language_transformer_config.language_max_sequence_length,
            parallel_output=parallel_output,
            position_embedding_type="mrope",
            rotary_percent=language_transformer_config.rotary_percent,
            pre_process=self.pre_process,
            post_process=self.post_process,
            rotary_base=language_transformer_config.rotary_base,
            fp16_lm_cross_entropy=language_transformer_config.fp16_lm_cross_entropy,
            share_embeddings_and_output_weights=language_transformer_config.share_embeddings_and_output_weights,
            scatter_embedding_sequence_parallel=False,
        )

        self.share_embeddings_and_output_weights = self.language_model.share_embeddings_and_output_weights


    def shared_embedding_or_output_weight(self):
        return self.language_model.shared_embedding_or_output_weight()

    def set_input_tensor(self, input_tensor):
        if not isinstance(input_tensor, list):
            input_tensor = [input_tensor]
        assert len(input_tensor) == 1
        if self.pre_process:
            self.encoder_hidden_state = input_tensor[0]
        else:
            self.language_model.set_input_tensor(input_tensor[0])


    def _get_image_features(self, pixel_values, image_grid_thw):
        pixel_values = pixel_values.to(dtype=self.vision_model.dtype)
        with torch.no_grad():
            return self.vision_model(pixel_values, grid_thw=image_grid_thw)


    @staticmethod
    def _get_vision_position_ids(
        start_position: int,
        grid_thw,
        temp_merge_size: int,
        spatial_merge_size: int,
        device,
    ) -> torch.Tensor:
        llm_grid_t = grid_thw[0].item() // temp_merge_size
        llm_grid_h = grid_thw[1].item() // spatial_merge_size
        llm_grid_w = grid_thw[2].item() // spatial_merge_size
        n_tokens = llm_grid_h * llm_grid_w * llm_grid_t

        pos_w = torch.arange(start_position, start_position + llm_grid_w, device=device)
        pos_w = pos_w.repeat(llm_grid_h * llm_grid_t)
        pos_h = torch.arange(start_position, start_position + llm_grid_h, device=device)
        pos_h = pos_h.repeat_interleave(llm_grid_w * llm_grid_t)
        pos_t = torch.full((n_tokens,), start_position, device=device, dtype=torch.long)
        return torch.stack([pos_t, pos_h, pos_w], dim=0)

    def _compute_mrope_position_ids(
        self,
        input_ids_bshd: torch.Tensor,
        image_grid_thw: torch.Tensor | None,
    ) -> torch.Tensor:
        bs, seq_len = input_ids_bshd.shape
        device = input_ids_bshd.device
        spatial_merge_size = self.spatial_merge_size

        position_ids = torch.zeros(3, bs, seq_len, dtype=torch.long, device=device)

        if image_grid_thw is None or image_grid_thw.numel() == 0:
            pos = torch.arange(seq_len, device=device).unsqueeze(0).expand(bs, -1)
            position_ids[0] = pos
            position_ids[1] = pos
            position_ids[2] = pos
            return position_ids

        grid_iter = iter(image_grid_thw)

        for b in range(bs):
            ids = input_ids_bshd[b]
            is_image = ids == self.image_token_id

            token_types = is_image.long()
            groups = []
            for key, group in itertools.groupby(enumerate(token_types.tolist()), lambda x: x[1]):
                g = list(group)
                groups.append((key, g[0][0], g[-1][0] + 1))

            current_pos = 0
            pos_list = []
            for modality, start, end in groups:
                if modality == 0:
                    n = end - start
                    pos_list.append(torch.arange(n, device=device).view(1, -1).expand(3, -1) + current_pos)
                    current_pos += n
                else:
                    grid_thw = next(grid_iter)
                    temp_merge_size = grid_thw[0]
                    vis_pos = self._get_vision_position_ids(
                        current_pos,
                        grid_thw,
                        temp_merge_size,
                        spatial_merge_size,
                        device,
                    )
                    pos_list.append(vis_pos)
                    current_pos += max(grid_thw[1], grid_thw[2]) // spatial_merge_size

            all_pos = torch.cat(pos_list, dim=1)
            position_ids[:, b, : all_pos.shape[1]] = all_pos

        return position_ids


    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor = None,
        attention_mask: torch.Tensor = None,
        labels: torch.Tensor = None,
        loss_mask: torch.Tensor = None,
        inference_params=None,
        packed_seq_params: PackedSeqParams = None,
        extra_block_kwargs: dict = None,
        pixel_values: torch.Tensor = None,
        image_grid_thw: torch.Tensor = None,
        pixel_values_videos: torch.Tensor = None,
        video_grid_thw: torch.Tensor = None,
        mm_token_type_ids: torch.Tensor = None,
        **kwargs,
    ) -> torch.Tensor:
        assert pixel_values_videos is None, "Video not supported yet"
        assert inference_params is None, "Inference not supported"

        cu_seqlens = None
        if packed_seq_params is not None:
            cu_seqlens = (
                packed_seq_params.cu_seqlens_q_padded
                if packed_seq_params.cu_seqlens_q_padded is not None
                else packed_seq_params.cu_seqlens_q
            )
        cp_size = parallel_state.get_context_parallel_world_size()
        full_input_ids = None

        combined_embeddings = None

        if self.pre_process:
            combined_embeddings = self.language_model.embedding(
                input_ids=input_ids,
                position_ids=None,
            ).clone()

            if pixel_values is not None and image_grid_thw is not None:
                image_embeds = self._get_image_features(pixel_values, image_grid_thw)
                image_embeds = image_embeds.to(combined_embeddings.device, combined_embeddings.dtype)

                if cp_size > 1 and cu_seqlens is not None:
                    full_input_ids = _gather_input_ids_from_cp(input_ids, cu_seqlens)
                    cp_rank = parallel_state.get_context_parallel_rank()
                    image_embeds = _select_local_image_embeds(
                        full_input_ids,
                        cu_seqlens,
                        self.image_token_id,
                        image_embeds,
                        cp_rank,
                        cp_size,
                    )

                image_mask = (input_ids == self.image_token_id).contiguous()
                combined_embeddings = combined_embeddings.transpose(0, 1).contiguous()
                if image_mask.any():
                    combined_embeddings[image_mask] = image_embeds
                combined_embeddings = combined_embeddings.transpose(0, 1).contiguous()

            if self.config.sequence_parallel:
                combined_embeddings = tensor_parallel.scatter_to_sequence_parallel_region(combined_embeddings)
                combined_embeddings = combined_embeddings.contiguous()

        pp_size = parallel_state.get_pipeline_model_parallel_world_size()

        if position_ids is None:
            if self.pre_process:
                if cu_seqlens is not None:
                    if cp_size > 1:
                        if full_input_ids is None:
                            full_input_ids = _gather_input_ids_from_cp(input_ids, cu_seqlens)
                    else:
                        full_input_ids = input_ids
                    input_ids_bshd = _thd_to_bshd(full_input_ids, cu_seqlens)
                    pos_bshd = self._compute_mrope_position_ids(input_ids_bshd, image_grid_thw)
                    pos_packed = _bshd_to_thd(pos_bshd.permute(1, 2, 0), cu_seqlens)
                    position_ids = pos_packed.permute(2, 0, 1).contiguous()
                else:
                    position_ids = self._compute_mrope_position_ids(input_ids, image_grid_thw)
            else:
                if cu_seqlens is not None:
                    T = cu_seqlens[-1].item()
                    position_ids = torch.zeros(3, 1, T, dtype=torch.long, device=torch.cuda.current_device())
                else:
                    raise NotImplementedError(
                        "Non-THD position_ids broadcast not yet supported for non-first PP stages"
                    )

            if pp_size > 1:
                src = parallel_state.get_pipeline_model_parallel_first_rank()
                torch.distributed.broadcast(
                    position_ids,
                    src=src,
                    group=parallel_state.get_pipeline_model_parallel_group(),
                )

        output = self.language_model(
            input_ids=None,
            position_ids=position_ids,
            attention_mask=attention_mask,
            decoder_input=combined_embeddings,
            labels=labels,
            loss_mask=loss_mask,
            inference_params=inference_params,
            packed_seq_params=packed_seq_params,
            **(extra_block_kwargs or {}),
        )

        return output


@dataclass
class Glm4vMoeVLModelProvider(Qwen3MoEModelProvider):

    image_token_id: int = 151363
    video_token_id: int = 151364
    spatial_merge_size: int = 2

    hf_vision_config: object = None
    hf_text_config: object = None

    position_embedding_type: str = "mrope"
    mrope_section: list[int] = field(default_factory=lambda: [8, 12, 12])
    scatter_embedding_sequence_parallel: bool = False

    language_max_sequence_length: int = 131072

    def provide(self, pre_process=None, post_process=None, vp_stage=None):

        if pre_process is None:
            pre_process = parallel_state.is_pipeline_first_stage(ignore_virtual=False, vp_stage=vp_stage)
        if post_process is None:
            post_process = parallel_state.is_pipeline_last_stage(ignore_virtual=False, vp_stage=vp_stage)

        transformer_layer_spec = get_gpt_decoder_block_spec(
            config=self,
            use_transformer_engine=True,
            vp_stage=vp_stage,
        )

        model = Glm4vMoeVLModel(
            language_transformer_config=self,
            language_transformer_layer_spec=transformer_layer_spec,
            hf_vision_config=self.hf_vision_config,
            parallel_output=True,
            pre_process=pre_process,
            post_process=post_process,
        )

        return model


try:
    from transformers import Glm4vMoeForConditionalGeneration as _Glm4vMoeHF
except ImportError:
    _Glm4vMoeHF = "Glm4vMoeForConditionalGeneration"


@MegatronModelBridge.register_bridge(source=_Glm4vMoeHF, target=Glm4vMoeVLModel)
class Glm4vMoeBridge(MegatronModelBridge):

    def provider_bridge(self, hf_pretrained):
        hf_config = hf_pretrained.config
        text_config = hf_config.text_config
        vision_config = deepcopy(hf_config.vision_config)

        model_dtype = self.dtype_from_hf(text_config, default=torch.bfloat16)
        vision_config.torch_dtype = model_dtype

        ProviderClass = Glm4vMoeVLModelProvider

        rope_params = getattr(text_config, "rope_parameters", {}) or {}
        mrope_section = rope_params.get("mrope_section", [8, 12, 12])
        rotary_base = rope_params.get("rope_theta", 500000)
        partial_rotary_factor = rope_params.get("partial_rotary_factor", 0.5)

        first_k_dense = getattr(text_config, "first_k_dense_replace", 1)
        num_layers = text_config.num_hidden_layers
        moe_layer_freq_list = [0] * first_k_dense + [1] * (num_layers - first_k_dense)

        n_shared = getattr(text_config, "n_shared_experts", 1)
        moe_ffn = getattr(text_config, "moe_intermediate_size", 1408)
        shared_expert_intermediate = moe_ffn * n_shared

        provider = ProviderClass(
            num_layers=num_layers,
            hidden_size=text_config.hidden_size,
            ffn_hidden_size=text_config.intermediate_size,
            num_attention_heads=text_config.num_attention_heads,
            num_query_groups=text_config.num_key_value_heads,
            kv_channels=getattr(text_config, "head_dim", 128),
            init_method_std=text_config.initializer_range,
            layernorm_epsilon=text_config.rms_norm_eps,
            gated_linear_unit=True,
            make_vocab_size_divisible_by=self.make_vocab_size_divisible_by(text_config.vocab_size),
            rotary_base=rotary_base,
            rotary_percent=partial_rotary_factor,
            share_embeddings_and_output_weights=getattr(text_config, "tie_word_embeddings", False),
            vocab_size=text_config.vocab_size,
            seq_length=text_config.max_position_embeddings,
            fp16=(model_dtype == torch.float16),
            bf16=(model_dtype == torch.bfloat16),
            params_dtype=model_dtype,
            num_moe_experts=getattr(text_config, "n_routed_experts", 128),
            moe_router_topk=getattr(text_config, "num_experts_per_tok", 8),
            moe_ffn_hidden_size=moe_ffn,
            moe_shared_expert_intermediate_size=shared_expert_intermediate,
            moe_layer_freq=moe_layer_freq_list,
            moe_grouped_gemm=True,
            moe_router_load_balancing_type="seq_aux_loss",
            moe_aux_loss_coeff=0,
            moe_router_score_function="sigmoid",
            moe_router_pre_softmax=True,
            moe_router_enable_expert_bias=True,
            moe_router_dtype="fp32",
            add_qkv_bias=getattr(text_config, "attention_bias", True),
            qk_layernorm=getattr(text_config, "qk_layernorm", False) or getattr(text_config, "use_qk_norm", False),
            mrope_section=mrope_section,
            position_embedding_type="mrope",
            scatter_embedding_sequence_parallel=False,
            hf_vision_config=vision_config,
            hf_text_config=text_config,
            image_token_id=getattr(hf_config, "image_token_id", 151363),
            video_token_id=getattr(hf_config, "video_token_id", 151364),
            spatial_merge_size=getattr(hf_config.vision_config, "spatial_merge_size", 2),
            language_max_sequence_length=text_config.max_position_embeddings,
        )

        return provider

    def mapping_registry(self) -> MegatronMappingRegistry:
        param_mappings = {
            "language_model.embedding.word_embeddings.weight": "model.language_model.embed_tokens.weight",
            "language_model.output_layer.weight": "lm_head.weight",
            "language_model.decoder.final_layernorm.weight": "model.language_model.norm.weight",
            "language_model.decoder.layers.*.self_attention.linear_qkv.layer_norm_weight": "model.language_model.layers.*.input_layernorm.weight",
            "language_model.decoder.layers.*.input_layernorm.weight": "model.language_model.layers.*.input_layernorm.weight",
            "language_model.decoder.layers.*.self_attention.linear_proj.weight": "model.language_model.layers.*.self_attn.o_proj.weight",
            "language_model.decoder.layers.*.pre_mlp_layernorm.weight": "model.language_model.layers.*.post_attention_layernorm.weight",
            "language_model.decoder.layers.*.mlp.linear_fc1.layer_norm_weight": "model.language_model.layers.*.post_attention_layernorm.weight",
            "language_model.decoder.layers.*.mlp.linear_fc2.weight": "model.language_model.layers.*.mlp.down_proj.weight",
            "language_model.decoder.layers.*.mlp.router.weight": "model.language_model.layers.*.mlp.gate.weight",
            "language_model.decoder.layers.*.mlp.router.expert_bias": "model.language_model.layers.*.mlp.gate.e_score_correction_bias",
            "language_model.decoder.layers.*.mlp.experts.linear_fc2.weight*": "model.language_model.layers.*.mlp.experts.*.down_proj.weight",
            "language_model.decoder.layers.*.mlp.shared_experts.linear_fc2.weight": "model.language_model.layers.*.mlp.shared_experts.down_proj.weight",
        }

        mapping_list = []
        for megatron_param, hf_param in param_mappings.items():
            mapping_list.append(AutoMapping(megatron_param=megatron_param, hf_param=hf_param))

        mapping_list.extend(
            [
                ReplicatedMapping(
                    megatron_param="vision_model.**",
                    hf_param="model.visual.**",
                ),
                QKVMapping(
                    megatron_param="language_model.decoder.layers.*.self_attention.linear_qkv.weight",
                    q="model.language_model.layers.*.self_attn.q_proj.weight",
                    k="model.language_model.layers.*.self_attn.k_proj.weight",
                    v="model.language_model.layers.*.self_attn.v_proj.weight",
                ),
                QKVMapping(
                    megatron_param="language_model.decoder.layers.*.self_attention.linear_qkv.bias",
                    q="model.language_model.layers.*.self_attn.q_proj.bias",
                    k="model.language_model.layers.*.self_attn.k_proj.bias",
                    v="model.language_model.layers.*.self_attn.v_proj.bias",
                ),
                GatedMLPMapping(
                    megatron_param="language_model.decoder.layers.*.mlp.linear_fc1.weight",
                    gate="model.language_model.layers.*.mlp.gate_proj.weight",
                    up="model.language_model.layers.*.mlp.up_proj.weight",
                ),
                GatedMLPMapping(
                    megatron_param="language_model.decoder.layers.*.mlp.experts.linear_fc1.weight*",
                    gate="model.language_model.layers.*.mlp.experts.*.gate_proj.weight",
                    up="model.language_model.layers.*.mlp.experts.*.up_proj.weight",
                ),
                GatedMLPMapping(
                    megatron_param="language_model.decoder.layers.*.mlp.experts.local_experts.*.linear_fc1.weight",
                    gate="model.language_model.layers.*.mlp.experts.*.gate_proj.weight",
                    up="model.language_model.layers.*.mlp.experts.*.up_proj.weight",
                ),
                AutoMapping(
                    megatron_param="language_model.decoder.layers.*.mlp.experts.local_experts.*.linear_fc2.weight",
                    hf_param="model.language_model.layers.*.mlp.experts.*.down_proj.weight",
                ),
                GatedMLPMapping(
                    megatron_param="language_model.decoder.layers.*.mlp.shared_experts.linear_fc1.weight",
                    gate="model.language_model.layers.*.mlp.shared_experts.gate_proj.weight",
                    up="model.language_model.layers.*.mlp.shared_experts.up_proj.weight",
                ),
            ]
        )

        return MegatronMappingRegistry(*mapping_list)
