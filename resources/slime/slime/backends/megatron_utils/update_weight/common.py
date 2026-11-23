# Modified by MAI-Gym contributors.
import inspect
import re
from argparse import Namespace
from collections.abc import Iterator, Sequence

import torch
import torch.distributed as dist
from megatron.core import mpu
from megatron.core.transformer.transformer_layer import get_transformer_layer_offset

from slime.backends.megatron_utils.misc_utils import strip_param_name_prefix
from slime.utils.qwen35_debug import qdebug, qwen35_debug_enabled, tstat
from slime.utils.types import ParamInfo


def all_gather_param(name: str, param: torch.nn.Parameter) -> torch.Tensor:
    _dbg = qwen35_debug_enabled()
    if "expert_bias" in name:
        if _dbg:
            qdebug("TPGATHER", f"{name}: expert_bias passthrough {tstat(param)}")
        return param

    assert hasattr(param, "tensor_model_parallel"), f"{name} does not have tensor_model_parallel attribute"
    if not param.tensor_model_parallel or getattr(param, "parallel_mode", None) == "duplicated":
        if _dbg:
            qdebug(
                "TPGATHER",
                f"{name}: non-TP passthrough tp_flag={param.tensor_model_parallel} "
                f"parallel_mode={getattr(param, 'parallel_mode', None)} {tstat(param.data)}",
            )
        return param.data

    if ".experts." in name:
        tp_size = mpu.get_expert_tensor_parallel_world_size()
        tp_group = mpu.get_expert_tensor_parallel_group()
    else:
        tp_size = mpu.get_tensor_model_parallel_world_size()
        tp_group = mpu.get_tensor_model_parallel_group()

    param_partitions = [torch.empty_like(param.data) for _ in range(tp_size)]
    dist.all_gather(param_partitions, param.data, group=tp_group)
    partition_dim = param.partition_dim
    assert param.partition_stride == 1 or (
        param.partition_stride == 2 and "linear_fc1" in name
    ), "partition_stride != 1 is not supported"
    _dbg_shard_shape = tuple(param.data.shape)
    _dbg_partition_dim = param.partition_dim
    _dbg_partition_stride = param.partition_stride
    glu_rechunk = ("linear_fc1.weight" in name or "linear_fc1.bias" in name) and "vision_model" not in name
    if glu_rechunk:
        param_partitions = [p.chunk(2, dim=0) for p in param_partitions]
        param_partitions = [p[0] for p in param_partitions] + [p[1] for p in param_partitions]
    fc2_dim_fix = "linear_fc2.weight" in name and partition_dim == 0
    if fc2_dim_fix:
        partition_dim = 1
    param = torch.cat(param_partitions, dim=partition_dim)
    if _dbg:
        qdebug(
            "TPGATHER",
            f"{name}: tp_size={tp_size} shard={_dbg_shard_shape} "
            f"partition_dim={_dbg_partition_dim} partition_stride={_dbg_partition_stride} "
            f"glu_rechunk={glu_rechunk} fc2_dim_fix={fc2_dim_fix} cat_dim={partition_dim} "
            f"-> {tstat(param)}",
        )
    return param


def all_gather_params_async(
    param_infos_and_params: list[tuple[ParamInfo, torch.Tensor]],
) -> list[torch.Tensor]:
    gather_tasks = []
    handles = []

    for info, param in param_infos_and_params:
        if "expert_bias" in info.name:
            gather_tasks.append((info, param, None, None, None))
            handles.append(None)
        elif not param.tensor_model_parallel or getattr(param, "parallel_mode", None) == "duplicated":
            gather_tasks.append((info, param.data, None, None, None))
            handles.append(None)
        else:
            if ".experts." in info.name:
                tp_size = mpu.get_expert_tensor_parallel_world_size()
                tp_group = mpu.get_expert_tensor_parallel_group()
            else:
                tp_size = mpu.get_tensor_model_parallel_world_size()
                tp_group = mpu.get_tensor_model_parallel_group()

            param_partitions = [torch.empty_like(param.data) for _ in range(tp_size)]
            handle = dist.all_gather(param_partitions, param.data, group=tp_group, async_op=True)
            gather_tasks.append((info, None, handle, param_partitions, param.partition_dim))
            handles.append(handle)

    for handle in handles:
        if handle is not None:
            handle.wait()

    gathered_params = []
    for info, direct_param, handle, param_partitions, partition_dim in gather_tasks:
        if handle is None:
            param = direct_param
        else:
            assert partition_dim is not None, "partition_stride != 1 is not supported"
            if ("linear_fc1.weight" in info.name or "linear_fc1.bias" in info.name) and "vision_model" not in info.name:
                param_partitions = [p.chunk(2, dim=0) for p in param_partitions]
                param_partitions = [p[0] for p in param_partitions] + [p[1] for p in param_partitions]
            if "linear_fc2.weight" in info.name:
                if partition_dim == 0:
                    partition_dim = 1
            param = torch.cat(param_partitions, dim=partition_dim)

        gathered_params.append(param)

    return gathered_params


def named_params_and_buffers(
    args: Namespace,
    model: Sequence[torch.nn.Module],
    convert_to_global_name: bool = True,
    translate_gpu_to_cpu: bool = False,
) -> Iterator[tuple[str, torch.Tensor]]:
    if convert_to_global_name:
        ans = _named_params_and_buffers_global(args, model)
    else:
        ans = _named_params_and_buffers_vanilla(model)

    if translate_gpu_to_cpu:
        ans = ((name, _maybe_get_cpu_backup(tensor)) for name, tensor in ans)

    return ans


def _maybe_get_cpu_backup(x: torch.Tensor):
    from torch_memory_saver import torch_memory_saver

    if (cpu_tensor := torch_memory_saver.get_cpu_backup(x)) is not None:
        return cpu_tensor

    return x


def _named_params_and_buffers_vanilla(model: Sequence[torch.nn.Module]) -> Iterator[tuple[str, torch.Tensor]]:
    for vp_stage, model_module in enumerate(model):

        def _compute_fqn(name, vp_stage=vp_stage):
            return f"vp_stages.{vp_stage}.{strip_param_name_prefix(name)}"

        for name, param in model_module.named_parameters():
            yield _compute_fqn(name), param

        for name, buffer in model_module.named_buffers():
            if "expert_bias" not in name:
                continue
            yield _compute_fqn(name), buffer


def _named_params_and_buffers_global(
    args: Namespace, model: Sequence[torch.nn.Module]
) -> Iterator[tuple[str, torch.Tensor]]:
    _dbg = qwen35_debug_enabled()
    ep_size = mpu.get_expert_model_parallel_world_size()
    ep_rank = mpu.get_expert_model_parallel_rank()
    if args.num_experts:
        expert_offset = ep_rank * args.num_experts // ep_size

    sig = inspect.signature(get_transformer_layer_offset)
    need_vp_stage = "vp_stage" in sig.parameters
    if _dbg:
        qdebug(
            "NAMES",
            f"_named_params_and_buffers_global: pp_rank={mpu.get_pipeline_model_parallel_rank()} "
            f"pp_size={mpu.get_pipeline_model_parallel_world_size()} ep_size={ep_size} ep_rank={ep_rank} "
            f"num_experts={args.num_experts} need_vp_stage={need_vp_stage} n_model_chunks={len(model)}",
        )

    for vp_stage, model_module in enumerate(model):
        if need_vp_stage:
            layer_offset = get_transformer_layer_offset(model_module.config, vp_stage)
        else:
            layer_offset = get_transformer_layer_offset(model_module.config)
        if _dbg:
            qdebug("NAMES", f"vp_stage={vp_stage} layer_offset={layer_offset}")
        for name, param in model_module.named_parameters():
            if not name.startswith("module.module."):
                name = "module." + name

            decoder_layers_pattern = (
                r"module\.module\.(language_model\.)?decoder\.layers\.(\d+)\.(.+)"
            )
            match = re.match(decoder_layers_pattern, name)
            if not match:
                mtp_layers_pattern = r"module\.module\.mtp\.layers\.(\d+)\.(.+)"
                match = re.match(mtp_layers_pattern, name)
                if not match:
                    if _dbg:
                        qdebug("NAMES", f"non-layer      vp={vp_stage} yield-as-is name='{name}'")
                    yield name, param
                    continue

                layer_idx, rest = match.groups()
                expert_pattern = r"transformer_layer\.mlp\.experts\.(.+)\.(weight|bias)(\d+)"
                match = re.match(expert_pattern, rest)
                if not match:
                    if _dbg:
                        qdebug("NAMES", f"mtp-plain      vp={vp_stage} yield-as-is name='{name}'")
                    yield name, param
                    continue

                rest, param_type, expert_idx = match.groups()
                expert_idx = int(expert_idx) + expert_offset
                _mtp_name = (
                    f"module.module.mtp.layers.{layer_idx}.transformer_layer.mlp.experts."
                    f"{rest}.{param_type}{expert_idx}"
                )
                if _dbg:
                    qdebug("NAMES", f"mtp-expert     vp={vp_stage} mtp_layer={layer_idx} -> '{_mtp_name}'")
                yield _mtp_name, param
                continue

            lm_infix, layer_idx, rest = match.groups()
            lm_infix = lm_infix or ""
            _local_layer_idx = int(layer_idx)
            layer_idx = _local_layer_idx + layer_offset

            expert_pattern = r"mlp\.experts\.(.+)\.(weight|bias)(\d+)"
            match = re.match(expert_pattern, rest)
            if match:
                rest, param_type, expert_idx = match.groups()
                _local_expert_idx = int(expert_idx)
                expert_idx = _local_expert_idx + expert_offset
                _exp_name = (
                    f"module.module.{lm_infix}decoder.layers.{layer_idx}.mlp.experts."
                    f"{rest}.{param_type}{expert_idx}"
                )
                if _dbg:
                    qdebug(
                        "NAMES",
                        f"decoder-expert vp={vp_stage} lm_infix='{lm_infix}' "
                        f"local_layer={_local_layer_idx}+off={layer_offset}=>global={layer_idx} "
                        f"local_expert={_local_expert_idx}+eoff={expert_offset}=>{expert_idx} -> '{_exp_name}'",
                    )
                yield _exp_name, param
            else:
                _dec_name = f"module.module.{lm_infix}decoder.layers.{layer_idx}.{rest}"
                if _dbg:
                    qdebug(
                        "NAMES",
                        f"decoder        vp={vp_stage} lm_infix='{lm_infix}' "
                        f"local_layer={_local_layer_idx}+off={layer_offset}=>global={layer_idx} "
                        f"rest='{rest}' -> '{_dec_name}'",
                    )
                yield _dec_name, param

        for name, buffer in model_module.named_buffers():
            if "expert_bias" not in name:
                continue
            if not name.startswith("module.module."):
                name = "module." + name

            if name.startswith("module.module.language_model."):
                name = "module.module." + name[len("module.module.language_model.") :]

            decoder_layers_pattern = r"module\.module\.decoder\.layers\.(\d+)\.(.+)"
            match = re.match(decoder_layers_pattern, name)
            if not match:
                if _dbg:
                    qdebug("NAMES", f"expert_bias-buf vp={vp_stage} yield-as-is name='{name}'")
                yield name, buffer
            else:
                layer_idx, rest = match.groups()
                layer_idx = int(layer_idx) + layer_offset
                _buf_name = f"module.module.decoder.layers.{layer_idx}.{rest}"
                if _dbg:
                    qdebug("NAMES", f"expert_bias-buf vp={vp_stage} global_layer={layer_idx} -> '{_buf_name}'")
                yield _buf_name, buffer
