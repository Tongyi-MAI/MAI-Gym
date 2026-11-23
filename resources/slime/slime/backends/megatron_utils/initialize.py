# Modified by MAI-Gym contributors.
import logging
import random

import numpy as np
import torch
from megatron.core import mpu, tensor_parallel
from megatron.core.config import set_experimental_flag
from megatron.core.num_microbatches_calculator import init_num_microbatches_calculator
from megatron.training.global_vars import _build_tokenizer, set_args

logger = logging.getLogger(__name__)


def _set_random_seed(
    seed_: int,
    data_parallel_random_init: bool = False,
    te_rng_tracker: bool = False,
    inference_rng_tracker: bool = False,
    use_cudagraphable_rng: bool = False,
):
    seed = seed_ + (100 * mpu.get_pipeline_model_parallel_rank())
    if data_parallel_random_init:
        seed = seed + (10 * mpu.get_data_parallel_rank(with_context_parallel=False))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    tensor_parallel.model_parallel_cuda_manual_seed(seed, te_rng_tracker, inference_rng_tracker, use_cudagraphable_rng)


def _initialize_distributed(args, get_embedding_ranks=None, get_position_embedding_ranks=None):
    mpu.initialize_model_parallel(
        args.tensor_model_parallel_size,
        args.pipeline_model_parallel_size,
        args.virtual_pipeline_model_parallel_size,
        pipeline_model_parallel_comm_backend=args.pipeline_model_parallel_comm_backend,
        context_parallel_size=args.context_parallel_size,
        hierarchical_context_parallel_sizes=args.hierarchical_context_parallel_sizes,
        expert_model_parallel_size=args.expert_model_parallel_size,
        num_distributed_optimizer_instances=args.num_distributed_optimizer_instances,
        expert_tensor_parallel_size=args.expert_tensor_parallel_size,
        distributed_timeout_minutes=args.distributed_timeout_minutes,
        nccl_communicator_config_path=args.nccl_communicator_config_path,
        order="tp-cp-ep-dp-pp" if not args.use_tp_pp_dp_mapping else "tp-cp-ep-pp-dp",
        get_embedding_ranks=get_embedding_ranks,
        get_position_embedding_ranks=get_position_embedding_ranks,
        create_gloo_process_groups=args.enable_gloo_process_groups,
    )


def init(args):
    set_args(args)
    if args.enable_experimental:
        logger.info("Enable megatron experimental")
        set_experimental_flag(True)

    _initialize_distributed(args)

    assert np.__version__.startswith("1."), "Megatron does not support numpy 2.x"

    if args.rank == 0:
        logger.info(f"> setting random seeds to {args.seed} ...")
    _set_random_seed(
        args.seed,
        args.data_parallel_random_init,
        args.te_rng_tracker,
        args.inference_rng_tracker,
    )
    _build_tokenizer(args)
    init_num_microbatches_calculator(
        args.rank,
        args.rampup_batch_size,
        args.global_batch_size,
        args.micro_batch_size,
        args.data_parallel_size,
        args.decrease_batch_size_if_needed,
    )

    if args.deterministic_mode:
        if args.rank == 0:
            logger.info("> running in deterministic mode")
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True, warn_only=False)

    if args.tp_comm_overlap:
        from megatron.training.initialize import _initialize_tp_communicators

        _initialize_tp_communicators()

    if getattr(args, "custom_megatron_init_path", None):
        from slime.utils.misc import load_function

        custom_init = load_function(args.custom_megatron_init_path)
        custom_init(args)


def is_megatron_main_rank():
    return (
        mpu.get_data_parallel_rank(with_context_parallel=True) == 0
        and mpu.get_tensor_model_parallel_rank() == 0
        and mpu.get_pipeline_model_parallel_rank() == mpu.get_pipeline_model_parallel_world_size() - 1
    )
