# Modified by MAI-Gym contributors.
import argparse

from sglang.srt.server_args import ServerArgs
from slime.utils.http_utils import _wrap_ipv6


def add_sglang_router_arguments(parser):
    parser.add_argument(
        "--sglang-router-ip",
        type=str,
        default=None,
        help="IP address of the SGLang router",
    )
    parser.add_argument(
        "--sglang-router-port",
        type=int,
        default=None,
        help="Port of the SGLang router",
    )
    parser.add_argument(
        "--sglang-router-request-timeout-secs",
        type=int,
        default=14400,
        help="Timeout for requests to the SGLang router in seconds",
    )
    return parser


def add_sglang_arguments(parser):
    parser = add_sglang_router_arguments(parser)
    parser.set_defaults(router_balance_abs_threshold=10, router_balance_rel_threshold=1.2)
    parser.add_argument("--sglang-server-concurrency", type=int, default=512)

    old_add_argument = parser.add_argument

    skipped_args = [
        "model_path",
        "config",
        "trust_remote_code",
        "random_seed",
        "enable_memory_saver",
        "tp_size",
        "port",
        "nnodes",
        "node_rank",
        "dist_init_addr",
        "gpu_id_step",
        "base_gpu_id",
        "nccl_port",
        "skip_server_warmup",
        "enable_return_routed_experts",
    ]

    def new_add_argument_wrapper(*name_or_flags, **kwargs):
        canonical_name_for_skip_check = None
        if "dest" in kwargs:
            canonical_name_for_skip_check = kwargs["dest"]
        else:
            for flag_name_candidate in name_or_flags:
                if isinstance(flag_name_candidate, str) and flag_name_candidate.startswith("--"):
                    stem = flag_name_candidate[2:]
                    canonical_name_for_skip_check = stem.replace("-", "_")
                    break

        if canonical_name_for_skip_check and canonical_name_for_skip_check in skipped_args:
            return

        new_name_or_flags_list = []
        for item_flag in name_or_flags:
            if isinstance(item_flag, str) and item_flag.startswith("-"):
                original_flag_stem = item_flag.lstrip("-")
                prefixed_item = f"--sglang-{original_flag_stem}"
                new_name_or_flags_list.append(prefixed_item)
            else:
                new_name_or_flags_list.append(item_flag)

        final_kwargs = kwargs.copy()

        if "dest" in final_kwargs and isinstance(final_kwargs["dest"], str):
            original_dest = final_kwargs["dest"]
            if not original_dest.startswith("sglang_"):
                final_kwargs["dest"] = f"sglang_{original_dest}"

        old_add_argument(*new_name_or_flags_list, **final_kwargs)

    parser.add_argument = new_add_argument_wrapper
    ServerArgs.add_cli_args(parser)
    parser.add_argument = old_add_argument

    parser.add_argument(
        "--prefill-num-servers",
        type=int,
        default=None,
        help="Number of prefill servers for disaggregation.",
    )
    parser.add_argument(
        "--sglang-config",
        type=str,
        default=None,
        help=(
            "Path to a YAML config for SGLang engine deployment. "
            "Defines server_groups with worker_type (regular/prefill/decode/placeholder), "
            "num_gpus per group, and optional per-group 'overrides' dict of "
            "ServerArgs field names that override the base --sglang-* CLI args. "
            "Placeholder groups reserve GPU slots without creating engines. "
            "Mutually exclusive with --prefill-num-servers."
        ),
    )

    return parser


def validate_args(args):
    args.sglang_dp_size = args.sglang_data_parallel_size
    args.sglang_pp_size = args.sglang_pipeline_parallel_size
    args.sglang_ep_size = args.sglang_expert_parallel_size

    if args.sglang_pp_size > 1:
        assert args.rollout_num_gpus_per_engine % args.sglang_pp_size == 0, (
            f"rollout_num_gpus_per_engine ({args.rollout_num_gpus_per_engine}) must be divisible by "
            f"sglang_pipeline_parallel_size ({args.sglang_pp_size})"
        )
        args.sglang_tp_size = args.rollout_num_gpus_per_engine // args.sglang_pp_size
    else:
        args.sglang_tp_size = args.rollout_num_gpus_per_engine

    if args.sglang_dp_size > 1:
        assert args.sglang_enable_dp_attention

    if getattr(args, "sglang_router_ip", None):
        args.sglang_router_ip = _wrap_ipv6(args.sglang_router_ip)

    assert not (
        getattr(args, "prefill_num_servers", None) is not None and args.rollout_external
    ), "prefill_num_servers cannot be set when rollout_external is set."

    assert not (
        getattr(args, "sglang_config", None) is not None and args.rollout_external
    ), "sglang_config cannot be set when rollout_external is set."

    assert not (
        getattr(args, "sglang_config", None) is not None and getattr(args, "prefill_num_servers", None) is not None
    ), "sglang_config and prefill_num_servers are mutually exclusive. Use server_groups in the YAML config instead."


def sglang_parse_args():
    parser = argparse.ArgumentParser(add_help=False)
    add_sglang_arguments(parser)

    temp_parser = argparse.ArgumentParser(add_help=False)
    temp_parser.add_argument("--rollout-num-gpus-per-engine", type=int, default=1)
    temp_parser.add_argument("--sglang-pp-size", type=int, default=1)
    temp_parser.add_argument("--sglang-pipeline-parallel-size", type=int, default=1)
    temp_args, _ = temp_parser.parse_known_args()
    pp_size = temp_args.sglang_pp_size if temp_args.sglang_pp_size != 1 else temp_args.sglang_pipeline_parallel_size
    sglang_tp_size = temp_args.rollout_num_gpus_per_engine // pp_size
    parser.set_defaults(sglang_tensor_parallel_size=sglang_tp_size)

    args, _ = parser.parse_known_args()
    return args
