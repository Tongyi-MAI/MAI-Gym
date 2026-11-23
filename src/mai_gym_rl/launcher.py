#!/usr/bin/env python3

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import runpy
import shlex
import sys
import urllib.error
import urllib.request
from pathlib import Path

CLI_DESCRIPTION = "Start colocated Qwen3.5-9B GUI-agent GRPO training on an existing Ray cluster."

ROOT = (
    Path(os.environ.get("MAI_GYM_ROOT") or Path(__file__).resolve().parents[2])
    .expanduser()
    .resolve()
)
TRAINER = ROOT / "resources/slime/train.py"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=CLI_DESCRIPTION, allow_abbrev=False)
    parser.add_argument("--model", type=Path, required=True, help="Complete local HF model export")
    parser.add_argument("--data", type=Path, required=True, help="Training .jsonl or .parquet file")
    parser.add_argument("--output", type=Path, required=True, help="Run output directory")
    parser.add_argument(
        "--adapter-config",
        type=Path,
        required=True,
        help="Environment-manager and GUI-agent configuration",
    )
    parser.add_argument("--resume", type=Path, help="Explicit Megatron checkpoint directory")
    parser.add_argument("--run-name", help="Stable manager run ID; otherwise derived from output")
    parser.add_argument("--ray-address", default=os.environ.get("RAY_ADDRESS", "auto"))
    parser.add_argument("--nodes", type=int, default=4)
    parser.add_argument("--gpus-per-node", type=int, default=8)
    parser.add_argument("--tp", type=int, default=4)
    parser.add_argument("--pp", type=int, default=2)
    parser.add_argument("--cp", type=int, default=4)
    parser.add_argument("--engine-gpus", type=int, default=2)
    parser.add_argument("--rollouts", type=int, default=2000)
    parser.add_argument("--rollout-batch-size", type=int, default=100, help="Tasks per rollout")
    parser.add_argument("--samples-per-prompt", type=int, default=8)
    parser.add_argument(
        "--global-batch-size",
        type=int,
        help="Step/window samples per optimizer step; defaults to 320",
    )
    parser.add_argument("--lr", type=float, default=3e-6)
    parser.add_argument(
        "--kl-loss-coef", type=float, default=0.0, help="Enable KL loss if positive"
    )
    parser.add_argument("--max-response-len", type=int, default=6144)
    parser.add_argument("--max-context-len", type=int, default=65536)
    parser.add_argument("--save-interval", type=int, default=1)
    parser.add_argument(
        "--check-env-pools",
        action="store_true",
        help="Check live manager backend availability before allocating GPUs (skipped in dry-run)",
    )
    parser.add_argument("--input-key", default="prompt")
    parser.add_argument("--metadata-key", default="metadata")
    parser.add_argument("--label-key", default="label")
    parser.add_argument("--only-env-type", help="Reject every data row outside this backend")
    parser.add_argument("--max-env-sessions", type=int, help="Cap total primary + backup sessions")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate local inputs and print the trainer command without connecting to Ray",
    )
    parser.add_argument("trainer_args", nargs=argparse.REMAINDER, help="Extra Slime args after --")
    args = parser.parse_args(argv)
    if args.trainer_args[:1] == ["--"]:
        args.trainer_args = args.trainer_args[1:]
    return args


def _records(path: Path):
    if path.suffix == ".jsonl":
        with path.open() as stream:
            for line in stream:
                if line.strip():
                    yield json.loads(line)
    elif path.suffix == ".parquet":
        import pyarrow.parquet as pq

        for batch in pq.ParquetFile(path).iter_batches(batch_size=1024):
            yield from batch.to_pylist()
    else:
        raise ValueError("Training data must be a .jsonl or .parquet file")


def _first_record(path: Path) -> dict:
    try:
        return next(_records(path))
    except StopIteration:
        raise ValueError("Training data is empty") from None


def config_environment(config: object) -> dict[str, str]:
    from mai_gym.utils.config_env import referenced_environment

    return referenced_environment(config)


def validate(args: argparse.Namespace) -> dict:
    import yaml

    from mai_gym.utils.config_env import resolve_environment

    if not Path(getattr(args, "trainer_entrypoint", TRAINER)).is_file():
        raise ValueError(
            "Training runtime is unavailable; set MAI_GYM_ROOT to the framework checkout"
        )
    for name in ("model", "data", "output", "adapter_config", "resume"):
        if getattr(args, name) is not None:
            setattr(args, name, getattr(args, name).expanduser().resolve())
    for name in (
        "nodes",
        "gpus_per_node",
        "tp",
        "pp",
        "cp",
        "engine_gpus",
        "rollouts",
        "rollout_batch_size",
        "samples_per_prompt",
        "max_response_len",
        "max_context_len",
        "save_interval",
    ):
        if getattr(args, name) < 1:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    if args.samples_per_prompt < 2:
        raise ValueError("GRPO needs at least two samples per prompt")
    if not math.isfinite(args.lr) or args.lr <= 0:
        raise ValueError("--lr must be finite and positive")
    if not math.isfinite(args.kl_loss_coef) or args.kl_loss_coef < 0:
        raise ValueError("--kl-loss-coef must be finite and nonnegative")
    if os.environ.get("DYNAMIC_TIMEOUT", "0") not in {"0", "1"}:
        raise ValueError("DYNAMIC_TIMEOUT must be 0 or 1")
    for name in ("TRAJ_TIMEOUT_S", "ROLLOUT_TIMEOUT_S"):
        if name in os.environ:
            value = float(os.environ[name])
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
    from types import SimpleNamespace

    from mai_gym_rl.slime_adapter.distributed_rollout import WorkerOptions

    workers = WorkerOptions.from_env()
    if workers.nodes != 0:
        workers.validate(SimpleNamespace(env_client_shards=1), SimpleNamespace())
    if "TIMEOUT_TRAJ_PER_HOUR" in os.environ:
        try:
            timeout_rate = float(os.environ["TIMEOUT_TRAJ_PER_HOUR"])
        except ValueError as exc:
            raise ValueError("TIMEOUT_TRAJ_PER_HOUR must be finite and positive") from exc
        if not math.isfinite(timeout_rate) or timeout_rate <= 0:
            raise ValueError("TIMEOUT_TRAJ_PER_HOUR must be finite and positive")
    world = args.nodes * args.gpus_per_node
    parallel = args.tp * args.pp * args.cp
    if world % parallel or args.gpus_per_node % args.tp or 4 % args.tp:
        raise ValueError("GPU topology must fit TP/PP/CP; TP must divide four query groups")
    if 32 % args.pp or args.tp * args.cp > 16:
        raise ValueError(
            "PP must divide 32 layers, and TP * CP cannot exceed 16 linear-attention heads"
        )
    if args.gpus_per_node % args.engine_gpus:
        raise ValueError("--engine-gpus must divide --gpus-per-node")
    dp = world // parallel
    if args.global_batch_size is None:
        args.global_batch_size = 320
    if args.global_batch_size < 1 or args.global_batch_size % dp:
        raise ValueError("--global-batch-size must be a positive multiple of data-parallel size")
    if args.max_response_len >= args.max_context_len:
        raise ValueError("--max-response-len must be smaller than --max-context-len")
    model_config = json.loads((args.model / "config.json").read_text())
    text = model_config.get("text_config", model_config)
    expected = {
        "hidden_size": 4096,
        "num_hidden_layers": 32,
        "num_attention_heads": 16,
        "num_key_value_heads": 4,
        "intermediate_size": 12288,
    }
    if any(text.get(key) != value for key, value in expected.items()):
        raise ValueError("Model dimensions do not match this Qwen3.5-9B preset")
    if (args.model / "latest_checkpointed_iteration.txt").exists():
        raise ValueError("--model must contain only an HF export; use --resume for Megatron state")
    if not any(args.model.glob("*.safetensors")) and not any(args.model.glob("pytorch_model*.bin")):
        raise ValueError("--model has no HF weight files")
    if args.resume and not (args.resume / "latest_checkpointed_iteration.txt").is_file():
        raise ValueError("--resume must contain latest_checkpointed_iteration.txt")
    if args.output.exists() and any(args.output.iterdir()) and not args.resume:
        raise ValueError("Output is not empty; choose a new directory or set --resume explicitly")
    if not args.run_name:
        digest = hashlib.sha256(str(args.output).encode()).hexdigest()[:8]
        args.run_name = f"{args.output.name}-{digest}"
    config = yaml.safe_load(args.adapter_config.read_text())
    if not isinstance(config, dict):
        raise ValueError("Adapter configuration must be a YAML mapping")
    args.adapter_environment = config_environment(config)
    config = resolve_environment(config)
    if config.get("rollout_agent", "gui") != "gui":
        raise ValueError("This preset uses GUI-agent rollouts")
    pools = config.get("env_pools") or {config.get("env_type"): config}
    if args.only_env_type and set(pools) != {args.only_env_type}:
        raise ValueError("Adapter pools must match --only-env-type")
    if args.max_env_sessions is not None:
        backup = config.get("num_backup_sessions", 0)
        counts = [
            pool.get("num_sessions", config.get("num_sessions", 0)) for pool in pools.values()
        ]
        if any(isinstance(n, bool) or not isinstance(n, int) or n < 0 for n in [backup, *counts]):
            raise ValueError("Session counts must be nonnegative integers")
        total = sum(counts) + len(pools) * backup
        if args.max_env_sessions < 1 or total > args.max_env_sessions:
            raise ValueError("Environment session count exceeds --max-env-sessions")
        concurrency = int(os.environ.get("ROLLOUT_CONCURRENCY", total))
        if not 1 <= concurrency <= total:
            raise ValueError("ROLLOUT_CONCURRENCY must be between 1 and the session count")
    for pool in pools.values():
        turns = pool.get("max_turns", config.get("max_turns", 20))
        if isinstance(turns, bool) or not isinstance(turns, int) or turns < 1:
            raise ValueError("Environment max_turns must be a positive integer")
        if int(pool.get("num_sessions", config.get("num_sessions", 0))) < 1:
            raise ValueError("Each environment pool needs a positive session count")
        if (
            "manager_url" in pool
            and "controller_url" in pool
            and pool["manager_url"] != pool["controller_url"]
        ):
            raise ValueError("Conflicting manager_url and legacy controller_url in pool")
        pool_url = (
            pool.get("manager_url")
            or pool.get("controller_url")
            or config.get("manager_url")
            or config.get("controller_url", "")
        )
        if not str(pool_url).startswith(("http://", "https://")):
            raise ValueError("Each environment pool needs an HTTP(S) manager_url")
    if config.get("max_total_length", 0) > args.max_context_len:
        raise ValueError("Adapter max_total_length exceeds --max-context-len")
    if config.get("max_turns", 20) < 1:
        raise ValueError("Adapter max_turns must be positive")
    if config.get("windowwise_train") and not (
        config.get("window_size", 5) > config.get("overlap", 1) >= 0
    ):
        raise ValueError("Adapter window_size must be greater than overlap >= 0")
    record = _first_record(args.data)
    if not isinstance(record, dict) or any(
        key not in record for key in (args.input_key, args.metadata_key, args.label_key)
    ):
        raise ValueError("Training records need the configured prompt, metadata, and label fields")
    meta = record[args.metadata_key]
    if not isinstance(record[args.input_key], (str, list)) or not record[args.input_key]:
        raise ValueError("Training prompt must be nonempty text or chat messages")
    if not isinstance(meta, dict) or not meta.get("task_id"):
        raise ValueError("Training metadata must contain task_id")
    if (meta.get("env_type") or config.get("env_type")) not in pools:
        raise ValueError("Training metadata env_type has no configured environment pool")
    if len(pools) > 1 or args.only_env_type:
        task_envs: dict[str, str] = {}
        for index, row in enumerate(_records(args.data), 1):
            metadata = row.get(args.metadata_key) if isinstance(row, dict) else None
            if not isinstance(metadata, dict) or not metadata.get("task_id"):
                raise ValueError(f"Mixed data row {index} needs metadata.task_id")
            env_type = metadata.get("env_type") or (args.only_env_type if len(pools) == 1 else None)
            if env_type not in pools:
                raise ValueError(f"Mixed data row {index} needs an explicit configured env_type")
            if (
                not isinstance(row.get(args.input_key), (str, list))
                or not row[args.input_key]
                or args.label_key not in row
            ):
                raise ValueError(f"Mixed data row {index} needs prompt and label fields")
            task_id = str(metadata["task_id"])
            if task_id in task_envs and task_envs[task_id] != env_type:
                raise ValueError(
                    f"Mixed data task_id {task_id!r} collides across environment pools"
                )
            task_envs[task_id] = env_type
    return config


def trainer_command(args: argparse.Namespace) -> list[str]:
    command = [
        sys.executable,
        str(getattr(args, "trainer_entrypoint", TRAINER)),
        "--spec",
        "slime_plugins.models.qwen3_5",
        "get_qwen3_5_spec",
        "--disable-bias-linear",
        "--qk-layernorm",
        "--group-query-attention",
        "--num-attention-heads",
        "16",
        "--num-query-groups",
        "4",
        "--kv-channels",
        "256",
        "--num-layers",
        "32",
        "--hidden-size",
        "4096",
        "--ffn-hidden-size",
        "12288",
        "--use-gated-attention",
        "--normalization",
        "RMSNorm",
        "--apply-layernorm-1p",
        "--position-embedding-type",
        "rope",
        "--norm-epsilon",
        "1e-6",
        "--rotary-percent",
        "0.25",
        "--swiglu",
        "--vocab-size",
        "248320",
        "--untie-embeddings-and-output-weights",
        "--rotary-base",
        "10000000",
        "--attention-output-gate",
        "--train-backend",
        "megatron",
        "--actor-num-nodes",
        str(args.nodes),
        "--actor-num-gpus-per-node",
        str(args.gpus_per_node),
        "--num-gpus-per-node",
        str(args.gpus_per_node),
        "--rollout-num-gpus-per-engine",
        str(args.engine_gpus),
        "--hf-checkpoint",
        str(args.model),
        "--load",
        str(args.resume or args.model),
        "--megatron-to-hf-mode",
        "bridge",
        "--save",
        str(args.output / "checkpoints"),
        "--save-hf",
        str(args.output / "hf/iter_{rollout_id:07d}"),
        "--save-interval",
        str(args.save_interval),
        "--prompt-data",
        str(args.data),
        "--input-key",
        args.input_key,
        "--metadata-key",
        args.metadata_key,
        "--label-key",
        args.label_key,
        "--multimodal-keys",
        '{"image": "images"}',
        "--rollout-shuffle",
        "--rollout-function-path",
        "mai_gym_rl.slime_adapter.rollout_mobile_stepwise.generate_rollout",
        "--custom-reward-post-process-path",
        "mai_gym_rl.slime_adapter.reward_post_process.grpo_normalize_by_group_id",
        "--num-rollout",
        str(args.rollouts),
        "--rollout-batch-size",
        str(args.rollout_batch_size),
        "--n-samples-per-prompt",
        str(args.samples_per_prompt),
        "--rollout-max-response-len",
        str(args.max_response_len),
        "--rollout-max-context-len",
        str(args.max_context_len),
        "--rollout-temperature",
        "1.0",
        "--global-batch-size",
        str(args.global_batch_size),
        "--balance-data",
        "--advantage-estimator",
        "grpo",
        "--kl-coef",
        "0",
        "--entropy-coef",
        "0",
        "--eps-clip",
        "0.2",
        "--eps-clip-high",
        "0.28",
        "--use-tis",
        "--optimizer",
        "adam",
        "--lr",
        str(args.lr),
        "--lr-decay-style",
        "constant",
        "--weight-decay",
        "0.1",
        "--adam-beta1",
        "0.9",
        "--adam-beta2",
        "0.98",
        "--optimizer-cpu-offload",
        "--overlap-cpu-optimizer-d2h-h2d",
        "--use-precision-aware-optimizer",
        "--override-opt_param-scheduler",
        "--tensor-model-parallel-size",
        str(args.tp),
        "--pipeline-model-parallel-size",
        str(args.pp),
        "--context-parallel-size",
        str(args.cp),
        "--sequence-parallel",
        "--expert-model-parallel-size",
        "1",
        "--expert-tensor-parallel-size",
        "1",
        "--recompute-granularity",
        "full",
        "--recompute-method",
        "uniform",
        "--recompute-num-layers",
        "1",
        "--qkv-format",
        "bshd",
        "--micro-batch-size",
        "1",
        "--attention-dropout",
        "0",
        "--hidden-dropout",
        "0",
        "--attention-softmax-in-fp32",
        "--attention-backend",
        "flash",
        "--loss-mask-type",
        "qwen3_5",
        "--accumulate-allreduce-grads-in-fp32",
        "--calculate-per-token-loss",
        "--clip-grad",
        "1.0",
        "--log-probs-chunk-size",
        "512",
        "--distributed-timeout-minutes",
        "60",
        "--sglang-mem-fraction-static",
        "0.35",
        "--sglang-cuda-graph-bs",
        "1",
        "2",
        "4",
        "8",
        "16",
        "24",
        "32",
        "--sglang-disable-custom-all-reduce",
        "--sglang-log-level",
        "warning",
        "--custom-megatron-init-path",
        "mai_gym_rl.slime_adapter.mem_selfcheck.custom_megatron_init",
        "--colocate",
        "--offload-rollout",
        "--offload-train",
    ]
    command.append("--skip-eval-before-train")
    if args.kl_loss_coef > 0:
        command += [
            "--use-kl-loss",
            "--kl-loss-coef",
            str(args.kl_loss_coef),
            "--kl-loss-type",
            "low_var_kl",
            "--ref-load",
            str(args.model),
        ]
    return command + args.trainer_args


def runtime_environment(args: argparse.Namespace, config: dict) -> dict[str, str]:
    pythonpath = [str(ROOT / "src"), str(ROOT / "resources/slime")]
    if os.environ.get("MEGATRON_LM_PATH"):
        pythonpath.append(os.environ["MEGATRON_LM_PATH"])
    if os.environ.get("PYTHONPATH"):
        pythonpath.append(os.environ["PYTHONPATH"])
    defaults = {
        "PYTHONPATH": os.pathsep.join(pythonpath),
        "SLIME_ADAPTER_CONFIG": str(args.adapter_config),
        "MAI_GYM_TRAIN_ID": args.run_name,
        "CUDA_DEVICE_MAX_CONNECTIONS": "1",
        "PYTORCH_ALLOC_CONF": "",
        "PYTORCH_CUDA_ALLOC_CONF": "",
        "LAZY_MULTIMODAL": "1",
        "MAI_VIT_RECOMPUTE": "1",
        "TRAJ_TIMEOUT_S": "21600",
        "ROLLOUT_TIMEOUT_S": "21600",
        "DYNAMIC_TIMEOUT": "0",
        "SCHED_DEFAULT_SIBLINGS": str(args.samples_per_prompt),
        "SCHED_MAX_SIBLINGS": "16",
        "SCHED_HOPELESS_SIBLINGS": "2",
        "SCHED_MASTERED_SIBLINGS": "2",
    }
    if args.max_env_sessions is not None:
        defaults["NUM_SESSIONS"] = str(config["num_sessions"])
    for name in (
        "ROLLOUT_CONCURRENCY",
        "ROLLOUT_THREAD_POOL_SIZE",
        "MAX_TRAIN_TRAJECTORIES",
        "MAX_EXTRA_ROUNDS",
        "MAX_ROLLOUT_GROUP_RETRIES",
        "MIN_TRAINABLE_TASKS",
        "MIN_SAMPLES_PER_TASK",
        "SCHED_HOPELESS_SIBLINGS",
        "MAX_EVAL_RETRIES",
        "DYNAMIC_TIMEOUT",
        "TIMEOUT_TRAJ_PER_HOUR",
        "TRAJ_TIMEOUT_S",
        "ROLLOUT_TIMEOUT_S",
        "NCCL_SOCKET_IFNAME",
        "GLOO_SOCKET_IFNAME",
        "NCCL_IB_HCA",
        "NCCL_IB_GID_INDEX",
        "NCCL_IB_DISABLE",
        "NCCL_NVLS_ENABLE",
        "MAI_GYM_ROLLOUT_WORKERS",
        "MAI_GYM_ROLLOUT_WORKER_CPUS",
        "MAI_GYM_ROLLOUT_WORKER_THREADS",
        "MAI_GYM_DISTRIBUTED_HTTP_LIMIT_PER_WORKER",
        "MAI_GYM_MAX_INFLIGHT_REQ",
        "MAI_GYM_MAX_INFLIGHT_OSS_GET",
    ):
        if os.environ.get(name):
            defaults[name] = os.environ[name]
    references = getattr(args, "adapter_environment", None)
    return (
        defaults
        | (config_environment(config) if references is None else references)
        | getattr(args, "environment_overrides", {})
    )


def check_environment_pools(config: dict) -> None:
    pools = config.get("env_pools") or {config.get("env_type"): config}
    for env_type, pool in pools.items():
        url = (
            pool.get("manager_url")
            or pool.get("controller_url")
            or config.get("manager_url")
            or config.get("controller_url", "")
        ).rstrip("/")
        try:
            with urllib.request.urlopen(url + "/health", timeout=20) as response:
                health = json.load(response)
        except (OSError, ValueError) as exc:
            raise ValueError(f"Environment health check failed for {env_type}: {url}") from exc
        if not isinstance(health, dict) or not health.get("ok"):
            raise ValueError(f"Manager {url} is not healthy for {env_type}")
        contract = health.get("environment_contract") or {}
        if env_type in contract and not contract[env_type]:
            raise ValueError(f"Manager {url} has no available backend configuration for {env_type}")
        distributed = health.get("distributed") or {}
        if distributed.get("admission_ready") is False:
            raise ValueError(f"Manager {url} is not ready to admit {env_type} sessions")


def launch(args: argparse.Namespace, config: dict, command: list[str]) -> None:
    import ray

    env = runtime_environment(args, config)
    os.environ.update(env)
    sys.path[:0] = env["PYTHONPATH"].split(os.pathsep)
    ray.init(address=args.ray_address, runtime_env={"env_vars": env})
    try:
        required = args.nodes * args.gpus_per_node
        nodes = [
            node
            for node in ray.nodes()
            if node["Alive"] and node["Resources"].get("GPU", 0) >= args.gpus_per_node
        ]
        if len(nodes) < args.nodes or ray.available_resources().get("GPU", 0) < required:
            raise ValueError("Ray cluster does not have the requested nodes and available GPUs")
        args.output.mkdir(parents=True, exist_ok=True)
        sys.argv = command[1:]
        runpy.run_path(command[1], run_name="__main__")
    finally:
        ray.shutdown()


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        config = validate(args)
        command = trainer_command(args)
        if args.dry_run:
            print(shlex.join(command))
            print("Local preflight passed; no Ray connection or training was started.")
        else:
            if args.check_env_pools:
                check_environment_pools(config)
            launch(args, config, command)
    except (ValueError, OSError, ImportError) as exc:
        print(f"Training startup failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
