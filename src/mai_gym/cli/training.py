import argparse
import json
import os
import shlex
import shutil
import tempfile
from pathlib import Path
from types import ModuleType
from typing import Any

import click
import yaml

from mai_gym.cli.evaluation import resolve_path
from mai_gym.models.workload import WorkloadConfig


def referenced_path(value: str, config_path: str) -> Path:
    if value.startswith("env/"):
        variable = value[4:]
        value = os.environ.get(variable, "")
        if not value:
            raise ValueError(f"Set environment variable {variable}")
    return resolve_path(value, config_path)


def adapter_config(cfg: WorkloadConfig) -> dict[str, Any]:
    pools: dict[str, dict[str, Any]] = {}
    for task in cfg.tasks:
        pool = {
            "manager_url": task.manager_url,
            "num_sessions": task.num_sessions or cfg.rollout.num_sessions,
        }
        if task.env_type in pools and pools[task.env_type] != pool:
            raise ValueError(f"Conflicting manager settings for environment {task.env_type!r}")
        pools[task.env_type] = pool
    if not pools:
        raise ValueError("Configure at least one task pool and environment")
    first = next(iter(pools))
    return {
        **cfg.model.runtime_conf,
        "manager_url": pools[first]["manager_url"],
        "env_type": first,
        "env_pools": pools,
        "num_sessions": cfg.rollout.num_sessions,
        "agent_variant": cfg.model.agent_variant,
        "unified_variant_map": cfg.model.unified_variant_map,
        "policy_variant": cfg.model.policy_variant,
        "tool_parser": cfg.model.tool_parser,
        "max_turns": cfg.rollout.max_turns,
        "max_total_length": cfg.rollout.max_total_length,
        "window_size": cfg.history.window_size,
        "overlap": cfg.history.overlap,
        "image_retention": cfg.history.image_retention,
        "token_native_history": cfg.history.token_native_history,
        "windowwise_train": cfg.history.windowwise_train,
        "reward_allocation": cfg.algorithm.reward_allocation,
        "screenshot_transport": cfg.observations.screenshot_transport,
        **({"oss": cfg.observations.oss} if cfg.observations.oss else {}),
    }


def combine_pools(cfg: WorkloadConfig, config_path: str, destination: Path) -> None:
    with destination.open("w") as stream:
        count = 0
        for pool in cfg.tasks:
            path = referenced_path(pool.pool, config_path)
            if path.suffix == ".parquet":
                import pyarrow.parquet as pq

                parquet_file: Any = pq.ParquetFile
                records = (
                    row for batch in parquet_file(path).iter_batches() for row in batch.to_pylist()
                )
            elif path.suffix == ".jsonl":
                records = (
                    json.loads(line) for line in path.read_text().splitlines() if line.strip()
                )
            else:
                raise ValueError(
                    "Task pools must be JSONL or Parquet, or set trainer.data to a combined dataset"
                )
            for record in records:
                if (
                    not isinstance(record, dict)
                    or not record.get("prompt")
                    or "label" not in record
                ):
                    raise ValueError("Every task pool record needs prompt and label fields")
                metadata = record.get("metadata")
                if not isinstance(metadata, dict) or not metadata.get("task_id"):
                    raise ValueError("Every task pool record needs metadata.task_id")
                if metadata.get("env_type", pool.env_type) != pool.env_type:
                    raise ValueError("Task record env_type disagrees with its configured pool")
                record["metadata"] = {**metadata, "env_type": pool.env_type}
                stream.write(json.dumps(record) + "\n")
                count += 1
        if not count:
            raise ValueError("Task pools contain no records")


def training_command(
    cfg: WorkloadConfig, args: argparse.Namespace, launcher: ModuleType
) -> list[str]:
    command: list[str] = launcher.trainer_command(args)
    algorithm = cfg.algorithm.name
    if algorithm == "sao" and not cfg.trainer.entrypoint:
        raise ValueError("SAO requires trainer.entrypoint and its backend arguments")
    estimator = {"dapo": "grpo", "reinforce++": "reinforce_plus_plus", "sao": "grpo"}.get(
        algorithm, algorithm
    )
    command[command.index("--advantage-estimator") + 1] = estimator
    command[command.index("--entropy-coef") + 1] = str(cfg.algorithm.entropy_coef)
    command[command.index("--rollout-temperature") + 1] = str(cfg.rollout.temperature)
    if not cfg.algorithm.per_token_loss:
        command.remove("--calculate-per-token-loss")
    if estimator not in {"grpo", "gspo"}:
        index = command.index("--custom-reward-post-process-path")
        del command[index : index + 2]
    if cfg.trainer.entrypoint:
        command[1] = str(args.trainer_entrypoint)
    return command


@click.command("train", help="Launch online RL training from a YAML configuration.")
@click.option("--config", required=True, type=click.Path(exists=True, dir_okay=False))
@click.option(
    "--dry-run",
    is_flag=True,
    help="Validate inputs and print the trainer command without starting Ray.",
)
def train(config: str, dry_run: bool) -> None:
    from mai_gym_rl import launcher

    try:
        cfg = WorkloadConfig.from_yaml(config)
        if not cfg.model.checkpoint:
            raise ValueError("model.checkpoint is required for training")
        adapter = adapter_config(cfg)
        trainer, rollout, algorithm = cfg.trainer, cfg.rollout, cfg.algorithm
        parallel = trainer.parallelism
        actual_dp = (
            trainer.nodes
            * trainer.gpus_per_node
            / (parallel.tensor * parallel.pipeline * parallel.context)
        )
        if parallel.data is not None and parallel.data != actual_dp:
            raise ValueError("trainer.parallelism.data disagrees with the configured GPU topology")
        with tempfile.TemporaryDirectory(prefix="mai-gym-train-") as temporary:
            staging = Path(temporary)
            adapter_path = staging / "adapter.yaml"
            adapter_path.write_text(yaml.safe_dump(adapter))
            adapter_path.chmod(0o600)
            data = (
                referenced_path(trainer.data, config) if trainer.data else staging / "tasks.jsonl"
            )
            if not trainer.data:
                combine_pools(cfg, config, data)
            argv = [
                "--model",
                str(referenced_path(cfg.model.checkpoint, config)),
                "--data",
                str(data),
                "--output",
                str(referenced_path(trainer.output, config)),
                "--adapter-config",
                str(adapter_path),
            ]
            values = {
                "nodes": trainer.nodes,
                "gpus-per-node": trainer.gpus_per_node,
                "tp": parallel.tensor,
                "pp": parallel.pipeline,
                "cp": parallel.context,
                "engine-gpus": trainer.engine_gpus,
                "rollouts": rollout.rounds,
                "rollout-batch-size": rollout.batch_size,
                "samples-per-prompt": algorithm.group_size,
                "global-batch-size": trainer.global_batch_size,
                "lr": algorithm.learning_rate,
                "kl-loss-coef": algorithm.kl_loss_coef,
                "max-context-len": rollout.max_total_length,
                "max-response-len": rollout.max_response_length,
                "save-interval": trainer.save_interval,
                "ray-address": trainer.ray_address,
            }
            if trainer.resume:
                values["resume"] = referenced_path(trainer.resume, config)
            if trainer.run_name:
                values["run-name"] = trainer.run_name
            for option, value in values.items():
                argv.extend(["--" + option, str(value)])
            args = launcher.parse_args(argv)
            if trainer.entrypoint:
                args.trainer_entrypoint = referenced_path(trainer.entrypoint, config)
                if not args.trainer_entrypoint.is_file():
                    raise ValueError("trainer.entrypoint does not exist")
            args.environment_overrides = {
                "ROLLOUT_CONCURRENCY": str(rollout.concurrency),
                "SCHED_DEFAULT_SIBLINGS": str(algorithm.group_size),
                "SCHED_HOPELESS_SIBLINGS": str(cfg.sampling.fail_budget),
                "SCHED_MASTERED_SIBLINGS": str(cfg.sampling.mastered_budget),
                "SCHED_MASTERED_ACC": str(cfg.sampling.success_threshold),
            }
            launcher.validate(args)
            command = training_command(cfg, args, launcher) + trainer.extra_args
            if dry_run:
                click.echo(shlex.join(command))
                click.echo("Local preflight passed; no Ray connection or training was started.")
                return
            artifacts = args.output / "artifacts"
            artifacts.mkdir(parents=True, exist_ok=True)
            args.adapter_config = artifacts / "adapter.yaml"
            shutil.copyfile(adapter_path, args.adapter_config)
            args.adapter_config.chmod(0o600)
            if not trainer.data:
                args.data = artifacts / "tasks.jsonl"
                shutil.copyfile(data, args.data)
            command = training_command(cfg, args, launcher) + trainer.extra_args
            launcher.launch(args, adapter, command)
    except (ValueError, OSError, ImportError, yaml.YAMLError) as exc:
        raise click.ClickException(str(exc)) from exc
