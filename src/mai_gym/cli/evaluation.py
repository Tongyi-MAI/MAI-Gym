import json
import os
import secrets
import signal
import socket
import subprocess
import sys
import tempfile
import time
from collections.abc import Iterator
from contextlib import ExitStack, contextmanager, suppress
from pathlib import Path

import click
import httpx
import yaml

from mai_gym.benchmarks import BENCHMARKS
from mai_gym.models.config import AgentConfig, MobileGymConfig, OSSConfig
from mai_gym.models.environment import identity
from mai_gym.models.workload import BenchmarkSettings, ModelSettings, WorkloadConfig


def load_config(
    path: str | None,
) -> tuple[MobileGymConfig, ModelSettings, dict[str, BenchmarkSettings], WorkloadConfig | None]:
    if not path:
        return MobileGymConfig(), ModelSettings(), {}, None
    data = yaml.safe_load(Path(path).read_text())
    if not isinstance(data, dict):
        raise ValueError("Configuration must be a YAML mapping")
    if "agent" in data:
        settings = {
            name: BenchmarkSettings.model_validate(value)
            for name, value in data.pop("benchmarks", {}).items()
        }
        cfg = MobileGymConfig.model_validate(data)
        if cfg.agent is None:
            raise ValueError("The agent configuration must be a mapping")
        model = ModelSettings(
            name=cfg.agent.model_name,
            base_url=cfg.agent.base_url,
            api_key=cfg.agent.api_key,
            agent_variant=cfg.agent.type,
            runtime_conf=cfg.agent.runtime_conf,
        )
        return cfg, model, settings, None
    workload = WorkloadConfig.model_validate(data)
    cfg = MobileGymConfig(manager=workload.manager, devices=workload.devices, judge=workload.judge)
    cfg.rollout.num_envs = workload.rollout.concurrency
    cfg.rollout.screenshot_transport = workload.observations.screenshot_transport
    if workload.observations.oss:
        cfg.manager.oss = OSSConfig.model_validate(workload.observations.oss)
    return cfg, workload.model, workload.benchmarks, workload


def resolve_path(value: str | Path, config_path: str | None) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = (Path(config_path).resolve().parent if config_path else Path.cwd()) / path
    return path.resolve()


def endpoint_url(value: str) -> str:
    url = value.rstrip("/")
    if url.endswith("/chat/completions"):
        url = url[: -len("/chat/completions")]
    return url if url.endswith("/v1") else url + "/v1"


def api_key(value: str) -> str:
    return os.environ.get(value[4:], "EMPTY") if value.startswith("env/") else value


@contextmanager
def _runtime_environment(name: str, value: str) -> Iterator[None]:
    previous = os.environ.get(name)
    os.environ[name] = value
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = previous


@contextmanager
def model_endpoint(
    model: str, settings: ModelSettings, *, dry_run: bool = False
) -> Iterator[tuple[str, str, str]]:
    key = api_key(settings.api_key)
    path = Path(model).expanduser()
    local = path.is_dir()
    if not local and (model.startswith(("/", "./", "../", "~"))):
        raise ValueError("Local model checkpoint directory does not exist")
    process = None
    name: str | None
    try:
        if local:
            name = settings.name or path.resolve().name
            with socket.socket() as sock:
                sock.bind(("127.0.0.1", 0))
                port = sock.getsockname()[1]
            base_url = f"http://127.0.0.1:{port}/v1"
            if not dry_run:
                command = [
                    sys.executable,
                    "-m",
                    "sglang.launch_server",
                    "--model-path",
                    str(path.resolve()),
                    "--served-model-name",
                    name,
                    "--host",
                    "127.0.0.1",
                    "--port",
                    str(port),
                ]
                with tempfile.TemporaryFile(mode="w+") as log:
                    process = subprocess.Popen(
                        command, stdout=log, stderr=log, start_new_session=True
                    )
                with httpx.Client(timeout=2) as client:
                    deadline = time.monotonic() + 600
                    while time.monotonic() < deadline:
                        if process.poll() is not None:
                            raise ValueError(
                                "Local inference server exited; verify the SGLang/GPU installation"
                            )
                        try:
                            if client.get(f"{base_url}/models").status_code == 200:
                                break
                        except httpx.TransportError:
                            pass
                        time.sleep(1)
                    else:
                        raise ValueError(
                            "Local inference server did not become ready within 600 seconds"
                        )
        else:
            is_url = model.startswith(("http://", "https://"))
            base_url = endpoint_url(
                model
                if is_url
                else settings.base_url
                or os.environ.get("OPENAI_BASE_URL", "http://127.0.0.1:8000/v1")
            )
            name = settings.name if is_url else model
            if not name and not dry_run:
                with httpx.Client(timeout=30, headers={"Authorization": f"Bearer {key}"}) as client:
                    response = client.get(f"{base_url}/models")
                    response.raise_for_status()
                    models = response.json().get("data", [])
                if len(models) != 1:
                    raise ValueError("Endpoint must expose one model, or specify --model-name")
                name = models[0]["id"]
            name = name or "endpoint-model"
        yield name, base_url, key
    finally:
        if process is not None:
            with suppress(ProcessLookupError):
                os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                with suppress(ProcessLookupError):
                    os.killpg(process.pid, signal.SIGKILL)
                process.wait()
            finally:
                with suppress(ProcessLookupError):
                    os.killpg(process.pid, signal.SIGKILL)


def benchmark_config(
    template: MobileGymConfig,
    model: ModelSettings,
    workload: WorkloadConfig | None,
    settings: BenchmarkSettings,
    name: str,
    config_path: str | None,
) -> MobileGymConfig:
    spec = BENCHMARKS[name]
    cfg = (
        MobileGymConfig.from_yaml(resolve_path(settings.manager_config, config_path))
        if settings.manager_config
        else template.model_copy(deep=True)
    )
    runtime = dict(model.runtime_conf)
    runtime.setdefault("unified_variant_map", model.unified_variant_map)
    if workload:
        runtime.update(
            window_size=workload.history.window_size,
            overlap=workload.history.overlap,
            image_retention=workload.history.image_retention,
        )
    runtime["temperature"] = 0.0
    runtime["device_type"] = identity(settings.env_type or spec.env_type).device_type
    variant = settings.agent_variant or model.agent_variant
    if name == "mocked-apps":
        runtime.setdefault("screen_size", [390, 844])
    if name == "mobilegym":
        from mai_gym.envs.mobilegym.agent import configure_agent

        variant = configure_agent(settings.agent_variant, runtime)
    if variant == "unified" and name in {
        "webvoyager",
        "onlinemind2web",
        "webarena",
        "mocked-apps",
    }:
        variant = "general_e2e_agent"
    cfg.agent = AgentConfig(
        type=variant,
        model_name="pending",
        base_url="http://localhost/v1",
        max_steps=spec.max_steps or 50,
        runtime_conf=runtime,
    )
    cfg.rollout.env_type = settings.env_type or spec.env_type
    cfg.rollout.manager_url = settings.manager_url or cfg.rollout.manager_url
    cfg.rollout.num_envs = settings.num_envs or cfg.rollout.num_envs
    cfg.rollout.resolve_task_metadata = True
    cfg.rollout.require_score = True
    cfg.rollout.is_eval = True
    cfg.rollout.require_task_steps = spec.max_steps is None
    cfg.rollout.task_metadata = dict(settings.task_metadata)
    cfg.rollout.benchmark.suite = name
    if settings.tasks is not None:
        if not settings.tasks or len(settings.tasks) != len(set(settings.tasks)):
            raise ValueError("Benchmark tasks must be a nonempty list of unique task IDs")
        cfg.rollout.tasks = ",".join(settings.tasks)
    if settings.tasks_file:
        data = json.loads(resolve_path(settings.tasks_file, config_path).read_text())
        data = data.get("tasks", data) if isinstance(data, dict) else data
        if not isinstance(data, list):
            raise ValueError("Benchmark task file must contain a task list")
        entries = [{"task_id": item} if isinstance(item, str) else item for item in data]
        if not entries or any(
            not isinstance(item, dict) or not item.get("task_id") for item in entries
        ):
            raise ValueError("Benchmark task file must contain nonempty task IDs")
        if len(entries) != len({item["task_id"] for item in entries}):
            raise ValueError("Benchmark task IDs must be unique")
        cfg.rollout.tasks = ",".join(item["task_id"] for item in entries)
        cfg.rollout.task_metadata.update({item["task_id"]: item for item in entries})
    if settings.tasks_dir:
        cfg.rollout.tasks_dir = str(resolve_path(settings.tasks_dir, config_path))
    return cfg


@click.command("eval", help="Evaluate a checkpoint or served model on registered benchmarks.")
@click.option("--model", default=None, help="Local checkpoint, served model ID, or API endpoint.")
@click.option("--benchmarks", required=True, help="Benchmark names separated by spaces.")
@click.argument("benchmark_names", nargs=-1)
@click.option("--trials", default=3, type=click.IntRange(min=1), show_default=True)
@click.option("--config", type=click.Path(exists=True, dir_okay=False), envvar="MAI_GYM_CONFIG")
@click.option("--model-name", default=None, help="Served model ID when --model is an endpoint.")
@click.option("--base-url", default=None, help="API base URL when --model is a served model ID.")
@click.option(
    "--manager-url",
    "--server",
    "manager_url",
    default=None,
    help="Override the environment manager URL for all benchmarks (alias: --server).",
)
@click.option("--num-envs", default=None, type=click.IntRange(min=1))
@click.option("--output", type=click.Path(file_okay=False), default=None)
@click.option(
    "--max-retry-rounds",
    default=5,
    type=click.IntRange(min=0),
    show_default=True,
    help="Additional rounds for unscored trials; zero disables retries.",
)
@click.option("--dry-run", is_flag=True, help="Print the execution plan without starting services.")
def evaluate(
    model: str | None,
    benchmarks: str,
    benchmark_names: tuple[str, ...],
    trials: int,
    config: str | None,
    model_name: str | None,
    base_url: str | None,
    manager_url: str | None,
    num_envs: int | None,
    output: str | None,
    max_retry_rounds: int,
    dry_run: bool,
) -> None:
    from mai_gym.cli.main import _auto_serve
    from mai_gym.runner.batch import RolloutSetupError
    from mai_gym.runner.passn import run_passn_rollout

    names = list(
        dict.fromkeys(
            part for value in (benchmarks, *benchmark_names) for part in value.split(",") if part
        )
    )
    unknown = set(names) - BENCHMARKS.keys()
    if not names:
        raise click.BadParameter("Select at least one benchmark", param_hint="--benchmarks")
    if unknown:
        raise click.BadParameter(
            f"Unknown benchmarks: {', '.join(sorted(unknown))}", param_hint="--benchmarks"
        )
    try:
        template, model_settings, suites, workload = load_config(config)
        if model_name:
            model_settings.name = model_name
        if base_url:
            model_settings.base_url = base_url
        configured_model = model is None
        model = model or model_settings.checkpoint or model_settings.name or model_settings.base_url
        if not model:
            raise ValueError(
                "Provide --model or model.checkpoint/name/base_url in the configuration"
            )
        if model.startswith("env/"):
            variable = model[4:]
            model = os.environ.get(variable)
            if not model:
                raise ValueError(f"Set environment variable {variable}")
        if configured_model and config and not model.startswith(("http://", "https://")):
            candidate = resolve_path(model, config)
            if candidate.is_dir():
                model = str(candidate)
        configurations = [
            benchmark_config(
                template,
                model_settings,
                workload,
                suites.get(name, BenchmarkSettings()),
                name,
                config,
            )
            for name in names
        ]
        run_id = f"eval_{time.strftime('%Y%m%d_%H%M%S')}_{secrets.token_hex(4)}"
        root = Path(output or template.manager.traj_log_dir or "./traj_logs") / run_id
        with model_endpoint(model, model_settings, dry_run=dry_run) as (served_name, url, key):
            if dry_run:
                click.echo(
                    json.dumps(
                        {
                            "model": served_name,
                            "trials": trials,
                            "benchmarks": [
                                {
                                    "name": name,
                                    "env_type": cfg.rollout.env_type,
                                    "max_steps": BENCHMARKS[name].max_steps,
                                    "metric": BENCHMARKS[name].metric,
                                    "num_envs": num_envs or cfg.rollout.num_envs,
                                }
                                for name, cfg in zip(names, configurations, strict=True)
                            ],
                        },
                        indent=2,
                    )
                )
                return
            root.mkdir(parents=True)
            results = {}
            incomplete = []
            with ExitStack() as stack:
                directory = Path(
                    stack.enter_context(tempfile.TemporaryDirectory(prefix="mai-gym-eval-"))
                )
                managers = set()
                for name, cfg in zip(names, configurations, strict=True):
                    assert cfg.agent is not None
                    cfg.agent.model_name, cfg.agent.base_url, cfg.agent.api_key = (
                        served_name,
                        url,
                        key,
                    )
                    cfg.rollout.pass_n = trials
                    cfg.rollout.retry_until_scored = max_retry_rounds > 0
                    cfg.rollout.max_retry_rounds = max_retry_rounds
                    cfg.rollout.train_id = f"{run_id}_{name}"
                    cfg.rollout.benchmark.model = cfg.rollout.benchmark.model or served_name
                    cfg.rollout.benchmark.run_id = cfg.rollout.benchmark.run_id or run_id
                    cfg.manager.traj_log_dir = str(root / name)
                    manager = (
                        manager_url
                        or cfg.rollout.manager_url
                        or f"http://127.0.0.1:{cfg.manager.port}"
                    )
                    cfg.rollout.num_envs = num_envs or cfg.rollout.num_envs
                    if manager not in managers:
                        path = directory / f"{name}.yaml"
                        serve_config = cfg.model_dump(exclude_none=True, exclude={"agent"})
                        if cfg.judge is not None:
                            variable = "MAI_GYM_EVAL_JUDGE_KEY"
                            stack.enter_context(_runtime_environment(variable, cfg.judge.api_key))
                            serve_config["judge"]["api_key"] = f"env/{variable}"
                        path.write_text(yaml.safe_dump(serve_config))
                        path.chmod(0o600)
                        stack.enter_context(_auto_serve(str(path), manager))
                        managers.add(manager)
                    report = run_passn_rollout(
                        cfg,
                        server=manager,
                        num_envs=cfg.rollout.num_envs,
                        env_type=cfg.rollout.env_type,
                        task_filter=cfg.rollout.tasks,
                        no_resume=False,
                        tasks_dir=cfg.rollout.tasks_dir,
                        domains=set(cfg.rollout.domains) if cfg.rollout.domains else None,
                        pass_n=trials,
                        retry_until_scored=max_retry_rounds > 0,
                        max_retry_rounds=max_retry_rounds,
                    )
                    missing = sum(task.env_failed_count for task in report.per_task)
                    metric = BENCHMARKS[name].metric
                    results[name] = {
                        "metric": metric,
                        "score": report.avg_score if metric == "partial_score" else report.avg_sr,
                        "success_rate": report.avg_sr,
                        "partial_score": report.avg_score,
                        "pass_at_n": report.pass_at_n,
                        "pass_all_n": report.pass_all_n,
                        "trials": trials,
                        "unscored_trials": missing,
                        "tasks": report.n_tasks,
                    }
                    if missing:
                        incomplete.append(name)
                    (root / "results.json").write_text(json.dumps(results, indent=2))
            click.echo(f"Evaluation results: {root / 'results.json'}")
            if incomplete:
                raise click.ClickException("Unscored trials remain for: " + ", ".join(incomplete))
    except (ValueError, OSError, yaml.YAMLError, httpx.HTTPError, RolloutSetupError) as exc:
        raise click.ClickException(str(exc)) from exc
