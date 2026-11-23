import contextlib
import ctypes
import ctypes.util
import logging
import os
import secrets
import signal
import subprocess
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from urllib.parse import quote, urlparse

import click
import httpx
from dotenv import load_dotenv

from mai_gym.cli.evaluation import evaluate
from mai_gym.cli.training import train

load_dotenv()

logging.basicConfig(
    level=os.environ.get("MAI_GYM_LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def _raise_fd_limit(minimum: int = 65536) -> None:

    try:
        import resource
    except ImportError:
        return
    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    target = minimum if hard == resource.RLIM_INFINITY else hard
    if soft < target:
        try:
            resource.setrlimit(resource.RLIMIT_NOFILE, (target, hard))
            logger.info("[serve] raised RLIMIT_NOFILE soft limit %d -> %d", soft, target)
        except (ValueError, OSError) as exc:
            logger.warning(
                "[serve] could not raise RLIMIT_NOFILE (soft=%d hard=%d): %s",
                soft,
                hard,
                exc,
            )
            target = soft
    if target < minimum:
        logger.warning(
            "[serve] RLIMIT_NOFILE is only %d — at high environment counts the "
            "manager may exhaust file descriptors. Raise the hard limit "
            "(e.g. LimitNOFILE=65536 in the systemd unit).",
            target,
        )


@click.group(help="MAI-Gym: training and rollout environments for GUI agents.")
@click.version_option(package_name="mai-gym")
def cli() -> None:
    pass


@cli.command(help="Start the environment manager.")
@click.option("--config", required=True, type=click.Path(exists=True), help="YAML config file.")
@click.option("--host", default=None, help="Override host from config.")
@click.option("--port", default=None, type=int, help="Override port from config.")
def serve(config: str, host: str | None, port: int | None) -> None:

    import uvicorn

    _raise_fd_limit()

    from mai_gym.models.config import MobileGymConfig
    from mai_gym.server.app import create_app

    cfg = MobileGymConfig.from_yaml(config)
    app = create_app(cfg)

    keepalive_timeout = int(os.environ.get("MAI_GYM_KEEPALIVE_TIMEOUT", "120"))
    uvicorn.run(
        app,
        host=host or cfg.manager.host,
        port=port or cfg.manager.port,
        timeout_keep_alive=keepalive_timeout,
    )


def _server_alive(url: str, timeout: float = 2.0) -> bool:
    try:
        r = httpx.get(f"{url.rstrip('/')}/health", timeout=timeout)
        return r.status_code == 200
    except Exception:
        return False


def _set_pdeathsig(sig: int = signal.SIGTERM) -> None:

    if sys.platform != "linux":
        return
    libc_name = ctypes.util.find_library("c") or "libc.so.6"
    libc = ctypes.CDLL(libc_name, use_errno=True)
    PR_SET_PDEATHSIG = 1
    if libc.prctl(PR_SET_PDEATHSIG, sig, 0, 0, 0) != 0:
        pass


def _tail(path: Path, n_bytes: int = 2000) -> str:
    try:
        with path.open("rb") as f:
            f.seek(0, 2)
            size = f.tell()
            f.seek(max(0, size - n_bytes))
            return f.read().decode("utf-8", errors="replace")
    except Exception:
        return ""


@contextmanager
def _auto_serve(config_path: str, server_url: str, ready_timeout: int = 600):

    if _server_alive(server_url):
        yield
        return

    parsed = urlparse(server_url)
    host = parsed.hostname or ""
    if host not in ("127.0.0.1", "localhost", "0.0.0.0"):
        raise click.ClickException(
            f"--server {server_url} is unreachable and host {host!r} is non-local; "
            "auto-serve only spawns processes on the local machine.",
        )

    log_dir = Path("./logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"serve_auto_{int(time.time())}_{secrets.token_hex(8)}.log"
    log_fp = log_path.open("w")

    cmd = [
        sys.executable,
        "-m",
        "mai_gym.cli",
        "serve",
        "--config",
        config_path,
    ]
    if parsed.port:
        cmd += ["--port", str(parsed.port)]
    cmd += ["--host", host]

    click.echo(f"Server unreachable at {server_url}; spawning serve subprocess. Logs → {log_path}")
    proc = subprocess.Popen(
        cmd,
        stdout=log_fp,
        stderr=log_fp,
        start_new_session=True,
        preexec_fn=lambda: _set_pdeathsig(signal.SIGTERM),
    )

    try:
        deadline = time.time() + ready_timeout
        while time.time() < deadline:
            if _server_alive(server_url):
                click.echo(f"Server ready at {server_url}")
                break
            if proc.poll() is not None:
                raise click.ClickException(
                    f"serve subprocess exited early (code={proc.returncode}). "
                    f"Last lines:\n{_tail(log_path)}",
                )
            time.sleep(2)
        else:
            raise click.ClickException(
                f"serve subprocess did not become ready within {ready_timeout}s. Check {log_path}",
            )
        yield
    finally:
        if proc.poll() is None:
            click.echo("Stopping auto-spawned serve subprocess...")
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
                proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                proc.kill()
                with contextlib.suppress(Exception):
                    proc.wait(timeout=5)
            except ProcessLookupError:
                pass
            except Exception as exc:
                click.echo(f"Error during serve cleanup: {exc}", err=True)
        with contextlib.suppress(Exception):
            log_fp.close()


@cli.command()
@click.option("--config", required=True, type=click.Path(exists=True), help="YAML config file.")
@click.option(
    "--manager-url",
    "--server",
    "manager_url",
    default=None,
    help=(
        "Environment manager URL (alias: --server). If omitted, defaults to "
        "rollout.manager_url or http://127.0.0.1:<manager.port> from the config. "
        "If the resulting URL is unreachable and points at localhost, mai-gym "
        "auto-spawns `mai-gym serve` as a background subprocess for the rollout's lifetime."
    ),
)
@click.option(
    "--num-envs",
    default=None,
    type=int,
    help="Number of environments to provision. Overrides rollout.num_envs in YAML.",
)
@click.option(
    "--tasks",
    default=None,
    help="Comma-separated task IDs. Overrides rollout.tasks in YAML.",
)
@click.option(
    "--tasks-dir",
    default=None,
    type=click.Path(exists=True, file_okay=False),
    help=(
        "Auto-discover generated OSWorld-style tasks under "
        "<dir>/<domain>/<seed_uuid>.json. Overrides rollout.tasks_dir in YAML."
    ),
)
@click.option(
    "--domains",
    default=None,
    help="Comma-separated domains. Overrides rollout.domains in YAML.",
)
@click.option(
    "--env-type",
    default=None,
    help="Environment type to provision. Overrides rollout.env_type in YAML.",
)
@click.option(
    "--no-resume",
    is_flag=True,
    default=False,
    help="Re-run every requested task even if a result.txt already exists.",
)
@click.option(
    "--exact-train-id",
    is_flag=True,
    default=False,
    help=(
        "Use rollout.train_id verbatim instead of adding a random suffix. "
        "Set it to a previous launch's full printed ID to resume that run."
    ),
)
@click.option(
    "--pass-n",
    "pass_n",
    default=None,
    type=int,
    help=(
        "Number of independent rollout passes (pass@n). Default 1. When >1, the "
        "rollout runs N times and an aggregate pass@n report (results.json + "
        "summary.md) is written. Overrides rollout.pass_n in YAML."
    ),
)
@click.option(
    "--retry-until-scored",
    "retry_until_scored",
    is_flag=True,
    default=None,
    help=(
        "After the pass@n grid, re-run only the env-failed (score is None) "
        "(pass, task) cells in extra concurrent rounds until all are scored or "
        "--max-retry-rounds is hit. Produces a pass@n report even for --pass-n 1. "
        "Overrides rollout.retry_until_scored in YAML."
    ),
)
@click.option(
    "--max-retry-rounds",
    "max_retry_rounds",
    default=None,
    type=int,
    help=(
        "Cap on retry rounds (excludes the initial round); 0 = unbounded. "
        "Overrides rollout.max_retry_rounds in YAML."
    ),
)
@click.option(
    "--evaluation-mode",
    type=click.Choice(["built_in", "llm_judge", "both"], case_sensitive=True),
    default=None,
    help=(
        "Evaluator selection for this rollout. Defaults to built_in; "
        "llm_judge and both require a judge-capable manager. Overrides "
        "rollout.evaluation_mode in YAML."
    ),
)
@click.option(
    "--set",
    "set_overrides",
    multiple=True,
    metavar="KEY=VALUE",
    help=(
        "Override any config field by dotted path, e.g. "
        "--set agent.model_name=foo --set agent.runtime_conf.temperature=0.5 "
        "--set rollout.train_id=run1. Repeatable. Applied after the YAML loads; "
        "explicit flags (--num-envs etc.) still win over --set."
    ),
)
def rollout(
    config: str,
    manager_url: str | None,
    num_envs: int | None,
    tasks: str | None,
    tasks_dir: str | None,
    domains: str | None,
    env_type: str | None,
    no_resume: bool,
    exact_train_id: bool,
    pass_n: int | None,
    retry_until_scored: bool | None,
    max_retry_rounds: int | None,
    evaluation_mode: str | None,
    set_overrides: tuple[str, ...],
) -> None:

    from mai_gym.models.config import MobileGymConfig, apply_cli_overrides
    from mai_gym.runner.batch import RolloutSetupError
    from mai_gym.runner.passn import run_passn_rollout
    from mai_gym.server.trajectory import model_slug

    cfg = MobileGymConfig.from_yaml(config)
    if set_overrides:
        try:
            cfg = apply_cli_overrides(cfg, set_overrides)
        except ValueError as exc:
            raise click.ClickException(f"Invalid --set override: {exc}") from exc
    if cfg.agent is None:
        raise click.ClickException("Config file must have an 'agent' section for rollout.")
    if exact_train_id and not cfg.rollout.train_id:
        raise click.ClickException("--exact-train-id requires rollout.train_id to be set.")

    run_id = cfg.rollout.train_id or f"eval_{model_slug(cfg.agent.model_name)}"
    if not exact_train_id:
        run_id = f"{run_id}_{secrets.token_hex(4)}"
    cfg.rollout.train_id = run_id

    log_root = cfg.manager.traj_log_dir or "./traj_logs"
    cfg.manager.traj_log_dir = f"{log_root.rstrip('/')}/run-{quote(run_id, safe='')}"
    click.echo(f"Resolved train_id: {run_id}")
    click.echo(f"Run log root: {cfg.manager.traj_log_dir}")

    server_url = manager_url or cfg.rollout.manager_url or f"http://127.0.0.1:{cfg.manager.port}"

    resolved_num_envs = num_envs if num_envs is not None else cfg.rollout.num_envs
    resolved_env_type = env_type or cfg.rollout.env_type
    resolved_tasks = tasks if tasks is not None else cfg.rollout.tasks
    resolved_tasks_dir = tasks_dir or cfg.rollout.tasks_dir
    resolved_no_resume = no_resume or cfg.rollout.no_resume
    resolved_pass_n = pass_n if pass_n is not None else cfg.rollout.pass_n
    resolved_retry_until_scored = (
        retry_until_scored if retry_until_scored is not None else cfg.rollout.retry_until_scored
    )
    resolved_max_retry_rounds = (
        max_retry_rounds if max_retry_rounds is not None else cfg.rollout.max_retry_rounds
    )
    if evaluation_mode is not None:
        cfg.rollout.evaluation_mode = evaluation_mode

    if domains is not None:
        domain_set = {d.strip() for d in domains.split(",") if d.strip()} or None
    elif cfg.rollout.domains:
        domain_set = set(cfg.rollout.domains)
    else:
        domain_set = None

    with _auto_serve(config, server_url):
        try:
            run_passn_rollout(
                cfg,
                server=server_url,
                num_envs=resolved_num_envs,
                env_type=resolved_env_type,
                task_filter=resolved_tasks,
                no_resume=resolved_no_resume,
                tasks_dir=resolved_tasks_dir,
                domains=domain_set,
                pass_n=resolved_pass_n,
                retry_until_scored=resolved_retry_until_scored,
                max_retry_rounds=resolved_max_retry_rounds,
            )
        except RolloutSetupError as exc:
            raise click.ClickException(str(exc)) from exc


cli.add_command(evaluate)
cli.add_command(train)
