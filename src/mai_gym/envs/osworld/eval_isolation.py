from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import os
import signal
import sys
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Sequence

    from mai_gym.envs.osworld.backend import OSWorldBackend

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class EvalIsolationSettings:
    enabled: bool = True
    timeout_seconds: float = 300.0
    queue_timeout_seconds: float = 300.0
    max_concurrency: int = 32
    worker_cpu_threads: int = 1
    max_ocr_concurrency: int = 2
    terminate_grace_seconds: float = 3.0


_settings = EvalIsolationSettings()
_semaphore: asyncio.Semaphore | None = None
_semaphore_loop: asyncio.AbstractEventLoop | None = None
_ocr_semaphore: asyncio.Semaphore | None = None
_ocr_semaphore_loop: asyncio.AbstractEventLoop | None = None
_stats: dict[str, int | float] = {
    "active": 0,
    "ocr_active": 0,
    "completed": 0,
    "timeouts": 0,
    "queue_timeouts": 0,
    "crashes": 0,
    "cancelled": 0,
    "total_queue_seconds": 0.0,
    "total_run_seconds": 0.0,
}
_SENSITIVE_ENV_MARKERS = (
    "ACCESS_KEY",
    "API_KEY",
    "CREDENTIAL",
    "PASSWORD",
    "SECRET",
    "SESSION_TOKEN",
)


def configure_eval_isolation(settings: dict[str, Any] | None = None) -> None:

    global _settings, _semaphore, _semaphore_loop
    global _ocr_semaphore, _ocr_semaphore_loop

    raw = settings or {}
    _settings = EvalIsolationSettings(
        enabled=bool(raw.get("enabled", True)),
        timeout_seconds=float(raw.get("timeout_seconds", 300.0)),
        queue_timeout_seconds=float(raw.get("queue_timeout_seconds", 300.0)),
        max_concurrency=int(raw.get("max_concurrency", 32)),
        worker_cpu_threads=max(1, int(raw.get("worker_cpu_threads", 1))),
        max_ocr_concurrency=max(1, int(raw.get("max_ocr_concurrency", 2))),
        terminate_grace_seconds=float(raw.get("terminate_grace_seconds", 3.0)),
    )
    _semaphore = None
    _semaphore_loop = None
    _ocr_semaphore = None
    _ocr_semaphore_loop = None
    for key in _stats:
        _stats[key] = 0
    logger.info(
        "[osworld-eval] isolation enabled=%s timeout=%.1fs queue_timeout=%.1fs "
        "max_concurrency=%d kill_grace=%.1fs worker_cpu_threads=%d max_ocr_concurrency=%d",
        _settings.enabled,
        _settings.timeout_seconds,
        _settings.queue_timeout_seconds,
        _settings.max_concurrency,
        _settings.terminate_grace_seconds,
        _settings.worker_cpu_threads,
        _settings.max_ocr_concurrency,
    )


def eval_isolation_settings() -> EvalIsolationSettings:
    return _settings


def eval_isolation_stats() -> dict[str, Any]:

    return {**asdict(_settings), **_stats}


def _get_semaphore() -> asyncio.Semaphore:
    global _semaphore, _semaphore_loop

    loop = asyncio.get_running_loop()
    if _semaphore is None or _semaphore_loop is not loop:
        _semaphore = asyncio.Semaphore(_settings.max_concurrency)
        _semaphore_loop = loop
    return _semaphore


def _get_ocr_semaphore() -> asyncio.Semaphore:
    global _ocr_semaphore, _ocr_semaphore_loop

    loop = asyncio.get_running_loop()
    if _ocr_semaphore is None or _ocr_semaphore_loop is not loop:
        _ocr_semaphore = asyncio.Semaphore(_settings.max_ocr_concurrency)
        _ocr_semaphore_loop = loop
    return _ocr_semaphore


def _uses_ocr(task_config: dict[str, Any]) -> bool:

    metrics = (task_config.get("evaluator") or {}).get("func")
    if isinstance(metrics, str):
        metrics = [metrics]
    return isinstance(metrics, list) and "compare_image_text" in metrics


def _worker_request(
    backend: OSWorldBackend,
    task_config: dict[str, Any],
    action_history: list[str] | None,
    cache_key: str | None,
    client_password: str,
) -> dict[str, Any]:
    metadata = backend.backend.metadata
    scalar_metadata = {
        key: metadata[key]
        for key in ("chromium_port", "vlc_port", "vnc_port")
        if isinstance(metadata.get(key), (str, int, float, bool))
    }
    running = backend.backend
    return {
        "backend": {
            "id": running.id,
            "url": running.url,
            "device_name": running.device_name,
            "env_type": running.env_type,
            "device_id": running.device_id,
            "metadata": scalar_metadata,
        },
        "cache_root": str(backend._cache_root),
        "task_config": task_config,
        "action_history": action_history,
        "cache_key": cache_key,
        "client_password": client_password,
    }


def _worker_environment() -> dict[str, str]:

    env = {
        key: value
        for key, value in os.environ.items()
        if not any(marker in key.upper() for marker in _SENSITIVE_ENV_MARKERS)
    }

    threads = str(_settings.worker_cpu_threads)
    for key in (
        "OMP_NUM_THREADS",
        "OMP_THREAD_LIMIT",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "GOTO_NUM_THREADS",
        "BLIS_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "NUMEXPR_MAX_THREADS",
    ):
        env[key] = threads
    env["OMP_MAX_ACTIVE_LEVELS"] = "1"
    env["MAI_GYM_OSWORLD_EVAL_WORKER_THREADS"] = threads
    return env


async def _terminate_process_group(
    process: asyncio.subprocess.Process,
    grace_seconds: float,
) -> None:

    if process.returncode is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    with contextlib.suppress(TimeoutError):
        await asyncio.wait_for(process.wait(), timeout=max(grace_seconds, 0.01))

    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        return
    if process.returncode is None:
        await process.wait()


def _kill_lingering_process_group(process_group_id: int) -> None:

    with contextlib.suppress(ProcessLookupError):
        os.killpg(process_group_id, signal.SIGKILL)


async def _run_worker(
    request: dict[str, Any],
    *,
    timeout_seconds: float,
    terminate_grace_seconds: float,
    worker_command: Sequence[str] | None = None,
) -> tuple[tuple[float, str] | None, str | None]:

    command = (
        list(worker_command)
        if worker_command is not None
        else [
            sys.executable,
            "-m",
            "mai_gym.envs.osworld.eval_worker",
        ]
    )
    started = time.monotonic()
    with tempfile.TemporaryDirectory(prefix="mai-gym-osw-eval-") as tmp:
        work_dir = Path(tmp)
        request_path = work_dir / "request.json"
        response_path = work_dir / "response.json"
        request_path.write_text(json.dumps(request), encoding="utf-8")
        request_path.chmod(0o600)
        command.extend(("--request", str(request_path), "--response", str(response_path)))

        process = await asyncio.create_subprocess_exec(
            *command,
            cwd=str(work_dir),
            stdin=asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.DEVNULL,
            start_new_session=True,
            env=_worker_environment(),
        )
        _stats["active"] += 1
        logger.info(
            "[osworld-eval] worker start pid=%d task=%s metric=%s",
            process.pid,
            request.get("task_config", {}).get("id"),
            request.get("task_config", {}).get("evaluator", {}).get("func"),
        )
        try:
            try:
                await asyncio.wait_for(process.wait(), timeout=timeout_seconds)
            except TimeoutError:
                _stats["timeouts"] += 1
                await _terminate_process_group(process, terminate_grace_seconds)
                elapsed = time.monotonic() - started
                return None, f"OSWorld evaluator timed out after {elapsed:.1f}s"
            except asyncio.CancelledError:
                _stats["cancelled"] += 1
                await _terminate_process_group(process, terminate_grace_seconds)
                raise

            _kill_lingering_process_group(process.pid)

            if process.returncode != 0:
                _stats["crashes"] += 1
                return None, (
                    f"OSWorld evaluator worker crashed with exit code {process.returncode}"
                )
            try:
                response = json.loads(response_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                _stats["crashes"] += 1
                return None, f"OSWorld evaluator returned no valid result: {exc}"
            if not response.get("ok"):
                _stats["crashes"] += 1
                error_type = str(response.get("error_type") or "Error")
                error = str(response.get("error") or "unknown worker error")
                return None, f"OSWorld evaluator {error_type}: {error}"
            raw_result = response.get("result")
            _stats["completed"] += 1
            if raw_result is None:
                return None, None
            return (float(raw_result[0]), str(raw_result[1])), None
        finally:
            _stats["active"] -= 1
            _stats["total_run_seconds"] += time.monotonic() - started


async def evaluate_in_subprocess(
    backend: OSWorldBackend,
    task_config: dict[str, Any],
    action_history: list[str] | None = None,
    cache_key: str | None = None,
    client_password: str = "password",
) -> tuple[float, str] | None:

    request = _worker_request(
        backend,
        task_config,
        action_history,
        cache_key,
        client_password,
    )
    semaphore = _get_semaphore()
    ocr_semaphore = _get_ocr_semaphore() if _uses_ocr(task_config) else None
    queued_at = time.monotonic()
    acquired = False
    ocr_acquired = False
    try:
        try:
            async with asyncio.timeout(_settings.queue_timeout_seconds):
                if ocr_semaphore is not None:
                    await ocr_semaphore.acquire()
                    ocr_acquired = True
                await semaphore.acquire()
                acquired = True
        except TimeoutError:
            _stats["queue_timeouts"] += 1
            return 0.0, (
                f"OSWorld evaluator queue timed out after {_settings.queue_timeout_seconds:.1f}s"
            )
        _stats["total_queue_seconds"] += time.monotonic() - queued_at
        if ocr_acquired:
            _stats["ocr_active"] += 1
        try:
            result, error = await _run_worker(
                request,
                timeout_seconds=_settings.timeout_seconds,
                terminate_grace_seconds=_settings.terminate_grace_seconds,
            )
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            _stats["crashes"] += 1
            error = (
                "OSWorld evaluator worker could not start or communicate: "
                f"{type(exc).__name__}: {exc}"
            )
            result = None
    finally:
        if acquired:
            semaphore.release()
            if ocr_acquired:
                _stats["ocr_active"] -= 1
        if ocr_acquired:
            ocr_semaphore.release()
    if error is not None:
        logger.error(
            "[osworld-eval] task=%s metric=%s failed: %s",
            task_config.get("id"),
            task_config.get("evaluator", {}).get("func"),
            error,
        )
        return 0.0, error
    return result
