import asyncio
import base64
import hashlib
import json
import logging
import os
import tempfile
import time
from collections import Counter, OrderedDict
from contextlib import nullcontext, suppress
from pathlib import Path
from typing import Any, Literal

import httpx

from mai_gym.envs import create_backend
from mai_gym.envs.shared.backend import BrowserBackend, PooledBackend
from mai_gym.envs.shared.strategy import EnvStrategy
from mai_gym.infra.backend import Backend, BackendCallError, BackendUnavailableError
from mai_gym.infra.manager import InfraManager
from mai_gym.infra.task_artifacts import parse_task_set_artifact_manifest
from mai_gym.models.config import DeviceConfig, K8sLaunchConfig, OSSConfig
from mai_gym.models.operation_error import OperationError, operation_error_detail
from mai_gym.models.session import (
    PendingOp,
    RunningBackend,
    SessionState,
    SessionStatus,
    SessionTask,
)
from mai_gym.server.config_writer import persist_max_sessions, persist_scale_settings
from mai_gym.server.env_strategy import create_strategy
from mai_gym.server.fleet_autoscaler import FleetAutoscaler, FleetRequestUnavailable
from mai_gym.server.reward import DeterministicEvaluator, RewardEvaluator
from mai_gym.server.scheduler import AcquireResult, SessionScheduler
from mai_gym.server.session import (
    InvalidTransitionError,
    SessionManager,
    SessionNotFoundError,
)
from mai_gym.server.trajectory import (
    alog_step,
    create_traj_logger,
    log_failure,
    log_score,
    model_slug,
)

logger = logging.getLogger(__name__)


class TaskArtifactPreparationError(RuntimeError):
    pass


_TERMINAL_ACTIONS = frozenset({"answer", "terminate", "finished"})


_DEEPLINK_RECHECK_WAIT = 2.0


_LEASE_SWEEP_INTERVAL_S = 60.0


_MW_ATTACH_DEADLINE_S = 210.0
_MW_RESET_DEADLINE_S = 420.0


_POOL_SNAPSHOT_INTERVAL_S = 60.0


_RECOVERY_SWEEP_INTERVAL_S = 60.0


_BUSY_REAP_AFTER_S = float(os.environ.get("MAI_GYM_BUSY_REAP_AFTER_S", "900"))


_RECOVERY_HEALTH_TIMEOUT_S = 120.0


_BATCH_INIT_CONCURRENCY = 512


_PREWARM_CHUNK = max(1, int(os.environ.get("MAI_GYM_PREWARM_CHUNK", "16") or "16"))
_PREWARM_INFLIGHT = max(1, int(os.environ.get("MAI_GYM_PREWARM_INFLIGHT", "64") or "64"))


_STEP_SCREENSHOT_RETRIES = max(
    0, int(os.environ.get("MAI_GYM_STEP_SCREENSHOT_RETRIES", "2") or "2")
)
_STEP_SCREENSHOT_RETRY_BACKOFF_S = 0.8


_REGISTER_FANOUT_CONCURRENCY = 16
_CLEANUP_RECEIPT_LIMIT = 4096
_CLEANUP_MAX_INFLIGHT = 4096


_TASK_LIST_CACHE_TTL_S = max(
    0.0, float(os.environ.get("MAI_GYM_TASK_LIST_CACHE_TTL_S", "300") or "300")
)

_DEFAULT_RESET_CONCURRENCY_PER_DEVICE = 256
_RESET_CONCURRENCY_PER_DEVICE = max(
    256,
    int(
        os.environ.get(
            "MAI_GYM_RESET_CONCURRENCY_PER_DEVICE",
            str(_DEFAULT_RESET_CONCURRENCY_PER_DEVICE),
        )
    ),
)


def _classify_backend_failure(exc: BaseException) -> tuple[bool, str]:

    if isinstance(exc, BackendCallError):
        return True, "conn"
    if isinstance(exc, BackendUnavailableError):
        return True, "circuit"
    if isinstance(exc, TaskArtifactPreparationError):
        return True, "task_artifact"
    if isinstance(exc, httpx.HTTPStatusError):
        code = exc.response.status_code
        if 500 <= code < 600:
            return True, f"http_{code}"
    return False, type(exc).__name__


def _is_transient_capture_error(exc: BaseException) -> bool:

    if isinstance(exc, httpx.HTTPStatusError):
        return 500 <= exc.response.status_code < 600
    return isinstance(exc, (httpx.TransportError, BackendCallError))


def _failure_log_extra(exc: BaseException) -> str:

    if isinstance(exc, BackendCallError):
        return f"cause={type(exc.cause).__name__}: {exc.cause}"
    if isinstance(exc, BackendUnavailableError):
        return f"retry_in={exc.retry_in:.1f}s"
    if isinstance(exc, httpx.HTTPStatusError):
        try:
            body = exc.response.text[:200].replace("\n", " ")
        except Exception:
            body = ""
        return f"status={exc.response.status_code} url={exc.request.url} body={body!r}"
    return f"{type(exc).__name__}: {exc}"


def _resize_screenshot(b64: str, width: int, height: int) -> str:

    if not b64:
        return b64
    from mai_gym.utils.images import b64_to_pil, pil_to_b64

    img = b64_to_pil(b64)
    if img.size == (width, height):
        return b64
    from PIL import Image

    return pil_to_b64(img.resize((width, height), Image.Resampling.LANCZOS))


def _encode_screenshot_sync(b64: str, codec: str, quality: int) -> str:

    from mai_gym.utils.images import transcode_b64

    return transcode_b64(b64, fmt=codec, quality=quality)


class EnvController:
    def __init__(
        self,
        infra: InfraManager,
        *,
        debug_log_dir: str | None = None,
        post_action_wait: float = 1.5,
        reward_evaluator: RewardEvaluator | None = None,
        judge_evaluator: RewardEvaluator | None = None,
        osworld_examples_dir: str | None = None,
        cua_osworld_tasks_dir: str | None = None,
        vscode_ext_mirror: dict | None = None,
        devices: list[DeviceConfig] | None = None,
        model_slug: str | None = None,
        oss_cfg: OSSConfig | None = None,
        screenshot_codec: str = "png",
        screenshot_quality: int = 85,
        screenshot_transport: str = "inline",
        screenshot_oss_prefix: str | None = None,
        config_path: str | None = None,
    ) -> None:
        self._infra = infra

        self._config_path = config_path

        self._screenshot_codec = screenshot_codec
        self._screenshot_quality = screenshot_quality

        self._screenshot_transport = screenshot_transport
        self._screenshot_oss_prefix = screenshot_oss_prefix
        self._debug_log_dir = debug_log_dir
        self._model_slug = model_slug

        self._oss_cfg = oss_cfg

        self._world_cache_dir = os.environ.get(
            "MAI_GYM_WORLD_CACHE_DIR",
            os.path.join(tempfile.gettempdir(), "mai_gym_world_cache"),
        )
        self._world_cache_locks: dict[str, asyncio.Lock] = {}

        if debug_log_dir and model_slug:
            self._traj_log_dir: str | None = f"{debug_log_dir.rstrip('/')}/{model_slug}"
        else:
            self._traj_log_dir = debug_log_dir
        self._post_action_wait = post_action_wait
        self._reward_evaluator = reward_evaluator or DeterministicEvaluator()

        self._judge_evaluator = judge_evaluator
        self._sessions = SessionManager()
        self._scheduler = SessionScheduler(self._sessions)
        self._backends: dict[str, Backend] = {}
        self._task_list_cache: list[dict[str, Any]] | None = None
        self._task_list_cache_expires_at = 0.0
        self._task_list_cache_ttl_s = _TASK_LIST_CACHE_TTL_S
        self._task_list_cache_lock = asyncio.Lock()

        self._relaunching: set[str] = set()

        self._run_config_written: set[str] = set()

        self._trained_written: set[tuple[str, int | None, bool]] = set()

        self._recovery_tasks: set[asyncio.Task[None]] = set()

        self._lazy_destroy_tasks: dict[str, asyncio.Task[None]] = {}
        self._op_tasks: set[asyncio.Task[dict[str, Any]]] = set()
        self._cleanup_receipts: OrderedDict[tuple[str, str], dict[str, Any]] = OrderedDict()
        self._cleanup_tasks: dict[tuple[str, str], asyncio.Task[dict[str, Any]]] = {}
        from mai_gym.server.sdk_operations import SDKOperations

        self.sdk_operations = SDKOperations(self)
        self._strategies: dict[str, EnvStrategy] = {}
        self._osworld_examples_dir = osworld_examples_dir
        self._cua_osworld_tasks_dir = cua_osworld_tasks_dir

        self._vscode_ext_mirror = vscode_ext_mirror
        self._devices: list[DeviceConfig] = list(devices) if devices else []

        self._lazy_pool_devices = {
            d.name
            for d in self._devices
            if isinstance(d.launch, K8sLaunchConfig) and d.launch.lazy_acquire
        }
        lazy_env_types = {d.type for d in self._devices if d.name in self._lazy_pool_devices}
        if len(lazy_env_types) > 1:
            raise ValueError(
                "lazy_acquire devices on one manager must share one env_type; "
                f"got {sorted(lazy_env_types)}"
            )
        for d in self._devices:
            if d.name in self._lazy_pool_devices and d.effective_sessions_per_backend != 1:
                raise ValueError(f"lazy_acquire device {d.name!r} requires sessions_per_backend=1")
        self._lazy_pool_env_type = next(iter(lazy_env_types), None)

        self._lazy_capacity_cond = asyncio.Condition()
        self._lazy_capacity_generation = 0
        self._lazy_provision_tasks: set[asyncio.Task[int]] = set()
        self._lazy_provision_lock = asyncio.Lock()
        self._lazy_acquire_waiters = 0
        self._lazy_scoped_acquire_waiters = 0
        self._lazy_retry_after = 0.0
        self._fleet_autoscaler = FleetAutoscaler(
            self._devices,
            resize_pool=self._infra.resize_sandbox_pool,
            pool_status=self._infra.sandbox_pool_status,
            activity_counts=self._fleet_activity_counts,
        )

        self._reset_semaphores: dict[str, asyncio.Semaphore] = {}
        ungated: list[str] = []
        for d in self._devices:
            if d.max_sessions <= 0:
                continue
            if d.reset_concurrency is not None:
                limit = d.reset_concurrency
            elif isinstance(d.launch, K8sLaunchConfig):
                limit = 0
            else:
                limit = min(_RESET_CONCURRENCY_PER_DEVICE, d.max_sessions)
            if limit <= 0:
                ungated.append(d.name)
                continue
            self._reset_semaphores[d.name] = asyncio.Semaphore(limit)
        if self._reset_semaphores or ungated:
            logger.info(
                "[EnvController] reset-throttle: cap=%d per device (env=%s); gated=%s ungated=%s",
                _RESET_CONCURRENCY_PER_DEVICE,
                os.environ.get("MAI_GYM_RESET_CONCURRENCY_PER_DEVICE", "unset"),
                sorted(self._reset_semaphores),
                sorted(ungated),
            )

        self._failure_counters: Counter[tuple[str, str, str]] = Counter()

        self._screenshot_upload_successes = 0
        self._screenshot_upload_inline_fallbacks = 0
        self._screenshot_inline_deliveries = 0
        self._screenshot_upload_bytes = 0
        self._screenshot_upload_seconds = 0.0

    def _record_backend_failure(
        self,
        phase: str,
        state: Any,
        exc: BaseException,
        kind: str,
    ) -> None:

        self._failure_counters[(phase, state.backend.device_name, kind)] += 1
        logger.warning(
            "[backend-fail] phase=%s sid=%s device=%s kind=%s %s",
            phase,
            state.session_id,
            state.backend.device_name,
            kind,
            _failure_log_extra(exc),
        )

    def _get_or_create_backend(self, backend: RunningBackend) -> Backend:
        if backend.id not in self._backends:
            self._backends[backend.id] = create_backend(backend)
            self._invalidate_task_list_cache()
        return self._backends[backend.id]

    def _invalidate_task_list_cache(self) -> None:
        self._task_list_cache = None
        self._task_list_cache_expires_at = 0.0

    def _get_strategy(self, env_type: str) -> EnvStrategy:
        if env_type not in self._strategies:
            self._strategies[env_type] = create_strategy(
                env_type,
                osworld_examples_dir=self._osworld_examples_dir,
                cua_osworld_tasks_dir=self._cua_osworld_tasks_dir,
                vscode_ext_mirror=self._vscode_ext_mirror,
            )
        return self._strategies[env_type]

    async def batch_start_env(self, num_envs: int, env_type: str = "real_device") -> dict[str, Any]:
        if self._fleet_autoscaler.supports_env_type(env_type):
            ready = sum(
                state.status == SessionStatus.READY
                and state.lease_id is None
                and state.backend.env_type == env_type
                for state in self._sessions.list_sessions()
            )
            hint = await self._fleet_autoscaler.request_legacy_warm_capacity(
                env_type,
                ready + num_envs,
            )
            return {
                "sessions": [],
                "total_count": num_envs,
                "accepted_count": num_envs,
                "requested_count": num_envs,
                "manager_ready_count": ready,
                "capacity_state": ("warm_ready" if hint["warm_ready"] else "scale_requested"),
                "warming": not hint["warm_ready"],
                **hint,
            }
        unscoped_device = await self._fleet_autoscaler.begin_unscoped_provision(
            env_type,
            num_envs,
        )
        try:
            return await self._batch_start_env(num_envs, env_type)
        finally:
            await self._fleet_autoscaler.complete_unscoped_provision(
                unscoped_device,
                num_envs,
            )

    async def _batch_start_env(
        self, num_envs: int, env_type: str = "real_device"
    ) -> dict[str, Any]:

        backends = await self._infra.provision(env_type, num_envs)
        clients = [self._get_or_create_backend(b) for b in backends]
        sem = asyncio.Semaphore(_BATCH_INIT_CONCURRENCY)

        async def _start_one(
            backend: RunningBackend,
            client: Backend,
        ) -> dict[str, Any] | None:
            async with sem:
                try:
                    await client.init_device()
                    context_id = (
                        await client.acquire_context()
                        if isinstance(client, PooledBackend)
                        else None
                    )

                    state = self._sessions.create_session(backend)
                    state.context_id = context_id
                    self._sessions.transition(state.session_id, SessionStatus.READY)
                    return {
                        "session_id": state.session_id,
                        "message": "started",
                        "worker_idx": backend.device_name,
                    }
                except Exception as exc:
                    logger.error(
                        "[EnvController] batch_start_env error for %s: %s",
                        backend.device_name,
                        exc,
                    )
                    await self._infra.release(backend.id)
                    if not isinstance(client, PooledBackend):
                        failed_client = self._backends.pop(backend.id, None)
                        if failed_client is not None:
                            with suppress(Exception):
                                await failed_client.close()
                    return None

        results = await asyncio.gather(
            *(_start_one(b, c) for b, c in zip(backends, clients, strict=True)),
        )

        ready_backends = [
            backend for backend, result in zip(backends, results, strict=True) if result is not None
        ]
        if self._infra.registered_set_ids():
            await self._reapply_task_sets(ready_backends)
        sessions_out = [r for r in results if r is not None]
        return {"sessions": sessions_out, "total_count": len(sessions_out)}

    async def close(self) -> None:
        for backend in self._backends.values():
            try:
                await backend.close()
            except Exception:
                logger.exception("Backend connection cleanup failed")
        self._backends.clear()

    async def register_task_set(
        self,
        set_id: str,
        package_bytes: bytes,
    ) -> dict[str, Any]:

        manifest = parse_task_set_artifact_manifest(package_bytes)
        lock = await self._infra.get_set_lock(set_id)
        async with lock:
            self._infra.store_task_set_artifact_manifest(set_id, manifest)
            self._infra.store_task_set(set_id, package_bytes)

            backends = self._infra.all_backends()
            sem = asyncio.Semaphore(_REGISTER_FANOUT_CONCURRENCY)

            async def _one(backend: RunningBackend) -> dict[str, Any]:
                async with sem:
                    client = self._get_or_create_backend(backend)
                    base = {
                        "device": backend.device_name,
                        "backend_id": backend.id,
                    }
                    try:
                        existing = await client.list_task_sets()
                        if any(s.get("set_id") == set_id for s in existing):
                            return {**base, "ok": True, "skipped": True}
                        resp = await client.register_task_set(set_id, package_bytes)
                        return {
                            **base,
                            "ok": True,
                            "skipped": False,
                            "registered": resp.get("registered", []),
                            "errors": resp.get("errors", {}),
                            "shadowed": resp.get("shadowed", []),
                        }
                    except httpx.HTTPStatusError as exc:
                        resp = exc.response
                        return {
                            **base,
                            "ok": False,
                            "status": resp.status_code if resp is not None else None,
                            "error": resp.text[:300] if resp is not None else str(exc),
                        }
                    except Exception as exc:
                        return {
                            **base,
                            "ok": False,
                            "error": f"{type(exc).__name__}: {exc}",
                        }

            results = await asyncio.gather(*(_one(b) for b in backends))

        num_ok = sum(1 for r in results if r["ok"])
        num_skipped = sum(1 for r in results if r.get("skipped"))
        num_registered = num_ok - num_skipped
        num_failed = len(results) - num_ok
        logger.info(
            "[EnvController] register_task_set set_id=%s backends=%d "
            "registered=%d skipped=%d failed=%d",
            set_id,
            len(results),
            num_registered,
            num_skipped,
            num_failed,
        )
        self._invalidate_task_list_cache()
        response = {
            "set_id": set_id,
            "num_backends": len(results),
            "num_registered": num_registered,
            "num_skipped": num_skipped,
            "num_failed": num_failed,
            "results": results,
        }
        if manifest is not None:
            response.update(
                {
                    "artifact_aware": True,
                    "artifact_count": len(manifest.artifacts),
                    "artifact_tasks": list(manifest.task_names),
                }
            )
        return response

    async def unregister_task_set(self, set_id: str) -> dict[str, Any]:

        lock = await self._infra.get_set_lock(set_id)
        async with lock:
            was_registered = self._infra.unregister_task_set(set_id)

            backends = self._infra.all_backends()
            sem = asyncio.Semaphore(_REGISTER_FANOUT_CONCURRENCY)

            async def _one(backend: RunningBackend) -> dict[str, Any]:
                async with sem:
                    client = self._get_or_create_backend(backend)
                    base = {
                        "device": backend.device_name,
                        "backend_id": backend.id,
                    }
                    try:
                        resp = await client.unregister_task_set(set_id)
                        return {**base, "ok": True, "removed": resp.get("removed", [])}
                    except httpx.HTTPStatusError as exc:
                        resp = exc.response
                        return {
                            **base,
                            "ok": False,
                            "status": resp.status_code if resp is not None else None,
                            "error": resp.text[:300] if resp is not None else str(exc),
                        }
                    except Exception as exc:
                        return {
                            **base,
                            "ok": False,
                            "error": f"{type(exc).__name__}: {exc}",
                        }

            results = await asyncio.gather(*(_one(b) for b in backends))

        num_removed = sum(1 for r in results if r["ok"])
        num_failed = len(results) - num_removed
        logger.info(
            "[EnvController] unregister_task_set set_id=%s was_registered=%s "
            "backends=%d removed=%d failed=%d",
            set_id,
            was_registered,
            len(results),
            num_removed,
            num_failed,
        )
        self._invalidate_task_list_cache()
        return {
            "set_id": set_id,
            "was_registered": was_registered,
            "num_backends": len(results),
            "num_removed": num_removed,
            "num_failed": num_failed,
            "results": results,
        }

    async def write_run_config(
        self,
        train_id: str | None,
        config: dict[str, Any],
        *,
        model_name: str | None = None,
    ) -> dict[str, Any]:

        if not train_id:
            return {"ok": False, "skipped": True, "reason": "no train_id"}
        if train_id in self._run_config_written:
            return {"ok": True, "skipped": True, "reason": "already written"}

        resolve = getattr(self, "_resolve_traj_target", None)
        traj_root = resolve(model_name)[0] if callable(resolve) else self._traj_log_dir
        if not traj_root:
            return {"ok": False, "skipped": True, "reason": "no debug_log_dir"}

        import re

        from mai_gym.io import oss_io

        seg = re.sub(r"[^A-Za-z0-9._-]", "_", train_id) or "unknown_train"
        path = oss_io.join(traj_root, seg, "run_config.json")
        is_oss = oss_io.is_oss_path(path)
        if is_oss and self._oss_cfg is None:
            logger.warning("write_run_config: oss path %r needs oss_cfg; skipping", path)
            return {"ok": False, "skipped": True, "reason": "oss path without oss_cfg"}

        try:
            fs = oss_io.get_async_fs(self._oss_cfg) if is_oss else None

            try:
                if await oss_io.exists(path, fs=fs):
                    self._run_config_written.add(train_id)
                    return {"ok": True, "skipped": True, "reason": "exists on sink"}
            except Exception:
                pass
            payload = json.dumps(config, ensure_ascii=False, indent=2, default=str)
            await oss_io.write_text(path, payload, fs=fs)
            self._run_config_written.add(train_id)
            logger.info(
                "[EnvController] wrote run_config train_id=%s -> %s",
                train_id,
                path,
            )
            return {"ok": True, "skipped": False, "path": path}
        except Exception as exc:
            logger.warning("write_run_config: %s failed (%s); continuing", path, exc)
            return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}

    async def write_trained_manifest(
        self,
        train_id: str | None,
        rollout_id: int | None,
        manifest: dict[str, Any],
        *,
        evaluation: bool = False,
        model_name: str | None = None,
    ) -> dict[str, Any]:

        if not train_id:
            return {"ok": False, "skipped": True, "reason": "no train_id"}
        dedup_key = (train_id, rollout_id, evaluation)
        if dedup_key in self._trained_written:
            return {"ok": True, "skipped": True, "reason": "already written"}

        resolve = getattr(self, "_resolve_traj_target", None)
        traj_root = resolve(model_name)[0] if callable(resolve) else self._traj_log_dir
        if not traj_root:
            return {"ok": False, "skipped": True, "reason": "no debug_log_dir"}

        from mai_gym.io import oss_io
        from mai_gym.server.trajectory import _rollout_seg, _seg_slug

        seg = _seg_slug(train_id, "unknown_train")
        rollout_seg = _rollout_seg(rollout_id, evaluation)
        path = oss_io.join(traj_root, seg, rollout_seg, "trained.json")
        is_oss = oss_io.is_oss_path(path)
        if is_oss and self._oss_cfg is None:
            logger.warning("write_trained_manifest: oss path %r needs oss_cfg; skipping", path)
            return {"ok": False, "skipped": True, "reason": "oss path without oss_cfg"}

        manifest = {**manifest, "train_id": train_id}
        try:
            fs = oss_io.get_async_fs(self._oss_cfg) if is_oss else None
            payload = json.dumps(manifest, ensure_ascii=False, indent=2, default=str)
            await oss_io.write_text(path, payload, fs=fs)
            self._trained_written.add(dedup_key)
            logger.info(
                "[EnvController] wrote trained manifest train_id=%s rollout=%s -> %s",
                train_id,
                rollout_id,
                path,
            )
            return {"ok": True, "skipped": False, "path": path}
        except Exception as exc:
            logger.warning("write_trained_manifest: %s failed (%s); continuing", path, exc)
            return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}

    async def log_acquire_failure(
        self,
        *,
        train_id: str | None,
        task_id: str | None,
        attempt_id: str | None,
        rollout_id: int | None,
        evaluation: bool,
        model_name: str | None,
        env_type: str | None,
        reason: str | None,
    ) -> dict[str, Any]:

        from mai_gym.io import oss_io
        from mai_gym.server.trajectory import (
            _MAX_FAILURE_REASON_CHARS,
            _traj_subpath,
        )

        resolve = getattr(self, "_resolve_traj_target", None)
        traj_root = resolve(model_name)[0] if callable(resolve) else self._traj_log_dir
        if not traj_root:
            return {"ok": False, "skipped": True, "reason": "no debug_log_dir"}

        traj_name = _traj_subpath(
            task_id,
            env_type,
            None,
            attempt_id=attempt_id,
            train_id=train_id,
            rollout_id=rollout_id,
            evaluation=evaluation,
        )
        path = oss_io.join(traj_root, traj_name, "error.txt")
        is_oss = oss_io.is_oss_path(path)
        if is_oss and self._oss_cfg is None:
            logger.warning(
                "log_acquire_failure: oss path %r needs oss_cfg; skipping",
                path,
            )
            return {"ok": False, "skipped": True, "reason": "oss path without oss_cfg"}

        text = (reason or "acquire failed (no session)").strip()
        if len(text) > _MAX_FAILURE_REASON_CHARS:
            text = text[:_MAX_FAILURE_REASON_CHARS] + "…"

        try:
            fs = oss_io.get_async_fs(self._oss_cfg) if is_oss else None
            await oss_io.write_text(path, text + "\n", fs=fs)
            logger.info(
                "[EnvController] wrote acquire-failure error.txt train_id=%s "
                "task=%s attempt=%s -> %s",
                train_id,
                task_id,
                attempt_id,
                path,
            )
            return {"ok": True, "path": path}
        except Exception as exc:
            logger.warning(
                "log_acquire_failure: %s failed (%s); continuing",
                path,
                exc,
            )
            return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}

    async def _reapply_task_sets(self, backends: list[RunningBackend]) -> None:

        sets = self._infra.task_set_bytes()
        if not sets or not backends:
            return
        sem = asyncio.Semaphore(_REGISTER_FANOUT_CONCURRENCY)

        async def _seed(backend: RunningBackend) -> None:
            async with sem:
                client = self._get_or_create_backend(backend)
                try:
                    have = {s.get("set_id") for s in await client.list_task_sets()}
                except Exception:
                    have = set()
                for set_id, pkg in sets.items():
                    if set_id in have:
                        continue
                    try:
                        await client.register_task_set(set_id, pkg)
                    except Exception as exc:
                        logger.warning(
                            "[EnvController] re-apply set %s to %s failed: %s",
                            set_id,
                            backend.device_name,
                            exc,
                        )

        seen: set[str] = set()
        uniq = [b for b in backends if not (b.id in seen or seen.add(b.id))]
        await asyncio.gather(*(_seed(b) for b in uniq))

    async def _prepare_registered_task_artifacts(
        self,
        client: Backend,
        state: SessionState,
        task_id: str | None,
    ) -> None:

        if not task_id:
            return
        binding = self._infra.task_artifact_binding(task_id)

        if not isinstance(binding, tuple) or len(binding) != 3:
            return
        set_id, package, _artifacts = binding
        try:
            have = {item.get("set_id") for item in await client.list_task_sets()}
            if set_id not in have:
                registration = await client.register_task_set(set_id, package)
                errors = registration.get("errors") or {}
                if task_id in errors:
                    raise RuntimeError(f"task code registration failed: {errors[task_id]}")
            results = await self._infra.prepare_task_artifacts(
                state.backend,
                task_id,
            )
        except Exception as exc:
            raise TaskArtifactPreparationError(
                f"failed to prepare artifacts for {task_id} on "
                f"{state.backend.id}: {type(exc).__name__}: {exc}"
            ) from exc
        state.backend.metadata["task_artifacts"] = {
            "task_id": task_id,
            "set_id": set_id,
            "results": results,
        }
        logger.info(
            "[EnvController] prepared %d artifact(s) for task=%s backend=%s set=%s",
            len(results),
            task_id,
            state.backend.id,
            set_id,
        )

    def _is_lazy_pool_backend(self, backend: RunningBackend) -> bool:
        return backend.device_name in self._lazy_pool_devices

    def _is_lazy_pool_state(self, state: SessionState) -> bool:
        return self._is_lazy_pool_backend(state.backend)

    async def _notify_lazy_capacity(self, n: int = 1) -> None:
        if n <= 0:
            return
        async with self._lazy_capacity_cond:
            self._lazy_capacity_generation += 1
            self._lazy_capacity_cond.notify(n)

    async def _provision_one_lazy_session(
        self,
        env_type: str,
        unscoped_device: str | None = None,
    ) -> int:

        try:
            result = await self._batch_start_env(1, env_type)
            created = int(result.get("total_count", 0))
            if created:
                await self._notify_lazy_capacity(created)
            else:
                self._lazy_retry_after = time.monotonic() + 1.0
            return created
        except Exception:
            logger.exception(
                "[EnvController] lazy acquire provision failed for env_type=%s",
                env_type,
            )
            self._lazy_retry_after = time.monotonic() + 1.0
            return 0
        finally:
            await self._fleet_autoscaler.complete_unscoped_provision(unscoped_device)

    def _start_lazy_provision(
        self,
        env_type: str,
        unscoped_device: str | None = None,
    ) -> asyncio.Task[int]:
        task = asyncio.create_task(self._provision_one_lazy_session(env_type, unscoped_device))
        self._lazy_provision_tasks.add(task)
        task.add_done_callback(self._lazy_provision_tasks.discard)
        return task

    def _lazy_ready_buffer_target(self) -> int:
        return sum(
            min(d.launch.ready_buffer_size, max(d.max_sessions, 0))
            for d in self._devices
            if d.name in self._lazy_pool_devices and isinstance(d.launch, K8sLaunchConfig)
        )

    async def _ensure_lazy_provisions(self, *, fleet_scoped: bool = False) -> None:

        env_type = self._lazy_pool_env_type
        if env_type is None or time.monotonic() < self._lazy_retry_after:
            return
        async with self._lazy_provision_lock:
            sessions = [s for s in self._sessions.list_sessions() if self._is_lazy_pool_state(s)]
            ready = sum(s.status == SessionStatus.READY and s.lease_id is None for s in sessions)
            pending = sum(not task.done() for task in self._lazy_provision_tasks)
            ceiling = sum(
                max(d.max_sessions, 0) for d in self._devices if d.name in self._lazy_pool_devices
            )

            waiter_deficit = max(self._lazy_scoped_acquire_waiters - ready - pending, 0)
            demand = self._lazy_acquire_waiters + self._lazy_ready_buffer_target()
            deficit = min(
                max(demand - ready - pending, 0),
                max(ceiling - len(sessions) - pending, 0),
            )
            for index in range(deficit):
                token = None
                if not fleet_scoped or index >= waiter_deficit:
                    try:
                        token = await self._fleet_autoscaler.begin_unscoped_provision(env_type)
                    except FleetRequestUnavailable:
                        break
                self._start_lazy_provision(env_type, token)

    async def run_lazy_ready_buffer(self) -> None:

        while True:
            if self._lazy_acquire_waiters or self._lazy_ready_buffer_target():
                try:
                    await self._ensure_lazy_provisions(
                        fleet_scoped=self._lazy_scoped_acquire_waiters > 0,
                    )
                except Exception:
                    logger.exception("[EnvController] ready buffer refill failed")
            await asyncio.sleep(1.0)

    def _fleet_activity_counts(self, device_name: str) -> tuple[int, int]:
        registered = sum(
            state.backend.device_name == device_name for state in self._sessions.list_sessions()
        )
        pending = (
            sum(not task.done() for task in self._lazy_provision_tasks)
            if device_name in self._lazy_pool_devices
            else 0
        )
        return registered, pending

    def fleet_capability(self) -> dict[str, Any] | None:
        return self._fleet_autoscaler.capability()

    async def run_fleet_autoscaler(self) -> None:
        await self._fleet_autoscaler.run()

    async def upsert_fleet_request(self, **kwargs: Any) -> dict[str, Any]:
        return await self._fleet_autoscaler.upsert_request(**kwargs)

    async def heartbeat_fleet_request(
        self,
        request_id: str,
        holder_id: str,
    ) -> dict[str, Any]:
        return await self._fleet_autoscaler.heartbeat(request_id, holder_id)

    async def release_fleet_holder(
        self,
        request_id: str,
        holder_id: str,
    ) -> dict[str, Any]:
        return await self._fleet_autoscaler.release_holder(request_id, holder_id)

    async def get_fleet_request(self, request_id: str) -> dict[str, Any]:
        return await self._fleet_autoscaler.get_request(request_id)

    async def list_fleet_requests(self) -> list[dict[str, Any]]:
        return await self._fleet_autoscaler.list_requests()

    def _pool_availability(self, sessions: list[SessionState]) -> dict[str, Any]:

        manager_idle = sum(s.status == SessionStatus.READY and s.lease_id is None for s in sessions)
        lazy_sessions = [s for s in sessions if self._is_lazy_pool_state(s)]
        pending_provisions = sum(not task.done() for task in self._lazy_provision_tasks)
        lazy_target = sum(
            max(0, device.max_sessions)
            for device in self._devices
            if device.name in self._lazy_pool_devices
        )
        on_demand_capacity = max(
            lazy_target - len(lazy_sessions) - pending_provisions,
            0,
        )
        buffer_status = (
            {
                "ready_buffer_target": self._lazy_ready_buffer_target(),
                "acquire_waiters": self._lazy_acquire_waiters,
            }
            if self._lazy_ready_buffer_target()
            else {}
        )
        return {
            "acquirable": manager_idle + on_demand_capacity,
            "manager_idle": manager_idle,
            "on_demand_capacity": on_demand_capacity,
            "pending_provisions": pending_provisions,
            "availability_mode": ("on_demand" if self._lazy_pool_devices else "registered"),
            **buffer_status,
        }

    async def acquire(
        self,
        target_apps: list[str] | None,
        timeout_s: float,
        fleet_request_id: str | None = None,
        env_type: str | None = None,
    ) -> AcquireResult | None:
        reservation: str | None = None
        if fleet_request_id is not None:
            reservation = await self._fleet_autoscaler.begin_acquire(fleet_request_id)
        try:
            result = await self._acquire_session(
                target_apps,
                timeout_s,
                fleet_scoped=reservation is not None,
                **({"requested_env_type": env_type} if env_type else {}),
            )
        except BaseException:
            if reservation is not None:
                await self._fleet_autoscaler.complete_acquire(reservation, None)
            raise
        if reservation is not None:
            await self._fleet_autoscaler.complete_acquire(
                reservation,
                result.session_id if result is not None else None,
            )
        return result

    async def _acquire_session(
        self,
        target_apps: list[str] | None,
        timeout_s: float,
        *,
        fleet_scoped: bool = False,
        requested_env_type: str | None = None,
    ) -> AcquireResult | None:

        env_type = self._lazy_pool_env_type
        selection = {"env_type": requested_env_type} if requested_env_type else {}
        if env_type is None:
            return await self._scheduler.acquire(target_apps, timeout_s, **selection)

        deadline = time.monotonic() + max(timeout_s, 0.0)
        self._lazy_acquire_waiters += 1
        self._lazy_scoped_acquire_waiters += int(fleet_scoped)
        try:
            while True:
                generation = self._lazy_capacity_generation
                ready = await self._scheduler.acquire(target_apps, 0.0, **selection)
                if ready is not None:
                    return ready
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return None
                await self._ensure_lazy_provisions(fleet_scoped=fleet_scoped)

                async with self._lazy_capacity_cond:
                    if self._lazy_capacity_generation == generation:
                        with suppress(TimeoutError):
                            await asyncio.wait_for(
                                self._lazy_capacity_cond.wait(),
                                timeout=max(deadline - time.monotonic(), 0.0),
                            )

        finally:
            self._lazy_acquire_waiters -= 1
            self._lazy_scoped_acquire_waiters -= int(fleet_scoped)

    def pool_summary(self) -> dict[str, Any]:

        sessions = self._sessions.list_sessions()
        by_status: dict[str, int] = {}
        leased = 0
        for s in sessions:
            by_status[str(s.status)] = by_status.get(str(s.status), 0) + 1
            if s.lease_id is not None:
                leased += 1
        availability = self._pool_availability(sessions)
        fleet_autoscale = self._fleet_autoscaler.device_summaries()
        if len(fleet_autoscale) == 1:
            availability.update(next(iter(fleet_autoscale.values())))
        return {
            "total": len(sessions),
            "by_status": by_status,
            "leased": leased,
            "target": sum(max(0, d.max_sessions) for d in self._devices),
            **availability,
            **({"fleet_autoscale": fleet_autoscale} if fleet_autoscale else {}),
        }

    async def _log_traj_failure(self, session_id: str, reason: str) -> None:

        try:
            state = self._sessions.get_session(session_id)
        except Exception:
            return
        if state.traj_logger is None or state.traj_scored:
            return

        state.traj_scored = True
        await log_failure(state.traj_logger, reason)

    async def _destroy_lazy_session(self, state: SessionState) -> None:

        task = self._lazy_destroy_tasks.get(state.session_id)
        if task is None:
            task = asyncio.create_task(self._destroy_lazy_session_once(state))
            self._lazy_destroy_tasks[state.session_id] = task
        try:
            await asyncio.shield(task)
        finally:
            if task.done() and self._lazy_destroy_tasks.get(state.session_id) is task:
                self._lazy_destroy_tasks.pop(state.session_id, None)

    async def _destroy_lazy_session_once(self, state: SessionState) -> None:

        session_id = state.session_id
        if state.status != SessionStatus.DESTROYING:
            try:
                self._sessions.transition(session_id, SessionStatus.DESTROYING)
            except (InvalidTransitionError, SessionNotFoundError):
                return

        await self._infra.release(state.backend.id)
        old_backend = self._backends.pop(state.backend.id, None)
        if old_backend is not None:
            with suppress(Exception):
                await old_backend.close()
        with suppress(SessionNotFoundError):
            self._sessions.remove_session(session_id)
        await self._fleet_autoscaler.detach_session(session_id)
        await self._notify_lazy_capacity()

    async def _cleanup_operation(
        self, session_id: str, lease_id: str, *, discard: bool, reason: str | None = None
    ) -> dict[str, Any]:
        key = (session_id, lease_id)
        if key in self._cleanup_receipts:
            return dict(self._cleanup_receipts[key])
        task = self._cleanup_tasks.get(key)
        if task is None:
            if len(self._cleanup_tasks) >= _CLEANUP_MAX_INFLIGHT:
                return {"ok": False, "error": "cleanup_capacity_exhausted"}

            async def run() -> dict[str, Any]:
                result = (
                    await self._discard(session_id, lease_id, reason)
                    if discard
                    else await self._release(session_id, lease_id)
                )
                if result.get("ok") is True:
                    self._cleanup_receipts[key] = dict(result)
                    while len(self._cleanup_receipts) > _CLEANUP_RECEIPT_LIMIT:
                        self._cleanup_receipts.popitem(last=False)
                return result

            task = asyncio.create_task(run())
            self._cleanup_tasks[key] = task

            def finished(done: asyncio.Task[dict[str, Any]]) -> None:
                if self._cleanup_tasks.get(key) is done:
                    self._cleanup_tasks.pop(key, None)
                if not done.cancelled():
                    done.exception()

            task.add_done_callback(finished)
        return dict(await asyncio.shield(task))

    async def release(self, session_id: str, lease_id: str) -> dict[str, Any]:
        return await self._cleanup_operation(session_id, lease_id, discard=False)

    async def _release(self, session_id: str, lease_id: str) -> dict[str, Any]:

        try:
            state = self._sessions.get_session(session_id)
        except KeyError:
            state = None

        if state is None:
            return {"ok": True, "session_gone": True}
        retry_cleanup = (
            state is not None
            and state.status == SessionStatus.DESTROYING
            and state.lease_id is None
            and state.cleanup_lease_id == lease_id
        )
        if state is not None and state.lease_id != lease_id and not retry_cleanup:
            return {"ok": False, "error": "lease_mismatch"}

        try:
            async with asyncio.timeout(10):
                await self._log_traj_failure(
                    session_id,
                    "trajectory released without evaluation (abandoned)",
                )
        except TimeoutError:
            logger.warning("Abandonment log timed out sid=%s", session_id)

        if state.sdk_session is not None and (
            state.sdk_session.dirty or self._is_lazy_pool_state(state)
        ):
            if self._is_lazy_pool_state(state):
                ok = await self._scheduler.release_and_destroy(
                    session_id,
                    lease_id,
                    reason="SDK worker requires retirement",
                )
                if ok:
                    task = asyncio.create_task(self._destroy_lazy_session(state))
                    self._recovery_tasks.add(task)
                    task.add_done_callback(self._recovery_tasks.discard)
            else:
                ok = await self._scheduler.release_and_recycle(session_id, lease_id)
                if ok:
                    state.force_relaunch = True
                    self._kick_relaunch(session_id, state.backend.id)
            return {"ok": ok, "recycling": ok}

        if state is not None and self._is_lazy_pool_state(state):
            state.cleanup_lease_id = lease_id
            ok = retry_cleanup or await self._scheduler.release_and_destroy(
                session_id,
                lease_id,
                reason="lazy pool release",
            )
            if ok:
                await self._destroy_lazy_session(state)
            return {"ok": ok, "recycling": ok, "destroyed": ok}

        backend_id: str | None = None
        state: SessionState | None = None
        ephemeral: SessionTask | None = None
        try:
            state = self._sessions.get_session(session_id)
            backend_id = state.backend.id
            ephemeral = state.session_task
        except (KeyError, SessionNotFoundError):
            state = None
            backend_id = None
        recycle = backend_id is not None and (
            ephemeral is not None
            or (state is not None and state.ephemeral_dirty)
            or self._infra.recycle_on_release(backend_id)
        )
        if recycle:
            ok = await self._scheduler.release_and_recycle(session_id, lease_id)
            if ok:
                if state is not None and (ephemeral is not None or state.ephemeral_dirty):
                    state.ephemeral_dirty = True
                    state.force_relaunch = True
                    if ephemeral is not None:
                        try:
                            client = self._get_or_create_backend(state.backend)
                            strategy = self._get_strategy(state.backend.env_type)
                            async with asyncio.timeout(10):
                                await strategy.on_detach_task(client, state, task=ephemeral)
                        except Exception as exc:
                            logger.warning("[release sid=%s] detach failed: %s", session_id, exc)
                        state.session_task = None
                self._kick_relaunch(session_id, backend_id)
            return {"ok": ok, "recycling": ok}
        ok = await self._scheduler.release(session_id, lease_id)
        return {"ok": ok}

    async def discard(
        self,
        session_id: str,
        lease_id: str,
        reason: str | None,
    ) -> dict[str, Any]:
        return await self._cleanup_operation(session_id, lease_id, discard=True, reason=reason)

    async def _discard(self, session_id: str, lease_id: str, reason: str | None) -> dict[str, Any]:
        try:
            state = self._sessions.get_session(session_id)
        except KeyError:
            state = None
        if state is None:
            return {"ok": True, "session_gone": True}
        if state.lease_id != lease_id:
            if state.status == SessionStatus.DESTROYING and state.cleanup_lease_id == lease_id:
                return await self._release(session_id, lease_id)
            return {"ok": False, "error": "lease_mismatch"}
        await self._log_traj_failure(session_id, reason or "session discarded")
        if state is not None and (
            state.ephemeral_dirty
            or state.session_task is not None
            or (state.sdk_session is not None and state.sdk_session.dirty)
        ):
            return await self._release(session_id, lease_id)
        if state is not None and self._is_lazy_pool_state(state):
            state.cleanup_lease_id = lease_id
            ok = await self._scheduler.release_and_destroy(
                session_id,
                lease_id,
                reason=reason or "lazy pool discard",
            )
            if ok:
                await self._destroy_lazy_session(state)
            return {"ok": ok, "destroyed": ok}
        ok = await self._scheduler.discard(session_id, lease_id, reason or "")
        return {"ok": ok}

    async def prewarm_all(self, devices: list[DeviceConfig] | None = None) -> None:

        requested_targets = devices if devices is not None else self._devices
        targets = [
            device for device in requested_targets if device.name not in self._lazy_pool_devices
        ]
        skipped = [
            device.name for device in requested_targets if device.name in self._lazy_pool_devices
        ]
        if skipped:
            logger.info(
                "[EnvController] prewarm skipped lazy-acquire device(s): %s",
                skipped,
            )
        wanted_by_type: dict[str, int] = {}
        for device in targets:
            if device.max_sessions <= 0:
                continue
            wanted_by_type[device.type] = wanted_by_type.get(device.type, 0) + device.max_sessions
        for env_type, wanted in wanted_by_type.items():
            chunk = min(_PREWARM_CHUNK, wanted)
            sizes = [min(chunk, wanted - i) for i in range(0, wanted, chunk)]

            sem = asyncio.Semaphore(max(1, _PREWARM_INFLIGHT // chunk))

            async def _one_chunk(
                n: int,
                *,
                etype: str = env_type,
                _sem: asyncio.Semaphore = sem,
                _wanted: int = wanted,
            ) -> int:
                async with _sem:
                    try:
                        result = await self.batch_start_env(n, etype)
                        got = int(result.get("total_count", 0))
                        logger.info(
                            "[EnvController] prewarm type=%s chunk +%d/%d",
                            etype,
                            got,
                            _wanted,
                        )
                        return got
                    except Exception as exc:
                        logger.warning(
                            "[EnvController] prewarm chunk failed for type=%s: %s",
                            etype,
                            exc,
                        )
                        return 0

            counts = await asyncio.gather(*(_one_chunk(n) for n in sizes))
            logger.info(
                "[EnvController] prewarm type=%s requested=%d ready=%d",
                env_type,
                wanted,
                sum(counts),
            )

    def _resolve_scale_device(
        self, env_type: str | None, device_name: str | None
    ) -> tuple[DeviceConfig | None, str | None]:

        devs = self._devices
        if not devs:
            return None, "no devices configured on this manager"
        if device_name:
            for d in devs:
                if d.name == device_name:
                    return d, None
            return None, f"no device named {device_name!r}"
        if env_type:
            matches = [d for d in devs if d.type == env_type]
            if len(matches) == 1:
                return matches[0], None
            if not matches:
                return None, f"no device of type {env_type!r}"
            return None, (f"{len(matches)} devices of type {env_type!r}; pass device=<name>")
        if len(devs) == 1:
            return devs[0], None
        return None, (f"{len(devs)} devices configured; pass env_type=<type> or device=<name>")

    async def scale_pool(
        self,
        target: int,
        env_type: str | None = None,
        device_name: str | None = None,
    ) -> dict[str, Any]:

        device, err = self._resolve_scale_device(env_type, device_name)
        if device is None:
            return {"ok": False, "error": err, "target": target}
        if target < 0:
            return {"ok": False, "error": "target must be >= 0", "device": device.name}

        dev_sessions = [
            s for s in self._sessions.list_sessions() if s.backend.device_name == device.name
        ]
        before = len(dev_sessions)

        if isinstance(device.launch, K8sLaunchConfig) and device.launch.lazy_acquire:
            if device.launch.fleet_autoscale.enabled:
                return await self._scale_fleet_capacity_limit(
                    device,
                    target,
                    dev_sessions,
                )
            return await self._scale_lazy_sandboxset(device, target, dev_sessions)

        warning: str | None = None

        if target > before:
            action = "growing"
            device.max_sessions = target
            if getattr(device.launch, "acquisition", None) == "adopt":
                ns = getattr(device.launch, "namespace", "?")
                warning = (
                    f"adopt-mode pool: this manager only claims already-running "
                    f"pods in namespace {ns!r}; it never cold-creates. This grow "
                    f"adds envs only if spare un-adopted pods exist there — "
                    f"otherwise it is a no-op. Create pods first (temp direct "
                    f"builder) or switch acquisition to pool/direct."
                )
                logger.warning(
                    "[scale] device=%s adopt-mode grow %d -> %d: will NOT "
                    "cold-create; claims spare pods in ns=%s only",
                    device.name,
                    before,
                    target,
                    ns,
                )
            logger.info(
                "[scale] device=%s grow %d -> %d (prewarm in background)",
                device.name,
                before,
                target,
            )

            task = asyncio.create_task(self.prewarm_all([device]))
            self._recovery_tasks.add(task)
            task.add_done_callback(self._recovery_tasks.discard)
        elif target < before:
            non_idle = [
                s for s in dev_sessions if s.lease_id is not None or s.status != SessionStatus.READY
            ]
            if non_idle:
                return {
                    "ok": False,
                    "error": (
                        f"cannot scale down: {len(non_idle)} of {before} env(s) "
                        f"busy/leased; scale down only when the pool is idle"
                    ),
                    "device": device.name,
                    "before": before,
                    "target": target,
                    "busy_or_leased": len(non_idle),
                }
            action = "drained"
            device.max_sessions = target
            logger.info(
                "[scale] device=%s shrink %d -> %d (drain idle)",
                device.name,
                before,
                target,
            )
            await self._drain_ready_sessions(device.name, before - target)
        else:
            action = "noop"
            device.max_sessions = target

        after = len(
            [s for s in self._sessions.list_sessions() if s.backend.device_name == device.name]
        )
        persisted = persist_max_sessions(self._config_path, device.name, target)
        logger.info(
            "[scale] device=%s done: %s before=%d after=%d target=%d persisted=%s",
            device.name,
            action,
            before,
            after,
            target,
            persisted,
        )
        result: dict[str, Any] = {
            "ok": True,
            "device": device.name,
            "action": action,
            "before": before,
            "after": after,
            "target": target,
            "persisted": persisted,
        }
        if warning:
            result["warning"] = warning
        return result

    async def _scale_fleet_capacity_limit(
        self,
        device: DeviceConfig,
        target: int,
        dev_sessions: list[SessionState],
    ) -> dict[str, Any]:

        assert isinstance(device.launch, K8sLaunchConfig)
        usage = await self._fleet_autoscaler.capacity_usage(device.name)
        requested = usage["reserved"]
        unreserved = usage["unreserved"]
        legacy_warm = usage["legacy_warm"]
        pending = sum(not task.done() for task in self._lazy_provision_tasks)
        minimum = max(
            usage["required"],
            device.launch.fleet_autoscale.min_replicas,
        )
        if target < minimum:
            return {
                "ok": False,
                "error": (
                    f"capacity limit {target} is below currently required "
                    f"capacity {minimum} (requested={requested}, "
                    f"legacy_warm={legacy_warm}, unreserved={unreserved}, "
                    f"registered={len(dev_sessions)}, "
                    f"pending={pending})"
                ),
                "device": device.name,
                "target": target,
                "capacity_limit": device.max_sessions,
                "requested_capacity": requested,
                "legacy_warm_replicas": legacy_warm,
                "unreserved_active_capacity": unreserved,
                "registered": len(dev_sessions),
                "pending_provisions": pending,
            }

        before = device.max_sessions
        device.max_sessions = target
        persisted = persist_max_sessions(self._config_path, device.name, target)
        self._fleet_autoscaler.notify_config_changed()
        action = (
            "ceiling_growing"
            if target > before
            else "ceiling_shrinking"
            if target < before
            else "noop"
        )
        logger.info(
            "[scale] dynamic fleet device=%s %s limit=%d->%d "
            "requested=%d registered=%d pending=%d persisted=%s",
            device.name,
            action,
            before,
            target,
            requested,
            len(dev_sessions),
            pending,
            persisted,
        )
        return {
            "ok": True,
            "device": device.name,
            "action": action,
            "before": before,
            "after": target,
            "target": target,
            "capacity_limit": target,
            "requested_capacity": requested,
            "legacy_warm_replicas": legacy_warm,
            "unreserved_active_capacity": unreserved,
            "registered": len(dev_sessions),
            "pending_provisions": pending,
            "persisted": persisted,
        }

    async def _scale_lazy_sandboxset(
        self,
        device: DeviceConfig,
        target: int,
        dev_sessions: list[SessionState],
    ) -> dict[str, Any]:

        assert isinstance(device.launch, K8sLaunchConfig)
        if target < 1:
            return {
                "ok": False,
                "error": "lazy SandboxSet target must be >= 1",
                "device": device.name,
                "target": target,
            }

        old_target = device.max_sessions
        pending = sum(not task.done() for task in self._lazy_provision_tasks)
        reserved = len(dev_sessions) + pending
        if target < reserved:
            return {
                "ok": False,
                "error": (
                    f"cannot scale lazy SandboxSet below {reserved} active/pending "
                    f"session(s); wait for claims to drain"
                ),
                "device": device.name,
                "before": old_target,
                "target": target,
                "registered": len(dev_sessions),
                "pending_provisions": pending,
                "active_or_pending": reserved,
            }

        if target > old_target:
            action = "growing"
        elif target < old_target:
            action = "shrinking"

            device.max_sessions = target
        else:
            action = "reconciling"

        try:
            sandboxset_action = await self._infra.resize_sandbox_pool(
                device.name,
                target,
            )
        except Exception as exc:
            device.max_sessions = old_target
            logger.exception(
                "[scale] lazy SandboxSet resize failed device=%s target=%d",
                device.name,
                target,
            )
            return {
                "ok": False,
                "error": f"SandboxSet resize failed: {type(exc).__name__}: {exc}",
                "device": device.name,
                "before": old_target,
                "target": target,
            }

        device.max_sessions = target
        device.launch.pool_size = target
        persisted = persist_scale_settings(
            self._config_path,
            device.name,
            target,
            pool_size=target,
            num_envs=target,
        )
        logger.info(
            "[scale] lazy SandboxSet device=%s action=%s %d -> %d "
            "registered=%d pending=%d sandboxset=%s persisted=%s",
            device.name,
            action,
            old_target,
            target,
            len(dev_sessions),
            pending,
            sandboxset_action,
            persisted,
        )
        return {
            "ok": True,
            "device": device.name,
            "action": action,
            "before": old_target,
            "after": target,
            "target": target,
            "registered": len(dev_sessions),
            "pending_provisions": pending,
            "sandboxset_replicas": target,
            "sandboxset_action": sandboxset_action,
            "persisted": persisted,
        }

    async def _drain_ready_sessions(self, device_name: str, n: int) -> int:

        if n <= 0:
            return 0
        drained = 0
        for s in self._sessions.list_sessions(SessionStatus.READY):
            if drained >= n:
                break
            if s.backend.device_name != device_name or s.lease_id is not None:
                continue
            sid = s.session_id
            try:
                self._sessions.transition(sid, SessionStatus.DESTROYING)
            except (InvalidTransitionError, SessionNotFoundError):
                continue
            client = self._backends.get(s.backend.id)
            if isinstance(client, PooledBackend) and s.context_id is not None:
                try:
                    await client.release_context(s.context_id)
                    s.context_id = None
                except Exception as exc:
                    logger.warning("[scale] context release failed sid=%s: %s", sid, exc)
            try:
                await self._infra.release(s.backend.id)
            except Exception as exc:
                logger.warning("[scale] drain release failed sid=%s: %s", sid, exc)
            with suppress(SessionNotFoundError):
                self._sessions.remove_session(sid)
            if not any(
                other.backend.id == s.backend.id for other in self._sessions.list_sessions()
            ):
                old = self._backends.pop(s.backend.id, None)
                if old is not None:
                    with suppress(Exception):
                        await old.close()
            drained += 1
        logger.info("[scale] drained %d/%d idle session(s) on %s", drained, n, device_name)
        return drained

    async def run_lease_sweep(self) -> None:

        while True:
            try:
                await asyncio.sleep(_LEASE_SWEEP_INTERVAL_S)
                _, destroy_ids = await self._scheduler.sweep_expired_details(
                    lambda state: (
                        self._is_lazy_pool_state(state)
                        or state.ephemeral_dirty
                        or state.session_task is not None
                        or (state.sdk_session is not None and state.sdk_session.dirty)
                    ),
                )
                for session_id in destroy_ids:
                    try:
                        state = self._sessions.get_session(session_id)
                    except KeyError:
                        continue
                    try:
                        if self._is_lazy_pool_state(state):
                            await self._destroy_lazy_session(state)
                        else:
                            self._sessions.transition(session_id, SessionStatus.INITIALIZING)
                            self._sessions.transition(session_id, SessionStatus.INVALID)
                            state.force_relaunch = True
                            self._kick_relaunch(session_id, state.backend.id)
                    except Exception:
                        logger.exception("Lease sweep cleanup failed sid=%s", session_id)

                for state in self._sessions.list_sessions(SessionStatus.DESTROYING):
                    if self._is_lazy_pool_state(state) and state.session_id not in destroy_ids:
                        try:
                            await self._destroy_lazy_session(state)
                        except Exception:
                            logger.exception(
                                "Lease sweep cleanup retry failed sid=%s",
                                state.session_id,
                            )
                self._reap_stuck_busy()
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("[EnvController] lease sweep iteration failed")

    def _reap_stuck_busy(self) -> None:

        if _BUSY_REAP_AFTER_S <= 0:
            return
        for state in self._sessions.reap_stuck_busy(time.monotonic(), _BUSY_REAP_AFTER_S):
            self._maybe_force_relaunch(state)

    def _maybe_force_relaunch(self, state: SessionState) -> None:

        if self._is_lazy_pool_state(state) or (state.ephemeral_dirty and state.lease_id):
            return
        backend_id = state.backend.id
        if backend_id in self._relaunching:
            return
        recovery = self._infra.device_recovery(backend_id)
        if recovery is None or recovery.strategy != "relaunch" or not recovery.auto_relaunch:
            logger.info(
                "[EnvController] session %s needs relaunch but device recovery "
                "is %s — leaving INVALID for the recovery sweep",
                state.session_id,
                recovery.strategy if recovery else "unknown",
            )
            return
        self._kick_relaunch(state.session_id, backend_id)

    async def run_pool_snapshot(self) -> None:

        while True:
            try:
                await asyncio.sleep(_POOL_SNAPSHOT_INTERVAL_S)
                sessions = self._sessions.list_sessions()
                by_status: dict[str, int] = {}
                leased = 0
                total_steps = 0
                for s in sessions:
                    by_status[str(s.status)] = by_status.get(str(s.status), 0) + 1
                    total_steps += s.step_count
                    if s.lease_id is not None:
                        leased += 1
                logger.info(
                    "[EnvController] pool: total=%d by_status=%s leased=%d "
                    "active_per_device=%s aggregate_steps=%d",
                    len(sessions),
                    dict(sorted(by_status.items())),
                    leased,
                    self._infra.active_counts_snapshot(),
                    total_steps,
                )
                if self._failure_counters:
                    top = sorted(
                        self._failure_counters.items(),
                        key=lambda kv: -kv[1],
                    )[:20]
                    fail_breakdown = {f"{phase}/{dev}/{kind}": n for (phase, dev, kind), n in top}
                    logger.info(
                        "[EnvController] backend-fail-counters total=%d top=%s",
                        sum(self._failure_counters.values()),
                        fail_breakdown,
                    )
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("[EnvController] pool snapshot iteration failed")

    async def run_recovery_sweep(self) -> None:

        while True:
            try:
                await asyncio.sleep(_RECOVERY_SWEEP_INTERVAL_S)
                await self._recover_invalid_sessions()
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("[EnvController] recovery sweep iteration failed")

    async def _recover_invalid_sessions(self) -> None:

        invalid = self._sessions.list_sessions(SessionStatus.INVALID)

        forced = [s for s in invalid if s.force_relaunch and s.backend.id not in self._relaunching]
        for state in forced:
            self._maybe_force_relaunch(state)
        relaunching = self._relaunching
        invalid = [s for s in invalid if not (s.force_relaunch and s.backend.id in relaunching)]
        if not invalid:
            return

        lazy_unleased = [
            state for state in invalid if self._is_lazy_pool_state(state) and state.lease_id is None
        ]
        for state in lazy_unleased:
            await self._destroy_lazy_session(state)
        invalid = [state for state in invalid if not self._is_lazy_pool_state(state)]

        invalid = [
            state
            for state in invalid
            if state.backend.id not in self._relaunching
            and not state.ephemeral_dirty
            and state.session_task is None
            and not (state.sdk_session is not None and state.sdk_session.dirty)
        ]
        if not invalid:
            return
        sem = asyncio.Semaphore(_BATCH_INIT_CONCURRENCY)

        async def _probe(state: Any) -> tuple[Any, bool]:
            async with sem:
                try:
                    client = self._get_or_create_backend(state.backend)
                    return state, await client.health_check(
                        timeout=_RECOVERY_HEALTH_TIMEOUT_S,
                    )
                except Exception:
                    return state, False

        results = await asyncio.gather(*(_probe(s) for s in invalid))

        healthy_ids = [s.session_id for s, ok in results if ok]
        if healthy_ids:
            await self._scheduler.recover_sessions(healthy_ids)

        for state, ok in results:
            if ok:
                continue
            backend_id = state.backend.id
            if backend_id in self._relaunching:
                continue
            recovery = self._infra.device_recovery(backend_id)
            if recovery is None or recovery.strategy != "relaunch" or not recovery.auto_relaunch:
                logger.info(
                    "[EnvController] session %s INVALID, backend %s unhealthy; "
                    "recovery strategy=%s — leaving INVALID",
                    state.session_id,
                    backend_id,
                    recovery.strategy if recovery else "unknown",
                )
                continue
            self._kick_relaunch(state.session_id, backend_id)

    def _kick_relaunch(self, session_id: str, backend_id: str) -> None:

        if backend_id in self._relaunching:
            return
        self._relaunching.add(backend_id)
        task = asyncio.create_task(self._relaunch_and_recover(session_id, backend_id))
        self._recovery_tasks.add(task)
        task.add_done_callback(self._recovery_tasks.discard)

    async def _relaunch_and_recover(self, session_id: str, backend_id: str) -> None:

        try:
            ok = await asyncio.to_thread(self._infra.relaunch_backend, backend_id)
            if ok:
                try:
                    state = self._sessions.get_session(session_id)
                except KeyError:
                    state = None
                if state is not None:
                    stale = self._backends.pop(state.backend.id, None)
                    if stale is not None:
                        await stale.close()
                    state.ephemeral_dirty = False
                    state.sdk_session = None
                    state.session_task = None
                    state.context_id = None
                    state.verifier_reset_ops.clear()

                    if self._infra.registered_set_ids():
                        try:
                            await self._reapply_task_sets([state.backend])
                        except Exception:
                            logger.exception(
                                "[EnvController] re-apply task sets after relaunch "
                                "of backend %s failed",
                                backend_id,
                            )
                await self._scheduler.recover_sessions([session_id])
            else:
                logger.warning(
                    "[EnvController] relaunch of backend %s failed; session %s stays INVALID",
                    backend_id,
                    session_id,
                )
        except Exception:
            logger.exception("[EnvController] relaunch task for backend %s failed", backend_id)
        finally:
            self._relaunching.discard(backend_id)

    @property
    def traj_log_dir(self) -> str | None:

        return self._traj_log_dir

    @property
    def screenshot_transport_stats(self) -> dict[str, int | float]:
        successes = self._screenshot_upload_successes
        return {
            "upload_successes": successes,
            "inline_fallbacks": self._screenshot_upload_inline_fallbacks,
            "inline_deliveries": self._screenshot_inline_deliveries,
            "upload_bytes": self._screenshot_upload_bytes,
            "upload_seconds_total": self._screenshot_upload_seconds,
            "upload_ms_mean": (
                self._screenshot_upload_seconds * 1000 / successes if successes else 0.0
            ),
        }

    def _resolve_traj_target(
        self,
        model_name: str | None,
    ) -> tuple[str | None, str | None]:

        if model_name:
            req_slug = model_slug(model_name)
            base = self._debug_log_dir
            root = f"{base.rstrip('/')}/{req_slug}" if base else base
            return root, req_slug
        return self._traj_log_dir, self._model_slug

    def list_sessions(self) -> dict[str, Any]:

        sessions = self._sessions.list_sessions()
        now = time.time()
        leased = 0
        by_device: dict[str, dict[str, int]] = {}
        session_list = []

        for s in sessions:
            device = s.backend.device_name
            if device not in by_device:
                by_device[device] = {
                    "total": 0,
                    "leased": 0,
                    "ready": 0,
                    "busy": 0,
                    "invalid": 0,
                }
            by_device[device]["total"] += 1
            if s.lease_id is not None:
                leased += 1
                by_device[device]["leased"] += 1
            if s.status == SessionStatus.READY and s.lease_id is None:
                by_device[device]["ready"] += 1
            if s.status == SessionStatus.BUSY:
                by_device[device]["busy"] += 1
            if s.status == SessionStatus.INVALID:
                by_device[device]["invalid"] += 1

            lease_duration_m = None
            if s.leased_since is not None:
                lease_duration_m = round((now - s.leased_since) / 60, 1)

            session_list.append(
                {
                    "session_id": s.session_id,
                    "status": str(s.status),
                    "backend_idx": s.backend.device_name,
                    "env_type": s.backend.env_type,
                    "infra_type": s.backend.metadata.get("infra_type"),
                    "device_type": s.backend.metadata.get("device_type"),
                    "environment_profile": s.backend.metadata.get("environment_profile"),
                    "leased": s.lease_id is not None,
                    "step_count": s.step_count,
                    "current_task": s.current_task,
                    "train_id": s.train_id,
                    "traj_name": s.traj_name,
                    "created_at": s.created_at,
                    "leased_since": s.leased_since,
                    "lease_duration_minutes": lease_duration_m,
                }
            )

        availability = self._pool_availability(sessions)
        fleet_autoscale = self._fleet_autoscaler.device_summaries()
        if len(fleet_autoscale) == 1:
            availability.update(next(iter(fleet_autoscale.values())))
        return {
            "summary": {
                "total": len(sessions),
                "leased": leased,
                "target": sum(max(0, d.max_sessions) for d in self._devices),
                **availability,
            },
            "by_device": by_device,
            "sessions": session_list,
            **({"fleet_autoscale": fleet_autoscale} if fleet_autoscale else {}),
        }

    async def admin_force_release(self, session_id: str) -> dict[str, Any]:

        destroy_state: SessionState | None = None
        async with self._scheduler._cond:
            had_lease = self._sessions.force_clear_lease(session_id)
            state = self._sessions.get_session(session_id)
            if self._is_lazy_pool_state(state):
                try:
                    self._sessions.transition(session_id, SessionStatus.DESTROYING)
                    destroy_state = state
                except Exception as exc:
                    logger.warning("[Admin] lazy force-release transition: %s", exc)
            elif state.sdk_session is not None and state.sdk_session.dirty:
                if state.status != SessionStatus.INVALID:
                    self._sessions.transition(session_id, SessionStatus.INVALID)
                state.force_relaunch = True
                self._kick_relaunch(session_id, state.backend.id)
            elif state.status == SessionStatus.BUSY:
                try:
                    self._sessions.transition(session_id, SessionStatus.READY)
                except Exception as exc:
                    logger.warning("[Admin] force-release transition: %s", exc)
            if had_lease and destroy_state is None:
                self._scheduler._cond.notify(1)
        if destroy_state is not None:
            await self._destroy_lazy_session(destroy_state)
        logger.info("[Admin] force-released session %s (had_lease=%s)", session_id, had_lease)
        return {
            "ok": True,
            "session_id": session_id,
            "action": "force_release",
            "destroyed": destroy_state is not None,
        }

    async def admin_force_invalidate(self, session_id: str) -> dict[str, Any]:

        destroy_state: SessionState | None = None
        async with self._scheduler._cond:
            self._sessions.force_clear_lease(session_id)
            state = self._sessions.get_session(session_id)
            try:
                target = (
                    SessionStatus.DESTROYING
                    if self._is_lazy_pool_state(state)
                    else SessionStatus.INVALID
                )
                self._sessions.transition(session_id, target)
                if target == SessionStatus.DESTROYING:
                    destroy_state = state
            except Exception as exc:
                logger.warning("[Admin] force-invalidate transition: %s", exc)
            if destroy_state is None:
                self._scheduler._cond.notify(1)
        if destroy_state is not None:
            await self._destroy_lazy_session(destroy_state)
        logger.info("[Admin] force-invalidated session %s", session_id)
        return {
            "ok": True,
            "session_id": session_id,
            "action": "force_invalidate",
            "destroyed": destroy_state is not None,
        }

    async def _encode_screenshot(self, b64: str) -> str:

        if not b64:
            return b64
        from mai_gym.utils.images import sniff_format

        codec = self._screenshot_codec
        target = "jpeg" if codec in ("jpg", "jpeg") else codec
        if sniff_format(b64) == target:
            return b64
        out = await asyncio.to_thread(
            _encode_screenshot_sync,
            b64,
            codec,
            self._screenshot_quality,
        )
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                "[screenshot] codec=%s in=%dB out=%dB ratio=%.2f",
                codec,
                len(b64),
                len(out),
                len(out) / max(1, len(b64)),
            )
        return out

    async def _maybe_upload_screenshot(
        self,
        b64: str,
        session_id: str,
        step: int,
        screenshot_transport: str = "inline",
    ) -> dict[str, str]:

        if screenshot_transport != "oss":
            self._screenshot_inline_deliveries += 1
            return {"image": b64}
        if self._screenshot_transport != "oss" or not b64 or self._oss_cfg is None:
            self._screenshot_upload_inline_fallbacks += 1
            return {"image": b64}
        started = time.perf_counter()
        try:
            from mai_gym.io.oss_io import get_async_fs, join, write_bytes
            from mai_gym.utils.images import sniff_format

            fmt = sniff_format(b64)
            ext = "jpg" if fmt == "jpeg" else fmt
            key = join(self._screenshot_oss_prefix, session_id, f"{step:04d}.{ext}")

            data = await asyncio.to_thread(base64.b64decode, b64)
            await write_bytes(key, data, fs=get_async_fs(self._oss_cfg))
            self._screenshot_upload_successes += 1
            self._screenshot_upload_bytes += len(data)
            self._screenshot_upload_seconds += time.perf_counter() - started
            return {"image_uri": key}
        except Exception as exc:
            self._screenshot_upload_inline_fallbacks += 1
            logger.warning(
                "[screenshot] OSS upload failed (session=%s step=%d); inline fallback: %s",
                session_id,
                step,
                exc,
            )
            return {"image": b64}

    async def _ensure_world_cached(
        self,
        oss_uri: str,
        sha256: str | None,
    ) -> str:

        if self._oss_cfg is None:
            raise RuntimeError(
                f"world archive {oss_uri!r} requested but no OSS config is set on "
                "this controller (manager.oss)"
            )
        key = sha256 or hashlib.sha256(oss_uri.encode()).hexdigest()
        dest = os.path.join(self._world_cache_dir, f"{key}.tar.gz")
        lock = self._world_cache_locks.setdefault(key, asyncio.Lock())
        async with lock:
            if os.path.isfile(dest) and os.path.getsize(dest) > 0:
                return dest
            from mai_gym.io import oss_io

            data = await oss_io.read_bytes(
                oss_uri,
                fs=oss_io.get_async_fs(self._oss_cfg),
            )
            if sha256:
                got = hashlib.sha256(data).hexdigest()
                if got != sha256:
                    raise RuntimeError(
                        f"world archive {oss_uri!r} sha256 mismatch: expected {sha256}, got {got}"
                    )
            os.makedirs(self._world_cache_dir, exist_ok=True)

            tmp = f"{dest}.{os.getpid()}.part"

            def _write() -> None:
                with open(tmp, "wb") as fp:
                    fp.write(data)

            await asyncio.to_thread(_write)
            os.replace(tmp, dest)
            logger.info(
                "[world-cache] fetched %s (%d bytes) -> %s",
                oss_uri,
                len(data),
                dest,
            )
            return dest

    async def attach_session_task(
        self,
        session_id: str,
        *,
        env_type: str,
        lease_id: str | None = None,
        set_id: str | None = None,
        task_class: str | None = None,
        golden_class: str | None = None,
        package_b64: str | None = None,
        initial_setup: str | None = None,
        golden_patch: str | None = None,
        reward: str | None = None,
        world_oss_uri: str | None = None,
        world_sha256: str | None = None,
        world_extract_to: str | None = None,
    ) -> dict[str, Any]:

        self.sdk_operations.guard(session_id)
        state = self._sessions.get_session(session_id)
        self._sessions.check_lease(session_id, lease_id)
        self._scheduler.extend_lease(session_id)

        task = SessionTask(
            kind=env_type,
            set_id=set_id,
            task_class=task_class,
            golden_class=golden_class,
            initial_setup=initial_setup,
            golden_patch=golden_patch,
            reward=reward,
            world_oss_uri=world_oss_uri,
            world_sha256=world_sha256,
            world_extract_to=world_extract_to,
        )

        package = base64.b64decode(package_b64) if package_b64 else None

        world_path = (
            await self._ensure_world_cached(world_oss_uri, world_sha256) if world_oss_uri else None
        )
        client = self._get_or_create_backend(state.backend)
        strategy = self._get_strategy(state.backend.env_type)

        epoch = state.epoch
        self._sessions.transition(session_id, SessionStatus.BUSY)

        state.ephemeral_dirty = True
        try:
            deadline = _MW_ATTACH_DEADLINE_S if task.kind == "mobileworld" else None
            async with asyncio.timeout(deadline):
                await strategy.on_attach_task(
                    client,
                    state,
                    task=task,
                    package=package,
                    world_path=world_path,
                )
        except BaseException as exc:
            if state.epoch == epoch:
                with suppress(InvalidTransitionError, SessionNotFoundError):
                    self._sessions.transition(session_id, SessionStatus.INVALID)
                state.force_relaunch = True
                state.last_op_failure_at = time.monotonic()
                is_backend_fail, kind = _classify_backend_failure(exc)
                if is_backend_fail:
                    self._record_backend_failure("attach", state, exc, kind)
                self._maybe_force_relaunch(state)
            raise
        if state.epoch != epoch or state.status != SessionStatus.BUSY:
            raise InvalidTransitionError(session_id, state.status, SessionStatus.READY)
        self._sessions.transition(session_id, SessionStatus.READY)
        state.session_task = task
        state.last_op_failure_at = None
        logger.info(
            "[SessionTask sid=%s] attached kind=%s set_id=%s (backend=%s)",
            session_id,
            task.kind,
            task.set_id,
            state.backend.id,
        )
        return {"ok": True, "kind": task.kind}

    async def reset(
        self,
        session_id: str,
        category: str,
        task_id: str | None,
        goal: str | None = None,
        target_apps: list[str] | None = None,
        seed_task_id: str | None = None,
        attempt_id: str | None = None,
        train_id: str | None = None,
        rollout_id: int | None = None,
        evaluation: bool = False,
        screenshot_width: int | None = None,
        screenshot_height: int | None = None,
        lease_id: str | None = None,
        model_name: str | None = None,
        action_id: str | None = None,
        screenshot_transport: str = "inline",
        task_phase: str | None = None,
    ) -> dict[str, Any]:
        self.sdk_operations.guard(session_id)
        state = self._sessions.get_session(session_id)
        self._sessions.check_lease(session_id, lease_id)
        self._scheduler.extend_lease(session_id)

        if (
            action_id is not None
            and action_id == state.last_reset_id
            and state.last_reset_result is not None
        ):
            logger.info("[Reset sid=%s] idempotent replay action_id=%s", session_id, action_id)
            return state.last_reset_result
        return await self._run_reset(
            session_id,
            category,
            task_id,
            goal=goal,
            target_apps=target_apps,
            seed_task_id=seed_task_id,
            attempt_id=attempt_id,
            train_id=train_id,
            rollout_id=rollout_id,
            evaluation=evaluation,
            screenshot_width=screenshot_width,
            screenshot_height=screenshot_height,
            model_name=model_name,
            action_id=action_id,
            screenshot_transport=screenshot_transport,
            task_phase=task_phase,
            already_busy=False,
        )

    async def _run_reset(self, session_id: str, *args: Any, **kwargs: Any) -> dict[str, Any]:

        state = self._sessions.get_session(session_id)
        ephemeral_mw = state.session_task is not None and state.session_task.kind == "mobileworld"
        if not ephemeral_mw:
            return await self._run_reset_body(session_id, *args, **kwargs)
        epoch = state.epoch
        try:
            async with asyncio.timeout(_MW_RESET_DEADLINE_S):
                return await self._run_reset_body(session_id, *args, **kwargs)
        except BaseException as exc:
            if state.epoch == epoch:
                with suppress(InvalidTransitionError, SessionNotFoundError):
                    self._sessions.transition(session_id, SessionStatus.INVALID)
            if state.status == SessionStatus.INVALID and state.epoch == epoch + 1:
                state.force_relaunch = True
                state.last_op_failure_at = time.monotonic()
            if isinstance(exc, asyncio.CancelledError):
                raise
            detail = operation_error_detail(exc, "reset", operation_id=kwargs.get("action_id"))
            if isinstance(exc, TimeoutError):
                detail.update(code="operation_deadline_exceeded", stage="reset_deadline")
            status = detail.get("upstream_status", 504 if isinstance(exc, TimeoutError) else 500)
            message = detail.get("message") or detail.get("upstream_detail") or str(exc)
            raise OperationError(str(message), error_detail=detail, status_code=status) from exc

    async def _run_reset_body(
        self,
        session_id: str,
        category: str,
        task_id: str | None,
        goal: str | None = None,
        target_apps: list[str] | None = None,
        seed_task_id: str | None = None,
        attempt_id: str | None = None,
        train_id: str | None = None,
        rollout_id: int | None = None,
        evaluation: bool = False,
        screenshot_width: int | None = None,
        screenshot_height: int | None = None,
        model_name: str | None = None,
        action_id: str | None = None,
        screenshot_transport: str = "inline",
        task_phase: str | None = None,
        already_busy: bool = False,
    ) -> dict[str, Any]:

        state = self._sessions.get_session(session_id)

        transitioned_busy = already_busy

        epoch = state.epoch
        logger.info(
            "[Reset sid=%s] category=%s task_id=%s attempt_id=%s target_apps=%s prior_steps=%d",
            session_id,
            category,
            task_id,
            attempt_id,
            target_apps or [],
            state.step_count,
        )
        try:
            client = self._get_or_create_backend(state.backend)

            if isinstance(client, PooledBackend) and (state.context_id is None or state.step_count):
                if state.context_id is not None:
                    await client.release_context(state.context_id)
                    state.context_id = None
                state.context_id = await client.acquire_context()

            state.current_task = task_id
            state.train_id = train_id
            state.step_count = 0
            state.action_history.clear()
            state.traj_logger = None
            state.traj_name = None
            state.traj_scored = False

            state.last_action_id = None
            state.last_step_result = None

            traj_root, traj_model = self._resolve_traj_target(model_name)
            if traj_root:
                state.traj_logger, state.traj_name = await create_traj_logger(
                    traj_root,
                    task_id=task_id,
                    env_type=state.backend.env_type,
                    session_id=session_id,
                    attempt_id=attempt_id,
                    oss_cfg=self._oss_cfg,
                    train_id=train_id,
                    rollout_id=rollout_id,
                    evaluation=evaluation,
                    model=traj_model,
                )

            strategy = self._get_strategy(state.backend.env_type)

            sem = self._reset_semaphores.get(state.backend.device_name)
            throttle = sem if sem is not None else nullcontext()
            async with throttle:
                if not transitioned_busy:
                    self._sessions.transition(session_id, SessionStatus.BUSY)
                    transitioned_busy = True
                await self._prepare_registered_task_artifacts(
                    client,
                    state,
                    task_id,
                )
                resolved_goal = (
                    await strategy.on_reset(
                        client,
                        state,
                        task_id=task_id,
                        goal=goal,
                        target_apps=target_apps,
                        seed_task_id=seed_task_id,
                        task_phase=task_phase,
                    )
                    or goal
                )

            state.task_goal = resolved_goal or task_id or category

            if target_apps and isinstance(client, BrowserBackend) and state.context_id is not None:
                await client.prepare_apps(state.context_id, target_apps)

            b64_image = await client.screenshot(state.context_id)
            if screenshot_width and screenshot_height:
                b64_image = await asyncio.to_thread(
                    _resize_screenshot,
                    b64_image,
                    screenshot_width,
                    screenshot_height,
                )
            b64_image = await self._encode_screenshot(b64_image)
            state._last_screenshot_b64 = b64_image
            instruction = state.task_goal or task_id or category

            if state.epoch != epoch:
                raise InvalidTransitionError(session_id, state.status, SessionStatus.READY)
            image_payload = await self._maybe_upload_screenshot(
                b64_image,
                session_id,
                0,
                screenshot_transport,
            )
            result = {
                "env_idx": state.backend.device_name,
                "obs_messages": [{"instruction": instruction, **image_payload}],
                "is_done": False,
                "format_reward": 0.0,
            }
            if state.epoch != epoch or state.status != SessionStatus.BUSY:
                raise InvalidTransitionError(session_id, state.status, SessionStatus.READY)
            self._sessions.transition(session_id, SessionStatus.READY)
            state.last_op_failure_at = None

            if action_id is not None:
                state.last_reset_id = action_id
                state.last_reset_result = result
            return result
        except Exception as exc:
            if isinstance(exc, TaskArtifactPreparationError) and self._infra.recycle_on_release(
                state.backend.id
            ):
                await self._log_traj_failure(session_id, str(exc))
                if self._is_lazy_pool_state(state):
                    await self._destroy_lazy_session(state)
                else:
                    with suppress(InvalidTransitionError, SessionNotFoundError):
                        self._sessions.transition(session_id, SessionStatus.INVALID)
                    self._kick_relaunch(session_id, state.backend.id)
                raise
            is_backend_fail, kind = _classify_backend_failure(exc)
            target = SessionStatus.INVALID if is_backend_fail else SessionStatus.READY
            try:
                if state.epoch == epoch and (transitioned_busy or target == SessionStatus.INVALID):
                    self._sessions.transition(session_id, target)
            except Exception as cleanup_exc:
                logger.warning(
                    "[EnvController] reset cleanup transition failed: %s",
                    cleanup_exc,
                )
            if is_backend_fail:
                self._record_backend_failure("reset", state, exc, kind)

                state.last_op_failure_at = time.monotonic()
            raise

    async def step(
        self,
        session_id: str,
        action: dict[str, Any],
        screenshot_width: int | None = None,
        screenshot_height: int | None = None,
        lease_id: str | None = None,
        action_id: str | None = None,
        screenshot_transport: str = "inline",
    ) -> dict[str, Any]:
        self.sdk_operations.guard(session_id)
        state = self._sessions.get_session(session_id)
        self._sessions.check_lease(session_id, lease_id)

        if (
            action_id is not None
            and action_id == state.last_action_id
            and state.last_step_result is not None
        ):
            logger.info(
                "[Step sid=%s] idempotent replay of action_id=%s — returning "
                "cached result (action NOT re-applied)",
                session_id,
                action_id,
            )
            return state.last_step_result
        self._scheduler.extend_lease(session_id)
        return await self._run_step(
            session_id,
            action,
            screenshot_width=screenshot_width,
            screenshot_height=screenshot_height,
            action_id=action_id,
            screenshot_transport=screenshot_transport,
            already_busy=False,
        )

    async def _recheck_landing(
        self,
        client: Any,
        session_id: str,
        result: dict[str, Any],
    ) -> None:

        expected = str(result.get("expected_activity") or "")
        if result.get("landed") or not expected:
            return
        device_state = getattr(client, "device_state", None)
        if device_state is None:
            return
        await asyncio.sleep(_DEEPLINK_RECHECK_WAIT)
        actual = str((await device_state()).get("current_activity") or "")
        if not actual:
            return
        pkg = expected.split("/")[0]
        activity = expected.split("/")[-1].rsplit(".", 1)[-1]
        if (pkg and pkg in actual) or (activity and activity in actual):
            result["landed"] = True
            result["actual_activity"] = actual
            logger.info(
                "[Step sid=%s] deeplink landed on re-read: %s",
                session_id,
                actual,
            )

    async def _capture_step_screenshot(
        self,
        client: Backend,
        display_id: Any,
        session_id: str,
    ) -> str:

        last_exc: BaseException | None = None
        for attempt in range(_STEP_SCREENSHOT_RETRIES + 1):
            try:
                return await client.screenshot(display_id)
            except Exception as exc:
                if not _is_transient_capture_error(exc):
                    raise
                last_exc = exc
                if attempt < _STEP_SCREENSHOT_RETRIES:
                    logger.info(
                        "[Step sid=%s] transient screenshot failure (%s) — "
                        "retry %d/%d (not invalidating)",
                        session_id,
                        type(exc).__name__,
                        attempt + 1,
                        _STEP_SCREENSHOT_RETRIES,
                    )
                    await asyncio.sleep(_STEP_SCREENSHOT_RETRY_BACKOFF_S * (attempt + 1))
        assert last_exc is not None
        raise last_exc

    async def _run_step(
        self,
        session_id: str,
        action: dict[str, Any],
        screenshot_width: int | None = None,
        screenshot_height: int | None = None,
        action_id: str | None = None,
        screenshot_transport: str = "inline",
        already_busy: bool = False,
    ) -> dict[str, Any]:

        state = self._sessions.get_session(session_id)
        if not already_busy:
            self._sessions.transition(session_id, SessionStatus.BUSY)
        pre_action_b64 = state._last_screenshot_b64
        try:
            client = self._get_or_create_backend(state.backend)

            reasoning_content = action.pop("reasoning_content", None)
            mw_action = {k: v for k, v in action.items() if v is not None}
            action_type = mw_action.get("action_type", "")

            if action_type in ("DONE", "FAIL"):
                mw_action = {
                    **mw_action,
                    "action_type": "terminate",
                    "status": action_type.lower(),
                }
                action_type = "terminate"

            if action_type == "wait":
                state.action_history.append("WAIT")
            elif action_type == "terminate" or action_type == "finished":
                status = (action.get("status") or "").lower()
                state.action_history.append("DONE" if status == "success" else "FAIL")
            else:
                state.action_history.append(action_type or "UNKNOWN")

            is_done = action_type in _TERMINAL_ACTIONS

            record_terminal = getattr(type(client), "record_terminal", None)
            record_episode_terminal = getattr(type(client), "record_episode_terminal", None)
            if is_done and action_type != "answer":
                if callable(record_episode_terminal):
                    await record_episode_terminal(client, mw_action, state.context_id)
                elif callable(record_terminal):
                    await record_terminal(client, mw_action)

            step_result: dict | None = None
            if action_type == "answer" or not is_done:
                execute_step = getattr(type(client), "execute_step", None)
                if callable(execute_step):
                    step_result = await execute_step(client, mw_action, state.context_id, action_id)
                    if isinstance(step_result, dict):
                        is_done = is_done or bool(step_result.get("done"))
                else:
                    step_result = await client.do_step(mw_action, state.context_id)
                if not is_done and self._post_action_wait > 0:
                    await asyncio.sleep(self._post_action_wait)

            b64_image = await self._capture_step_screenshot(
                client,
                state.context_id,
                session_id,
            )
            if screenshot_width and screenshot_height:
                b64_image = await asyncio.to_thread(
                    _resize_screenshot,
                    b64_image,
                    screenshot_width,
                    screenshot_height,
                )
            b64_image = await self._encode_screenshot(b64_image)
            state._last_screenshot_b64 = b64_image

            state.step_count += 1
            logger.debug(
                "[Step sid=%s] step=%d action_type=%s is_done=%s",
                session_id,
                state.step_count,
                action_type,
                is_done,
            )
            if is_done or state.step_count % 10 == 0:
                logger.info(
                    "[Step sid=%s] step=%d action_type=%s is_done=%s task_id=%s",
                    session_id,
                    state.step_count,
                    action_type,
                    is_done,
                    state.current_task,
                )
            if state.traj_logger is not None:
                log_action = mw_action
                if reasoning_content:
                    log_action = {**mw_action, "reasoning_content": reasoning_content}
                await alog_step(
                    traj_logger=state.traj_logger,
                    traj_name=state.traj_name or session_id,
                    task_goal=state.task_goal or "",
                    step=state.step_count,
                    prediction=str(action.get("prediction") or ""),
                    action=log_action,
                    screenshot_b64=pre_action_b64,
                )

            self._sessions.transition(session_id, SessionStatus.READY)
            image_payload = await self._maybe_upload_screenshot(
                b64_image,
                session_id,
                state.step_count,
                screenshot_transport,
            )

            if isinstance(step_result, dict) and (
                "output" in step_result or "error" in step_result
            ):
                image_payload["bash_output"] = step_result.get("output", "")
                image_payload["bash_error"] = step_result.get("error", "")
                image_payload["bash_returncode"] = step_result.get("returncode", 0)

            if action_type == "intent_call" and isinstance(step_result, dict):
                tc = step_result.get("result")
                if isinstance(tc, dict):
                    await self._recheck_landing(client, session_id, tc)
                    image_payload["tool_call"] = tc
            result = {
                "env_idx": state.backend.device_name,
                "obs_messages": [image_payload],
                "is_done": is_done,
                "format_reward": 0.0,
            }

            if action_id is not None:
                state.last_action_id = action_id
                state.last_step_result = result
            return result
        except Exception as exc:
            is_backend_fail, kind = _classify_backend_failure(exc)
            target = SessionStatus.INVALID if is_backend_fail else SessionStatus.READY
            try:
                self._sessions.transition(session_id, target)
            except Exception as cleanup_exc:
                logger.warning(
                    "[EnvController] step cleanup transition failed: %s",
                    cleanup_exc,
                )
            if is_backend_fail:
                self._record_backend_failure("step", state, exc, kind)

                if self._infra.recycle_on_release(
                    state.backend.id
                ) and not self._is_lazy_pool_state(state):
                    self._kick_relaunch(session_id, state.backend.id)
            raise

    def _finalize_op(
        self, session_id: str, op: PendingOp, task: "asyncio.Task[dict[str, Any]]"
    ) -> None:

        self._op_tasks.discard(task)

        exc = None if task.cancelled() else task.exception()
        try:
            state = self._sessions.get_session(session_id)
            current = state.pending_op
            retained = state.verifier_reset_ops.get(op.action_id) is op
        except KeyError:
            current, retained = None, False
        if current is not op and not retained:
            return
        if task.cancelled():
            op.error = "cancelled"
            op.retryable = op.kind != "reset"
            if op.kind == "reset":
                op.error_detail = operation_error_detail(
                    asyncio.CancelledError(),
                    "reset",
                    operation_id=op.action_id,
                )
            op.done = True
            return
        if exc is not None:
            is_backend_fail, kind = _classify_backend_failure(exc)
            op.error = f"{type(exc).__name__}: {exc}"
            op.retryable = is_backend_fail if op.kind != "reset" else False
            if op.kind == "reset":
                op.error_detail = operation_error_detail(exc, "reset", operation_id=op.action_id)
                op.status_code = (
                    exc.status_code
                    if isinstance(exc, OperationError)
                    else exc.response.status_code
                    if isinstance(exc, httpx.HTTPStatusError)
                    else 502
                    if isinstance(exc, BackendCallError)
                    else 503
                    if isinstance(exc, BackendUnavailableError)
                    else 504
                    if isinstance(exc, TimeoutError)
                    else 500
                )
                if op.error_detail.get("upstream_detail") is not None:
                    op.error = str(op.error_detail["upstream_detail"])[:2000]
            logger.info(
                "[async-op sid=%s kind=%s] action_id=%s failed: %s (retryable=%s)",
                session_id,
                op.kind,
                op.action_id,
                op.error,
                op.retryable,
            )
        else:
            op.result = task.result()
        op.done = True

    def _launch_op(self, session_id: str, op: PendingOp, coro: "object") -> dict[str, str]:

        try:
            self._sessions.transition(session_id, SessionStatus.BUSY)
            state = self._sessions.get_session(session_id)
        except BaseException:
            coro.close()
            raise
        state.pending_op = op
        if op.kind == "reset" and state.session_task is not None:
            state.verifier_reset_ops[op.action_id] = op
        task: asyncio.Task[dict[str, Any]] = asyncio.create_task(coro)
        op.task = task
        self._op_tasks.add(task)
        task.add_done_callback(lambda t, sid=session_id, o=op: self._finalize_op(sid, o, t))
        return {"op_id": op.action_id, "status": "pending"}

    async def submit_reset(
        self,
        session_id: str,
        category: str,
        task_id: str | None,
        goal: str | None = None,
        target_apps: list[str] | None = None,
        seed_task_id: str | None = None,
        attempt_id: str | None = None,
        train_id: str | None = None,
        rollout_id: int | None = None,
        evaluation: bool = False,
        screenshot_width: int | None = None,
        screenshot_height: int | None = None,
        lease_id: str | None = None,
        model_name: str | None = None,
        action_id: str | None = None,
        screenshot_transport: str = "inline",
        task_phase: str | None = None,
    ) -> dict[str, str]:

        if action_id is None:
            raise ValueError("submit_reset requires action_id (the op handle)")
        self.sdk_operations.guard(session_id)
        state = self._sessions.get_session(session_id)
        self._sessions.check_lease(session_id, lease_id)
        self._scheduler.extend_lease(session_id)

        op = state.verifier_reset_ops.get(action_id) or state.pending_op
        if op is not None and op.kind == "reset" and op.action_id == action_id:
            logger.info(
                "[submit_reset sid=%s] idempotent re-submit action_id=%s "
                "(done=%s) — not relaunching",
                session_id,
                action_id,
                op.done,
            )
            return {"op_id": action_id, "status": "done" if op.done else "pending"}

        if action_id == state.last_reset_id and state.last_reset_result is not None:
            logger.info(
                "[submit_reset sid=%s] action_id=%s already completed (cached) — not relaunching",
                session_id,
                action_id,
            )
            return {"op_id": action_id, "status": "done"}
        if state.session_task is not None and len(state.verifier_reset_ops) >= 32:
            raise OperationError(
                "Verifier operation history is full; acquire a fresh session",
                status_code=409,
                error_detail={
                    "code": "operation_history_full",
                    "phase": "reset",
                    "failure_kind": "environment",
                    "recovery": "fresh_session",
                    "operation_id": action_id,
                },
            )
        new_op = PendingOp(kind="reset", action_id=action_id)
        coro = self._run_reset(
            session_id,
            category,
            task_id,
            goal=goal,
            target_apps=target_apps,
            seed_task_id=seed_task_id,
            attempt_id=attempt_id,
            train_id=train_id,
            rollout_id=rollout_id,
            evaluation=evaluation,
            screenshot_width=screenshot_width,
            screenshot_height=screenshot_height,
            model_name=model_name,
            action_id=action_id,
            screenshot_transport=screenshot_transport,
            task_phase=task_phase,
            already_busy=True,
        )
        logger.info(
            "[submit_reset sid=%s] launching detached reset action_id=%s category=%s task_id=%s",
            session_id,
            action_id,
            category,
            task_id,
        )
        return self._launch_op(session_id, new_op, coro)

    async def reset_readiness(
        self,
        session_id: str,
        action_id: str,
        lease_id: str | None = None,
    ) -> dict[str, Any]:

        state = self._sessions.get_session(session_id)
        self._sessions.check_lease(session_id, lease_id)
        self._scheduler.extend_lease(session_id)
        return self._readiness("reset", state, action_id)

    async def submit_step(
        self,
        session_id: str,
        action: dict[str, Any],
        screenshot_width: int | None = None,
        screenshot_height: int | None = None,
        lease_id: str | None = None,
        action_id: str | None = None,
        screenshot_transport: str = "inline",
    ) -> dict[str, str]:

        if action_id is None:
            raise ValueError("submit_step requires action_id (the op handle)")
        self.sdk_operations.guard(session_id)
        state = self._sessions.get_session(session_id)
        self._sessions.check_lease(session_id, lease_id)
        self._scheduler.extend_lease(session_id)

        op = state.pending_op
        if op is not None and op.kind == "step" and op.action_id == action_id:
            logger.info(
                "[submit_step sid=%s] idempotent re-submit action_id=%s "
                "(done=%s) — not relaunching",
                session_id,
                action_id,
                op.done,
            )
            return {"op_id": action_id, "status": "done" if op.done else "pending"}
        if action_id == state.last_action_id and state.last_step_result is not None:
            logger.info(
                "[submit_step sid=%s] action_id=%s already completed (cached) — not relaunching",
                session_id,
                action_id,
            )
            return {"op_id": action_id, "status": "done"}
        new_op = PendingOp(kind="step", action_id=action_id)
        coro = self._run_step(
            session_id,
            action,
            screenshot_width=screenshot_width,
            screenshot_height=screenshot_height,
            action_id=action_id,
            screenshot_transport=screenshot_transport,
            already_busy=True,
        )
        logger.info(
            "[submit_step sid=%s] launching detached step action_id=%s",
            session_id,
            action_id,
        )
        return self._launch_op(session_id, new_op, coro)

    async def step_readiness(
        self,
        session_id: str,
        action_id: str,
        lease_id: str | None = None,
    ) -> dict[str, Any]:

        state = self._sessions.get_session(session_id)
        self._sessions.check_lease(session_id, lease_id)
        self._scheduler.extend_lease(session_id)
        return self._readiness("step", state, action_id)

    def _readiness(
        self,
        kind: str,
        state: SessionState,
        action_id: str,
    ) -> dict[str, Any]:

        op = (
            state.verifier_reset_ops.get(action_id) if kind == "reset" else None
        ) or state.pending_op
        if op is not None and op.kind == kind and op.action_id == action_id:
            if not op.done:
                return {"ready": False, "pending": True}
            if op.error is not None:
                return {
                    "ready": False,
                    "error": op.error,
                    "retryable": op.retryable,
                    **({"error_detail": op.error_detail} if op.error_detail is not None else {}),
                    **({"status_code": op.status_code} if op.status_code is not None else {}),
                }
            return {"ready": True, **(op.result or {})}

        if kind == "reset":
            if action_id == state.last_reset_id and state.last_reset_result is not None:
                return {"ready": True, **state.last_reset_result}
        else:
            if action_id == state.last_action_id and state.last_step_result is not None:
                return {"ready": True, **state.last_step_result}
        result: dict[str, Any] = {"ready": False, "unknown": True}
        if kind == "reset":
            result["error_detail"] = {
                "code": "operation_unknown",
                "phase": "reset",
                "failure_kind": "environment",
                "recovery": "fresh_session",
                "operation_id": action_id,
            }
        return result

    async def evaluate(
        self,
        session_id: str,
        category: str,
        task_id: str | None,
        trajectory_messages: str | None,
        use_judge: bool = False,
        evaluation_mode: Literal["built_in", "llm_judge", "both"] = "built_in",
        *,
        _sdk: bool = False,
        require_score: bool = False,
    ) -> dict[str, Any]:
        if not _sdk:
            self.sdk_operations.guard(session_id)
        state = self._sessions.get_session(session_id)
        self._scheduler.extend_lease(session_id)
        task_goal = state.task_goal or task_id or category

        mode = "llm_judge" if use_judge else evaluation_mode
        if mode not in {"built_in", "llm_judge", "both"}:
            raise ValueError(f"unsupported evaluation_mode: {mode!r}")
        wants_judge = mode in {"llm_judge", "both"}
        judge = self._judge_evaluator if wants_judge else None
        if wants_judge and judge is None:
            logger.warning(
                "[Evaluate sid=%s] task_id=%s requested evaluation_mode=%s but the "
                "manager has no judge configured (config.judge unset) — "
                "falling back to built-in evaluation",
                session_id,
                task_id,
                mode,
            )

        if mode == "llm_judge" and judge is not None:
            env_reward, reason = None, "llm_judge"
        else:
            client = self._get_or_create_backend(state.backend)
            strategy = self._get_strategy(state.backend.env_type)
            env_reward, reason = await strategy.on_evaluate(
                client,
                state,
                task_id=task_id,
                category=category,
            )

        if (
            require_score
            and env_reward is None
            and isinstance(self._reward_evaluator, DeterministicEvaluator)
            and not (mode == "llm_judge" and judge is not None)
        ):
            raise RuntimeError("The environment verifier did not return a score")

        from mai_gym.io import oss_io

        if state.traj_logger is not None:
            traj_dir: str | None = state.traj_logger.log_file_dir
        elif self._traj_log_dir and state.traj_name:
            traj_dir = oss_io.join(self._traj_log_dir, state.traj_name)
        else:
            traj_dir = None
        built_in_reward: float | None = None
        judge_reward: float | None = None
        if mode == "both" and judge is not None:
            built_in_reward = await self._reward_evaluator.evaluate(
                session_id=session_id,
                task_goal=task_goal,
                env_reward=env_reward,
                traj_dir=traj_dir,
            )
            judge_reward = await self._judge_evaluator.evaluate(
                session_id=session_id,
                task_goal=task_goal,
                env_reward=env_reward,
                traj_dir=traj_dir,
            )
            reward = built_in_reward
            backend_reason = reason
            reason = "built_in+llm_judge"
            if backend_reason:
                reason += f"\n{backend_reason}"
        else:
            evaluator = (
                judge if mode == "llm_judge" and judge is not None else self._reward_evaluator
            )
            reward = await evaluator.evaluate(
                session_id=session_id,
                task_goal=task_goal,
                env_reward=env_reward,
                traj_dir=traj_dir,
            )
            if mode == "llm_judge" and judge is not None:
                judge_reward = reward
        if state.traj_logger is not None:
            await log_score(state.traj_logger, reward, reason=reason)

        state.traj_scored = True
        logger.info(
            "[Evaluate sid=%s] task_id=%s mode=%s reward=%s steps=%d reason=%s",
            session_id,
            task_id,
            mode,
            reward,
            state.step_count,
            reason or "",
        )
        result: dict[str, Any] = {"reward": reward, "reason": reason}
        success = getattr(reward, "success", None)
        if success is not None:
            result["success"] = success
        if built_in_reward is not None:
            result["built_in_reward"] = built_in_reward
        if judge_reward is not None:
            result["judge_reward"] = judge_reward
        return result

    async def list_tasks(self) -> dict[str, Any]:

        now = time.monotonic()
        if self._task_list_cache is not None and now < self._task_list_cache_expires_at:
            return {"tasks": self._task_list_cache}

        async with self._task_list_cache_lock:
            now = time.monotonic()
            if self._task_list_cache is not None and now < self._task_list_cache_expires_at:
                return {"tasks": self._task_list_cache}

            backends = tuple(self._backends.values())
            per_backend = await asyncio.gather(*(backend.list_tasks() for backend in backends))
            all_tasks: list[dict[str, Any]] = []
            seen: set[tuple[str | None, str | None, str | None, str]] = set()
            for backend, tasks in zip(backends, per_backend, strict=True):
                running_backend = getattr(backend, "backend", None)
                env_type = getattr(running_backend, "env_type", None)
                env_type = env_type if isinstance(env_type, str) else None
                metadata = getattr(running_backend, "metadata", {})
                metadata = metadata if isinstance(metadata, dict) else {}
                routing = {
                    key: metadata[key]
                    for key in ("infra_type", "environment_profile")
                    if metadata.get(key) is not None
                }
                for task in tasks:
                    tid = task.get("task_id", "")
                    key = (
                        task.get("env_type", env_type),
                        routing.get("infra_type"),
                        routing.get("environment_profile"),
                        tid,
                    )
                    if tid and key not in seen:
                        all_tasks.append(
                            {**task, **routing, **({"env_type": env_type} if env_type else {})}
                        )
                        seen.add(key)

            if self._osworld_examples_dir:
                directory = Path(self._osworld_examples_dir)
                for path in sorted(directory.rglob("*.json")):
                    try:
                        data = json.loads(path.read_text())
                    except (OSError, ValueError):
                        continue
                    if (
                        not isinstance(data, dict)
                        or not data.get("instruction")
                        or "evaluator" not in data
                    ):
                        continue
                    tid = str(path.relative_to(directory).with_suffix(""))
                    key = ("osworld", None, None, tid)
                    if key not in seen:
                        all_tasks.append(
                            {
                                "task_id": tid,
                                "instruction": data["instruction"],
                                "env_type": "osworld",
                                "max_steps": data.get("max_steps", 100),
                            }
                        )
                        seen.add(key)

            self._task_list_cache = all_tasks
            self._task_list_cache_expires_at = time.monotonic() + self._task_list_cache_ttl_s
            logger.info(
                "[EnvController] task-list cache refreshed: tasks=%d backends=%d ttl=%.1fs",
                len(all_tasks),
                len(backends),
                self._task_list_cache_ttl_s,
            )
            return {"tasks": all_tasks}

    async def log_messages(
        self,
        session_id: str,
        step: int,
        payload: dict[str, Any],
    ) -> dict[str, Any]:

        state = self._sessions.get_session(session_id)
        traj_name = state.traj_name
        if not traj_name:
            return {"logged": False, "reason": "no traj_name for session"}

        from mai_gym.io import oss_io

        if state.traj_logger is not None:
            traj_dir = state.traj_logger.log_file_dir
            oss_cfg = state.traj_logger.oss_cfg
        else:
            log_root = self._traj_log_dir or self._debug_log_dir
            if not log_root:
                return {"logged": False, "reason": "debug_log_dir not set"}
            traj_dir = oss_io.join(log_root, traj_name)
            oss_cfg = self._oss_cfg

        path = oss_io.join(traj_dir, "messages", f"step_{step:04d}.json")
        is_oss = oss_io.is_oss_path(path)
        if is_oss and oss_cfg is None:
            logger.warning(
                "log_messages: oss path %r needs oss_cfg; skipping",
                path,
            )
            return {
                "logged": False,
                "reason": "oss path without oss_cfg",
            }

        try:
            fs = oss_io.get_async_fs(oss_cfg) if is_oss else None
            text = json.dumps(payload, ensure_ascii=False, indent=2)
            await oss_io.write_text(path, text, fs=fs)
            return {"logged": True}
        except Exception as exc:
            logger.warning("[EnvController] log_messages failed: %s", exc)
            return {"logged": False, "reason": str(exc)}

    async def tear_down(
        self,
        session_id: str,
        category: str,
        task_id: str | None,
        *,
        _sdk: bool = False,
    ) -> dict[str, Any]:
        if not _sdk:
            self.sdk_operations.guard(session_id)
        state = self._sessions.get_session(session_id)
        remove_session = state.backend.env_type == "osworld_k8s" and self._infra.recycle_on_release(
            state.backend.id
        )
        logger.info(
            "[TearDown sid=%s] category=%s task_id=%s steps=%d remove_session=%s",
            session_id,
            category,
            task_id,
            state.step_count,
            remove_session,
        )
        if remove_session:
            self._sessions.transition(session_id, SessionStatus.DESTROYING)
        client = self._get_or_create_backend(state.backend)
        strategy = self._get_strategy(state.backend.env_type)
        await strategy.on_tear_down(client, state, task_id=task_id, category=category)

        if remove_session and state.session_task is not None:
            try:
                await strategy.on_detach_task(
                    client,
                    state,
                    task=state.session_task,
                )
            except Exception as exc:
                logger.warning(
                    "[TearDown sid=%s] on_detach_task failed: %s",
                    session_id,
                    exc,
                )
            state.session_task = None

        if not remove_session:
            return {"message": "ok"}

        if isinstance(client, PooledBackend) and state.context_id is not None:
            await client.release_context(state.context_id)
            state.context_id = None
        await self._infra.release(state.backend.id)
        old_backend = self._backends.pop(state.backend.id, None)
        if old_backend is not None:
            await old_backend.close()
        self._sessions.remove_session(session_id)
        await self._fleet_autoscaler.detach_session(session_id)
        if self._is_lazy_pool_state(state):
            await self._notify_lazy_capacity()
        return {"message": "ok"}
