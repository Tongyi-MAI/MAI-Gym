import asyncio
import concurrent.futures
import contextlib
import logging
import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

from mai_gym.envs.osworld.launch_k8s import pod_bootstrap_contract
from mai_gym.infra.backend import (
    BackendCallError,
    BackendUnavailableError,
    UnscoredTaskEvaluationError,
)
from mai_gym.infra.manager import InfraManager
from mai_gym.infra.task_artifacts import TaskArtifactManifestError
from mai_gym.models.config import K8sLaunchConfig, MobileGymConfig
from mai_gym.models.operation_error import OperationError, operation_error_detail
from mai_gym.models.requests import (
    AcquireFailureRequest,
    AcquireRequest,
    BatchStartEnvRequest,
    DiscardRequest,
    EvaluateRequest,
    FleetHeartbeatRequest,
    FleetRequestCreate,
    ReadinessRequest,
    ReleaseRequest,
    ResetRequest,
    RunConfigRequest,
    SessionTaskRequest,
    StepRequest,
    TearDownRequest,
    TrainedManifestRequest,
)
from mai_gym.models.responses import AcquireResponse, SessionTaskResponse
from mai_gym.models.sdk import (
    SDK_CAPABILITY,
    SDKOperationRef,
    SDKOperationRequest,
    SDKResultRequest,
)
from mai_gym.server.controller import EnvController
from mai_gym.server.fleet_autoscaler import (
    FleetCapacityExceeded,
    FleetRequestConflict,
    FleetRequestNotFound,
    FleetRequestUnavailable,
)
from mai_gym.server.response_runtime import (
    BoundedGZipMiddleware,
    CompressionPool,
    EventLoopWatchdog,
)
from mai_gym.server.reward import DeterministicEvaluator, LLMJudgeEvaluator, RewardEvaluator
from mai_gym.server.session import (
    InvalidTransitionError,
    LeaseMismatchError,
    SessionNotFoundError,
)
from mai_gym.server.trajectory import model_slug as _model_slug

logger = logging.getLogger(__name__)


def _environment_contract(config: MobileGymConfig) -> dict[str, Any]:

    osworld_k8s: list[dict[str, Any]] = []
    for device in config.devices:
        launch = device.launch
        if not device.type.startswith("osworld") or not isinstance(launch, K8sLaunchConfig):
            continue
        osworld_k8s.append(
            {
                "device": device.name,
                "template_id": launch.template_id,
                "acquisition": launch.acquisition,
                "pool_name": launch.pool_name,
                "pool_size": launch.pool_size,
                "lazy_acquire": launch.lazy_acquire,
                **(
                    {"fleet_autoscale": launch.fleet_autoscale.model_dump()}
                    if launch.fleet_autoscale.enabled
                    else {}
                ),
                "max_concurrent_launches": launch.max_concurrent_launches,
                "ready_request_timeout": launch.ready_request_timeout,
                "pod_bootstrap": pod_bootstrap_contract(launch),
            }
        )
    contract: dict[str, Any] = {"osworld_k8s": osworld_k8s}
    if any(device.type.startswith("osworld") for device in config.devices):
        contract["osworld_eval_isolation"] = config.manager.osworld_eval_isolation.model_dump()
    return contract


def _register_exception_handlers(app: FastAPI) -> None:

    @app.exception_handler(UnscoredTaskEvaluationError)
    async def _unscored_task_evaluation(
        _req: Request,
        exc: UnscoredTaskEvaluationError,
    ) -> JSONResponse:
        return JSONResponse(
            status_code=502,
            content={
                "error": "task_evaluation_unscored",
                "message": str(exc),
                "detail": str(exc),
                "evaluation_status": exc.status,
                "verifier_result": exc.result,
            },
        )

    def request_phase(req: Request) -> str:
        route = req.url.path.strip("/").split("/")[0]
        return {"session_task": "attach", "areset": "reset", "evaluate": "eval"}.get(route, route)

    @app.exception_handler(OperationError)
    async def _operation_error(_req: Request, exc: OperationError) -> JSONResponse:
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "error": exc.error_detail["code"],
                "message": str(exc),
                "error_detail": exc.error_detail,
            },
        )

    @app.exception_handler(TimeoutError)
    async def _operation_timeout(req: Request, exc: TimeoutError) -> JSONResponse:
        detail = operation_error_detail(exc, request_phase(req))
        detail["code"] = "operation_deadline_exceeded"
        return JSONResponse(
            status_code=504,
            content={
                "error": detail["code"],
                "message": "Operation deadline exceeded",
                "error_detail": detail,
            },
        )

    @app.exception_handler(SessionNotFoundError)
    async def _session_not_found(_req: Request, exc: SessionNotFoundError) -> JSONResponse:
        return JSONResponse(
            status_code=404,
            content={
                "error": "session_not_found",
                "message": f"Session {exc.session_id!r} not found",
                "session_id": exc.session_id,
                "hint": "the session was released, torn down, or never existed; re-/acquire",
            },
        )

    @app.exception_handler(InvalidTransitionError)
    async def _invalid_transition(_req: Request, exc: InvalidTransitionError) -> JSONResponse:

        return JSONResponse(
            status_code=409,
            content={
                "error": "session_busy" if exc.current.value == "busy" else "invalid_session_state",
                "message": str(exc),
                "session_id": exc.session_id,
                "current_state": exc.current.value,
                "requested_state": exc.target.value,
                "hint": (
                    "a prior /reset or /step on this session is still running; "
                    "wait for it to return before issuing another, or acquire a "
                    "fresh session"
                ),
            },
        )

    @app.exception_handler(LeaseMismatchError)
    async def _lease_mismatch(_req: Request, exc: LeaseMismatchError) -> JSONResponse:

        return JSONResponse(
            status_code=409,
            content={
                "error": "lease_mismatch",
                "message": str(exc),
                "session_id": exc.session_id,
                "hint": (
                    "your lease expired and the session was re-leased to "
                    "another worker; discard this session_id and acquire a "
                    "fresh one — do not retry this request"
                ),
            },
        )

    @app.exception_handler(BackendUnavailableError)
    async def _backend_unavailable(_req: Request, exc: BackendUnavailableError) -> JSONResponse:
        retry_in = max(1, int(exc.retry_in))
        return JSONResponse(
            status_code=503,
            content={
                "error": "backend_unavailable",
                "message": str(exc),
                "device": exc.backend.device_name,
                "retry_in_s": round(exc.retry_in, 1),
                "error_detail": operation_error_detail(exc, request_phase(_req)),
            },
            headers={"Retry-After": str(retry_in)},
        )

    @app.exception_handler(BackendCallError)
    async def _backend_call_error(_req: Request, exc: BackendCallError) -> JSONResponse:
        return JSONResponse(
            status_code=502,
            content={
                "error": "backend_call_failed",
                "message": str(exc),
                "error_detail": operation_error_detail(exc, request_phase(_req)),
                "device": exc.backend.device_name,
                "hint": "the backend container dropped the connection; it may be "
                "restarting — retry after a short backoff",
            },
        )

    @app.exception_handler(FleetCapacityExceeded)
    async def _fleet_capacity_exceeded(
        _req: Request,
        exc: FleetCapacityExceeded,
    ) -> JSONResponse:
        return JSONResponse(
            status_code=409,
            content={
                "error": "fleet_capacity_exceeded",
                "message": str(exc),
                "requested": exc.requested,
                "available": exc.available,
                "capacity_limit": exc.limit,
                "reserved": exc.reserved,
                "active_unreserved": exc.active_unreserved,
                "legacy_warm": exc.legacy_warm,
            },
        )

    @app.exception_handler(FleetRequestConflict)
    async def _fleet_request_conflict(
        _req: Request,
        exc: FleetRequestConflict,
    ) -> JSONResponse:
        return JSONResponse(
            status_code=409,
            content={"error": "fleet_request_conflict", "message": str(exc)},
        )

    @app.exception_handler(FleetRequestNotFound)
    async def _fleet_request_not_found(
        _req: Request,
        exc: FleetRequestNotFound,
    ) -> JSONResponse:
        return JSONResponse(
            status_code=404,
            content={"error": "fleet_request_not_found", "message": str(exc)},
        )

    @app.exception_handler(FleetRequestUnavailable)
    async def _fleet_request_unavailable(
        _req: Request,
        exc: FleetRequestUnavailable,
    ) -> JSONResponse:

        return JSONResponse(
            status_code=503,
            headers={"Retry-After": "2"},
            content={"error": "fleet_request_unavailable", "message": str(exc)},
        )

    @app.exception_handler(httpx.HTTPStatusError)
    async def _upstream_status(_req: Request, exc: httpx.HTTPStatusError) -> JSONResponse:

        code = exc.response.status_code
        status = code if 400 <= code < 600 else 502
        detail = operation_error_detail(exc, request_phase(_req))
        upstream_detail = detail.get("upstream_detail")
        return JSONResponse(
            status_code=status,
            content={
                "error": "backend_error",
                "message": f"backend returned HTTP {code} for {exc.request.url.path}",
                "upstream_status": code,
                "upstream_detail": upstream_detail,
                "error_detail": detail,
            },
        )


def _build_judge_evaluator(config: MobileGymConfig) -> RewardEvaluator | None:

    if config.judge is None:
        return None
    try:
        from openai import AsyncOpenAI

        client = AsyncOpenAI(
            api_key=config.judge.api_key,
            base_url=config.judge.base_url,
        )
        return LLMJudgeEvaluator(
            client=client,
            model=config.judge.model,
            image_scale=config.judge.image_scale,
            oss_cfg=config.manager.oss,
        )
    except ImportError:
        logger.warning(
            "openai package not installed; LLM judge disabled — "
            "falling back to DeterministicEvaluator. "
            "Install with: uv sync --extra judge"
        )
        return None


def create_app(config: MobileGymConfig) -> FastAPI:

    from mai_gym.envs.osworld.verifier_patches import (
        fixed_verifier_enabled,
        set_fixed_verifier,
    )

    set_fixed_verifier(config.rollout.use_fixed_verifier)
    logger.info(
        "OSWorld fixed-verifier patches: %s (configured=%r)",
        "enabled" if fixed_verifier_enabled() else "disabled",
        config.rollout.use_fixed_verifier,
    )
    environment_contract = _environment_contract(config)

    try:
        from mai_gym.envs.osworld.backend import configure_osw_proxy

        configure_osw_proxy(
            config.manager.osworld_proxy.model_dump()
            if config.manager.osworld_proxy.enabled
            else None
        )
    except Exception as exc:
        logger.warning("OSWorld proxy registration failed: %s", exc)

    from mai_gym.envs.osworld.eval_isolation import configure_eval_isolation

    configure_eval_isolation(config.manager.osworld_eval_isolation.model_dump())

    infra = InfraManager(config)

    judge_evaluator = _build_judge_evaluator(config)
    judge_scope = config.judge.scope if config.judge is not None else "all"
    judge_mode = config.judge.mode if config.judge is not None else "replace"

    controller = EnvController(
        infra=infra,
        debug_log_dir=config.manager.traj_log_dir,
        post_action_wait=config.manager.post_action_wait,
        reward_evaluator=DeterministicEvaluator(),
        judge_evaluator=judge_evaluator,
        osworld_examples_dir=config.manager.osworld_examples_dir,
        cua_osworld_tasks_dir=config.manager.cua_osworld_tasks_dir,
        vscode_ext_mirror=(
            config.manager.vscode_ext_mirror.model_dump()
            if config.manager.vscode_ext_mirror.base_url
            and config.manager.vscode_ext_mirror.extensions
            else None
        ),
        devices=config.devices,
        model_slug=(_model_slug(config.agent.model_name) if config.agent is not None else None),
        oss_cfg=config.manager.oss,
        screenshot_codec=config.manager.screenshot.codec,
        screenshot_quality=config.manager.screenshot.quality,
        screenshot_transport=config.manager.screenshot.transport,
        screenshot_oss_prefix=config.manager.screenshot.oss_prefix,
        config_path=config.source_path,
    )

    compression_config = config.manager.response_compression
    compression_pool = CompressionPool(compression_config.workers, compression_config.level)
    loop_watchdog = EventLoopWatchdog()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:

        pool_size = int(os.environ.get("MAI_GYM_BLOCKING_IO_THREADS", "2048") or "2048")
        asyncio.get_running_loop().set_default_executor(
            concurrent.futures.ThreadPoolExecutor(
                max_workers=pool_size,
                thread_name_prefix="blocking-io",
            )
        )
        logger.info("[lifespan] default thread executor sized to %d workers", pool_size)
        watchdog_task = asyncio.create_task(loop_watchdog.run())
        sweep_task = asyncio.create_task(controller.run_lease_sweep())
        snapshot_task = asyncio.create_task(controller.run_pool_snapshot())
        recovery_task = asyncio.create_task(controller.run_recovery_sweep())
        ready_buffer_task = asyncio.create_task(controller.run_lazy_ready_buffer())
        fleet_task: asyncio.Task[None] | None = (
            asyncio.create_task(controller.run_fleet_autoscaler())
            if controller.fleet_capability() is not None
            else None
        )
        prewarm_task: asyncio.Task[None] | None = (
            asyncio.create_task(controller.prewarm_all()) if config.manager.auto_prewarm else None
        )
        try:
            yield
        finally:
            tasks = [sweep_task, snapshot_task, recovery_task, ready_buffer_task, watchdog_task]
            if fleet_task is not None:
                tasks.append(fleet_task)
            if prewarm_task is not None:
                tasks.append(prewarm_task)
            for task in tasks:
                task.cancel()
                with contextlib.suppress(asyncio.CancelledError, Exception):
                    await task
            try:
                await controller.close()
                await infra.shutdown_all()
            finally:
                compression_pool.close()

    app = FastAPI(title="mai-gym EnvController", lifespan=lifespan)

    if compression_config.enabled:
        app.add_middleware(
            BoundedGZipMiddleware,
            pool=compression_pool,
            minimum_size=compression_config.minimum_size,
        )
    _register_exception_handlers(app)

    @app.post("/sdk/capabilities")
    async def sdk_capabilities() -> dict[str, Any]:
        return {"capabilities": [SDK_CAPABILITY]}

    @app.post("/sdk/{session_id}/submit")
    async def sdk_submit(session_id: str, req: SDKOperationRequest) -> dict[str, Any]:
        return controller.sdk_operations.submit(session_id, req)

    @app.post("/sdk/{session_id}/status")
    async def sdk_status(session_id: str, req: SDKOperationRef) -> dict[str, Any]:
        return controller.sdk_operations.status(session_id, req.lease_id, req.operation_id)

    @app.post("/sdk/{session_id}/result")
    async def sdk_result(session_id: str, req: SDKResultRequest) -> dict[str, Any]:
        return controller.sdk_operations.result(
            session_id,
            req.lease_id,
            req.operation_id,
            req.chunk,
        )

    @app.get("/health")
    async def health() -> dict[str, Any]:

        from mai_gym.envs.osworld.eval_isolation import eval_isolation_stats

        fleet_capability = controller.fleet_capability()
        return {
            "ok": True,
            "event_loop": loop_watchdog.snapshot(),
            "response_compression": {
                "enabled": compression_config.enabled,
                **compression_pool.snapshot(),
            },
            "capabilities": {
                "verifier_errors.v1": {"version": 1},
                **({"fleet_requests.v1": fleet_capability} if fleet_capability is not None else {}),
            },
            "traj_log_dir": controller.traj_log_dir,
            "screenshot_transport": config.manager.screenshot.transport,
            "screenshot_transport_selection": "client",
            "screenshot_transport_default": "inline",
            "screenshot_transports": (
                ["inline", "oss"] if config.manager.screenshot.transport == "oss" else ["inline"]
            ),
            "screenshot_codec": config.manager.screenshot.codec,
            "screenshot_upload": controller.screenshot_transport_stats,
            "evaluator_contract": {
                "fixed_verifier_patches": fixed_verifier_enabled(),
                "selection": "client",
                "default_mode": "built_in",
                "supported_modes": (
                    ["built_in", "llm_judge", "both"]
                    if judge_evaluator is not None
                    else ["built_in"]
                ),
                "judge": (
                    {
                        "model": config.judge.model,
                        "scope": judge_scope,
                        "mode": judge_mode,
                    }
                    if config.judge is not None
                    else None
                ),
                **(
                    {"osworld_process_isolation": eval_isolation_stats()}
                    if any(device.type.startswith("osworld") for device in config.devices)
                    else {}
                ),
            },
            "environment_contract": environment_contract,
            "model_slug": (
                _model_slug(config.agent.model_name) if config.agent is not None else None
            ),
        }

    @app.post("/batch_start_env")
    async def batch_start_env(req: BatchStartEnvRequest) -> dict[str, Any]:
        return await controller.batch_start_env(req.num_envs, req.env_type)

    @app.get("/sessions")
    async def list_sessions() -> dict[str, Any]:
        return controller.list_sessions()

    @app.post("/fleet/requests", status_code=202)
    async def create_fleet_request(req: FleetRequestCreate) -> dict[str, Any]:
        return await controller.upsert_fleet_request(
            job_id=req.job_id,
            holder_id=req.holder_id,
            idempotency_key=req.idempotency_key,
            env_type=req.env_type,
            capacity=req.capacity,
        )

    @app.get("/fleet/requests")
    async def list_fleet_requests() -> dict[str, Any]:
        return {"requests": await controller.list_fleet_requests()}

    @app.get("/fleet/requests/{request_id}")
    async def get_fleet_request(request_id: str) -> dict[str, Any]:
        return await controller.get_fleet_request(request_id)

    @app.post("/fleet/requests/{request_id}/heartbeat")
    async def heartbeat_fleet_request(
        request_id: str,
        req: FleetHeartbeatRequest,
    ) -> dict[str, Any]:
        return await controller.heartbeat_fleet_request(
            request_id,
            req.holder_id,
        )

    @app.delete("/fleet/requests/{request_id}/holders/{holder_id}")
    async def release_fleet_holder(
        request_id: str,
        holder_id: str,
    ) -> dict[str, Any]:
        return await controller.release_fleet_holder(request_id, holder_id)

    @app.get("/tasks")
    async def list_tasks() -> dict[str, Any]:
        return await controller.list_tasks()

    @app.post("/reset/{session_id}")
    async def reset(session_id: str, req: ResetRequest) -> dict[str, Any]:
        return await controller.reset(
            session_id,
            req.category,
            req.task_id,
            req.goal,
            req.target_apps,
            seed_task_id=req.seed_task_id,
            attempt_id=req.attempt_id,
            train_id=req.train_id,
            rollout_id=req.rollout_id,
            evaluation=req.evaluation,
            screenshot_width=req.screenshot_width,
            screenshot_height=req.screenshot_height,
            lease_id=req.lease_id,
            model_name=req.model_name,
            action_id=req.action_id,
            screenshot_transport=req.screenshot_transport or "inline",
            task_phase=req.task_phase,
        )

    @app.post("/session_task/{session_id}", response_model=SessionTaskResponse)
    async def session_task(
        session_id: str,
        req: SessionTaskRequest,
    ) -> SessionTaskResponse:

        out = await controller.attach_session_task(
            session_id,
            env_type=req.env_type,
            lease_id=req.lease_id,
            set_id=req.set_id,
            task_class=req.task_class,
            golden_class=req.golden_class,
            package_b64=req.package_b64,
            initial_setup=req.initial_setup,
            golden_patch=req.golden_patch,
            reward=req.reward,
            world_oss_uri=req.world_oss_uri,
            world_sha256=req.world_sha256,
            world_extract_to=req.world_extract_to,
        )
        return SessionTaskResponse(ok=out["ok"], kind=out["kind"])

    @app.post("/step/{session_id}")
    async def step(session_id: str, req: StepRequest) -> dict[str, Any]:
        return await controller.step(
            session_id,
            req.action,
            screenshot_width=req.screenshot_width,
            screenshot_height=req.screenshot_height,
            lease_id=req.lease_id,
            action_id=req.action_id,
            screenshot_transport=req.screenshot_transport or "inline",
        )

    @app.post("/areset/{session_id}")
    async def areset(session_id: str, req: ResetRequest) -> dict[str, Any]:
        if req.action_id is None:
            raise HTTPException(
                status_code=400,
                detail="areset requires action_id (the op handle to poll)",
            )
        return await controller.submit_reset(
            session_id,
            req.category,
            req.task_id,
            req.goal,
            req.target_apps,
            seed_task_id=req.seed_task_id,
            attempt_id=req.attempt_id,
            train_id=req.train_id,
            rollout_id=req.rollout_id,
            evaluation=req.evaluation,
            screenshot_width=req.screenshot_width,
            screenshot_height=req.screenshot_height,
            lease_id=req.lease_id,
            model_name=req.model_name,
            action_id=req.action_id,
            screenshot_transport=req.screenshot_transport or "inline",
            task_phase=req.task_phase,
        )

    @app.post("/areset_readiness/{session_id}")
    async def areset_readiness(session_id: str, req: ReadinessRequest) -> dict[str, Any]:
        return await controller.reset_readiness(
            session_id,
            req.action_id,
            lease_id=req.lease_id,
        )

    @app.post("/astep/{session_id}")
    async def astep(session_id: str, req: StepRequest) -> dict[str, Any]:
        if req.action_id is None:
            raise HTTPException(
                status_code=400,
                detail="astep requires action_id (the op handle to poll)",
            )
        return await controller.submit_step(
            session_id,
            req.action,
            screenshot_width=req.screenshot_width,
            screenshot_height=req.screenshot_height,
            lease_id=req.lease_id,
            action_id=req.action_id,
            screenshot_transport=req.screenshot_transport or "inline",
        )

    @app.post("/astep_readiness/{session_id}")
    async def astep_readiness(session_id: str, req: ReadinessRequest) -> dict[str, Any]:
        return await controller.step_readiness(
            session_id,
            req.action_id,
            lease_id=req.lease_id,
        )

    @app.post("/evaluate/{session_id}")
    async def evaluate(session_id: str, req: EvaluateRequest) -> dict[str, Any]:
        return await controller.evaluate(
            session_id,
            req.category,
            req.task_id,
            req.trajectory_messages,
            use_judge=req.use_judge,
            evaluation_mode=req.evaluation_mode,
            require_score=req.require_score,
        )

    @app.post("/tear_down/{session_id}")
    async def tear_down(session_id: str, req: TearDownRequest) -> dict[str, Any]:
        return await controller.tear_down(session_id, req.category, req.task_id)

    @app.post("/debug/log_messages/{session_id}")
    async def log_messages(session_id: str, req: dict[str, Any]) -> dict[str, Any]:
        step_num = int(req.get("step", 0))
        payload = req.get("payload", {})
        return await controller.log_messages(session_id, step_num, payload)

    @app.post("/acquire", response_model=AcquireResponse)
    async def acquire(req: AcquireRequest) -> AcquireResponse:
        result = await controller.acquire(
            req.target_apps,
            req.timeout_s,
            req.fleet_request_id,
            **({"env_type": req.env_type} if req.env_type else {}),
        )
        if result is None:
            pool = controller.pool_summary()
            sandboxset = ""
            if "desired_warm_replicas" in pool:
                sandboxset = (
                    f", sandboxset_desired={pool['desired_warm_replicas']}"
                    f", sandboxset_available={pool['available_warm_replicas']}"
                )
                if pool.get("last_error"):
                    sandboxset += f", scale_error={pool['last_error']}"
            raise HTTPException(
                status_code=408,
                detail=(
                    f"No session available within {req.timeout_s}s — "
                    f"pool: {pool['total']} total, "
                    f"{pool['manager_idle']} ready, "
                    f"{pool['pending_provisions']} provisioning, "
                    f"{pool['on_demand_capacity']} remaining_capacity, "
                    f"{pool['leased']} leased, "
                    f"by_status={pool['by_status']}"
                    f"{sandboxset}"
                ),
            )
        return AcquireResponse(
            session_id=result.session_id,
            backend_idx=result.backend_idx,
            lease_id=result.lease_id,
            lease_expires_at=result.lease_expires_at,
        )

    @app.post("/release/{session_id}")
    async def release(session_id: str, req: ReleaseRequest) -> dict[str, Any]:
        return await controller.release(session_id, req.lease_id)

    @app.post("/discard/{session_id}")
    async def discard(session_id: str, req: DiscardRequest) -> dict[str, Any]:
        return await controller.discard(session_id, req.lease_id, req.reason)

    @app.post("/admin/force_release/{session_id}")
    async def admin_force_release(session_id: str) -> dict[str, Any]:
        return await controller.admin_force_release(session_id)

    @app.post("/admin/force_invalidate/{session_id}")
    async def admin_force_invalidate(session_id: str) -> dict[str, Any]:
        return await controller.admin_force_invalidate(session_id)

    @app.post("/admin/scale")
    async def admin_scale(
        size: int,
        env_type: str | None = None,
        device: str | None = None,
    ) -> dict[str, Any]:

        return await controller.scale_pool(size, env_type=env_type, device_name=device)

    @app.post("/register_task_set")
    async def register_task_set(set_id: str, request: Request) -> dict[str, Any]:

        package = await request.body()
        if not package:
            raise HTTPException(status_code=400, detail="empty request body (zip)")
        try:
            return await controller.register_task_set(set_id, package)
        except TaskArtifactManifestError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.get("/task_sets")
    async def task_sets() -> dict[str, Any]:

        return {"set_ids": controller._infra.registered_set_ids()}

    @app.delete("/register_task_set/{set_id}")
    async def unregister_task_set(set_id: str) -> dict[str, Any]:

        return await controller.unregister_task_set(set_id)

    @app.post("/run_config")
    async def run_config(req: RunConfigRequest) -> dict[str, Any]:

        snapshot = dict(req.config)
        requested = snapshot.get("evaluator_contract")
        evaluator_contract = dict(requested) if isinstance(requested, dict) else {}
        evaluator_contract["fixed_verifier_patches"] = fixed_verifier_enabled()
        evaluator_contract["selection"] = "client"
        evaluator_contract["default_mode"] = "built_in"
        evaluator_contract["supported_modes"] = (
            ["built_in", "llm_judge", "both"] if judge_evaluator is not None else ["built_in"]
        )
        evaluator_contract["judge"] = (
            {
                "model": config.judge.model,
                "scope": judge_scope,
                "mode": judge_mode,
            }
            if config.judge is not None
            else None
        )
        snapshot["evaluator_contract"] = evaluator_contract
        snapshot["environment_contract"] = environment_contract
        return await controller.write_run_config(
            req.train_id,
            snapshot,
            model_name=req.model_name,
        )

    @app.post("/trained")
    async def trained(req: TrainedManifestRequest) -> dict[str, Any]:

        return await controller.write_trained_manifest(
            req.train_id,
            req.rollout_id,
            req.manifest,
            evaluation=req.evaluation,
            model_name=req.model_name,
        )

    @app.post("/log_acquire_failure")
    async def log_acquire_failure(req: AcquireFailureRequest) -> dict[str, Any]:

        return await controller.log_acquire_failure(
            train_id=req.train_id,
            task_id=req.task_id,
            attempt_id=req.attempt_id,
            rollout_id=req.rollout_id,
            evaluation=req.evaluation,
            model_name=req.model_name,
            env_type=req.env_type,
            reason=req.reason,
        )

    return app
