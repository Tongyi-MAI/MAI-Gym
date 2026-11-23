from __future__ import annotations

import asyncio
import contextlib
import logging
import time
import uuid
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

from mai_gym.models.config import DeviceConfig, K8sLaunchConfig

logger = logging.getLogger(__name__)


class FleetRequestError(RuntimeError):
    pass


class FleetRequestNotFound(FleetRequestError):
    pass


class FleetRequestConflict(FleetRequestError):
    pass


class FleetCapacityExceeded(FleetRequestError):
    def __init__(
        self,
        *,
        requested: int,
        available: int,
        limit: int,
        reserved: int,
        active_unreserved: int = 0,
        legacy_warm: int = 0,
    ) -> None:
        self.requested = requested
        self.available = available
        self.limit = limit
        self.reserved = reserved
        self.active_unreserved = active_unreserved
        self.legacy_warm = legacy_warm
        super().__init__(
            f"fleet request needs {requested} slot(s), but only {available} "
            f"of {limit} remain ({reserved} reserved, "
            f"{active_unreserved} active outside reservations, "
            f"{legacy_warm} held for legacy warm capacity)"
        )


class FleetRequestUnavailable(FleetRequestError):
    pass


@dataclass
class _Holder:
    holder_id: str
    idempotency_key: str
    capacity: int
    expires_at: float
    last_heartbeat: float


@dataclass
class _FleetRequest:
    request_id: str
    job_id: str
    env_type: str
    device_name: str
    created_at: float
    holders: dict[str, _Holder] = field(default_factory=dict)
    attached_sessions: set[str] = field(default_factory=set)
    pending_acquires: int = 0
    inactive_reason: str | None = None
    inactive_at: float | None = None

    warmed_capacity: int = 0
    warmup_started_at: float | None = None

    @property
    def capacity(self) -> int:
        return max((holder.capacity for holder in self.holders.values()), default=0)

    @property
    def active(self) -> bool:
        return bool(self.holders)


@dataclass
class _DeviceState:
    desired_replicas: int = 0
    last_nonzero_target: int = 0
    idle_since: float | None = None
    pool_status: dict[str, Any] = field(default_factory=dict)
    last_error: str | None = None
    last_reconcile_at: float | None = None
    retry_delay: float = 1.0
    retry_after: float = 0.0
    last_scale_up_attempt_at: float | None = None

    legacy_warm_target: int = 0
    legacy_warm_expires_at: float | None = None


ResizePool = Callable[[str, int], Awaitable[str]]
PoolStatus = Callable[[str], Awaitable[dict[str, Any]]]
ActivityCounts = Callable[[str], tuple[int, int]]


class FleetAutoscaler:
    def __init__(
        self,
        devices: list[DeviceConfig],
        *,
        resize_pool: ResizePool,
        pool_status: PoolStatus,
        activity_counts: ActivityCounts,
        clock: Callable[[], float] = time.monotonic,
        wall_clock: Callable[[], float] = time.time,
    ) -> None:
        self._devices = {
            device.name: device
            for device in devices
            if isinstance(device.launch, K8sLaunchConfig) and device.launch.fleet_autoscale.enabled
        }
        self._by_env_type: dict[str, list[DeviceConfig]] = {}
        for device in self._devices.values():
            self._by_env_type.setdefault(device.type, []).append(device)
        self._resize_pool = resize_pool
        self._pool_status = pool_status
        self._activity_counts = activity_counts
        self._clock = clock
        self._wall_clock = wall_clock
        self._lock = asyncio.Lock()
        self._wake = asyncio.Event()
        self._requests: dict[str, _FleetRequest] = {}
        self._by_job: dict[tuple[str, str], str] = {}
        self._session_to_request: dict[str, str] = {}
        self._unscoped_pending = {name: 0 for name in self._devices}
        self._device_state = {
            name: _DeviceState(desired_replicas=device.launch.fleet_autoscale.min_replicas)
            for name, device in self._devices.items()
            if isinstance(device.launch, K8sLaunchConfig)
        }

    @property
    def enabled(self) -> bool:
        return bool(self._devices)

    def capability(self) -> dict[str, Any] | None:
        if not self.enabled:
            return None
        configs = [
            device.launch.fleet_autoscale
            for device in self._devices.values()
            if isinstance(device.launch, K8sLaunchConfig)
        ]
        return {
            "version": 1,
            "admission": "full_or_409",
            "warm_policy": "fully_warm",
            "polling": True,
            "heartbeat_interval_seconds": min(
                config.heartbeat_interval_seconds for config in configs
            ),
            "request_ttl_seconds": min(config.request_ttl_seconds for config in configs),
            "warmup_timeout_seconds": max(config.warmup_timeout_seconds for config in configs),
        }

    def supports_env_type(self, env_type: str) -> bool:

        return env_type in self._by_env_type

    def _device_for_env_type(self, env_type: str) -> DeviceConfig:
        matches = self._by_env_type.get(env_type, [])
        if not matches:
            raise FleetRequestUnavailable(
                f"fleet autoscaling is not enabled for env_type={env_type!r}"
            )
        if len(matches) > 1:
            raise FleetRequestConflict(
                f"{len(matches)} autoscaled devices serve env_type={env_type!r}; "
                "request routing is ambiguous"
            )
        return matches[0]

    @staticmethod
    def _config(device: DeviceConfig):
        assert isinstance(device.launch, K8sLaunchConfig)
        return device.launch.fleet_autoscale

    def _expire_holders_locked(self, now: float) -> None:
        for request in self._requests.values():
            expired = [
                holder_id
                for holder_id, holder in request.holders.items()
                if holder.expires_at <= now
            ]
            for holder_id in expired:
                request.holders.pop(holder_id, None)
            if expired and not request.holders:
                request.inactive_reason = "expired"
                request.inactive_at = now

    def _reserved_locked(
        self,
        device_name: str,
        *,
        exclude_request_id: str | None = None,
    ) -> int:
        return sum(
            request.capacity
            for request in self._requests.values()
            if request.device_name == device_name
            and request.active
            and request.request_id != exclude_request_id
        )

    def _unreserved_usage_locked(
        self,
        device_name: str,
        *,
        capacity_override: tuple[str, int] | None = None,
    ) -> int:

        registered, pending = self._activity_counts(device_name)
        covered_registered = 0
        covered_pending = 0
        override_id = capacity_override[0] if capacity_override else None
        for request in self._requests.values():
            if request.device_name != device_name:
                continue
            if request.request_id == override_id:
                capacity = capacity_override[1]
                active = True
            else:
                capacity = request.capacity
                active = request.active
            if not active or capacity <= 0:
                continue
            attached = min(len(request.attached_sessions), capacity)
            covered_registered += attached
            covered_pending += min(
                request.pending_acquires,
                max(capacity - attached, 0),
            )
        unreserved_registered = max(registered - covered_registered, 0)
        callback_pending = max(pending - covered_pending, 0)

        unreserved_pending = max(
            callback_pending,
            self._unscoped_pending.get(device_name, 0),
        )
        return unreserved_registered + unreserved_pending

    def _refresh_legacy_warm_locked(
        self,
        device: DeviceConfig,
        now: float,
    ) -> None:

        state = self._device_state[device.name]
        if state.legacy_warm_target <= 0:
            return
        config = self._config(device)
        if self._unreserved_usage_locked(device.name) > 0:
            state.legacy_warm_expires_at = now + config.idle_timeout_seconds
            return
        if state.legacy_warm_expires_at is None:
            state.legacy_warm_expires_at = now + config.idle_timeout_seconds
            return
        if now < state.legacy_warm_expires_at:
            return
        state.legacy_warm_target = 0
        state.legacy_warm_expires_at = None

        state.idle_since = now - config.idle_timeout_seconds

    async def capacity_usage(self, device_name: str) -> dict[str, int]:

        async with self._lock:
            now = self._clock()
            self._expire_holders_locked(now)
            self._refresh_legacy_warm_locked(self._devices[device_name], now)
            reserved = self._reserved_locked(device_name)
            unreserved = self._unreserved_usage_locked(device_name)
            legacy_warm = self._device_state[device_name].legacy_warm_target
            return {
                "reserved": reserved,
                "unreserved": unreserved,
                "legacy_warm": legacy_warm,
                "required": reserved + unreserved + legacy_warm,
            }

    async def request_legacy_warm_capacity(
        self,
        env_type: str,
        capacity: int,
    ) -> dict[str, Any]:

        if capacity < 1:
            raise ValueError("legacy warm capacity must be >= 1")
        device = self._device_for_env_type(env_type)
        now = self._clock()
        config = self._config(device)
        async with self._lock:
            self._expire_holders_locked(now)
            self._refresh_legacy_warm_locked(device, now)
            state = self._device_state[device.name]
            candidate = max(state.legacy_warm_target, capacity)
            reserved = self._reserved_locked(device.name)
            unreserved = self._unreserved_usage_locked(device.name)
            required = reserved + unreserved + candidate
            if required > device.max_sessions:
                raise FleetCapacityExceeded(
                    requested=capacity,
                    available=max(
                        device.max_sessions - reserved - unreserved,
                        0,
                    ),
                    limit=device.max_sessions,
                    reserved=reserved,
                    active_unreserved=unreserved,
                    legacy_warm=state.legacy_warm_target,
                )
            state.legacy_warm_target = candidate
            state.legacy_warm_expires_at = now + config.idle_timeout_seconds
            state.last_nonzero_target = max(
                state.last_nonzero_target,
                reserved + candidate,
                config.min_replicas,
            )
            state.idle_since = None
            self._wake.set()
            target = reserved + candidate
            current = int(state.pool_status.get("desired_replicas") or 0)
            available = int(state.pool_status.get("available_replicas") or 0)
            warm_ready = current >= target and available >= target and state.last_error is None
            return {
                "state": "ready" if warm_ready else "warming",
                "compatibility_mode": "legacy_ensure_pool",
                "scale_request_accepted": True,
                "requested_warm_replicas": capacity,
                "legacy_warm_replicas": candidate,
                "aggregate_requested_capacity": reserved,
                "desired_warm_replicas": target,
                "current_sandboxset_replicas": current,
                "available_warm_replicas": available,
                "warm_ready": warm_ready,
                "capacity_limit": device.max_sessions,
                "poll_after_seconds": min(config.reconcile_interval_seconds, 2.0),
                "last_error": state.last_error,
            }

    async def upsert_request(
        self,
        *,
        job_id: str,
        holder_id: str,
        idempotency_key: str,
        env_type: str,
        capacity: int,
    ) -> dict[str, Any]:
        device = self._device_for_env_type(env_type)
        now = self._clock()
        config = self._config(device)
        async with self._lock:
            self._expire_holders_locked(now)
            self._refresh_legacy_warm_locked(device, now)
            key = (device.name, job_id)
            request_id = self._by_job.get(key)
            request = self._requests.get(request_id or "")
            if request is None:
                request = _FleetRequest(
                    request_id=uuid.uuid4().hex,
                    job_id=job_id,
                    env_type=env_type,
                    device_name=device.name,
                    created_at=now,
                )

            existing = request.holders.get(holder_id)
            if existing is not None and existing.idempotency_key != idempotency_key:
                raise FleetRequestConflict(
                    f"holder_id {holder_id!r} already exists with a different idempotency key"
                )
            candidate_capacity = max(
                capacity,
                max(
                    (
                        holder.capacity
                        for hid, holder in request.holders.items()
                        if hid != holder_id
                    ),
                    default=0,
                ),
            )
            reserved = self._reserved_locked(
                device.name,
                exclude_request_id=request.request_id,
            )
            unreserved = self._unreserved_usage_locked(
                device.name,
                capacity_override=(request.request_id, candidate_capacity),
            )
            state = self._device_state[device.name]
            legacy_warm = state.legacy_warm_target
            if reserved + candidate_capacity + unreserved + legacy_warm > device.max_sessions:
                raise FleetCapacityExceeded(
                    requested=candidate_capacity,
                    available=max(
                        device.max_sessions - reserved - unreserved - legacy_warm,
                        0,
                    ),
                    limit=device.max_sessions,
                    reserved=reserved,
                    active_unreserved=unreserved,
                    legacy_warm=legacy_warm,
                )

            if request_id is None or request.request_id not in self._requests:
                self._requests[request.request_id] = request
                self._by_job[key] = request.request_id
            if not request.active:
                request.created_at = now
                request.warmed_capacity = 0
                request.warmup_started_at = now
            elif candidate_capacity > max(request.capacity, request.warmed_capacity):
                request.warmup_started_at = now
            request.holders[holder_id] = _Holder(
                holder_id=holder_id,
                idempotency_key=idempotency_key,
                capacity=capacity,
                expires_at=now + config.request_ttl_seconds,
                last_heartbeat=now,
            )
            request.inactive_reason = None
            request.inactive_at = None
            self._wake.set()
            return self._request_snapshot_locked(request, now)

    async def heartbeat(self, request_id: str, holder_id: str) -> dict[str, Any]:
        now = self._clock()
        async with self._lock:
            self._expire_holders_locked(now)
            request = self._requests.get(request_id)
            if request is None:
                raise FleetRequestNotFound(request_id)
            holder = request.holders.get(holder_id)
            if holder is None:
                raise FleetRequestNotFound(
                    f"holder {holder_id!r} is not active on request {request_id}"
                )
            device = self._devices[request.device_name]
            holder.last_heartbeat = now
            holder.expires_at = now + self._config(device).request_ttl_seconds
            return self._request_snapshot_locked(request, now)

    async def release_holder(
        self,
        request_id: str,
        holder_id: str,
    ) -> dict[str, Any]:
        now = self._clock()
        async with self._lock:
            request = self._requests.get(request_id)
            if request is None:
                raise FleetRequestNotFound(request_id)
            request.holders.pop(holder_id, None)
            if not request.holders:
                request.inactive_reason = "released"
                request.inactive_at = now
            self._wake.set()
            return self._request_snapshot_locked(request, now)

    async def get_request(self, request_id: str) -> dict[str, Any]:
        now = self._clock()
        async with self._lock:
            self._expire_holders_locked(now)
            request = self._requests.get(request_id)
            if request is None:
                raise FleetRequestNotFound(request_id)
            return self._request_snapshot_locked(request, now)

    async def list_requests(self) -> list[dict[str, Any]]:
        now = self._clock()
        async with self._lock:
            self._expire_holders_locked(now)
            return [
                self._request_snapshot_locked(request, now)
                for request in sorted(self._requests.values(), key=lambda item: item.created_at)
            ]

    async def begin_acquire(self, request_id: str) -> str:
        now = self._clock()
        async with self._lock:
            self._expire_holders_locked(now)
            request = self._requests.get(request_id)
            if request is None:
                raise FleetRequestNotFound(request_id)
            if not request.active:
                raise FleetRequestUnavailable(f"fleet request {request_id} is no longer active")
            state = self._request_state_locked(request, now)
            if state != "ready":
                raise FleetRequestUnavailable(
                    f"fleet request {request_id} is {state}; wait for warmup"
                )
            in_use = len(request.attached_sessions) + request.pending_acquires
            if in_use >= request.capacity:
                raise FleetRequestUnavailable(
                    f"fleet request {request_id} has {in_use}/{request.capacity} "
                    "sessions active or pending"
                )
            request.pending_acquires += 1
            return request.request_id

    async def begin_unscoped_provision(
        self,
        env_type: str | None,
        capacity: int = 1,
    ) -> str | None:

        if env_type is None or env_type not in self._by_env_type:
            return None
        device = self._device_for_env_type(env_type)
        now = self._clock()
        async with self._lock:
            self._expire_holders_locked(now)
            usage = self._reserved_locked(device.name)
            usage += self._unreserved_usage_locked(device.name)
            if usage + capacity > device.max_sessions:
                raise FleetRequestUnavailable(
                    f"unscoped provision needs {capacity} slot(s), but "
                    f"{usage}/{device.max_sessions} are reserved or active"
                )
            self._unscoped_pending[device.name] += capacity
            return device.name

    async def complete_unscoped_provision(
        self,
        device_name: str | None,
        capacity: int = 1,
    ) -> None:
        if device_name is None:
            return
        async with self._lock:
            self._unscoped_pending[device_name] = max(
                self._unscoped_pending.get(device_name, 0) - capacity,
                0,
            )
            self._wake.set()

    def requested_capacity(self, device_name: str) -> int:

        return sum(
            request.capacity
            for request in self._requests.values()
            if request.device_name == device_name and request.active
        )

    def notify_config_changed(self) -> None:
        self._wake.set()

    async def complete_acquire(
        self,
        request_id: str,
        session_id: str | None,
    ) -> None:
        async with self._lock:
            request = self._requests.get(request_id)
            if request is None:
                return
            request.pending_acquires = max(request.pending_acquires - 1, 0)
            if session_id is not None:
                request.attached_sessions.add(session_id)
                self._session_to_request[session_id] = request_id
            self._wake.set()

    async def detach_session(self, session_id: str) -> None:
        async with self._lock:
            request_id = self._session_to_request.pop(session_id, None)
            if request_id is None:
                return
            request = self._requests.get(request_id)
            if request is not None:
                request.attached_sessions.discard(session_id)
            self._wake.set()

    def _request_state_locked(self, request: _FleetRequest, now: float) -> str:
        if not request.active:
            if request.attached_sessions or request.pending_acquires:
                return "draining"
            return "expired"
        if request.warmed_capacity >= request.capacity:
            return "ready"
        device_state = self._device_state[request.device_name]
        status = device_state.pool_status
        aggregate = self._reserved_locked(request.device_name)
        generation = int(status.get("generation") or 0)
        observed = int(status.get("observed_generation") or 0)
        desired = int(status.get("desired_replicas") or -1)
        available = int(status.get("available_replicas") or 0)
        if desired >= aggregate and observed >= generation and available >= aggregate:
            request.warmed_capacity = max(request.warmed_capacity, request.capacity)
            return "ready"
        config = self._config(self._devices[request.device_name])
        started = request.warmup_started_at
        if started is None:
            started = request.created_at
        if now - started >= config.warmup_timeout_seconds:
            return "failed"
        return "warming"

    def _request_snapshot_locked(
        self,
        request: _FleetRequest,
        now: float,
    ) -> dict[str, Any]:
        device = self._devices[request.device_name]
        config = self._config(device)
        device_state = self._device_state[request.device_name]
        status = device_state.pool_status
        request_state = self._request_state_locked(request, now)
        last_error = device_state.last_error
        if request_state == "failed":
            last_error = (
                f"warmup timeout after {config.warmup_timeout_seconds:g}s: "
                f"requested={request.capacity}, "
                f"aggregate={self._reserved_locked(request.device_name)}, "
                f"target={device_state.desired_replicas}, "
                f"current={status.get('desired_replicas', 0)}, "
                f"available={status.get('available_replicas', 0)}, "
                f"observed_generation={status.get('observed_generation', 0)}/"
                f"{status.get('generation', 0)}"
                + (f"; last reconcile error: {last_error}" if last_error else "")
            )
        return {
            "request_id": request.request_id,
            "job_id": request.job_id,
            "env_type": request.env_type,
            "device": request.device_name,
            "state": request_state,
            "warmed_capacity": request.warmed_capacity,
            "requested_capacity": request.capacity,
            "aggregate_requested_capacity": self._reserved_locked(request.device_name),
            "capacity_limit": device.max_sessions,
            "desired_warm_replicas": device_state.desired_replicas,
            "current_sandboxset_replicas": int(status.get("desired_replicas") or 0),
            "available_warm_replicas": int(status.get("available_replicas") or 0),
            "observed_generation": int(status.get("observed_generation") or 0),
            "generation": int(status.get("generation") or 0),
            "holders": len(request.holders),
            "attached_sessions": len(request.attached_sessions),
            "pending_acquires": request.pending_acquires,
            "inactive_reason": request.inactive_reason,
            "expires_in_seconds": (
                max(
                    max(holder.expires_at for holder in request.holders.values()) - now,
                    0.0,
                )
                if request.holders
                else 0.0
            ),
            "poll_after_seconds": min(config.reconcile_interval_seconds, 2.0),
            "heartbeat_interval_seconds": config.heartbeat_interval_seconds,
            "warmup_timeout_seconds": config.warmup_timeout_seconds,
            "last_error": last_error,
        }

    def device_summaries(self) -> dict[str, dict[str, Any]]:

        now = self._clock()
        out: dict[str, dict[str, Any]] = {}
        for device_name, device in self._devices.items():
            state = self._device_state[device_name]
            requested = sum(
                request.capacity
                for request in self._requests.values()
                if request.device_name == device_name and request.active
            )
            active_requests = sum(
                request.active
                for request in self._requests.values()
                if request.device_name == device_name
            )
            config = self._config(device)
            idle_remaining: float | None = None
            if state.idle_since is not None:
                idle_remaining = max(config.idle_timeout_seconds - (now - state.idle_since), 0.0)
            out[device_name] = {
                "enabled": True,
                "capacity_limit": device.max_sessions,
                "requested_capacity": requested,
                "desired_warm_replicas": state.desired_replicas,
                "current_sandboxset_replicas": int(state.pool_status.get("desired_replicas") or 0),
                "available_warm_replicas": int(state.pool_status.get("available_replicas") or 0),
                "active_requests": active_requests,
                "legacy_warm_replicas": state.legacy_warm_target,
                "legacy_warm_expires_at": (
                    self._wall_clock() + max(state.legacy_warm_expires_at - now, 0.0)
                    if state.legacy_warm_expires_at is not None
                    else None
                ),
                "unreserved_active_capacity": self._unreserved_usage_locked(device_name),
                "idle_scale_down_in_seconds": idle_remaining,
                "idle_scale_down_at": (
                    self._wall_clock() + idle_remaining if idle_remaining is not None else None
                ),
                "last_error": state.last_error,
            }
        return out

    def _reconcile_interval(self) -> float:
        if not self._devices:
            return 60.0
        return min(
            self._config(device).reconcile_interval_seconds for device in self._devices.values()
        )

    def _next_scale_up_target(
        self,
        device: DeviceConfig,
        current: int,
        now: float,
    ) -> int | None:

        config = self._config(device)
        state = self._device_state[device.name]
        if config.scale_up_batch_size == 0:
            return state.desired_replicas
        if current < 0:
            return None
        if (
            state.last_scale_up_attempt_at is not None
            and now - state.last_scale_up_attempt_at < config.scale_up_interval_seconds
        ):
            return None
        status = state.pool_status
        if int(status.get("observed_generation") or 0) < int(status.get("generation") or 0):
            return None
        pending = max(current - int(status.get("available_replicas") or 0), 0)
        limit = config.scale_up_max_pending or config.scale_up_batch_size
        increment = min(config.scale_up_batch_size, max(limit - pending, 0))
        if increment == 0:
            return None
        return min(state.desired_replicas, current + increment)

    async def _reconcile_device(self, device: DeviceConfig) -> None:
        now = self._clock()
        config = self._config(device)
        state = self._device_state[device.name]
        if now < state.retry_after:
            return
        status_ok = False
        try:
            status = await self._pool_status(device.name)
            state.pool_status = dict(status)
            status_ok = True
            if state.last_nonzero_target <= 0:
                current = int(status.get("desired_replicas") or 0)
                if current > 0:
                    state.last_nonzero_target = current
        except Exception as exc:
            state.last_error = f"{type(exc).__name__}: {exc}"

        async with self._lock:
            self._expire_holders_locked(now)
            self._refresh_legacy_warm_locked(device, now)
            requested = self._reserved_locked(device.name)
            legacy_warm = state.legacy_warm_target
            registered, pending = self._activity_counts(device.name)
            if requested > 0 or legacy_warm > 0:
                target = max(requested + legacy_warm, config.min_replicas)
                state.last_nonzero_target = target
                state.idle_since = None
            elif registered > 0 or pending > 0:
                target = max(state.last_nonzero_target, config.min_replicas)
                state.idle_since = None
            else:
                if state.idle_since is None:
                    state.idle_since = now
                if now - state.idle_since >= config.idle_timeout_seconds:
                    target = config.min_replicas
                else:
                    target = max(state.last_nonzero_target, config.min_replicas)
            state.desired_replicas = target

        raw_current = state.pool_status.get("desired_replicas")
        current = -1 if raw_current is None else int(raw_current)
        if status_ok and current == state.desired_replicas:
            state.last_error = None
            state.retry_delay = 1.0
            state.retry_after = 0.0
            state.last_reconcile_at = now
            return
        if not status_ok:
            state.retry_after = now + state.retry_delay
            state.retry_delay = min(state.retry_delay * 2.0, 60.0)
            state.last_reconcile_at = now
            return
        next_target = state.desired_replicas
        if next_target > current:
            paced_target = self._next_scale_up_target(device, current, now)
            if paced_target is None:
                state.last_reconcile_at = now
                return
            next_target = paced_target
            if config.scale_up_batch_size > 0:
                state.last_scale_up_attempt_at = now
        try:
            action = await self._resize_pool(device.name, next_target)
            logger.info(
                "[fleet-autoscale] device=%s replicas %d -> %d (%s; target=%d)",
                device.name,
                current,
                next_target,
                action,
                state.desired_replicas,
            )
            state.pool_status = dict(await self._pool_status(device.name))
            state.last_error = None
            state.retry_delay = 1.0
            state.retry_after = 0.0
        except Exception as exc:
            state.last_error = f"{type(exc).__name__}: {exc}"
            state.retry_after = now + state.retry_delay
            state.retry_delay = min(state.retry_delay * 2.0, 60.0)
            logger.warning(
                "[fleet-autoscale] reconcile failed device=%s target=%d: %s",
                device.name,
                state.desired_replicas,
                state.last_error,
            )
        state.last_reconcile_at = now

    async def reconcile_once(self) -> None:
        for device in self._devices.values():
            await self._reconcile_device(device)

            async with self._lock:
                now = self._clock()
                self._expire_holders_locked(now)
                for request in self._requests.values():
                    if request.device_name == device.name:
                        self._request_state_locked(request, now)

    async def run(self) -> None:
        if not self.enabled:
            return
        while True:
            try:
                self._wake.clear()
                await self.reconcile_once()
                with contextlib.suppress(TimeoutError):
                    await asyncio.wait_for(self._wake.wait(), timeout=self._reconcile_interval())
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("[fleet-autoscale] reconciliation iteration failed")
                await asyncio.sleep(self._reconcile_interval())
