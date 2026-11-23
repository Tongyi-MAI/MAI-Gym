from __future__ import annotations

import asyncio
import logging
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass

from mai_gym.models.environment import matches_environment
from mai_gym.models.session import SessionState, SessionStatus
from mai_gym.server.session import InvalidTransitionError, SessionManager

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class AcquireResult:
    session_id: str
    backend_idx: str
    lease_id: str
    lease_expires_at: float


_SINGLE_DEVICE_ENV_TYPES = frozenset({"real_device", "mobileworld", "mobileworld_k8s"})


class SessionScheduler:
    DEFAULT_LEASE_TTL_S = 600.0
    DEFAULT_ACQUIRE_TIMEOUT_S = 30.0

    DEFAULT_FAILURE_COOLDOWN_S = 120.0

    def __init__(
        self,
        session_mgr: SessionManager,
        lease_ttl_s: float = DEFAULT_LEASE_TTL_S,
        failure_cooldown_s: float = DEFAULT_FAILURE_COOLDOWN_S,
    ) -> None:
        self._mgr = session_mgr
        self._cond = asyncio.Condition()
        self._lease_ttl_s = lease_ttl_s
        self._failure_cooldown_s = failure_cooldown_s

        self._acquire_seq: int = 0

    async def acquire(
        self,
        target_apps: list[str] | None,
        timeout_s: float = DEFAULT_ACQUIRE_TIMEOUT_S,
        env_type: str | None = None,
    ) -> AcquireResult | None:

        target = set(target_apps or [])
        deadline = time.monotonic() + timeout_s
        async with self._cond:
            woken = False
            while True:
                result = self._try_allocate(target, env_type)
                if result is not None:
                    return result
                if woken:
                    self._cond.notify(1)
                    woken = False
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return None
                try:
                    await asyncio.wait_for(self._cond.wait(), timeout=remaining)
                    woken = True
                except TimeoutError:
                    return None

    def _try_allocate(
        self, target_apps: set[str], env_type: str | None = None
    ) -> AcquireResult | None:

        sessions = self._mgr.list_sessions()
        leased = [s for s in sessions if s.lease_id is not None]
        leased_backends = {s.backend.id for s in leased}

        load_per_device: dict[str, int] = {}
        for s in leased:
            load_per_device[s.backend.device_name] = (
                load_per_device.get(s.backend.device_name, 0) + 1
            )

        devices_sorted = sorted({s.backend.device_name for s in sessions})
        n_devices = len(devices_sorted)
        rotation: dict[str, int] = (
            {name: (i - self._acquire_seq) % n_devices for i, name in enumerate(devices_sorted)}
            if n_devices
            else {}
        )

        candidates = [s for s in sessions if s.status == SessionStatus.READY and s.lease_id is None]

        now = time.monotonic()
        order = {s.session_id: i for i, s in enumerate(candidates)}
        n_cand = len(candidates) or 1
        candidates.sort(
            key=lambda s: (
                1
                if (
                    s.last_op_failure_at is not None
                    and now - s.last_op_failure_at < self._failure_cooldown_s
                )
                else 0,
                load_per_device.get(s.backend.device_name, 0),
                rotation.get(s.backend.device_name, 0),
                (order[s.session_id] - self._acquire_seq) % n_cand,
            )
        )

        for s in candidates:
            if env_type and not matches_environment(
                env_type,
                s.backend.env_type,
                infra_type=s.backend.metadata.get("infra_type"),
                profile=s.backend.metadata.get("environment_profile"),
            ):
                continue
            if s.backend.env_type in _SINGLE_DEVICE_ENV_TYPES and s.backend.id in leased_backends:
                continue
            lease_id = str(uuid.uuid4())
            expires = time.monotonic() + self._lease_ttl_s
            self._mgr.set_lease(s.session_id, lease_id, expires)
            self._acquire_seq += 1
            logger.info(
                "Acquired session %s on %s (lease=%s, target_apps=%s)",
                s.session_id,
                s.backend.device_name,
                lease_id[:8],
                sorted(target_apps),
            )
            return AcquireResult(
                session_id=s.session_id,
                backend_idx=s.backend.device_name,
                lease_id=lease_id,
                lease_expires_at=expires,
            )
        return None

    async def release(self, session_id: str, lease_id: str) -> bool:

        async with self._cond:
            ok = self._mgr.clear_lease(session_id, lease_id)
            if ok:
                logger.info("Released session %s (lease=%s)", session_id, lease_id[:8])
                self._cond.notify(1)
            return ok

    async def release_and_recycle(self, session_id: str, lease_id: str) -> bool:

        async with self._cond:
            ok = self._mgr.clear_lease(session_id, lease_id)
            if ok:
                try:
                    self._mgr.transition(session_id, SessionStatus.INVALID)
                except Exception as exc:
                    logger.warning(
                        "Recycle %s: status transition skipped: %s",
                        session_id,
                        exc,
                    )
                logger.info(
                    "Released session %s (lease=%s) for recycle",
                    session_id,
                    lease_id[:8],
                )

                self._cond.notify(1)
            return ok

    async def release_and_destroy(
        self,
        session_id: str,
        lease_id: str,
        *,
        reason: str = "",
    ) -> bool:

        async with self._cond:
            ok = self._mgr.clear_lease(session_id, lease_id)
            if ok:
                try:
                    self._mgr.transition(session_id, SessionStatus.DESTROYING)
                except Exception as exc:
                    logger.warning(
                        "Destroy %s: status transition skipped: %s",
                        session_id,
                        exc,
                    )
                logger.info(
                    "Released session %s (lease=%s) for destroy%s",
                    session_id,
                    lease_id[:8],
                    f": {reason}" if reason else "",
                )

                self._cond.notify(1)
            return ok

    async def discard(
        self,
        session_id: str,
        lease_id: str,
        reason: str = "",
    ) -> bool:

        async with self._cond:
            ok = self._mgr.clear_lease(session_id, lease_id)
            if ok:
                try:
                    self._mgr.transition(session_id, SessionStatus.INVALID)
                except Exception as exc:
                    logger.warning(
                        "Discard %s: status transition skipped: %s",
                        session_id,
                        exc,
                    )
                logger.warning(
                    "Discarded session %s (lease=%s): %s",
                    session_id,
                    lease_id[:8],
                    reason or "(no reason)",
                )
                self._cond.notify(1)
            return ok

    def extend_lease(self, session_id: str, ttl_s: float | None = None) -> None:

        ttl = ttl_s if ttl_s is not None else self._lease_ttl_s
        self._mgr.extend_lease(session_id, ttl)

    async def recover_sessions(self, session_ids: list[str]) -> int:

        async with self._cond:
            recovered = 0
            for sid in session_ids:
                try:
                    self._mgr.transition(sid, SessionStatus.RECOVERING)
                    self._mgr.transition(sid, SessionStatus.READY)
                    recovered += 1
                except (KeyError, InvalidTransitionError) as exc:
                    logger.warning("Recovery skipped %s: %s", sid, exc)
            if recovered:
                logger.info("Recovery promoted %d session(s) back to READY", recovered)
                self._cond.notify(recovered)
            return recovered

    async def sweep_expired(self) -> int:

        cleared, _ = await self.sweep_expired_details()
        return len(cleared)

    async def sweep_expired_details(
        self,
        destroy_predicate: Callable[[SessionState], bool] | None = None,
    ) -> tuple[list[str], list[str]]:

        async with self._cond:
            cleared = self._mgr.clear_expired_leases(time.monotonic())
            destroy_ids: list[str] = []
            if destroy_predicate is not None:
                for sid in cleared:
                    try:
                        state = self._mgr.get_session(sid)
                    except KeyError:
                        continue
                    if not destroy_predicate(state):
                        continue
                    try:
                        self._mgr.transition(sid, SessionStatus.DESTROYING)
                        destroy_ids.append(sid)
                    except Exception as exc:
                        logger.warning(
                            "Lease sweep destroy transition skipped %s: %s",
                            sid,
                            exc,
                        )
            if cleared:
                logger.warning(
                    "Lease sweep cleared %d expired session(s): %s",
                    len(cleared),
                    cleared,
                )
                reusable = len(cleared) - len(destroy_ids)
                if reusable:
                    self._cond.notify(reusable)
            return cleared, destroy_ids
