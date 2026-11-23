import logging
import threading
import time
import uuid

from mai_gym.models.session import (
    RunningBackend,
    SessionState,
    SessionStatus,
)
from mai_gym.models.session import (
    SessionTask as SessionTask,
)

logger = logging.getLogger(__name__)


_S = SessionStatus

_TRANSITIONS: dict[SessionStatus, set[SessionStatus]] = {
    _S.INITIALIZING: {_S.READY, _S.INVALID, _S.DESTROYING},
    _S.READY: {_S.BUSY, _S.INVALID, _S.DESTROYING},
    _S.BUSY: {_S.READY, _S.INVALID, _S.DESTROYING},
    _S.INVALID: {_S.RECOVERING, _S.DESTROYING},
    _S.RECOVERING: {_S.READY, _S.DESTROYING, _S.INVALID},
    _S.DESTROYING: {_S.INITIALIZING},
}


class SessionNotFoundError(KeyError):
    def __init__(self, session_id: str) -> None:
        self.session_id = session_id
        super().__init__(f"Session {session_id!r} not found")


class InvalidTransitionError(Exception):
    def __init__(self, session_id: str, current: SessionStatus, target: SessionStatus) -> None:
        super().__init__(
            f"Session {session_id!r}: transition {current!r} → {target!r} is not allowed"
        )
        self.session_id = session_id
        self.current = current
        self.target = target


class LeaseMismatchError(Exception):
    def __init__(self, session_id: str, presented: str, current: str | None) -> None:
        super().__init__(
            f"Session {session_id!r}: lease {presented[:8]!r} does not own this "
            f"session (current lease {(current[:8] if current else None)!r})"
        )
        self.session_id = session_id
        self.presented = presented
        self.current = current


class SessionManager:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._sessions: dict[str, SessionState] = {}

    def create_session(self, backend: RunningBackend) -> SessionState:

        session_id = str(uuid.uuid4())
        state = SessionState(session_id=session_id, backend=backend)
        with self._lock:
            self._sessions[session_id] = state

        namespace = backend.metadata.get("namespace")
        pod_name = backend.metadata.get("pod_name")
        logger.info(
            "Session %s created on backend %s (backend_id=%s namespace=%r pod_name=%r)",
            session_id,
            backend.device_name,
            backend.id,
            namespace[:253] if isinstance(namespace, str) else None,
            pod_name[:253] if isinstance(pod_name, str) else None,
        )
        return state

    def get_session(self, session_id: str) -> SessionState:

        with self._lock:
            state = self._sessions.get(session_id)
        if state is None:
            raise SessionNotFoundError(session_id)
        return state

    def transition(self, session_id: str, target: SessionStatus) -> None:

        with self._lock:
            state = self._sessions.get(session_id)
            if state is None:
                raise SessionNotFoundError(session_id)
            allowed = _TRANSITIONS.get(state.status, set())
            if target not in allowed:
                raise InvalidTransitionError(session_id, state.status, target)
            old = state.status
            state.status = target

            state.busy_since = time.monotonic() if target == SessionStatus.BUSY else None
            if target == SessionStatus.INVALID:
                state.epoch += 1
            elif target == SessionStatus.READY:
                state.force_relaunch = False
        logger.info("Session %s: %s → %s", session_id, old, target)

    def list_sessions(self, status_filter: SessionStatus | None = None) -> list[SessionState]:

        with self._lock:
            sessions = list(self._sessions.values())
        if status_filter is not None:
            sessions = [s for s in sessions if s.status == status_filter]
        return sessions

    def remove_session(self, session_id: str) -> None:

        with self._lock:
            if session_id not in self._sessions:
                raise SessionNotFoundError(session_id)
            del self._sessions[session_id]
        logger.info("Session %s removed", session_id)

    def set_lease(self, session_id: str, lease_id: str, expires_at: float) -> None:

        with self._lock:
            state = self._sessions.get(session_id)
            if state is None:
                raise SessionNotFoundError(session_id)
            state.lease_id = lease_id
            state.lease_expires_at = expires_at
            state.leased_since = time.time()

    def clear_lease(self, session_id: str, expected_lease_id: str) -> bool:

        with self._lock:
            state = self._sessions.get(session_id)
            if state is None or state.lease_id != expected_lease_id:
                return False
            state.lease_id = None
            state.lease_expires_at = None
            state.leased_since = None
            return True

    def force_clear_lease(self, session_id: str) -> bool:

        with self._lock:
            state = self._sessions.get(session_id)
            if state is None:
                raise SessionNotFoundError(session_id)
            had_lease = state.lease_id is not None
            state.lease_id = None
            state.lease_expires_at = None
            state.leased_since = None
            return had_lease

    def check_lease(self, session_id: str, lease_id: str | None) -> None:

        with self._lock:
            state = self._sessions.get(session_id)
            if state is None:
                raise SessionNotFoundError(session_id)
            if lease_id is not None and state.lease_id is not None and state.lease_id != lease_id:
                raise LeaseMismatchError(session_id, lease_id, state.lease_id)

    def extend_lease(self, session_id: str, ttl_s: float) -> None:

        with self._lock:
            state = self._sessions.get(session_id)
            if state is None or state.lease_id is None:
                return
            state.lease_expires_at = time.monotonic() + ttl_s

    def reap_stuck_busy(self, now: float, max_busy_s: float) -> list[SessionState]:

        reaped: list[SessionState] = []
        with self._lock:
            for state in self._sessions.values():
                if (
                    state.status == SessionStatus.BUSY
                    and state.busy_since is not None
                    and now - state.busy_since > max_busy_s
                ):
                    state.status = SessionStatus.INVALID
                    state.epoch += 1
                    state.busy_since = None
                    state.lease_id = None
                    state.lease_expires_at = None
                    state.leased_since = None
                    state.force_relaunch = True
                    state.last_op_failure_at = now
                    reaped.append(state)
        for state in reaped:
            logger.warning(
                "Session %s: reaped after BUSY > %.0fs — INVALID, lease cleared, relaunch forced",
                state.session_id,
                max_busy_s,
            )
        return reaped

    def clear_expired_leases(self, now: float) -> list[str]:

        cleared: list[str] = []
        with self._lock:
            for sid, state in self._sessions.items():
                if (
                    state.lease_id is not None
                    and state.lease_expires_at is not None
                    and state.lease_expires_at < now
                    and state.status != SessionStatus.BUSY
                ):
                    state.lease_id = None
                    state.lease_expires_at = None
                    state.leased_since = None
                    cleared.append(sid)
        return cleared
