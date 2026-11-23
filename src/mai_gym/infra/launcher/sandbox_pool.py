from __future__ import annotations

import logging
import threading
import time
import uuid
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from mai_gym.vendor.acs import AcsClient

logger = logging.getLogger(__name__)


CLAIM_ID_LABEL = "mai-gym/claim-id"


_CLAIM_POLL_INTERVAL = 0.5
_SET_VISIBILITY_TIMEOUT = 30.0
_DNS_LABEL_MAX_LEN = 63
_CLAIM_SUFFIX_LEN = len("-claim-") + 12


@dataclass
class PoolAcquisition:
    sandbox_name: str
    pod_ip: str
    claim_name: str


def _extract_pod_ip(sandbox: dict[str, Any]) -> str:

    status = sandbox.get("status") or {}
    pod_info = status.get("podInfo") or {}
    ip = pod_info.get("podIP") or status.get("sandboxIp")
    if ip:
        return str(ip)
    raise RuntimeError("pod_ip not ready")


class SandboxPool:
    def __init__(
        self,
        client: AcsClient,
        *,
        namespace: str,
        set_name: str,
        template_id: str,
        replicas: int,
        cpu: int,
        memory: int,
        os_type: str = "linux",
        claim_timeout: int = 300,
        claim_ttl: int = 900,
        create_on_no_stock: bool = True,
        env: dict[str, str] | None = None,
        reconcile_existing_replicas: bool = True,
    ) -> None:
        self._client = client
        self.namespace = namespace
        self.set_name = set_name
        self._template_id = template_id
        self._replicas = replicas
        self._cpu = cpu
        self._memory = memory
        self._os_type = os_type
        self._claim_timeout = max(int(claim_timeout), 1)
        self._claim_ttl = max(int(claim_ttl), 1)
        self._create_on_no_stock = create_on_no_stock
        self._env = dict(env or {})
        self._reconcile_existing_replicas = reconcile_existing_replicas
        self._ensured = False
        self._ensure_lock = threading.Lock()

    def ensure(self) -> None:

        if self._ensured:
            return
        with self._ensure_lock:
            if self._ensured:
                return
            result = self._client.ensure_sandbox_set(
                self.namespace,
                self.set_name,
                template_id=self._template_id,
                replicas=self._replicas,
                cpu=self._cpu,
                memory=self._memory,
                os_type=self._os_type,
                env=self._env,
                reconcile_existing=self._reconcile_existing_replicas,
            )
            logger.info(
                "[SandboxPool] SandboxSet %s/%s %s (replicas=%d)",
                self.namespace,
                self.set_name,
                result,
                self._replicas,
            )
            if self._env and result != "created":
                logger.warning(
                    "[SandboxPool] existing SandboxSet %s/%s was %s; pod env "
                    "is not reconciled in place. Use a fresh pool_name to apply "
                    "new bootstrap proxy env to every warm pod.",
                    self.namespace,
                    self.set_name,
                    result,
                )
            self._await_sandbox_set_visible()
            self._ensured = True

    def resize(self, replicas: int) -> str:

        replicas = int(replicas)
        if replicas < 0:
            raise ValueError("SandboxSet replicas must be >= 0")
        with self._ensure_lock:
            result = self._client.ensure_sandbox_set(
                self.namespace,
                self.set_name,
                template_id=self._template_id,
                replicas=replicas,
                cpu=self._cpu,
                memory=self._memory,
                os_type=self._os_type,
                env=self._env,
            )
            self._await_sandbox_set_visible()
            self._replicas = replicas
            self._ensured = True
        logger.info(
            "[SandboxPool] resized %s/%s to replicas=%d (%s)",
            self.namespace,
            self.set_name,
            replicas,
            result,
        )
        return result

    def status(self) -> dict[str, Any]:

        data = self._client.get_sandbox_set(self.namespace, self.set_name)
        metadata = data.get("metadata") or {}
        spec = data.get("spec") or {}
        status = data.get("status") or {}
        return {
            "generation": int(metadata.get("generation") or 0),
            "observed_generation": int(status.get("observedGeneration") or 0),
            "desired_replicas": int(spec.get("replicas") or 0),
            "replicas": int(status.get("replicas") or 0),
            "available_replicas": int(status.get("availableReplicas") or 0),
            "updated_replicas": int(status.get("updatedReplicas") or 0),
            "updated_available_replicas": int(status.get("updatedAvailableReplicas") or 0),
        }

    def _await_sandbox_set_visible(self) -> None:

        deadline = time.monotonic() + _SET_VISIBILITY_TIMEOUT
        last_error = ""
        while time.monotonic() < deadline:
            try:
                self._client.get_sandbox_set(self.namespace, self.set_name)
                return
            except Exception as exc:
                last_error = f"{type(exc).__name__}: {exc}"
                time.sleep(_CLAIM_POLL_INTERVAL)
        raise TimeoutError(
            f"SandboxSet {self.namespace}/{self.set_name} not visible after "
            f"{_SET_VISIBILITY_TIMEOUT:.0f}s (last={last_error})"
        )

    def acquire(self) -> PoolAcquisition:

        self.ensure()
        claim_id = uuid.uuid4().hex[:12]

        claim_prefix = self.set_name[: _DNS_LABEL_MAX_LEN - _CLAIM_SUFFIX_LEN]
        claim_prefix = claim_prefix.rstrip("-") or "mai-gym"
        claim_name = f"{claim_prefix}-claim-{claim_id}"
        self._client.create_sandbox_claim(
            self.namespace,
            claim_name,
            self.set_name,
            replicas=1,
            claim_timeout=f"{self._claim_timeout}s",
            ttl_after_completed=f"{self._claim_ttl}s",
            create_on_no_stock=self._create_on_no_stock,
            labels={CLAIM_ID_LABEL: claim_id},
        )
        try:
            sandbox = self._await_claim(claim_name, claim_id)
        except BaseException:
            self._safe_delete_claim(claim_name)
            raise

        sandbox_name = (sandbox.get("metadata") or {}).get("name") or ""
        pod_ip = _extract_pod_ip(sandbox)
        logger.info(
            "[SandboxPool] claimed %s/%s -> %s (claim=%s)",
            self.namespace,
            sandbox_name,
            pod_ip,
            claim_name,
        )
        return PoolAcquisition(sandbox_name=sandbox_name, pod_ip=pod_ip, claim_name=claim_name)

    def _await_claim(self, claim_name: str, claim_id: str) -> dict[str, Any]:
        deadline = time.time() + self._claim_timeout
        selectors = (
            f"{CLAIM_ID_LABEL}={claim_id}",
            f"agents.kruise.io/claim-name={claim_name}",
        )
        last = ""
        while time.time() < deadline:
            try:
                claim = self._client.get_sandbox_claim(self.namespace, claim_name)
            except Exception as exc:
                last = f"get-error:{type(exc).__name__}"
                time.sleep(_CLAIM_POLL_INTERVAL)
                continue
            status = claim.get("status") or {}
            phase = status.get("phase") or ""
            last = phase
            if phase == "Failed":
                raise RuntimeError(f"claim {claim_name} failed: {status.get('message') or phase}")
            claimed = status.get("claimedReplicas") or 0
            if phase == "Completed" and claimed < 1:
                raise RuntimeError(
                    f"claim {claim_name} completed with 0 replicas: "
                    f"{status.get('message') or status}"
                )
            if phase == "Completed" or claimed >= 1:
                for selector in selectors:
                    items = self._client.list_sandboxes(self.namespace, label_selector=selector)
                    for sandbox in items:
                        try:
                            _extract_pod_ip(sandbox)
                        except RuntimeError:
                            continue
                        return sandbox
            time.sleep(_CLAIM_POLL_INTERVAL)
        raise TimeoutError(
            f"claim {claim_name} not ready within {self._claim_timeout}s (last={last!r})"
        )

    def release(self, sandbox_name: str | None, claim_name: str | None) -> None:

        if sandbox_name:
            self._client.delete_sandbox(self.namespace, sandbox_name, wait=False)
        if claim_name:
            self._client.delete_sandbox_claim(self.namespace, claim_name, wait=False)

    def teardown(self) -> None:

        try:
            self._client.delete_sandbox_set(self.namespace, self.set_name, wait=False)
        except Exception as exc:
            logger.warning(
                "[SandboxPool] teardown: delete_sandbox_set %s/%s failed: %s",
                self.namespace,
                self.set_name,
                exc,
            )

    def _safe_delete_claim(self, claim_name: str) -> None:
        try:
            self._client.delete_sandbox_claim(self.namespace, claim_name, wait=False)
        except Exception as exc:
            logger.warning(
                "[SandboxPool] delete_sandbox_claim %s/%s failed: %s",
                self.namespace,
                claim_name,
                exc,
            )
