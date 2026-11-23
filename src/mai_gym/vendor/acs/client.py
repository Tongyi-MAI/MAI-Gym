from __future__ import annotations

import contextlib
import logging
import time
from dataclasses import dataclass
from typing import Any

import urllib3.exceptions
from kubernetes.client import ApiException
from kubernetes.stream import stream

from . import util

logger = logging.getLogger(__name__)

_SANDBOXSET_PATCH_ATTEMPTS = 3
_SANDBOXSET_PATCH_RETRY_DELAY = 0.25


@dataclass(frozen=True)
class PodExecResult:
    returncode: int
    stdout: str
    stderr: str


class AcsClient:
    def __init__(self, kubeconfig: str | None = None) -> None:
        self._kubeconfig = kubeconfig
        self._pod_api, self._sandbox_api, self._in_cluster = util.load_clients(kubeconfig)

    @staticmethod
    def _sandboxset_replicas(data: dict[str, Any]) -> int:
        return int((data.get("spec") or {}).get("replicas") or 0)

    @staticmethod
    def _retryable_sandboxset_patch_error(exc: Exception) -> bool:
        if isinstance(exc, AttributeError):
            return "decode" in str(exc) and "NoneType" in str(exc)
        if isinstance(exc, ApiException):
            status = int(exc.status or 0)
            return status in {401, 403, 408, 409, 429} or status >= 500
        return isinstance(exc, urllib3.exceptions.HTTPError)

    def _patch_sandboxset_replicas(
        self,
        sandbox_api: Any,
        *,
        namespace: str,
        name: str,
        replicas: int,
    ) -> None:

        last_error: Exception | None = None
        for attempt in range(1, _SANDBOXSET_PATCH_ATTEMPTS + 1):
            try:
                sandbox_api.patch_namespaced_custom_object(
                    group=util.SANDBOX_GROUP,
                    version=util.SANDBOX_VERSION,
                    namespace=namespace,
                    plural=util.SANDBOXSET_PLURAL,
                    name=name,
                    body={"spec": {"replicas": replicas}},
                )
            except Exception as exc:
                if not self._retryable_sandboxset_patch_error(exc):
                    raise
                last_error = exc
                logger.warning(
                    "SandboxSet patch attempt %d/%d returned an ambiguous "
                    "%s; verifying %s/%s replicas=%d",
                    attempt,
                    _SANDBOXSET_PATCH_ATTEMPTS,
                    type(exc).__name__,
                    namespace,
                    name,
                    replicas,
                )

            try:
                observed = sandbox_api.get_namespaced_custom_object(
                    group=util.SANDBOX_GROUP,
                    version=util.SANDBOX_VERSION,
                    namespace=namespace,
                    plural=util.SANDBOXSET_PLURAL,
                    name=name,
                )
                if self._sandboxset_replicas(observed) == replicas:
                    return
            except Exception as exc:
                last_error = exc

            if attempt < _SANDBOXSET_PATCH_ATTEMPTS:
                time.sleep(_SANDBOXSET_PATCH_RETRY_DELAY)

                _, sandbox_api, _ = util.load_clients(self._kubeconfig)

        raise RuntimeError(
            f"SandboxSet {namespace}/{name} did not reach replicas={replicas} "
            f"after {_SANDBOXSET_PATCH_ATTEMPTS} patch attempts"
        ) from last_error

    def create_sandbox(
        self,
        template_id: str,
        namespace: str,
        *,
        cpu: int = 2,
        memory: int = 4,
        os_type: str = "linux",
        owner_name: str | None = None,
        timeout: int = 600,
        env: dict[str, str] | None = None,
    ) -> dict[str, str]:
        template_id = template_id.strip()
        namespace = namespace.strip()
        if not template_id or not namespace:
            raise ValueError("template_id and namespace are required")

        os_type = util.os_type(os_type)
        mem = util.validate_resources(cpu, memory)

        try:
            self._pod_api.read_namespace(name=namespace)
        except ApiException as e:
            if e.status == 404:
                raise RuntimeError(f"namespace {namespace!r} does not exist") from e
            raise

        owner = (owner_name or "").strip()
        if not owner:
            if self._in_cluster:
                raise ValueError("in-cluster: owner_name is required")
            owner_ref = None
        else:
            owner_ref = util.owner_reference_for_pod(self._pod_api, namespace, owner)

        sandbox_name = util.allocate_sandbox_name(self._pod_api, self._sandbox_api, namespace)
        body = util.build_sandbox_body(
            sandbox_name=sandbox_name,
            template_id=template_id,
            cpu=cpu,
            memory=mem,
            os_type=os_type,
            owner_reference=owner_ref,
            env=env,
        )
        created = False
        try:
            self._sandbox_api.create_namespaced_custom_object(
                group=util.SANDBOX_GROUP,
                version=util.SANDBOX_VERSION,
                namespace=namespace,
                plural=util.SANDBOX_PLURAL,
                body=body,
            )
            created = True

            deadline = time.time() + timeout
            last_phase = ""
            while time.time() < deadline:
                try:
                    sandbox = self._read_sandbox(namespace, sandbox_name)
                except urllib3.exceptions.TimeoutError:
                    time.sleep(3)
                    continue
                except ApiException as e:
                    if e.status and e.status >= 500:
                        time.sleep(3)
                        continue
                    raise

                status = sandbox.get("status") or {}
                phase = status.get("phase") or ""
                last_phase = phase
                if phase in ("Failed", "Unknown"):
                    raise RuntimeError(f"sandbox {namespace}/{sandbox_name} phase={phase!r}")
                if phase != "Running":
                    time.sleep(3)
                    continue
                try:
                    ip = util.sandbox_pod_ip(sandbox)
                    return {"namespace": namespace, "name": sandbox_name, "ip": ip}
                except RuntimeError:
                    time.sleep(3)

            raise TimeoutError(
                f"sandbox {namespace}/{sandbox_name} not ready within {timeout}s "
                f"(phase={last_phase!r})"
            )
        except BaseException:
            if created:
                self._cleanup_sandbox(namespace, sandbox_name)
            raise

    def delete_sandbox(
        self,
        namespace: str,
        name: str,
        *,
        timeout: int = 600,
        wait: bool = True,
    ) -> None:

        try:
            self._sandbox_api.delete_namespaced_custom_object(
                group=util.SANDBOX_GROUP,
                version=util.SANDBOX_VERSION,
                namespace=namespace,
                plural=util.SANDBOX_PLURAL,
                name=name,
            )
        except ApiException as e:
            if e.status != 404:
                raise
        if not wait:
            return
        deadline = time.time() + timeout
        while time.time() < deadline:
            try:
                self._read_sandbox(namespace, name)
            except ApiException as e:
                if e.status == 404:
                    return
                raise
            time.sleep(2)
        raise TimeoutError(f"sandbox {namespace}/{name} not deleted within {timeout}s")

    def _cleanup_sandbox(self, namespace: str, name: str) -> None:

        with contextlib.suppress(ApiException):
            self.delete_sandbox(namespace, name, wait=False)

    def _read_sandbox(self, namespace: str, name: str) -> dict[str, Any]:
        return self._sandbox_api.get_namespaced_custom_object(
            group=util.SANDBOX_GROUP,
            version=util.SANDBOX_VERSION,
            namespace=namespace,
            plural=util.SANDBOX_PLURAL,
            name=name,
        )

    def exec_pod(
        self,
        namespace: str,
        pod_name: str,
        command: list[str],
        *,
        stdin_text: str | None = None,
        timeout: float = 180.0,
    ) -> PodExecResult:

        if not namespace.strip() or not pod_name.strip():
            raise ValueError("namespace and pod_name are required")
        if not command:
            raise ValueError("command is required")
        if timeout <= 0:
            raise ValueError("timeout must be positive")

        ws = stream(
            self._pod_api.connect_get_namespaced_pod_exec,
            pod_name,
            namespace,
            command=command,
            stderr=True,
            stdin=stdin_text is not None,
            stdout=True,
            tty=False,
            _preload_content=False,
        )
        try:
            if stdin_text is not None:
                ws.write_stdin(stdin_text)
            ws.run_forever(timeout=timeout)
            if ws.is_open():
                raise TimeoutError(f"pod exec {namespace}/{pod_name} timed out after {timeout}s")
            returncode = ws.returncode
            if returncode is None:
                raise RuntimeError(f"pod exec {namespace}/{pod_name} closed without exit status")
            return PodExecResult(
                returncode=int(returncode),
                stdout=ws.read_stdout(),
                stderr=ws.read_stderr(),
            )
        finally:
            ws.close()

    def ensure_sandbox_set(
        self,
        namespace: str,
        name: str,
        *,
        template_id: str,
        replicas: int,
        cpu: int = 2,
        memory: int = 4,
        os_type: str = "linux",
        env: dict[str, str] | None = None,
        reconcile_existing: bool = True,
    ) -> str:

        template_id = template_id.strip()
        namespace = namespace.strip()
        if not template_id or not namespace:
            raise ValueError("template_id and namespace are required")
        os_type = util.os_type(os_type)
        mem = util.validate_resources(cpu, memory)

        pod_api, sandbox_api, _ = util.load_clients(self._kubeconfig)

        try:
            pod_api.read_namespace(name=namespace)
        except ApiException as e:
            if e.status == 404:
                raise RuntimeError(f"namespace {namespace!r} does not exist") from e
            raise

        try:
            existing = sandbox_api.get_namespaced_custom_object(
                group=util.SANDBOX_GROUP,
                version=util.SANDBOX_VERSION,
                namespace=namespace,
                plural=util.SANDBOXSET_PLURAL,
                name=name,
            )
        except ApiException as e:
            if e.status != 404:
                raise
            existing = None

        if existing is None:
            body = util.build_sandboxset_body(
                name=name,
                template_id=template_id,
                replicas=replicas,
                cpu=cpu,
                memory=mem,
                os_type=os_type,
                env=env,
            )
            sandbox_api.create_namespaced_custom_object(
                group=util.SANDBOX_GROUP,
                version=util.SANDBOX_VERSION,
                namespace=namespace,
                plural=util.SANDBOXSET_PLURAL,
                body=body,
            )
            return "created"

        current = (existing.get("spec") or {}).get("replicas")
        if not reconcile_existing:
            return "adopted"
        if current != replicas:
            self._patch_sandboxset_replicas(
                sandbox_api,
                namespace=namespace,
                name=name,
                replicas=replicas,
            )
            return "scaled"
        return "adopted"

    def delete_sandbox_set(self, namespace: str, name: str, *, wait: bool = False) -> None:

        try:
            self._sandbox_api.delete_namespaced_custom_object(
                group=util.SANDBOX_GROUP,
                version=util.SANDBOX_VERSION,
                namespace=namespace,
                plural=util.SANDBOXSET_PLURAL,
                name=name,
            )
        except ApiException as e:
            if e.status != 404:
                raise

    def get_sandbox_set(self, namespace: str, name: str) -> dict[str, Any]:

        return self._sandbox_api.get_namespaced_custom_object(
            group=util.SANDBOX_GROUP,
            version=util.SANDBOX_VERSION,
            namespace=namespace,
            plural=util.SANDBOXSET_PLURAL,
            name=name,
        )

    def create_sandbox_claim(
        self,
        namespace: str,
        name: str,
        set_name: str,
        *,
        replicas: int = 1,
        claim_timeout: str = "300s",
        ttl_after_completed: str = "900s",
        create_on_no_stock: bool = True,
        labels: dict[str, str] | None = None,
    ) -> None:
        body = util.build_sandboxclaim_body(
            name=name,
            set_name=set_name,
            replicas=replicas,
            claim_timeout=claim_timeout,
            ttl_after_completed=ttl_after_completed,
            create_on_no_stock=create_on_no_stock,
            labels=labels or {},
        )
        self._sandbox_api.create_namespaced_custom_object(
            group=util.SANDBOX_GROUP,
            version=util.SANDBOX_VERSION,
            namespace=namespace,
            plural=util.SANDBOXCLAIM_PLURAL,
            body=body,
        )

    def get_sandbox_claim(self, namespace: str, name: str) -> dict[str, Any]:
        return self._sandbox_api.get_namespaced_custom_object(
            group=util.SANDBOX_GROUP,
            version=util.SANDBOX_VERSION,
            namespace=namespace,
            plural=util.SANDBOXCLAIM_PLURAL,
            name=name,
        )

    def delete_sandbox_claim(self, namespace: str, name: str, *, wait: bool = False) -> None:
        try:
            self._sandbox_api.delete_namespaced_custom_object(
                group=util.SANDBOX_GROUP,
                version=util.SANDBOX_VERSION,
                namespace=namespace,
                plural=util.SANDBOXCLAIM_PLURAL,
                name=name,
            )
        except ApiException as e:
            if e.status != 404:
                raise

    def list_sandboxes(
        self, namespace: str, *, label_selector: str | None = None
    ) -> list[dict[str, Any]]:
        resp = self._sandbox_api.list_namespaced_custom_object(
            group=util.SANDBOX_GROUP,
            version=util.SANDBOX_VERSION,
            namespace=namespace,
            plural=util.SANDBOX_PLURAL,
            label_selector=label_selector,
        )
        return resp.get("items", [])
