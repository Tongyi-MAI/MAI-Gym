from __future__ import annotations

import os
import uuid
from typing import Any

from kubernetes import client, config
from kubernetes.client import ApiException, CustomObjectsApi, V1OwnerReference

SANDBOX_GROUP = "agents.kruise.io"
SANDBOX_VERSION = "v1alpha1"
SANDBOX_PLURAL = "sandboxes"
SANDBOXSET_PLURAL = "sandboxsets"
SANDBOXCLAIM_PLURAL = "sandboxclaims"


def os_type(value: str) -> str:
    s = (value or "").strip().lower()
    if s not in ("linux", "windows"):
        raise ValueError(f'os_type must be "linux" or "windows", got {value!r}')
    return s


def allocate_sandbox_name(
    pod_api: client.CoreV1Api,
    sandbox_api: CustomObjectsApi,
    namespace: str,
) -> str:

    prefix = "sandbox-"
    suffix_len = 10
    max_attempts = 5
    for _ in range(max_attempts):
        candidate = f"{prefix}{uuid.uuid4().hex[:suffix_len]}"
        try:
            sandbox_api.get_namespaced_custom_object(
                group=SANDBOX_GROUP,
                version=SANDBOX_VERSION,
                namespace=namespace,
                plural=SANDBOX_PLURAL,
                name=candidate,
            )
        except ApiException as e:
            if e.status != 404:
                raise
        else:
            continue
        try:
            pod_api.read_namespaced_pod(name=candidate, namespace=namespace)
        except ApiException as e:
            if e.status == 404:
                return candidate
            raise
    raise RuntimeError(
        f"could not allocate unique sandbox name in namespace {namespace!r} "
        f"after {max_attempts} attempts"
    )


def validate_resources(cpu: int, memory: int) -> str:
    if cpu < 2:
        raise ValueError(f"cpu must be >= 2, got {cpu}")
    if memory < 4:
        raise ValueError(f"memory must be >= 4 (GiB), got {memory}")
    return f"{memory}Gi"


def load_clients(
    kubeconfig: str | None = None,
) -> tuple[client.CoreV1Api, CustomObjectsApi, bool]:

    candidates: list[str] = []
    if kubeconfig:
        candidates.append(os.path.expanduser(kubeconfig))
    default_kube = os.path.expanduser("~/.kube/config")
    if default_kube not in candidates:
        candidates.append(default_kube)

    for path in candidates:
        if os.path.isfile(path):
            config.load_kube_config(config_file=path)
            return client.CoreV1Api(), client.CustomObjectsApi(), False

    config.load_incluster_config()
    return client.CoreV1Api(), client.CustomObjectsApi(), True


def owner_reference_for_pod(pod_api: client.CoreV1Api, namespace: str, name: str) -> dict[str, Any]:

    try:
        pod = pod_api.read_namespaced_pod(name=name, namespace=namespace)
    except ApiException as e:
        if e.status == 404:
            raise RuntimeError(f"owner pod {namespace}/{name} not found") from e
        raise
    phase = (pod.status.phase if pod.status else None) or ""
    if phase != "Running":
        raise RuntimeError(f"owner pod {namespace}/{name} is not Running (phase={phase!r})")

    meta = pod.metadata
    if not meta or not meta.name or not meta.uid:
        raise RuntimeError(f"owner pod {namespace}/{name} missing metadata name or uid")
    return owner_reference_to_dict(
        client.V1OwnerReference(
            api_version="v1",
            kind="Pod",
            name=meta.name,
            uid=meta.uid,
            controller=True,
            block_owner_deletion=True,
        )
    )


def owner_reference_to_dict(ref: V1OwnerReference) -> dict[str, Any]:
    return {
        "apiVersion": ref.api_version,
        "kind": ref.kind,
        "name": ref.name,
        "uid": ref.uid,
        "controller": ref.controller,
        "blockOwnerDeletion": ref.block_owner_deletion,
    }


def build_sandbox_body(
    *,
    sandbox_name: str,
    template_id: str,
    cpu: int,
    memory: str,
    os_type: str,
    owner_reference: dict[str, Any] | None = None,
    env: dict[str, str] | None = None,
) -> dict[str, Any]:
    ann = {"alibabacloud.com/acs-sandbox-template-id": template_id}
    labels = {
        "alibabacloud.com/acs": "true",
        "alibabacloud.com/compute-class": "agent-sandbox",
    }

    container: dict[str, Any] = {
        "name": "container",
        "image": "none",
        "resources": {
            "limits": {"cpu": str(cpu), "memory": memory},
            "requests": {"cpu": str(cpu), "memory": memory},
        },
    }
    if env:
        container["env"] = [{"name": key, "value": value} for key, value in sorted(env.items())]
    template_spec: dict[str, Any] = {"containers": [container]}
    if os_type == "windows":
        template_spec["nodeSelector"] = {"kubernetes.io/os": "windows"}

    metadata: dict[str, Any] = {"name": sandbox_name}
    if owner_reference is not None:
        metadata["ownerReferences"] = [owner_reference]

    return {
        "apiVersion": f"{SANDBOX_GROUP}/{SANDBOX_VERSION}",
        "kind": "Sandbox",
        "metadata": metadata,
        "spec": {
            "template": {
                "metadata": {
                    "annotations": ann,
                    "labels": labels,
                },
                "spec": template_spec,
            }
        },
    }


def build_sandboxset_body(
    *,
    name: str,
    template_id: str,
    replicas: int,
    cpu: int,
    memory: str,
    os_type: str,
    env: dict[str, str] | None = None,
) -> dict[str, Any]:

    ann = {"alibabacloud.com/acs-sandbox-template-id": template_id}
    labels = {
        "alibabacloud.com/acs": "true",
        "alibabacloud.com/compute-class": "agent-sandbox",
    }
    container: dict[str, Any] = {
        "name": "sandbox",
        "image": "none",
        "imagePullPolicy": "IfNotPresent",
        "resources": {
            "limits": {"cpu": str(cpu), "memory": memory},
            "requests": {"cpu": str(cpu), "memory": memory},
        },
    }
    if env:
        container["env"] = [{"name": key, "value": value} for key, value in sorted(env.items())]
    template_spec: dict[str, Any] = {"containers": [container]}
    if os_type == "windows":
        template_spec["nodeSelector"] = {"kubernetes.io/os": "windows"}

    return {
        "apiVersion": f"{SANDBOX_GROUP}/{SANDBOX_VERSION}",
        "kind": "SandboxSet",
        "metadata": {"name": name},
        "spec": {
            "replicas": replicas,
            "template": {
                "metadata": {"annotations": ann, "labels": labels},
                "spec": template_spec,
            },
        },
    }


def build_sandboxclaim_body(
    *,
    name: str,
    set_name: str,
    replicas: int,
    claim_timeout: str,
    ttl_after_completed: str,
    create_on_no_stock: bool,
    labels: dict[str, str],
) -> dict[str, Any]:

    return {
        "apiVersion": f"{SANDBOX_GROUP}/{SANDBOX_VERSION}",
        "kind": "SandboxClaim",
        "metadata": {"name": name},
        "spec": {
            "templateName": set_name,
            "replicas": replicas,
            "claimTimeout": claim_timeout,
            "ttlAfterCompleted": ttl_after_completed,
            "createOnNoStock": create_on_no_stock,
            "labels": labels,
        },
    }


def sandbox_pod_ip(sandbox: dict[str, Any]) -> str:

    status = sandbox.get("status") or {}
    pod_info = status.get("podInfo") or {}
    ip = pod_info.get("podIP") or status.get("sandboxIp")
    if ip:
        return str(ip)
    raise RuntimeError("pod_ip not ready")
