from __future__ import annotations

import logging
import threading
import time
from typing import Any

import httpx

from mai_gym.envs.mobileworld.artifacts import (
    MobileWorldTaskArtifactStager,
)
from mai_gym.envs.mobileworld.patch import MobileWorldPodPatcher, discover_patch_bundle
from mai_gym.infra.launcher._helpers import _discover_device_id
from mai_gym.infra.launcher.k8s_base import K8sSandboxLauncherBase
from mai_gym.infra.task_artifacts import TaskArtifactSpec
from mai_gym.models.config import DeviceConfig, K8sLaunchConfig
from mai_gym.models.session import RunningBackend

logger = logging.getLogger(__name__)


_MW_DEFAULT_DEVICE_ID = "emulator-5554"


class MobileWorldK8sProfile(K8sSandboxLauncherBase):
    _ready_path = "/health"
    _ready_poll_interval = 3.0

    _ready_default_timeout = 600.0

    _ready_request_timeout = 95.0

    def __init__(self) -> None:
        super().__init__()
        self._patcher: MobileWorldPodPatcher | None = None
        self._patcher_key: tuple[str, str, float] | None = None
        self._patcher_lock = threading.Lock()
        self._patch_limit: tuple[int, threading.BoundedSemaphore] | None = None
        self._patch_limit_lock = threading.Lock()
        self._artifact_stager: MobileWorldTaskArtifactStager | None = None
        self._artifact_stager_key: tuple[str, float, int] | None = None
        self._artifact_stager_lock = threading.Lock()
        self._artifact_limit: tuple[int, threading.BoundedSemaphore] | None = None
        self._artifact_limit_lock = threading.Lock()
        self._artifact_locks: dict[tuple[str, str], threading.Lock] = {}
        self._artifact_locks_guard = threading.Lock()

    def _get_patcher(self, launch: K8sLaunchConfig) -> MobileWorldPodPatcher:
        config = launch.mobileworld_patch
        if config.directory is not None:
            with self._patcher_lock:
                bundle = discover_patch_bundle(config.directory)
                key = (config.directory, bundle.digest, config.restart_timeout_seconds)
                if self._patcher is None or self._patcher_key != key:
                    self._patcher = MobileWorldPodPatcher(
                        self._get_client(launch),
                        config,
                        bundle=bundle,
                    )
                    self._patcher_key = key
                    logger.info(
                        "[MobileWorldK8sProfile] discovered patch snapshot %s (patch_ids=%s)",
                        bundle.patch_id,
                        ",".join(bundle.source_patch_ids),
                    )
                return self._patcher
        assert config.bundle_path is not None
        assert config.bundle_sha256 is not None
        key = (
            config.bundle_path,
            config.bundle_sha256,
            config.restart_timeout_seconds,
        )
        if self._patcher is not None and self._patcher_key == key:
            return self._patcher
        with self._patcher_lock:
            if self._patcher is None or self._patcher_key != key:
                self._patcher = MobileWorldPodPatcher(
                    self._get_client(launch),
                    config,
                )
                self._patcher_key = key
        return self._patcher

    def _get_patch_limit(self, launch: K8sLaunchConfig) -> threading.BoundedSemaphore:
        limit = launch.mobileworld_patch.max_concurrent_patches
        current = self._patch_limit
        if current is not None and current[0] == limit:
            return current[1]
        with self._patch_limit_lock:
            current = self._patch_limit
            if current is None or current[0] != limit:
                current = (limit, threading.BoundedSemaphore(limit))
                self._patch_limit = current
        return current[1]

    def _get_artifact_stager(
        self,
        launch: K8sLaunchConfig,
    ) -> MobileWorldTaskArtifactStager:
        config = launch.mobileworld_task_artifacts
        if not config.enabled or not config.root_path:
            raise RuntimeError("MobileWorld task artifact staging is disabled for this device")
        key = (
            config.root_path,
            config.upload_timeout_seconds,
            config.chunk_size_bytes,
        )
        if self._artifact_stager is not None and self._artifact_stager_key == key:
            return self._artifact_stager
        with self._artifact_stager_lock:
            if self._artifact_stager is None or self._artifact_stager_key != key:
                self._artifact_stager = MobileWorldTaskArtifactStager(
                    self._get_client(launch),
                    config,
                )
                self._artifact_stager_key = key
        return self._artifact_stager

    def _get_artifact_limit(
        self,
        launch: K8sLaunchConfig,
    ) -> threading.BoundedSemaphore:
        limit = launch.mobileworld_task_artifacts.max_concurrent_uploads
        current = self._artifact_limit
        if current is not None and current[0] == limit:
            return current[1]
        with self._artifact_limit_lock:
            current = self._artifact_limit
            if current is None or current[0] != limit:
                current = (limit, threading.BoundedSemaphore(limit))
                self._artifact_limit = current
        return current[1]

    def _get_artifact_lock(
        self,
        pod_name: str,
        digest: str,
    ) -> threading.Lock:
        key = (pod_name, digest)
        with self._artifact_locks_guard:
            return self._artifact_locks.setdefault(key, threading.Lock())

    def prepare_task_artifacts(
        self,
        device: DeviceConfig,
        backend: RunningBackend,
        artifacts: tuple[TaskArtifactSpec, ...],
    ) -> list[dict[str, object]]:

        if not artifacts:
            return []
        launch = device.launch
        if not isinstance(launch, K8sLaunchConfig):
            raise RuntimeError("MobileWorld task artifacts require K8s launch config")
        namespace = str(backend.metadata.get("namespace") or "")
        pod_name = str(backend.metadata.get("pod_name") or "")
        if not namespace or not pod_name:
            raise RuntimeError(f"backend {backend.id} is missing Kubernetes pod metadata")
        stager = self._get_artifact_stager(launch)
        results: list[dict[str, object]] = []
        with self._get_artifact_limit(launch):
            for spec in artifacts:
                with self._get_artifact_lock(pod_name, spec.sha256):
                    results.append(stager.stage(namespace, pod_name, spec))
        return results

    def _prepare_backend(
        self,
        *,
        url: str,
        namespace: str,
        sandbox_name: str,
        launch: K8sLaunchConfig,
    ) -> dict[str, Any]:
        config = launch.mobileworld_patch
        if not config.enabled:
            return {}

        patcher = self._get_patcher(launch)
        logger.info(
            "[MobileWorldK8sProfile] patch %s start on %s/%s",
            patcher.patch_id,
            namespace,
            sandbox_name,
        )
        started = time.monotonic()
        try:
            with self._get_patch_limit(launch):
                result = patcher.apply(
                    namespace=namespace,
                    pod_name=sandbox_name,
                    url=url,
                )
        except Exception as exc:
            logger.error(
                "[MobileWorldK8sProfile] patch %s failed on %s/%s (error=%s duration=%.3fs)",
                patcher.patch_id,
                namespace,
                sandbox_name,
                type(exc).__name__,
                time.monotonic() - started,
            )
            raise
        logger.info(
            "[MobileWorldK8sProfile] patch %s complete on %s/%s (status=%s duration=%.3fs)",
            patcher.patch_id,
            namespace,
            sandbox_name,
            result["status"],
            result["duration_seconds"],
        )
        return {"mobileworld_patch": result}

    def _ready_predicate(self, response: httpx.Response) -> bool:

        return response.status_code == 200 and bool(response.json().get("ok", True))

    def _describe_backend(
        self,
        *,
        url: str,
        pod_ip: str,
        sandbox_name: str,
        namespace: str,
        launch: K8sLaunchConfig,
    ) -> tuple[str | None, dict[str, Any]]:

        device_id = _discover_device_id(url) or _MW_DEFAULT_DEVICE_ID
        metadata: dict[str, Any] = {
            "pod_ip": pod_ip,
            "pod_name": sandbox_name,
            "namespace": namespace,
        }
        return device_id, metadata
