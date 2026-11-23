from __future__ import annotations

import abc
import logging
import re
import threading
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import httpx

from mai_gym.infra.launcher._base import Launcher
from mai_gym.infra.launcher._helpers import _make_backend_id
from mai_gym.infra.launcher.sandbox_pool import SandboxPool
from mai_gym.models.config import DeviceConfig, K8sLaunchConfig
from mai_gym.models.session import RunningBackend

if TYPE_CHECKING:
    from mai_gym.vendor.acs import AcsClient

logger = logging.getLogger(__name__)


@dataclass
class _BackendHandle:
    namespace: str
    sandbox_name: str
    claim_name: str | None = None
    pool: SandboxPool | None = None

    adopted: bool = False


@dataclass(frozen=True)
class _AdoptCandidate:
    sandbox_name: str
    pod_ip: str
    claim_name: str | None = None


def _safe_set_name(device_name: str) -> str:

    base = re.sub(r"[^a-z0-9-]", "-", f"mai-gym-{device_name}".lower())
    return base.strip("-")[:63].strip("-") or "mai-gym-pool"


def _sandbox_env(launch: K8sLaunchConfig) -> dict[str, str]:

    proxy = (launch.http_proxy or "").strip()
    if not launch.pod_bootstrap_enabled or not proxy:
        return {}
    env = {
        "http_proxy": proxy,
        "https_proxy": proxy,
        "HTTP_PROXY": proxy,
        "HTTPS_PROXY": proxy,
    }
    no_proxy = launch.no_proxy.strip()
    if no_proxy:
        env.update({"no_proxy": no_proxy, "NO_PROXY": no_proxy})
    return env


class K8sSandboxLauncherBase(Launcher):
    _ready_path: str = "/health"
    _ready_poll_interval: float = 3.0
    _ready_default_timeout: float = 300.0

    _ready_request_timeout: float = 5.0

    def __init__(self) -> None:
        self._default_ready_request_timeout = float(type(self)._ready_request_timeout)

        self._backends: dict[str, _BackendHandle] = {}

        self._cleanup_lock = threading.Lock()
        self._failed_cleanups: dict[tuple[str, str, str | None], _BackendHandle] = {}

        self._client: AcsClient | None = None
        self._client_lock = threading.Lock()

        self._launch_limits: dict[str, tuple[int, threading.BoundedSemaphore]] = {}
        self._launch_limit_lock = threading.Lock()

        self._pools: dict[str, SandboxPool] = {}
        self._pool_lock = threading.Lock()

        self._adopt_queue: deque[_AdoptCandidate] | None = None
        self._adopt_lock = threading.Lock()

        self._reserved_pods: set[tuple[str, str]] = set()

    @abc.abstractmethod
    def _ready_predicate(self, response: httpx.Response) -> bool:
        pass

    @abc.abstractmethod
    def _describe_backend(
        self,
        *,
        url: str,
        pod_ip: str,
        sandbox_name: str,
        namespace: str,
        launch: K8sLaunchConfig,
    ) -> tuple[str | None, dict[str, Any]]:
        pass

    def _prepare_backend(
        self,
        *,
        url: str,
        namespace: str,
        sandbox_name: str,
        launch: K8sLaunchConfig,
    ) -> dict[str, Any]:

        return {}

    @staticmethod
    def _configured_launch_limit(launch: K8sLaunchConfig) -> int:

        import os

        limit = max(1, int(launch.max_concurrent_launches))
        env_limit = int(os.environ.get("MAX_K8S_LAUNCH_CONCURRENCY", "0") or "0")
        return min(limit, env_limit) if env_limit > 0 else limit

    def _get_launch_limit(
        self, device: DeviceConfig, launch: K8sLaunchConfig
    ) -> threading.BoundedSemaphore:
        limit = self._configured_launch_limit(launch)
        current = self._launch_limits.get(device.name)
        if current is not None and current[0] == limit:
            return current[1]
        with self._launch_limit_lock:
            current = self._launch_limits.get(device.name)
            if current is None or current[0] != limit:
                current = (limit, threading.BoundedSemaphore(limit))
                self._launch_limits[device.name] = current
        return current[1]

    def _get_client(self, launch: K8sLaunchConfig) -> AcsClient:
        if self._client is not None:
            return self._client
        with self._client_lock:
            if self._client is None:
                try:
                    from mai_gym.vendor.acs import AcsClient
                except ImportError as exc:
                    raise RuntimeError(
                        f"{type(self).__name__} requires the `kubernetes` "
                        "package; install with `uv pip install kubernetes`."
                    ) from exc
                self._client = AcsClient(kubeconfig=launch.kubeconfig)
        return self._client

    def _get_pool(
        self, device: DeviceConfig, launch: K8sLaunchConfig, client: AcsClient
    ) -> SandboxPool:

        pool = self._pools.get(device.name)
        if pool is not None:
            return pool
        with self._pool_lock:
            pool = self._pools.get(device.name)
            if pool is None:
                if launch.fleet_autoscale.enabled:
                    size = launch.fleet_autoscale.min_replicas
                else:
                    size = launch.pool_size if launch.pool_size is not None else device.max_sessions
                pool = SandboxPool(
                    client,
                    namespace=launch.namespace,
                    set_name=launch.pool_name or _safe_set_name(device.name),
                    template_id=launch.template_id,
                    replicas=max(0, int(size)),
                    cpu=launch.cpu,
                    memory=launch.memory,
                    os_type=launch.os_type,
                    claim_timeout=launch.claim_timeout,
                    claim_ttl=launch.claim_ttl,
                    create_on_no_stock=launch.create_on_no_stock,
                    env=_sandbox_env(launch),
                    reconcile_existing_replicas=(not launch.fleet_autoscale.enabled),
                )
                pool.ensure()
                self._pools[device.name] = pool
        return pool

    def resize_pool(self, device: DeviceConfig, replicas: int) -> str:

        if not isinstance(device.launch, K8sLaunchConfig):
            raise TypeError("resize_pool requires K8sLaunchConfig")
        launch = device.launch
        if launch.acquisition != "pool":
            raise ValueError("resize_pool requires acquisition='pool'")
        replicas = int(replicas)
        if replicas < 0:
            raise ValueError("SandboxSet replicas must be >= 0")

        client = self._get_client(launch)
        pool = self._pools.get(device.name)
        if pool is None:
            old_size = launch.pool_size
            launch.pool_size = replicas
            try:
                self._get_pool(device, launch, client)
            except BaseException:
                launch.pool_size = old_size
                raise
            return "ensured"

        result = pool.resize(replicas)
        launch.pool_size = replicas
        return result

    def pool_status(self, device: DeviceConfig) -> dict[str, Any]:

        if not isinstance(device.launch, K8sLaunchConfig):
            raise TypeError("pool_status requires K8sLaunchConfig")
        launch = device.launch
        if launch.acquisition != "pool":
            raise ValueError("pool_status requires acquisition='pool'")
        client = self._get_client(launch)
        pool = self._get_pool(device, launch, client)
        return pool.status()

    def _acquire_backend(
        self, device: DeviceConfig, launch: K8sLaunchConfig, client: AcsClient
    ) -> tuple[_BackendHandle, str]:

        if launch.acquisition == "pool":
            pool = self._get_pool(device, launch, client)
            if launch.pool_reuse_existing:
                adopted = self._try_pop_adopted(
                    launch,
                    client,
                    expected_pool=pool.set_name,
                )
                if adopted is not None:
                    if adopted.claim_name:
                        return _BackendHandle(
                            namespace=launch.namespace,
                            sandbox_name=adopted.sandbox_name,
                            claim_name=adopted.claim_name,
                            pool=pool,
                        ), adopted.pod_ip
                    return _BackendHandle(
                        namespace=launch.namespace,
                        sandbox_name=adopted.sandbox_name,
                        adopted=True,
                    ), adopted.pod_ip
            acq = pool.acquire()
            handle = _BackendHandle(
                namespace=pool.namespace,
                sandbox_name=acq.sandbox_name,
                claim_name=acq.claim_name,
                pool=pool,
            )
            return handle, acq.pod_ip
        if launch.acquisition == "adopt":
            adopted = self._pop_adopted(launch, client)
            handle = _BackendHandle(
                namespace=launch.namespace,
                sandbox_name=adopted.sandbox_name,
                adopted=True,
            )
            return handle, adopted.pod_ip
        result = client.create_sandbox(
            template_id=launch.template_id,
            namespace=launch.namespace,
            cpu=launch.cpu,
            memory=launch.memory,
            os_type=launch.os_type,
            owner_name=launch.owner_name,
            timeout=launch.launch_timeout,
            env=_sandbox_env(launch),
        )
        handle = _BackendHandle(namespace=result["namespace"], sandbox_name=result["name"])
        return handle, result["ip"]

    def _acquire_replacement(
        self, device: DeviceConfig, launch: K8sLaunchConfig, client: AcsClient
    ) -> tuple[_BackendHandle, str]:

        if launch.acquisition != "adopt":
            return self._acquire_backend(device, launch, client)
        spare = self._next_spare_adopt(launch, client)
        if spare is not None:
            name, pod_ip = spare
            return _BackendHandle(
                namespace=launch.namespace,
                sandbox_name=name,
                adopted=True,
            ), pod_ip
        logger.info(
            "[%s] relaunch: no spare adopt pod in %s — cold-creating a replacement",
            type(self).__name__,
            launch.namespace,
        )
        result = client.create_sandbox(
            template_id=launch.template_id,
            namespace=launch.namespace,
            cpu=launch.cpu,
            memory=launch.memory,
            os_type=launch.os_type,
            owner_name=launch.owner_name,
            timeout=launch.launch_timeout,
            env=_sandbox_env(launch),
        )

        self._reserve_pod(result["namespace"], result["name"])
        return _BackendHandle(
            namespace=result["namespace"],
            sandbox_name=result["name"],
        ), result["ip"]

    def _pop_adopted(self, launch: K8sLaunchConfig, client: AcsClient) -> _AdoptCandidate:

        skipped = 0
        while True:
            adopted = self._try_pop_adopted(launch, client)
            if adopted is None:
                raise RuntimeError(
                    "adopt: no more running sandboxes to adopt in namespace "
                    f"{launch.namespace!r}" + (f" ({skipped} unhealthy skipped)" if skipped else "")
                )
            if self._probe_spare_ready(adopted.pod_ip, launch):
                return adopted
            skipped += 1
            if skipped == 1 or skipped % 25 == 0:
                logger.warning(
                    "[%s] adopt: skipped %d unhealthy pod(s) while acquiring a warm sandbox in %s",
                    type(self).__name__,
                    skipped,
                    launch.namespace,
                )

    def _try_pop_adopted(
        self,
        launch: K8sLaunchConfig,
        client: AcsClient,
        *,
        expected_pool: str | None = None,
    ) -> _AdoptCandidate | None:

        with self._adopt_lock:
            if self._adopt_queue is None:
                self._adopt_queue = self._build_adopt_queue(
                    launch,
                    client,
                    expected_pool=expected_pool,
                )
            while self._adopt_queue:
                candidate = self._adopt_queue.popleft()
                with self._cleanup_lock:
                    dirty = any(
                        h.namespace == launch.namespace and h.sandbox_name == candidate.sandbox_name
                        for h in self._failed_cleanups.values()
                    )
                if not dirty:
                    return candidate
            return None

    def _build_adopt_queue(
        self,
        launch: K8sLaunchConfig,
        client: AcsClient,
        *,
        expected_pool: str | None = None,
    ) -> deque[_AdoptCandidate]:
        from mai_gym.vendor.acs import util as acs_util

        items = client.list_sandboxes(launch.namespace)
        out: deque[_AdoptCandidate] = deque()
        skipped = 0
        expected_pool = expected_pool or launch.pool_name
        for sb in items:
            metadata = sb.get("metadata") or {}
            labels = metadata.get("labels") or {}
            name = metadata.get("name")
            phase = (sb.get("status") or {}).get("phase")

            if not name or phase != "Running" or metadata.get("ownerReferences"):
                skipped += 1
                continue
            pool_name = labels.get("agents.kruise.io/sandbox-pool")
            claim_name = labels.get("agents.kruise.io/claim-name")
            claimed = str(labels.get("agents.kruise.io/sandbox-claimed") or "").lower() == "true"
            if launch.acquisition == "pool" and pool_name:
                if (
                    expected_pool is None
                    or pool_name != expected_pool
                    or not claimed
                    or not claim_name
                ):
                    skipped += 1
                    continue
            elif pool_name or claimed or claim_name:
                skipped += 1
                continue
            try:
                pod_ip = acs_util.sandbox_pod_ip(sb)
            except Exception:
                skipped += 1
                continue
            out.append(
                _AdoptCandidate(
                    sandbox_name=name,
                    pod_ip=pod_ip,
                    claim_name=claim_name,
                )
            )
        logger.info(
            "[%s] adopt: queued %d running sandbox(es) in namespace %s "
            "(%d skipped: not Running / no pod IP)",
            type(self).__name__,
            len(out),
            launch.namespace,
            skipped,
        )
        return out

    def _next_spare_adopt(
        self, launch: K8sLaunchConfig, client: AcsClient
    ) -> tuple[str, str] | None:

        from mai_gym.vendor.acs import util as acs_util

        listing = client.list_sandboxes(launch.namespace)
        dead: set[str] = set()
        dead_probes = 0
        while True:
            claimed: tuple[str, str] | None = None
            with self._adopt_lock:
                in_use = {h.sandbox_name for h in self._backends.values()}
                with self._cleanup_lock:
                    in_use.update(
                        h.sandbox_name
                        for h in self._failed_cleanups.values()
                        if h.namespace == launch.namespace
                    )
                for sb in listing:
                    metadata = sb.get("metadata") or {}
                    labels = metadata.get("labels") or {}
                    name = metadata.get("name")
                    phase = (sb.get("status") or {}).get("phase")
                    if (
                        not name
                        or name in in_use
                        or name in dead
                        or (launch.namespace, name) in self._reserved_pods
                        or phase != "Running"
                        or metadata.get("ownerReferences")
                        or labels.get("agents.kruise.io/sandbox-pool")
                        or labels.get("agents.kruise.io/claim-name")
                    ):
                        continue
                    try:
                        pod_ip = acs_util.sandbox_pod_ip(sb)
                    except Exception:
                        continue
                    self._reserved_pods.add((launch.namespace, name))
                    claimed = (name, pod_ip)
                    break
            if claimed is None:
                return None
            name, pod_ip = claimed

            if self._probe_spare_ready(pod_ip, launch):
                return name, pod_ip
            self._unreserve_pod(launch.namespace, name)
            dead.add(name)
            dead_probes += 1
            if dead_probes >= self._max_spare_probes:
                logger.warning(
                    "[%s] no healthy spare adopt pod after %d failed "
                    "probes — falling through to cold create",
                    type(self).__name__,
                    dead_probes,
                )
                return None

    def _reserve_pod(self, namespace: str, sandbox_name: str) -> None:

        with self._adopt_lock:
            self._reserved_pods.add((namespace, sandbox_name))

    def _unreserve_pod(self, namespace: str, sandbox_name: str) -> None:

        with self._adopt_lock:
            self._reserved_pods.discard((namespace, sandbox_name))

    _spare_probe_timeout: float = 5.0
    _max_spare_probes: int = 40

    def _probe_spare_ready(self, pod_ip: str, launch: K8sLaunchConfig) -> bool:

        url = f"http://{pod_ip}:{launch.service_port}{self._ready_path}"
        try:
            with httpx.Client(timeout=self._spare_probe_timeout) as client:
                r = client.get(url)
        except httpx.HTTPError:
            return False
        try:
            return self._ready_predicate(r)
        except Exception:
            return False

    def _release_handle(self, handle: _BackendHandle) -> None:

        key = (handle.namespace, handle.sandbox_name, handle.claim_name)
        try:
            if handle.pool is not None:
                handle.pool.release(handle.sandbox_name, handle.claim_name)
            else:
                if self._client is None:
                    raise RuntimeError(
                        f"Cannot release sandbox {handle.namespace}/{handle.sandbox_name}: "
                        "ACS client is not initialized"
                    )
                self._client.delete_sandbox(handle.namespace, handle.sandbox_name, wait=False)
        except Exception:
            with self._cleanup_lock:
                self._failed_cleanups[key] = handle
            raise
        else:
            with self._cleanup_lock:
                self._failed_cleanups.pop(key, None)

    def retry_failed_cleanup(self) -> None:

        with self._cleanup_lock:
            pending = list(self._failed_cleanups.values())
        for handle in pending:
            try:
                self._release_handle(handle)
            except Exception:
                logger.exception(
                    "[%s] retained cleanup still pending for %s/%s",
                    type(self).__name__,
                    handle.namespace,
                    handle.sandbox_name,
                )

    def _discard_handle(self, handle: _BackendHandle) -> None:

        self._release_handle(handle)

    def launch(self, device: DeviceConfig) -> RunningBackend:

        assert isinstance(device.launch, K8sLaunchConfig)
        with self._get_launch_limit(device, device.launch):
            return self._launch_unlimited(device)

    def _launch_unlimited(self, device: DeviceConfig) -> RunningBackend:
        self.retry_failed_cleanup()
        assert isinstance(device.launch, K8sLaunchConfig), (
            f"{type(self).__name__} only handles K8sLaunchConfig, got "
            f"{type(device.launch).__name__}"
        )
        launch = device.launch
        self._ready_request_timeout = (
            float(launch.ready_request_timeout)
            if launch.ready_request_timeout is not None
            else self._default_ready_request_timeout
        )
        client = self._get_client(launch)

        backend_id = _make_backend_id()
        try:
            handle, pod_ip = self._acquire_backend(device, launch, client)
        except Exception as exc:
            logger.error(
                "[%s] acquire (%s) failed for device %s: %s",
                type(self).__name__,
                launch.acquisition,
                device.name,
                exc,
            )
            raise

        namespace = handle.namespace
        sandbox_name = handle.sandbox_name
        url = f"http://{pod_ip}:{launch.service_port}"
        self._backends[backend_id] = handle

        if not self._wait_for_service(url, timeout=launch.launch_timeout):
            try:
                self._release_handle(handle)
            except Exception as exc:
                logger.warning(
                    "[%s] cleanup-on-fail release %s/%s failed: %s",
                    type(self).__name__,
                    namespace,
                    sandbox_name,
                    exc,
                )
            self._backends.pop(backend_id, None)
            raise RuntimeError(
                f"{type(self).__name__} service at {url} (pod "
                f"{namespace}/{sandbox_name}) did not become ready on "
                f"{self._ready_path} within {launch.launch_timeout}s"
            )

        try:
            bootstrap_metadata = self._prepare_backend(
                url=url,
                namespace=namespace,
                sandbox_name=sandbox_name,
                launch=launch,
            )
        except Exception:
            try:
                self._discard_handle(handle)
            except Exception as cleanup_exc:
                logger.warning(
                    "[%s] cleanup after prepare failure for %s/%s failed: %s",
                    type(self).__name__,
                    namespace,
                    sandbox_name,
                    cleanup_exc,
                )
            self._backends.pop(backend_id, None)
            raise

        device_id, metadata = self._describe_backend(
            url=url,
            pod_ip=pod_ip,
            sandbox_name=sandbox_name,
            namespace=namespace,
            launch=launch,
        )
        metadata.update(bootstrap_metadata)
        logger.info(
            "[%s] Backend %s ready at %s (pod=%s/%s, device_id=%s)",
            type(self).__name__,
            backend_id,
            url,
            namespace,
            sandbox_name,
            device_id,
        )
        return RunningBackend(
            id=backend_id,
            url=url,
            device_name=device.name,
            env_type=device.type,
            device_id=device_id,
            metadata=metadata,
        )

    def _wait_for_service(self, url: str, *, timeout: float) -> bool:

        deadline = time.time() + max(timeout, self._ready_default_timeout)
        last_err: str | None = None
        with httpx.Client(timeout=self._ready_request_timeout) as client:
            while time.time() < deadline:
                try:
                    r = client.get(f"{url}{self._ready_path}")
                except httpx.HTTPError as exc:
                    last_err = type(exc).__name__
                else:
                    try:
                        if self._ready_predicate(r):
                            return True
                        last_err = f"HTTP {r.status_code}"
                    except Exception as exc:
                        last_err = f"{type(exc).__name__} on HTTP {r.status_code}"
                time.sleep(self._ready_poll_interval)
        logger.warning(
            "[%s] %s%s not ready after %.0fs (last=%s)",
            type(self).__name__,
            url,
            self._ready_path,
            timeout,
            last_err,
        )
        return False

    def relaunch(self, device: DeviceConfig, backend: RunningBackend) -> bool:

        assert isinstance(device.launch, K8sLaunchConfig)
        with self._get_launch_limit(device, device.launch):
            return self._relaunch_unlimited(device, backend)

    def _relaunch_unlimited(self, device: DeviceConfig, backend: RunningBackend) -> bool:

        self.retry_failed_cleanup()
        assert isinstance(device.launch, K8sLaunchConfig), (
            f"{type(self).__name__} only handles K8sLaunchConfig, got "
            f"{type(device.launch).__name__}"
        )
        launch = device.launch
        self._ready_request_timeout = (
            float(launch.ready_request_timeout)
            if launch.ready_request_timeout is not None
            else self._default_ready_request_timeout
        )
        client = self._get_client(launch)
        old_handle = self._backends.get(backend.id)

        try:
            new_handle, pod_ip = self._acquire_replacement(device, launch, client)
        except Exception as exc:
            logger.warning(
                "[%s] relaunch: could not acquire a replacement pod for %s: %s",
                type(self).__name__,
                backend.id,
                exc,
            )
            return False

        try:
            new_url = f"http://{pod_ip}:{launch.service_port}"
            if not self._wait_for_service(new_url, timeout=launch.launch_timeout):
                try:
                    self._release_handle(new_handle)
                except Exception as exc:
                    logger.warning(
                        "[%s] relaunch: cleanup of unready replacement %s/%s failed: %s",
                        type(self).__name__,
                        new_handle.namespace,
                        new_handle.sandbox_name,
                        exc,
                    )
                return False

            try:
                bootstrap_metadata = self._prepare_backend(
                    url=new_url,
                    namespace=new_handle.namespace,
                    sandbox_name=new_handle.sandbox_name,
                    launch=launch,
                )
            except Exception as exc:
                logger.warning(
                    "[%s] relaunch: replacement bootstrap failed for %s/%s: %s",
                    type(self).__name__,
                    new_handle.namespace,
                    new_handle.sandbox_name,
                    type(exc).__name__,
                )
                try:
                    self._discard_handle(new_handle)
                except Exception as cleanup_exc:
                    logger.warning(
                        "[%s] relaunch: cleanup after bootstrap failure for %s/%s failed: %s",
                        type(self).__name__,
                        new_handle.namespace,
                        new_handle.sandbox_name,
                        cleanup_exc,
                    )
                return False

            device_id, metadata = self._describe_backend(
                url=new_url,
                pod_ip=pod_ip,
                sandbox_name=new_handle.sandbox_name,
                namespace=new_handle.namespace,
                launch=launch,
            )
            metadata.update(bootstrap_metadata)

            backend.url = new_url
            backend.device_id = device_id
            backend.metadata = metadata
            self._backends[backend.id] = new_handle

            if old_handle is not None:
                try:
                    self._discard_handle(old_handle)
                except Exception as exc:
                    logger.warning(
                        "[%s] relaunch: failed to tear down dead pod %s/%s: %s",
                        type(self).__name__,
                        old_handle.namespace,
                        old_handle.sandbox_name,
                        exc,
                    )
            logger.info(
                "[%s] relaunch: backend %s swapped onto fresh pod %s/%s at %s",
                type(self).__name__,
                backend.id,
                new_handle.namespace,
                new_handle.sandbox_name,
                new_url,
            )
            return True
        finally:
            self._unreserve_pod(new_handle.namespace, new_handle.sandbox_name)

    def batch_launch(self, device: DeviceConfig, count: int) -> list[RunningBackend]:

        assert isinstance(device.launch, K8sLaunchConfig)

        self._get_client(device.launch)

        if count <= 0:
            return []

        concurrency = min(count, self._configured_launch_limit(device.launch))
        logger.info(
            "[%s] %s: launching %d pod(s), max %d concurrent",
            type(self).__name__,
            device.name,
            count,
            concurrency,
        )

        backends: list[RunningBackend] = []
        errors: list[BaseException] = []
        with ThreadPoolExecutor(max_workers=concurrency) as pool:
            futures = [pool.submit(self.launch, device) for _ in range(count)]
            for fut in as_completed(futures):
                try:
                    backends.append(fut.result())
                except BaseException as exc:
                    errors.append(exc)
                    logger.warning(
                        "[%s] one pod launch failed: %s",
                        type(self).__name__,
                        exc,
                    )
        if errors:
            logger.warning(
                "[%s] %s: %d/%d pod launches failed",
                type(self).__name__,
                device.name,
                len(errors),
                count,
            )
        return backends

    def shutdown(self, backend: RunningBackend) -> None:
        handle = self._backends.get(backend.id)
        if handle is None:
            return

        try:
            self._release_handle(handle)
            if self._backends.get(backend.id) is handle:
                self._backends.pop(backend.id, None)
            logger.info(
                "[%s] Released sandbox %s/%s (backend=%s)",
                type(self).__name__,
                handle.namespace,
                handle.sandbox_name,
                backend.id,
            )
        except Exception as exc:
            logger.warning(
                "[%s] release %s/%s failed (backend=%s): %s",
                type(self).__name__,
                handle.namespace,
                handle.sandbox_name,
                backend.id,
                exc,
            )
            raise

    def shutdown_for_manager_restart(self, backend: RunningBackend) -> None:

        handle = self._backends.get(backend.id)
        if handle is not None and (handle.pool is not None or handle.adopted):
            self._backends.pop(backend.id, None)
            logger.info(
                "[%s] manager restart: leaving reusable sandbox %s/%s in place (backend=%s)",
                type(self).__name__,
                handle.namespace,
                handle.sandbox_name,
                backend.id,
            )
            return
        self.shutdown(backend)

    def shutdown_all(self) -> None:

        self.retry_failed_cleanup()
        if not self._backends:
            return
        snapshot = dict(self._backends)
        for backend_id, handle in snapshot.items():
            try:
                self._release_handle(handle)
                logger.info(
                    "[%s] sweep-released sandbox %s/%s (backend=%s)",
                    type(self).__name__,
                    handle.namespace,
                    handle.sandbox_name,
                    backend_id,
                )
                if self._backends.get(backend_id) is handle:
                    self._backends.pop(backend_id, None)
            except Exception as exc:
                logger.warning(
                    "[%s] sweep release %s/%s failed: %s",
                    type(self).__name__,
                    handle.namespace,
                    handle.sandbox_name,
                    exc,
                )
