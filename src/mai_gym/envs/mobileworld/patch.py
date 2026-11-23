from __future__ import annotations

import base64
import contextlib
import hashlib
import json
import logging
import re
import time
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING, Any

import httpx

from mai_gym.models.launch_profiles import MobileWorldPatchConfig

if TYPE_CHECKING:
    from mai_gym.vendor.acs.client import AcsClient, PodExecResult

logger = logging.getLogger(__name__)

_PATCH_ID_RE = re.compile(r"^[a-z0-9][a-z0-9._-]{0,62}$")
_OUTER_SOURCE_ROOT = "/root/MobileWorld/src"
_INNER_SOURCE_ROOT = "/app/service/src"
_SERVER_CONTAINER = "mobileworld"


@dataclass(frozen=True)
class PatchFile:
    path: str
    patch_name: str
    patch_bytes: bytes
    before_sha256: frozenset[str]
    after_sha256: str


@dataclass(frozen=True)
class PatchBundle:
    patch_id: str
    target_root: str
    inner_root: str
    server_container: str
    files: tuple[PatchFile, ...]
    digest: str
    error_contract_version: int | None = None
    source_patch_ids: tuple[str, ...] = ()

    health_patch_id: str | None = None


def _safe_relative_path(value: str, *, label: str) -> str:
    path = PurePosixPath(value)
    if not value or path.is_absolute() or ".." in path.parts:
        raise ValueError(f"unsafe {label}: {value!r}")
    return str(path)


def load_patch_bundle(config: MobileWorldPatchConfig) -> PatchBundle:

    if not config.enabled or not config.bundle_path or not config.bundle_sha256:
        raise ValueError("MobileWorld patch bundle is not enabled/configured")

    bundle_dir = Path(config.bundle_path).expanduser().resolve()
    return _read_patch_bundle(bundle_dir, expected_digest=config.bundle_sha256)


def _read_patch_bundle(
    bundle_dir: Path,
    *,
    expected_digest: str | None = None,
    manifest_bytes: bytes | None = None,
) -> PatchBundle:

    manifest_path = bundle_dir / "manifest.json"
    if manifest_path.resolve().parent != bundle_dir:
        raise ValueError("manifest path escapes bundle")
    if manifest_bytes is None:
        manifest_bytes = manifest_path.read_bytes()
    manifest = json.loads(manifest_bytes)
    if not isinstance(manifest, dict):
        raise ValueError("MobileWorld patch manifest must be an object")

    patch_id = str(manifest.get("patch_id") or "")
    if not _PATCH_ID_RE.fullmatch(patch_id):
        raise ValueError(f"invalid MobileWorld patch_id: {patch_id!r}")
    target_root = str(manifest.get("target_root") or "")
    inner_root = str(manifest.get("inner_root") or "")
    server_container = str(manifest.get("server_container") or "")
    if target_root != _OUTER_SOURCE_ROOT:
        raise ValueError(f"unsupported MobileWorld target_root: {target_root!r}")
    if inner_root != _INNER_SOURCE_ROOT:
        raise ValueError(f"unsupported MobileWorld inner_root: {inner_root!r}")
    if server_container != _SERVER_CONTAINER:
        raise ValueError(f"unsupported MobileWorld server_container: {server_container!r}")

    raw_files = manifest.get("files")
    if not isinstance(raw_files, list) or not raw_files:
        raise ValueError("MobileWorld patch manifest must contain files")

    digest = hashlib.sha256()
    digest.update(manifest_bytes)
    files: list[PatchFile] = []
    seen_targets: set[str] = set()
    for raw in raw_files:
        if not isinstance(raw, dict):
            raise ValueError("MobileWorld patch file entry must be an object")
        path = _safe_relative_path(str(raw.get("path") or ""), label="target path")
        patch_name = _safe_relative_path(str(raw.get("patch") or ""), label="patch path")
        if path in seen_targets:
            raise ValueError(f"duplicate MobileWorld patch target: {path}")
        seen_targets.add(path)
        patch_path = (bundle_dir / patch_name).resolve()
        if bundle_dir not in patch_path.parents:
            raise ValueError(f"patch path escapes bundle: {patch_name!r}")
        patch_bytes = patch_path.read_bytes()
        digest.update(b"\0")
        digest.update(patch_name.encode())
        digest.update(b"\0")
        digest.update(patch_bytes)

        before = raw.get("before_sha256")
        after = str(raw.get("after_sha256") or "")
        if (
            not isinstance(before, list)
            or not before
            or any(not re.fullmatch(r"[0-9a-f]{64}", str(item)) for item in before)
            or not re.fullmatch(r"[0-9a-f]{64}", after)
        ):
            raise ValueError(f"invalid hashes for MobileWorld patch target {path}")
        files.append(
            PatchFile(
                path=path,
                patch_name=patch_name,
                patch_bytes=patch_bytes,
                before_sha256=frozenset(str(item) for item in before),
                after_sha256=after,
            )
        )

    actual_digest = digest.hexdigest()
    if expected_digest is not None and actual_digest != expected_digest:
        raise ValueError(
            "MobileWorld patch bundle SHA256 mismatch: "
            f"expected {expected_digest}, got {actual_digest}"
        )
    return PatchBundle(
        patch_id=patch_id,
        target_root=target_root,
        inner_root=inner_root,
        server_container=server_container,
        files=tuple(files),
        digest=actual_digest,
        error_contract_version=manifest.get("error_contract_version"),
    )


def discover_patch_bundle(directory: str) -> PatchBundle:

    root = Path(directory).expanduser().resolve()
    if not root.is_dir():
        raise ValueError(f"MobileWorld patch root is not a directory: {root}")
    bundles: list[PatchBundle] = []
    seen_ids: set[str] = set()
    seen_targets: set[str] = set()
    digest = hashlib.sha256(b"mai-gym-mobileworld-discovery-v1\0")
    for entry in sorted(root.iterdir(), key=lambda path: path.name):
        if entry.name.startswith("."):
            continue
        if entry.is_symlink():
            raise ValueError(f"symlinked MobileWorld bundle entry is not allowed: {entry.name}")
        if not entry.is_dir():
            if entry.suffix == ".patch":
                raise ValueError(f"patch requires a bundle directory and manifest: {entry.name}")
            continue
        manifest_path = entry / "manifest.json"
        if manifest_path.resolve().parent != entry or not manifest_path.is_file():
            raise ValueError(f"missing or unsafe manifest in MobileWorld bundle: {entry.name}")
        manifest_bytes = manifest_path.read_bytes()
        manifest = json.loads(manifest_bytes)
        if not isinstance(manifest, dict):
            raise ValueError("MobileWorld patch manifest must be an object")
        enabled = manifest.get("enabled", True)
        if not isinstance(enabled, bool):
            raise ValueError("MobileWorld manifest enabled must be boolean")
        if not enabled:
            continue
        bundle = _read_patch_bundle(entry, manifest_bytes=manifest_bytes)
        if bundle.patch_id in seen_ids:
            raise ValueError(f"duplicate MobileWorld patch_id: {bundle.patch_id}")
        seen_ids.add(bundle.patch_id)
        for item in bundle.files:
            if item.path in seen_targets:
                raise ValueError(f"conflicting MobileWorld patch target: {item.path}")
            seen_targets.add(item.path)
        digest.update(entry.name.encode())
        digest.update(b"\0")
        digest.update(bundle.digest.encode())
        digest.update(b"\0")
        bundles.append(bundle)
    if not bundles:
        raise ValueError(f"no enabled MobileWorld patch bundles in {root}")
    contracts = [bundle for bundle in bundles if bundle.error_contract_version is not None]
    if len(contracts) > 1:
        raise ValueError("conflicting MobileWorld server error contract bundles")
    contract = contracts[0] if contracts else None
    fingerprint = digest.hexdigest()
    return PatchBundle(
        patch_id=f"dynamic-{fingerprint[:24]}",
        target_root=_OUTER_SOURCE_ROOT,
        inner_root=_INNER_SOURCE_ROOT,
        server_container=_SERVER_CONTAINER,
        files=tuple(item for bundle in bundles for item in bundle.files),
        digest=fingerprint,
        source_patch_ids=tuple(bundle.patch_id for bundle in bundles),
        error_contract_version=contract.error_contract_version if contract else None,
        health_patch_id=contract.patch_id if contract else None,
    )


def _require_exec(result: PodExecResult, label: str) -> str:
    if result.returncode != 0:
        detail = (result.stderr or result.stdout).strip()[:500]
        raise RuntimeError(f"{label} failed with exit {result.returncode}: {detail}")
    return result.stdout.strip()


class MobileWorldPodPatcher:
    def __init__(
        self,
        client: AcsClient,
        config: MobileWorldPatchConfig,
        *,
        bundle: PatchBundle | None = None,
    ) -> None:
        self._client = client
        self._config = config
        self._bundle = bundle if bundle is not None else load_patch_bundle(config)

    @property
    def patch_id(self) -> str:
        return self._bundle.patch_id

    def _exec(
        self,
        namespace: str,
        pod_name: str,
        command: list[str],
        *,
        stdin_text: str | None = None,
        timeout: float | None = None,
    ) -> PodExecResult:
        return self._client.exec_pod(
            namespace,
            pod_name,
            command,
            stdin_text=stdin_text,
            timeout=timeout or self._config.restart_timeout_seconds,
        )

    def _sha256(self, namespace: str, pod_name: str, path: str) -> str:
        output = _require_exec(
            self._exec(namespace, pod_name, ["sha256sum", "--", path]),
            f"sha256 {path}",
        )
        digest = output.split(maxsplit=1)[0]
        if not re.fullmatch(r"[0-9a-f]{64}", digest):
            raise RuntimeError(f"invalid sha256 output for {path}: {output!r}")
        return digest

    def _upload(
        self,
        namespace: str,
        pod_name: str,
        remote_path: str,
        payload: bytes,
    ) -> None:
        encoded = base64.b64encode(payload).decode("ascii")
        program = (
            "import base64,pathlib,sys;"
            f"data=sys.stdin.read({len(encoded)});"
            f"pathlib.Path({remote_path!r}).write_bytes(base64.b64decode(data))"
        )
        _require_exec(
            self._exec(
                namespace,
                pod_name,
                ["python3", "-c", program],
                stdin_text=encoded,
            ),
            f"upload {remote_path}",
        )

    def _restart_server(self, namespace: str, pod_name: str) -> None:
        stop_program = r"""
import os, signal, time
targets = {
    ("uv", "run", "mobile-world", "server", "--port", "6800"),
    (
        "/app/service/.venv/bin/python",
        "/app/service/.venv/bin/mobile-world",
        "server", "--port", "6800",
    ),
}
found = []
for entry in os.scandir("/proc"):
    if not entry.name.isdigit():
        continue
    try:
        argv = tuple(
            part.decode(errors="replace")
            for part in open(f"/proc/{entry.name}/cmdline", "rb").read().rstrip(b"\0").split(b"\0")
        )
    except OSError:
        continue
    if argv in targets:
        found.append(int(entry.name))
for pid in found:
    try:
        os.kill(pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
deadline = time.monotonic() + 15
while time.monotonic() < deadline:
    alive = [pid for pid in found if os.path.exists(f"/proc/{pid}")]
    if not alive:
        break
    time.sleep(0.2)
else:
    for pid in alive:
        try:
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
print("stopped=" + ",".join(str(pid) for pid in found))
""".strip()
        stopped = self._exec(
            namespace,
            pod_name,
            [
                "docker",
                "exec",
                self._bundle.server_container,
                "python",
                "-c",
                stop_program,
            ],
        )
        _require_exec(stopped, "stop nested MobileWorld server")

        start = self._exec(
            namespace,
            pod_name,
            [
                "docker",
                "exec",
                "-d",
                "-w",
                "/app/service",
                "-e",
                "UV_NO_SYNC=1",
                "-e",
                "UV_FROZEN=1",
                "-e",
                "LD_PRELOAD=/usr/local/lib/libtimeshift.so",
                "-e",
                "TIMESHIFT_OFFSET_FILE=/run/timeshift_offset",
                self._bundle.server_container,
                "sh",
                "-lc",
                "exec uv run mobile-world server --port 6800 >>/var/log/server.log 2>&1",
            ],
        )
        _require_exec(start, "start nested MobileWorld server")

    def _verify_server_processes(self, namespace: str, pod_name: str) -> None:
        probe = r"""
import json, os
targets = {
    ("uv", "run", "mobile-world", "server", "--port", "6800"),
    (
        "/app/service/.venv/bin/python",
        "/app/service/.venv/bin/mobile-world",
        "server", "--port", "6800",
    ),
}
found = []
for entry in os.scandir("/proc"):
    if not entry.name.isdigit():
        continue
    try:
        argv = tuple(
            part.decode(errors="replace")
            for part in open(f"/proc/{entry.name}/cmdline", "rb").read().rstrip(b"\0").split(b"\0")
        )
    except OSError:
        continue
    if argv in targets:
        found.append([int(entry.name), argv])
print(json.dumps(found))
""".strip()
        deadline = time.monotonic() + self._config.restart_timeout_seconds
        last: Any = None
        while time.monotonic() < deadline:
            result = self._exec(
                namespace,
                pod_name,
                ["docker", "exec", self._bundle.server_container, "python", "-c", probe],
                timeout=30,
            )
            if result.returncode == 0:
                try:
                    last = json.loads(result.stdout)
                except json.JSONDecodeError:
                    last = result.stdout.strip()
                if isinstance(last, list) and len(last) == 2:
                    return
            time.sleep(1)
        raise RuntimeError(f"nested MobileWorld server process contract failed: {last!r}")

    def _verify_health(self, url: str, require_contract: bool = True) -> None:
        deadline = time.monotonic() + self._config.restart_timeout_seconds
        last = "not attempted"
        with httpx.Client(timeout=15.0) as client:
            while time.monotonic() < deadline:
                try:
                    response = client.get(f"{url}/health")
                    last = f"HTTP {response.status_code}"
                    body = response.json()
                    if response.status_code == 200 and bool(body.get("ok", True)):
                        version = self._bundle.error_contract_version
                        if (
                            not require_contract
                            or version is None
                            or (
                                body.get("error_contract_version") == version
                                and body.get("patch_id")
                                == (self._bundle.health_patch_id or self._bundle.patch_id)
                            )
                        ):
                            return
                        last = "live server does not expose the expected patch/error contract"
                except Exception as exc:
                    last = f"{type(exc).__name__}: {exc}"
                time.sleep(2)
        raise RuntimeError(f"patched MobileWorld server did not become healthy: {last}")

    def _verify_inner_source(self, namespace: str, pod_name: str) -> None:
        for item in self._bundle.files:
            inner_path = f"{self._bundle.inner_root}/{item.path}"
            output = _require_exec(
                self._exec(
                    namespace,
                    pod_name,
                    [
                        "docker",
                        "exec",
                        self._bundle.server_container,
                        "sha256sum",
                        "--",
                        inner_path,
                    ],
                ),
                f"inner sha256 {inner_path}",
            )
            if output.split(maxsplit=1)[0] != item.after_sha256:
                raise RuntimeError(f"inner bind-mount hash mismatch for {inner_path}")

        paths = [
            f"{self._bundle.inner_root}/{item.path}"
            for item in self._bundle.files
            if item.path.endswith(".py")
        ]
        program = (
            "import ast,pathlib;"
            f"paths={paths!r};"
            "[ast.parse(pathlib.Path(p).read_bytes(),filename=p) for p in paths];"
        )
        if any(
            item.path == "mobile_world/runtime/utils/redroid_device.py"
            for item in self._bundle.files
        ):
            program += (
                "from mobile_world.runtime.utils.redroid_device import "
                "_wait_adb_root, _wait_redroid_stable;"
            )
        program += "print('ok')"
        _require_exec(
            self._exec(
                namespace,
                pod_name,
                [
                    "docker",
                    "exec",
                    "-w",
                    "/app/service",
                    self._bundle.server_container,
                    "/app/service/.venv/bin/python",
                    "-c",
                    program,
                ],
            ),
            "patched MobileWorld import probe",
        )

    def apply(self, *, namespace: str, pod_name: str, url: str) -> dict[str, Any]:
        started = time.monotonic()
        stage = f"/tmp/mai-gym-{self._bundle.patch_id}"
        changed: list[tuple[str, str]] = []
        candidates: list[str] = []
        backups: list[str] = []
        status = "already_applied"
        _require_exec(
            self._exec(namespace, pod_name, ["mkdir", "-p", "--", stage]),
            "create MobileWorld patch staging directory",
        )
        try:
            for index, item in enumerate(self._bundle.files):
                target = f"{self._bundle.target_root}/{item.path}"
                candidate = f"{target}.mai-gym-candidate-{self._bundle.patch_id}"
                candidates.append(candidate)
                current = self._sha256(namespace, pod_name, target)
                if current == item.after_sha256:
                    continue
                if current not in item.before_sha256:
                    raise RuntimeError(f"unexpected pre-patch hash for {item.path}: {current}")

                remote_patch = f"{stage}/{index}.patch"
                backup = f"{target}.mai-gym-backup-{self._bundle.patch_id}"
                self._upload(namespace, pod_name, remote_patch, item.patch_bytes)
                _require_exec(
                    self._exec(
                        namespace,
                        pod_name,
                        ["cp", "-a", "--", target, candidate],
                    ),
                    f"stage {item.path}",
                )
                _require_exec(
                    self._exec(
                        namespace,
                        pod_name,
                        [
                            "patch",
                            "--dry-run",
                            "--batch",
                            "--forward",
                            candidate,
                            "-i",
                            remote_patch,
                        ],
                    ),
                    f"dry-run patch {item.path}",
                )
                _require_exec(
                    self._exec(namespace, pod_name, ["cp", "-a", "--", target, backup]),
                    f"backup {item.path}",
                )
                backups.append(backup)
                _require_exec(
                    self._exec(
                        namespace,
                        pod_name,
                        [
                            "patch",
                            "--batch",
                            "--forward",
                            candidate,
                            "-i",
                            remote_patch,
                        ],
                    ),
                    f"apply patch {item.path}",
                )
                if self._sha256(namespace, pod_name, candidate) != item.after_sha256:
                    raise RuntimeError(f"post-patch hash mismatch for {item.path}")
                _require_exec(
                    self._exec(
                        namespace,
                        pod_name,
                        ["mv", "-f", "--", candidate, target],
                    ),
                    f"install {item.path}",
                )
                changed.append((target, backup))
                status = "applied"

            self._restart_server(namespace, pod_name)
            self._verify_server_processes(namespace, pod_name)
            self._verify_inner_source(namespace, pod_name)
            self._verify_health(url)
        except BaseException:
            if changed:
                logger.warning(
                    "MobileWorld patch %s failed on %s/%s; rolling back",
                    self._bundle.patch_id,
                    namespace,
                    pod_name,
                )
                try:
                    for target, backup in reversed(changed):
                        _require_exec(
                            self._exec(
                                namespace,
                                pod_name,
                                ["mv", "-f", "--", backup, target],
                            ),
                            f"rollback {target}",
                        )
                    self._restart_server(namespace, pod_name)
                    self._verify_server_processes(namespace, pod_name)
                    self._verify_health(url, False)
                except Exception:
                    logger.exception(
                        "MobileWorld patch rollback failed on %s/%s",
                        namespace,
                        pod_name,
                    )
            raise
        else:
            return {
                "patch_id": self._bundle.patch_id,
                "status": status,
                "duration_seconds": round(time.monotonic() - started, 3),
                **(
                    {
                        "bundle_sha256": self._bundle.digest,
                        "patch_ids": list(self._bundle.source_patch_ids),
                    }
                    if self._bundle.source_patch_ids
                    else {}
                ),
            }
        finally:
            for temporary in [*candidates, *backups]:
                with contextlib.suppress(Exception):
                    self._exec(
                        namespace,
                        pod_name,
                        ["rm", "-f", "--", temporary],
                    )
            with contextlib.suppress(Exception):
                self._exec(namespace, pod_name, ["rm", "-rf", "--", stage])
