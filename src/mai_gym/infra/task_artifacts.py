from __future__ import annotations

import hashlib
import io
import json
import re
import zipfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

MANIFEST_NAME = "mai_gym_task_set.json"
_MAX_MANIFEST_BYTES = 1024 * 1024
_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_TASK_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]{0,255}$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class TaskArtifactManifestError(ValueError):
    pass


@dataclass(frozen=True)
class TaskArtifactSpec:
    artifact_id: str
    kind: str
    source_path: str
    sha256: str

    @property
    def target_path(self) -> str:
        name = PurePosixPath(self.source_path).name
        return f"/app/service/task_artifacts/{self.sha256}/{name}"


@dataclass(frozen=True)
class TaskSetArtifactManifest:
    schema_version: int
    delivery: str
    artifacts: tuple[TaskArtifactSpec, ...]
    task_artifact_ids: tuple[tuple[str, tuple[str, ...]], ...]

    def artifacts_for(self, task_name: str) -> tuple[TaskArtifactSpec, ...]:
        artifact_by_id = {item.artifact_id: item for item in self.artifacts}
        for name, artifact_ids in self.task_artifact_ids:
            if name == task_name:
                return tuple(artifact_by_id[item] for item in artifact_ids)
        return ()

    @property
    def task_names(self) -> tuple[str, ...]:
        return tuple(name for name, _ in self.task_artifact_ids)


def _require_object(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise TaskArtifactManifestError(f"{label} must be an object")
    return value


def _safe_source_path(value: Any) -> str:
    if not isinstance(value, str) or not value or "\\" in value:
        raise TaskArtifactManifestError("artifact source_path must be a safe path")
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or path.name in {"", "."}:
        raise TaskArtifactManifestError(f"unsafe artifact source_path: {value!r}")
    return str(path)


def _parse_manifest(data: dict[str, Any]) -> TaskSetArtifactManifest:
    allowed = {"schema_version", "delivery", "artifacts", "tasks"}
    extra = sorted(set(data) - allowed)
    if extra:
        raise TaskArtifactManifestError(f"unknown artifact manifest fields: {', '.join(extra)}")
    if data.get("schema_version") != 1:
        raise TaskArtifactManifestError("artifact manifest schema_version must be 1")
    delivery = data.get("delivery")
    if delivery != "eager_code_lazy_artifacts":
        raise TaskArtifactManifestError(
            "artifact manifest delivery must be eager_code_lazy_artifacts"
        )

    raw_artifacts = data.get("artifacts")
    if not isinstance(raw_artifacts, list) or not raw_artifacts:
        raise TaskArtifactManifestError("artifact manifest requires artifacts")
    artifacts: list[TaskArtifactSpec] = []
    artifact_ids: set[str] = set()
    for index, raw_value in enumerate(raw_artifacts):
        raw = _require_object(raw_value, f"artifacts[{index}]")
        if set(raw) != {"id", "kind", "source_path", "sha256"}:
            raise TaskArtifactManifestError(
                f"artifacts[{index}] must contain id, kind, source_path, sha256"
            )
        artifact_id = raw.get("id")
        if not isinstance(artifact_id, str) or not _ID_RE.fullmatch(artifact_id):
            raise TaskArtifactManifestError(f"invalid artifact id at artifacts[{index}]")
        if artifact_id in artifact_ids:
            raise TaskArtifactManifestError(f"duplicate artifact id: {artifact_id}")
        artifact_ids.add(artifact_id)
        kind = raw.get("kind")
        if kind != "android_apk":
            raise TaskArtifactManifestError(
                f"unsupported artifact kind for {artifact_id}: {kind!r}"
            )
        source_path = _safe_source_path(raw.get("source_path"))
        if not source_path.lower().endswith(".apk"):
            raise TaskArtifactManifestError(
                f"android_apk artifact must end in .apk: {source_path!r}"
            )
        sha256 = raw.get("sha256")
        if not isinstance(sha256, str) or not _SHA256_RE.fullmatch(sha256):
            raise TaskArtifactManifestError(f"invalid sha256 for artifact {artifact_id}")
        artifacts.append(
            TaskArtifactSpec(
                artifact_id=artifact_id,
                kind=kind,
                source_path=source_path,
                sha256=sha256,
            )
        )

    raw_tasks = _require_object(data.get("tasks"), "tasks")
    if not raw_tasks:
        raise TaskArtifactManifestError("artifact manifest requires task mappings")
    task_artifact_ids: list[tuple[str, tuple[str, ...]]] = []
    referenced: set[str] = set()
    for task_name, raw_ids in raw_tasks.items():
        if not isinstance(task_name, str) or not _TASK_RE.fullmatch(task_name):
            raise TaskArtifactManifestError(f"invalid task name: {task_name!r}")
        if (
            not isinstance(raw_ids, list)
            or not raw_ids
            or any(not isinstance(item, str) for item in raw_ids)
        ):
            raise TaskArtifactManifestError(
                f"task {task_name} must reference one or more artifact ids"
            )
        ids = tuple(raw_ids)
        if len(ids) != len(set(ids)):
            raise TaskArtifactManifestError(f"task {task_name} contains duplicate artifact ids")
        unknown = sorted(set(ids) - artifact_ids)
        if unknown:
            raise TaskArtifactManifestError(
                f"task {task_name} references unknown artifacts: {unknown}"
            )
        referenced.update(ids)
        task_artifact_ids.append((task_name, ids))
    unreferenced = sorted(artifact_ids - referenced)
    if unreferenced:
        raise TaskArtifactManifestError(f"unreferenced artifacts: {unreferenced}")

    return TaskSetArtifactManifest(
        schema_version=1,
        delivery=delivery,
        artifacts=tuple(artifacts),
        task_artifact_ids=tuple(sorted(task_artifact_ids)),
    )


def parse_task_set_artifact_manifest(
    package_bytes: bytes,
) -> TaskSetArtifactManifest | None:

    try:
        with zipfile.ZipFile(io.BytesIO(package_bytes)) as archive:
            matches = [item for item in archive.infolist() if item.filename == MANIFEST_NAME]
            if not matches:
                return None
            if len(matches) != 1:
                raise TaskArtifactManifestError(
                    f"task ZIP contains duplicate {MANIFEST_NAME} entries"
                )
            info = matches[0]
            if info.file_size > _MAX_MANIFEST_BYTES:
                raise TaskArtifactManifestError("task artifact manifest is too large")
            raw = archive.read(info)
    except zipfile.BadZipFile:
        return None
    try:
        decoded = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise TaskArtifactManifestError(f"invalid {MANIFEST_NAME}: {exc}") from exc
    return _parse_manifest(_require_object(decoded, MANIFEST_NAME))


def resolve_artifact_source(root_path: str, spec: TaskArtifactSpec) -> Path:

    root = Path(root_path).expanduser().resolve()
    path = (root / spec.source_path).resolve()
    if path == root or root not in path.parents:
        raise TaskArtifactManifestError(
            f"artifact source escapes configured root: {spec.source_path!r}"
        )
    if not path.is_file():
        raise FileNotFoundError(f"task artifact does not exist: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    actual = digest.hexdigest()
    if actual != spec.sha256:
        raise TaskArtifactManifestError(
            f"task artifact SHA256 mismatch for {spec.artifact_id}: "
            f"expected {spec.sha256}, got {actual}"
        )
    return path
