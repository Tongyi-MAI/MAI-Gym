from __future__ import annotations

import base64
import contextlib
import logging
import re
import time
from pathlib import Path
from typing import TYPE_CHECKING

from mai_gym.infra.task_artifacts import (
    TaskArtifactSpec,
    resolve_artifact_source,
)
from mai_gym.models.launch_profiles import MobileWorldTaskArtifactConfig

if TYPE_CHECKING:
    from mai_gym.vendor.acs.client import AcsClient, PodExecResult

logger = logging.getLogger(__name__)

_INNER_CONTAINER = "mobileworld"
_OUTER_ROOT = "/tmp/mai-gym-task-artifacts"
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


def _require_exec(result: PodExecResult, label: str) -> str:
    if result.returncode != 0:
        detail = (result.stderr or result.stdout).strip()[:500]
        raise RuntimeError(f"{label} failed with exit {result.returncode}: {detail}")
    return result.stdout.strip()


class MobileWorldTaskArtifactStager:
    def __init__(
        self,
        client: AcsClient,
        config: MobileWorldTaskArtifactConfig,
    ) -> None:
        if not config.enabled or not config.root_path:
            raise ValueError("MobileWorld task artifact staging is not enabled")
        self._client = client
        self._config = config

    def _exec(
        self,
        namespace: str,
        pod_name: str,
        command: list[str],
        *,
        deadline: float,
        stdin_text: str | None = None,
    ) -> PodExecResult:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError(
                f"task artifact staging exceeded {self._config.upload_timeout_seconds}s"
            )
        return self._client.exec_pod(
            namespace,
            pod_name,
            command,
            stdin_text=stdin_text,
            timeout=remaining,
        )

    def _inner_sha256(
        self,
        namespace: str,
        pod_name: str,
        path: str,
        *,
        deadline: float,
    ) -> str | None:
        result = self._exec(
            namespace,
            pod_name,
            ["docker", "exec", _INNER_CONTAINER, "sha256sum", "--", path],
            deadline=deadline,
        )
        if result.returncode != 0:
            return None
        digest = result.stdout.strip().split(maxsplit=1)[0]
        return digest if _SHA256_RE.fullmatch(digest) else None

    def _upload_outer(
        self,
        namespace: str,
        pod_name: str,
        source: Path,
        spec: TaskArtifactSpec,
        *,
        deadline: float,
    ) -> str:
        target_dir = f"{_OUTER_ROOT}/{spec.sha256}"
        target = f"{target_dir}/{source.name}"
        partial = f"{target}.part"
        current = self._exec(
            namespace,
            pod_name,
            ["sha256sum", "--", target],
            deadline=deadline,
        )
        if current.returncode == 0:
            digest = current.stdout.strip().split(maxsplit=1)[0]
            if digest == spec.sha256:
                return target

        init_program = (
            "import pathlib;"
            f"p=pathlib.Path({partial!r});"
            "p.parent.mkdir(parents=True,exist_ok=True);"
            "p.write_bytes(b'')"
        )
        _require_exec(
            self._exec(
                namespace,
                pod_name,
                ["python3", "-c", init_program],
                deadline=deadline,
            ),
            f"initialize task artifact {spec.artifact_id}",
        )

        try:
            with source.open("rb") as stream:
                while True:
                    chunk = stream.read(self._config.chunk_size_bytes)
                    if not chunk:
                        break
                    encoded = base64.b64encode(chunk).decode("ascii")
                    append_program = (
                        "import base64,sys;"
                        f"data=sys.stdin.read({len(encoded)});"
                        f"open({partial!r},'ab').write("
                        "base64.b64decode(data,validate=True))"
                    )
                    _require_exec(
                        self._exec(
                            namespace,
                            pod_name,
                            ["python3", "-c", append_program],
                            deadline=deadline,
                            stdin_text=encoded,
                        ),
                        f"upload task artifact {spec.artifact_id}",
                    )

            finalize_program = (
                "import hashlib,os,pathlib;"
                f"p=pathlib.Path({partial!r});"
                "h=hashlib.sha256();"
                "f=p.open('rb');"
                "[(h.update(c)) for c in iter(lambda:f.read(1048576),b'')];"
                "f.close();"
                f"expected={spec.sha256!r};"
                "actual=h.hexdigest();"
                "(_ for _ in ()).throw(RuntimeError("
                "f'sha256 mismatch: expected {expected}, got {actual}')) "
                "if actual!=expected else None;"
                f"os.replace({partial!r},{target!r})"
            )
            _require_exec(
                self._exec(
                    namespace,
                    pod_name,
                    ["python3", "-c", finalize_program],
                    deadline=deadline,
                ),
                f"verify task artifact {spec.artifact_id}",
            )
        except Exception:
            with contextlib.suppress(Exception):
                self._exec(
                    namespace,
                    pod_name,
                    ["rm", "-f", "--", partial],
                    deadline=max(deadline, time.monotonic() + 5),
                )
            raise
        return target

    def stage(
        self,
        namespace: str,
        pod_name: str,
        spec: TaskArtifactSpec,
    ) -> dict[str, object]:

        assert self._config.root_path is not None
        source = resolve_artifact_source(self._config.root_path, spec)
        deadline = time.monotonic() + self._config.upload_timeout_seconds
        started = time.monotonic()

        if (
            self._inner_sha256(
                namespace,
                pod_name,
                spec.target_path,
                deadline=deadline,
            )
            == spec.sha256
        ):
            return {
                "artifact_id": spec.artifact_id,
                "sha256": spec.sha256,
                "target_path": spec.target_path,
                "status": "already_staged",
                "duration_seconds": round(time.monotonic() - started, 3),
            }

        outer = self._upload_outer(
            namespace,
            pod_name,
            source,
            spec,
            deadline=deadline,
        )
        inner_partial = f"{spec.target_path}.part"
        inner_dir = str(Path(spec.target_path).parent)
        try:
            _require_exec(
                self._exec(
                    namespace,
                    pod_name,
                    ["docker", "exec", _INNER_CONTAINER, "mkdir", "-p", inner_dir],
                    deadline=deadline,
                ),
                f"create inner artifact directory {spec.artifact_id}",
            )
            _require_exec(
                self._exec(
                    namespace,
                    pod_name,
                    ["docker", "cp", outer, f"{_INNER_CONTAINER}:{inner_partial}"],
                    deadline=deadline,
                ),
                f"copy task artifact {spec.artifact_id} into MobileWorld",
            )
            copied = self._inner_sha256(
                namespace,
                pod_name,
                inner_partial,
                deadline=deadline,
            )
            if copied != spec.sha256:
                raise RuntimeError(
                    f"inner task artifact SHA256 mismatch for {spec.artifact_id}: "
                    f"expected {spec.sha256}, got {copied}"
                )
            _require_exec(
                self._exec(
                    namespace,
                    pod_name,
                    [
                        "docker",
                        "exec",
                        _INNER_CONTAINER,
                        "mv",
                        "-f",
                        "--",
                        inner_partial,
                        spec.target_path,
                    ],
                    deadline=deadline,
                ),
                f"activate task artifact {spec.artifact_id}",
            )
        except Exception:
            with contextlib.suppress(Exception):
                self._exec(
                    namespace,
                    pod_name,
                    [
                        "docker",
                        "exec",
                        _INNER_CONTAINER,
                        "rm",
                        "-f",
                        "--",
                        inner_partial,
                    ],
                    deadline=max(deadline, time.monotonic() + 5),
                )
            raise
        final = self._inner_sha256(
            namespace,
            pod_name,
            spec.target_path,
            deadline=deadline,
        )
        if final != spec.sha256:
            raise RuntimeError(f"activated task artifact SHA256 mismatch for {spec.artifact_id}")
        logger.info(
            "[MobileWorldTaskArtifact] staged id=%s sha256=%s pod=%s/%s duration=%.3fs",
            spec.artifact_id,
            spec.sha256,
            namespace,
            pod_name,
            time.monotonic() - started,
        )
        return {
            "artifact_id": spec.artifact_id,
            "sha256": spec.sha256,
            "target_path": spec.target_path,
            "status": "staged",
            "duration_seconds": round(time.monotonic() - started, 3),
        }
