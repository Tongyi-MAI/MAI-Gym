import json
from pathlib import Path
from typing import Any

from mai_gym.envs.shared.provenance import PreparationError, sha256


def load_tasks(profile: dict[str, Any]) -> list[dict[str, Any]]:
    manifest = json.loads(Path(profile["task_manifest"]["path"]).read_text())
    if not isinstance(manifest.get("files"), dict):
        raise PreparationError("OSWorld 2 requires its native task hash manifest")
    task_root = Path(
        profile.get("tasks_root")
        or Path(profile["sources"]["runtime"]["path"]) / "evaluation_examples/task_class"
    )
    tasks = []
    for filename, metadata in manifest["files"].items():
        path = task_root / filename
        if not path.is_file() or sha256(path) != metadata["sha256"]:
            raise PreparationError(
                f"OSWorld 2 native task is missing or does not match its recorded hash: {path}"
            )
        tasks.append(
            {
                "task_id": path.stem.removeprefix("task_"),
                "path": str(path.resolve()),
                "sha256": metadata["sha256"],
                "max_steps": 500,
            }
        )
    if len(tasks) != manifest.get("task_count"):
        raise PreparationError("OSWorld 2 task manifest count does not match its file inventory")
    return tasks
