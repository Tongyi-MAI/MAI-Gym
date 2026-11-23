from pathlib import Path
from typing import Any

from mai_gym.envs.shared.provenance import PreparationError
from mai_gym.envs.shared.tasks import read_manifest


def load_tasks(profile: dict[str, Any]) -> list[dict[str, Any]]:
    path = Path(profile["task_manifest"]["path"])
    if path.suffix == ".txt":
        ids = [line.strip() for line in path.read_text().splitlines() if line.strip()]
        if not ids or len(set(ids)) != len(ids):
            raise PreparationError(f"MobileGym split must contain distinct native task IDs: {path}")
        return [{"task_id": task_id, "max_steps": 100} for task_id in ids]
    return [{**task, "max_steps": 100} for task in read_manifest(profile)]
