import json
from pathlib import Path
from typing import Any

from mai_gym.envs.shared.tasks import read_manifest


def load_tasks(profile: dict[str, Any]) -> list[dict[str, Any]]:
    data = json.loads(Path(profile["task_manifest"]["path"]).read_text())
    if (
        isinstance(data, list)
        and data
        and all(
            isinstance(task, dict) and "intent" in task and "start_url" in task for task in data
        )
    ):
        ids = [str(task["task_id"]) for task in data]
        if len(set(ids)) != len(ids):
            raise ValueError("WebArena native task identities must be distinct")
        return [
            {"task_id": str(task["task_id"]), "native_config": task, "max_steps": 30}
            for task in data
        ]
    return [{**task, "max_steps": 30} for task in read_manifest(profile)]
