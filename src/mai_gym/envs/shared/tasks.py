import json
from pathlib import Path
from typing import Any

from mai_gym.envs.shared.provenance import PreparationError, sha256


def read_manifest(profile: dict[str, Any]) -> list[dict[str, Any]]:
    path = Path(profile["task_manifest"]["path"])
    if path.suffix == ".jsonl":
        data = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    else:
        data = json.loads(path.read_text())
    if isinstance(data, dict) and "tasks" in data:
        data = data["tasks"]
    if isinstance(data, dict):
        data = [
            {"task_id": task_id, "category": category}
            for category, tasks in data.items()
            for task_id in tasks
        ]
    if not isinstance(data, list) or not data:
        raise PreparationError(f"Empty or invalid task manifest: {path}")
    result = []
    for entry in data:
        entry = {"task_id": entry} if isinstance(entry, str) else dict(entry)
        task_id = entry.get("task_id", entry.get("id", entry.get("name")))
        if task_id is None:
            raise PreparationError(f"Task manifest entry has no identity in {path}")
        result.append({**entry, "task_id": str(task_id)})
    if len({entry["task_id"] for entry in result}) != len(result):
        raise PreparationError(
            f"Duplicate task identities in {path}; parameterized variants need distinct IDs"
        )
    return result


def task_file(profile: dict[str, Any], task: dict[str, Any], *, suffix: str = ".json") -> Path:
    root = Path(profile["sources"]["runtime"]["path"])
    relative = task.get("path") or task.get("config_file")
    path: Path
    if relative:
        path = (root / str(relative)).resolve()
    else:
        category = str(task.get("category", ""))
        path = (
            root / "evaluation_examples" / "examples" / category / (str(task["task_id"]) + suffix)
        )
    if not path.is_file():
        raise PreparationError(
            f"Native task asset missing: {path}; restore the recorded task package"
        )
    if task.get("sha256") and sha256(path) != task["sha256"]:
        raise PreparationError(f"Native task does not match its recorded hash: {path}")
    return path
