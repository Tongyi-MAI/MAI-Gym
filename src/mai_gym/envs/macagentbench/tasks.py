from typing import Any

from mai_gym.envs.shared.tasks import read_manifest


def load_tasks(profile: dict[str, Any]) -> list[dict[str, Any]]:
    return [{**task, "max_steps": 50} for task in read_manifest(profile)]
