from typing import Any

from mai_gym.envs.shared.provenance import PreparationError
from mai_gym.envs.shared.tasks import read_manifest


def load_tasks(profile: dict[str, Any]) -> list[dict[str, Any]]:
    from android_world import registry, suite_utils

    classes = registry.TaskRegistry().get_registry(family="android_world")
    result = []
    for task in read_manifest(profile):
        name = task.get("task_class", task["task_id"])
        if name not in classes or ("params" not in task and "seed" not in task):
            raise PreparationError(
                f"AndroidWorld manifest requires a native task class and recorded params/instance seed: {task['task_id']}"
            )
        native = suite_utils._instantiate_task(
            classes[name], params=task.get("params"), seed=task.get("seed")
        )
        budget = suite_utils._allocate_step_budget(native.complexity)
        if task.get("max_steps", budget) != budget:
            raise PreparationError(
                f"AndroidWorld budget disagrees with native task complexity: {name}"
            )
        result.append({**task, "params": native.params, "max_steps": budget, "goal": native.goal})
    return result
