from typing import Any

from mai_gym.envs.shared.prepare import prepare_native


def prepare(settings: dict[str, Any]) -> dict[str, Any]:
    profile = prepare_native("mypcbench", settings)
    return profile
