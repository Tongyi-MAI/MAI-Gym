from typing import Any

from mai_gym.envs.shared.provenance import verify_profile


def prepare(settings: dict[str, Any]) -> dict[str, Any]:
    return verify_profile("mobileworld", settings)
