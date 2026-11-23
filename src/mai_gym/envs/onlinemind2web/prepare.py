from typing import Any

from mai_gym.envs.shared.prepare import prepare_native
from mai_gym.envs.shared.provenance import PreparationError


def prepare(settings: dict[str, Any]) -> dict[str, Any]:
    profile = prepare_native("onlinemind2web", settings)
    if profile.get("label") not in {"paper", "official-latest"}:
        raise PreparationError(
            "OnlineMind2Web requires a labelled paper or official-latest profile"
        )
    return profile
