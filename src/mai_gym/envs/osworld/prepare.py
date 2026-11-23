from typing import Any

from mai_gym.envs.shared.prepare import prepare_native
from mai_gym.envs.shared.provenance import PreparationError


def prepare(settings: dict[str, Any]) -> dict[str, Any]:
    profile = prepare_native("osworld", settings)
    if settings.get("desktop", {}).get("provider_name", "vmware") == "docker":
        image = settings.get("docker_image", "")
        if "@sha256:" not in image or not any(
            asset.get("kind") == "docker" and asset.get("image") == image
            for asset in profile["assets"]
        ):
            raise PreparationError(
                "OSWorld docker_image must select an immutable verified Docker profile asset"
            )
    return profile
