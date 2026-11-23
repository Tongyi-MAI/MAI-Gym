from typing import Any

from mai_gym.envs.shared.prepare import prepare_native
from mai_gym.envs.shared.provenance import PreparationError


def prepare(settings: dict[str, Any]) -> dict[str, Any]:
    profile = prepare_native("osworld_2", settings)
    if settings.get("desktop", {}).get("provider_name", "vmware") == "docker":
        image = settings.get("docker_image", "")
        if "@sha256:" not in image or not any(
            asset.get("kind") == "docker" and asset.get("image") == image
            for asset in profile["assets"]
        ):
            raise PreparationError(
                "OSWorld 2 docker_image must select an immutable verified Docker profile asset"
            )
    if settings.get("tasks_root"):
        profile["tasks_root"] = settings["tasks_root"]
    manifest = profile["task_manifest"]
    if profile.get("label", "paper") == "paper":
        if (
            manifest.get("revision") != "0a1aadad95aa79b00b3783e717d865089ab06e26"
            or manifest["sha256"]
            != "c54d428329be5ca72742a6becd49ee83cf739df5dea429bba182e1f3f21badfb"
        ):
            raise PreparationError(
                "OSWorld 2 paper profile requires the recorded official 2.1 task revision and manifest"
            )
        if (
            profile["sources"]["evaluator"]["revision"]
            != "11cea575561fb7800b5fb6b6cafa56f7a91de11f"
        ):
            raise PreparationError("OSWorld 2 paper evaluator commit mismatch")
    return profile
