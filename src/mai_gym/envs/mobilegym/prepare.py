from pathlib import Path
from typing import Any

from mai_gym.envs.shared.prepare import prepare_native
from mai_gym.envs.shared.provenance import PreparationError


def prepare(settings: dict[str, Any]) -> dict[str, Any]:
    profile = prepare_native("mobilegym", settings)
    requested = settings.get("media_dir")
    if requested is None:
        media_assets = [
            asset
            for asset in profile["assets"]
            if asset.get("kind") == "tree" and asset.get("role") == "media"
        ]
        if len(media_assets) != 1:
            raise PreparationError(
                "MobileGym requires a verified companion-media tree. Extract the pinned "
                "archive, record its file inventory as a tree asset with role='media', "
                "or set media_dir to an explicitly pinned tree; see skills/envs/mobilegym.md."
            )
        directory = Path(media_assets[0]["path"])
    else:
        directory = (Path(profile["lock_path"]).parent / Path(requested).expanduser()).resolve()
    if not any(
        asset.get("kind") == "tree" and Path(asset["path"]).resolve() == directory.resolve()
        for asset in profile["assets"]
    ):
        raise PreparationError(
            "MobileGym media_dir must match a verified companion-media tree asset"
        )
    profile["media_dir"] = str(directory.resolve())
    return profile
