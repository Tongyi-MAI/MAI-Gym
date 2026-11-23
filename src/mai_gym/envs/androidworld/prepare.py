from pathlib import Path
from typing import Any

from mai_gym.envs.shared.prepare import prepare_native
from mai_gym.envs.shared.provenance import PreparationError


def prepare(settings: dict[str, Any]) -> dict[str, Any]:
    profile = prepare_native("androidworld", settings)
    if settings.get("a11y_forwarder_apk"):
        apk = str(Path(settings["a11y_forwarder_apk"]).expanduser().resolve())
        if not any(asset.get("path") == apk for asset in profile["assets"]):
            raise PreparationError(
                "AndroidWorld a11y_forwarder_apk must match a verified profile asset"
            )
    return profile
