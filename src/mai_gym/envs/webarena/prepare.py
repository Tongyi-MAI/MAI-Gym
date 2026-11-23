import os
from typing import Any

from mai_gym.envs.shared.prepare import prepare_native
from mai_gym.envs.shared.provenance import PreparationError


def prepare(settings: dict[str, Any]) -> dict[str, Any]:
    profile = prepare_native("webarena", settings)
    urls = settings.get("sites", {}).get("urls", {})
    names = {"REDDIT", "SHOPPING", "SHOPPING_ADMIN", "GITLAB", "WIKIPEDIA", "MAP", "HOMEPAGE"}
    if set(urls) != names or any(
        not str(url).startswith(("http://", "https://")) for url in urls.values()
    ):
        raise PreparationError(
            "WebArena sites.urls requires explicit URLs for all seven native site variables"
        )
    os.environ.update(urls)
    profile["site_urls"] = dict(urls)
    return profile
