import os
import sys
from typing import Any

from mai_gym.envs.shared.provenance import verify_profile


def prepare_native(benchmark: str, settings: dict[str, Any]) -> dict[str, Any]:
    profile = verify_profile(benchmark, settings)
    for source in reversed(list(profile["sources"].values())):
        if source["path"] not in sys.path:
            sys.path.insert(0, source["path"])
    os.chdir(profile["sources"]["runtime"]["path"])
    return profile
