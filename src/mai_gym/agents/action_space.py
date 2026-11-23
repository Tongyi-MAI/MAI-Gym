from __future__ import annotations

from enum import StrEnum


class ActionSpace(StrEnum):
    COMPUTER_USE = "computer_use"
    MOBILE_USE = "mobile_use"
    WEB_USE = "web_use"


_ENV_ACTION_SPACE: dict[str, ActionSpace] = {
    "real_device": ActionSpace.MOBILE_USE,
    "mobileworld": ActionSpace.MOBILE_USE,
    "mobileworld_k8s": ActionSpace.MOBILE_USE,
    "mobilegym": ActionSpace.MOBILE_USE,
    "androidworld": ActionSpace.MOBILE_USE,
    "osworld-2": ActionSpace.COMPUTER_USE,
    "osworld_2": ActionSpace.COMPUTER_USE,
    "cua_hub": ActionSpace.COMPUTER_USE,
    "macagentbench": ActionSpace.COMPUTER_USE,
    "mypcbench": ActionSpace.COMPUTER_USE,
    "webvoyager": ActionSpace.WEB_USE,
    "onlinemind2web": ActionSpace.WEB_USE,
    "webarena": ActionSpace.WEB_USE,
    "osworld": ActionSpace.COMPUTER_USE,
    "osworld_cua": ActionSpace.COMPUTER_USE,
    "osworld_k8s": ActionSpace.COMPUTER_USE,
    "web_env": ActionSpace.WEB_USE,
    "webenv": ActionSpace.WEB_USE,
    "mocked-apps": ActionSpace.WEB_USE,
    "mocked_mobile": ActionSpace.WEB_USE,
}


def _register_env_space(env_type: str, space: ActionSpace) -> None:
    _ENV_ACTION_SPACE[env_type] = space


def env_action_space(env_type: str) -> ActionSpace:

    try:
        return _ENV_ACTION_SPACE[env_type]
    except KeyError as exc:
        raise KeyError(
            f"env_type={env_type!r} has no registered action_space. "
            "Pass action_space= to register_backend() in the env's __init__.py."
        ) from exc


def registered_env_types() -> dict[str, ActionSpace]:
    return dict(_ENV_ACTION_SPACE)
