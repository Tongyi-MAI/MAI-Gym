from dataclasses import dataclass
from typing import Literal

DeviceType = Literal["mobile", "computer", "web"]
InfraType = Literal["local", "docker", "ssh", "k8s", "agent", "attach"]


@dataclass(frozen=True)
class Identity:
    name: str
    device_type: DeviceType
    infra_type: InfraType | None = None
    profile: str | None = None


_DOMAINS: dict[str, DeviceType] = {
    "mobileworld": "mobile",
    "androidworld": "mobile",
    "mobilegym": "mobile",
    "osworld": "computer",
    "osworld_2": "computer",
    "macagentbench": "computer",
    "mypcbench": "computer",
    "webvoyager": "web",
    "onlinemind2web": "web",
    "webarena": "web",
    "webenv": "web",
    "cua_hub": "computer",
}
_ALIASES = {
    "mobileworld_k8s": Identity("mobileworld", "mobile", "k8s"),
    "real_device": Identity("mobileworld", "mobile", profile="real_device"),
    "osworld_k8s": Identity("osworld", "computer", "k8s"),
    "osworld_cua": Identity("osworld", "computer", profile="cua"),
    "osworld-verified": Identity("osworld", "computer"),
    "osworld-2": Identity("osworld_2", "computer"),
    "web_env": Identity("webenv", "web"),
    "mocked-apps": Identity("webenv", "web", profile="mocked-apps"),
    "mocked_mobile": Identity("webenv", "web", profile="mocked_mobile"),
}


def identity(name: str) -> Identity:
    if name in _ALIASES:
        return _ALIASES[name]
    try:
        return Identity(name, _DOMAINS[name])
    except KeyError as exc:
        raise ValueError(f"Unknown environment {name!r}; supported: {sorted(_DOMAINS)}") from exc


def environment_names() -> tuple[str, ...]:
    return tuple(_DOMAINS)


def matches_environment(
    requested: str, actual: str, *, infra_type: str | None = None, profile: str | None = None
) -> bool:
    try:
        wanted, available = identity(requested), identity(actual)
    except ValueError:
        return requested == actual
    return (
        wanted.name == available.name
        and wanted.profile == (profile or available.profile)
        and (wanted.infra_type is None or wanted.infra_type == (infra_type or available.infra_type))
    )
