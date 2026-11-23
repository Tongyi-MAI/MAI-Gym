from dataclasses import dataclass
from importlib import import_module
from typing import Any

from mai_gym.models.environment import DeviceType, identity


def load_component(path: str) -> Any:
    module, name = path.split(":", 1)
    return getattr(import_module(module), name)


@dataclass(frozen=True)
class Environment:
    name: str
    device_type: DeviceType
    backend: str
    strategy: str
    launch_profile: str
    task_loader: str
    preparation: str

    def create_strategy(self, requested_type: str, **settings: Any) -> Any:
        return load_component(self.strategy)(requested_type=requested_type, **settings)


_ENVIRONMENTS: dict[str, Environment] = {}


def register_environment(environment: Environment) -> None:
    if environment.name in _ENVIRONMENTS:
        raise ValueError(f"Environment already registered: {environment.name}")
    _ENVIRONMENTS[environment.name] = environment


def get_environment(name: str) -> Environment:
    canonical = identity(name).name
    if canonical not in _ENVIRONMENTS:
        import_module(f"mai_gym.envs.{canonical}")
    try:
        return _ENVIRONMENTS[canonical]
    except KeyError as exc:
        raise ValueError(f"No environment integration registered for {name!r}") from exc


def define_environment(
    name: str, *, strategy: str, backend: str = "mai_gym.infra.backend:BackendClient"
) -> None:
    package = f"mai_gym.envs.{name}"
    register_environment(
        Environment(
            name,
            identity(name).device_type,
            backend,
            strategy.split(":", 1)[0] + ":create_strategy",
            f"{package}.launch:launch_profile",
            f"{package}.tasks:load_tasks",
            f"{package}.prepare:prepare",
        )
    )
