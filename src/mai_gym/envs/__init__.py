from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, cast

if TYPE_CHECKING:
    from mai_gym.agents.action_space import ActionSpace
    from mai_gym.infra.backend import Backend
    from mai_gym.models.session import RunningBackend

_REGISTRY: dict[str, Callable[[RunningBackend], Backend]] = {}


def create_backend(backend: RunningBackend) -> Backend:

    factory = _REGISTRY.get(backend.env_type)
    if factory is not None:
        return factory(backend)
    from mai_gym.envs.registry import get_environment, load_component

    return cast("Backend", load_component(get_environment(backend.env_type).backend)(backend))


def register_backend(
    env_type: str,
    factory: Callable[[RunningBackend], Backend],
    *,
    action_space: ActionSpace,
) -> None:
    from mai_gym.agents.action_space import _register_env_space

    _REGISTRY[env_type] = factory
    _register_env_space(env_type, action_space)
