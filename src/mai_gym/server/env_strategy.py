from mai_gym.envs.registry import get_environment
from mai_gym.envs.shared.strategy import EnvStrategy


def create_strategy(env_type: str, **settings: object) -> EnvStrategy:
    try:
        environment = get_environment(env_type)
    except ValueError as exc:
        raise ValueError(f"No EnvStrategy registered for env_type {env_type!r}: {exc}") from exc
    return environment.create_strategy(env_type, **settings)
