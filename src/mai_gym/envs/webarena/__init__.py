from mai_gym.envs.registry import define_environment

define_environment(
    "webarena",
    strategy="mai_gym.envs.webarena.strategy:WebArenaStrategy",
    backend="mai_gym.envs.shared.backend:NativeBackend",
)
