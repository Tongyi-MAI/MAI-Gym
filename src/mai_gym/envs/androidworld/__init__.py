from mai_gym.envs.registry import define_environment

define_environment(
    "androidworld",
    strategy="mai_gym.envs.androidworld.strategy:AndroidWorldStrategy",
    backend="mai_gym.envs.shared.backend:NativeBackend",
)
