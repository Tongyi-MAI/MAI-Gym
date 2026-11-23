from mai_gym.envs.registry import define_environment

define_environment(
    "mobileworld",
    strategy="mai_gym.envs.mobileworld.strategy:MobileWorldStrategy",
    backend="mai_gym.envs.mobileworld.backend:MobileWorldBackend",
)
