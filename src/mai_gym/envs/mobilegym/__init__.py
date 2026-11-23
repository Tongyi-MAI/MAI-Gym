from mai_gym.envs.registry import define_environment

define_environment(
    "mobilegym",
    strategy="mai_gym.envs.mobilegym.strategy:MobileGymStrategy",
    backend="mai_gym.envs.mobilegym.backend:MobileGymBackend",
)
