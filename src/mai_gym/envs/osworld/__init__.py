from mai_gym.envs.registry import define_environment

define_environment(
    "osworld",
    strategy="mai_gym.envs.osworld.strategy:OSWorldStrategy",
    backend="mai_gym.envs.osworld.backend:create_backend",
)
