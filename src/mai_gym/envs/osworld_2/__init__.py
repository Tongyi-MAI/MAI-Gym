from mai_gym.envs.registry import define_environment

define_environment(
    "osworld_2",
    strategy="mai_gym.envs.osworld_2.strategy:OSWorld2Strategy",
    backend="mai_gym.envs.shared.backend:NativeBackend",
)
