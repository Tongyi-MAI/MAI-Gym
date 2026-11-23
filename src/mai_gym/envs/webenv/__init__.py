from mai_gym.envs.registry import define_environment

define_environment(
    "webenv",
    strategy="mai_gym.envs.webenv.strategy:WebEnvStrategy",
    backend="mai_gym.envs.webenv.backend:WebEnvBackend",
)
