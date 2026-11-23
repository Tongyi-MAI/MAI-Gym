from mai_gym.envs.registry import define_environment

define_environment(
    "cua_hub",
    strategy="mai_gym.envs.cua_hub.strategy:CUAHubStrategy",
    backend="mai_gym.envs.shared.backend:BrowserBackend",
)
