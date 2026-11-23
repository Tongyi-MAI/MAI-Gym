from mai_gym.envs.registry import define_environment

define_environment(
    "onlinemind2web",
    strategy="mai_gym.envs.onlinemind2web.strategy:OnlineMind2WebStrategy",
    backend="mai_gym.envs.shared.backend:NativeBackend",
)
