from mai_gym.envs.registry import define_environment

define_environment(
    "webvoyager",
    strategy="mai_gym.envs.webvoyager.strategy:WebVoyagerStrategy",
    backend="mai_gym.envs.shared.backend:NativeBackend",
)
