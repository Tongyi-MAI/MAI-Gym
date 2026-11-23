from mai_gym.envs.registry import define_environment

define_environment(
    "mypcbench",
    strategy="mai_gym.envs.mypcbench.strategy:MyPCBenchStrategy",
    backend="mai_gym.envs.shared.backend:NativeBackend",
)
