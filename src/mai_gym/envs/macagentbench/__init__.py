from mai_gym.envs.registry import define_environment

define_environment(
    "macagentbench",
    strategy="mai_gym.envs.macagentbench.strategy:MacAgentBenchStrategy",
    backend="mai_gym.envs.shared.backend:NativeBackend",
)
