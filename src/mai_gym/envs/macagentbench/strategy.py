from typing import Any

from mai_gym.envs.shared.strategy_native import NativeStrategy


class MacAgentBenchStrategy(NativeStrategy):
    pass


def create_strategy(*, requested_type: str, **settings: Any) -> MacAgentBenchStrategy:
    return MacAgentBenchStrategy()
