from typing import Any

from mai_gym.envs.shared.strategy_native import NativeStrategy


class MyPCBenchStrategy(NativeStrategy):
    pass


def create_strategy(*, requested_type: str, **settings: Any) -> MyPCBenchStrategy:
    return MyPCBenchStrategy()
