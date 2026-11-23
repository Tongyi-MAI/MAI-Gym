from typing import Any

from mai_gym.envs.shared.strategy_native import NativeStrategy


class OnlineMind2WebStrategy(NativeStrategy):
    pass


def create_strategy(*, requested_type: str, **settings: Any) -> OnlineMind2WebStrategy:
    return OnlineMind2WebStrategy()
