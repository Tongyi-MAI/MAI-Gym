from typing import Any

from mai_gym.envs.shared.strategy_native import NativeStrategy


class OSWorld2Strategy(NativeStrategy):
    pass


def create_strategy(*, requested_type: str, **settings: Any) -> OSWorld2Strategy:
    return OSWorld2Strategy()
