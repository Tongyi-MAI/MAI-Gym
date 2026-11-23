from typing import Any


def configure_agent(variant: str | None, runtime: dict[str, Any]) -> str:
    from mai_gym.agents import AgentVariant, get_entry, resolve_variant_for_env_type
    from mai_gym.agents.action_space import ActionSpace
    from mai_gym.agents.implementations.mai_gym_mobile_use import (
        MAIGymMobileUsePolicy,
        render_system_prompt,
    )

    selected = variant or "mai_gym_mobile_use"
    entry = (
        resolve_variant_for_env_type("mobilegym", overrides=runtime.get("unified_variant_map"))
        if selected == "unified"
        else get_entry(selected)
    )
    policy = (
        entry.policy_cls if isinstance(entry, AgentVariant) else getattr(entry, "__self__", entry)
    )
    if getattr(policy, "NATIVE_ACTION_SPACE", None) != ActionSpace.MOBILE_USE:
        raise ValueError(
            "MobileGym eval requires a mobile_use agent; select mai_gym_mobile_use or another "
            "native mobile policy"
        )
    if isinstance(policy, type) and issubclass(policy, MAIGymMobileUsePolicy):
        runtime.setdefault(
            "system_prompt",
            render_system_prompt("my") + "\nCoordinates are normalized to [0, 999] on each axis. "
            "Use the mobile touch actions above, including long_press and drag; "
            "desktop mouse and keyboard APIs are unavailable. "
            'To scroll, use {"action_type":"swipe","start_coordinate":[x1,y1],'
            '"end_coordinate":[x2,y2]}. A swipe scrolls with inertia; '
            "a drag holds the touch while moving an item.",
        )
    return selected
