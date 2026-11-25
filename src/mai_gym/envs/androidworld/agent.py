from typing import Any


def configure_agent(variant: str | None, runtime: dict[str, Any]) -> str:
    from mai_gym.agents import AgentVariant, get_entry, resolve_variant_for_env_type
    from mai_gym.agents.action_space import ActionSpace
    from mai_gym.agents.implementations.mai_gym_mobile_use import (
        APP_LISTS,
        MAIGymMobileUsePolicy,
        render_system_prompt,
    )
    from mai_gym.agents.implementations.mai_gym_mobile_use_windows_wise import (
        MAIGymMobileUseWindowsWisePolicy,
        render_system_prompt as render_windows_wise_system_prompt,
    )

    selected = variant or "unified"
    entry = (
        resolve_variant_for_env_type("androidworld", overrides=runtime.get("unified_variant_map"))
        if selected == "unified"
        else get_entry(selected)
    )
    policy = (
        entry.policy_cls if isinstance(entry, AgentVariant) else getattr(entry, "__self__", entry)
    )
    if getattr(policy, "NATIVE_ACTION_SPACE", None) != ActionSpace.MOBILE_USE:
        raise ValueError(
            "AndroidWorld eval requires a mobile_use agent; select unified, "
            "mai_gym_mobile_use, or another native mobile policy"
        )
    if isinstance(policy, type) and issubclass(policy, MAIGymMobileUsePolicy):
        base_prompt = (
            render_windows_wise_system_prompt(apps=APP_LISTS["aw"])
            if issubclass(policy, MAIGymMobileUseWindowsWisePolicy)
            else render_system_prompt("aw")
        )
        runtime.setdefault(
            "system_prompt",
            "\n".join(
                [
                    base_prompt,
                    "- Coordinates are normalized to [0, 999] on each axis.",
                    '- You can also use {"action_type": "swipe", "start_coordinate": [x1, y1], "end_coordinate": [x2, y2]} to scroll.',
                    "- Do not call ask_user on this benchmark; complete the task autonomously on the Android device.",
                ]
            ),
        )
    return selected
