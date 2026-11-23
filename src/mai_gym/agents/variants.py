from __future__ import annotations

import importlib
import logging
import pkgutil
from collections.abc import Callable, Mapping
from typing import Any

from mai_gym.models.config import AgentConfig

from ._base import AgentPolicy, AgentVariant

logger = logging.getLogger(__name__)


PolicyFactory = Callable[[AgentConfig], Any]


UNIFIED_AGENT_NAME = "unified"


_DEFAULT_UNIFIED_MAP: dict[str, str] = {
    "mobile_use": "mai_gym_mobile_use",
    "computer_use": "mai_gym_computer_use",
}

__all__ = [
    "UNIFIED_AGENT_NAME",
    "AgentPolicy",
    "AgentVariant",
    "PolicyFactory",
    "get_agent",
    "get_entry",
    "list_agents",
    "load_builtin_agents",
    "register_policy_factory",
    "register_variant",
    "resolve_variant_for_env_type",
]


_REGISTRY: dict[str, AgentVariant | PolicyFactory] = {}


def register_variant(variant: AgentVariant) -> None:

    if variant.name in _REGISTRY:
        raise ValueError(f"Agent {variant.name!r} already registered")
    _REGISTRY[variant.name] = variant
    logger.debug("Registered agent variant: %s", variant.name)


def register_policy_factory(agent_type: str, factory: PolicyFactory) -> None:

    if agent_type in _REGISTRY:
        raise ValueError(f"Agent {agent_type!r} already registered")
    _REGISTRY[agent_type] = factory
    logger.debug("Registered policy factory: %s", agent_type)


def get_entry(name: str) -> AgentVariant | PolicyFactory | None:

    load_builtin_agents()
    return _REGISTRY.get(name)


def get_agent(name: str) -> AgentVariant:

    entry = get_entry(name)
    if isinstance(entry, AgentVariant):
        return entry
    available = ", ".join(list_agents())
    raise ValueError(f"Unknown agent variant: {name!r}. Available: [{available}]")


def list_agents() -> list[str]:

    load_builtin_agents()
    return sorted(n for n, e in _REGISTRY.items() if isinstance(e, AgentVariant))


def resolve_variant_for_env_type(
    env_type: str, *, overrides: Mapping[str, str] | None = None
) -> AgentVariant:

    from .action_space import env_action_space

    space = env_action_space(env_type)
    mapping = {**_DEFAULT_UNIFIED_MAP, **(overrides or {})}
    name = mapping.get(str(space))
    if name is None:
        raise ValueError(
            f"unified agent has no variant mapped for action space {space!r} "
            f"(env_type={env_type!r}); configure unified_variant_map. "
            f"Known mappings: {sorted(mapping)}"
        )
    variant = get_agent(name)

    native = getattr(variant.policy_cls, "NATIVE_ACTION_SPACE", None)
    if native is not None and native != space:
        raise ValueError(
            f"unified agent mapped {space!r} -> {name!r}, but that variant's "
            f"NATIVE_ACTION_SPACE is {native!r}; the mapping is inconsistent."
        )
    return variant


_loaded = False


def load_builtin_agents() -> None:

    global _loaded
    if _loaded:
        return
    _loaded = True
    from mai_gym.agents import implementations

    for mod in pkgutil.iter_modules(implementations.__path__):
        try:
            importlib.import_module(f"{implementations.__name__}.{mod.name}")
        except Exception as exc:
            logger.warning("agent module %s failed to import: %s", mod.name, exc)

    from .implementations.mai_gym_computer_use import MAIGymComputerUsePolicy
    from .implementations.mai_gym_mobile_use import SYSTEM_PROMPT, MAIGymMobileUsePolicy
    from .implementations.mai_gym_mobile_use_windows_wise import (
        OVERLAP,
        WINDOW_SIZE,
        MAIGymMobileUseWindowsWisePolicy,
    )
    from .implementations.mai_gym_mobile_use_windows_wise import (
        SYSTEM_PROMPT as WINDOW_SYSTEM_PROMPT,
    )

    register_variant(
        AgentVariant(
            name="mai_gym_mobile_use",
            policy_cls=MAIGymMobileUsePolicy,
            parser_name="mai_gym_mobile_use",
            system_prompt=SYSTEM_PROMPT,
            preserve_thinking=True,
            extra={"release_server_pixels": True, "enable_thinking": True},
        )
    )
    register_variant(
        AgentVariant(
            name="mai_gym_mobile_use_windows_wise",
            policy_cls=MAIGymMobileUseWindowsWisePolicy,
            parser_name="mai_gym_mobile_use",
            system_prompt=WINDOW_SYSTEM_PROMPT,
            default_image_retention=WINDOW_SIZE,
            preserve_thinking=True,
            extra={"windowed": True, "window_size": WINDOW_SIZE, "overlap": OVERLAP},
        )
    )
    register_variant(
        AgentVariant(
            name="mai_gym_computer_use",
            policy_cls=MAIGymComputerUsePolicy,
            parser_name="mai_gym_computer_use",
            system_prompt="",
            default_image_retention=None,
            preserve_thinking=True,
        )
    )
