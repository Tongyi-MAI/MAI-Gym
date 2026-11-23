import inspect
from typing import Any

from mai_gym.models.config import AgentConfig

from .variants import (
    _REGISTRY as _REGISTRY,
)
from .variants import (
    UNIFIED_AGENT_NAME,
    AgentVariant,
    PolicyFactory,
    get_agent,
    get_entry,
    list_agents,
    load_builtin_agents,
    register_policy_factory,
    register_variant,
    resolve_variant_for_env_type,
)

__all__ = [
    "UNIFIED_AGENT_NAME",
    "AgentVariant",
    "PolicyFactory",
    "create_policy",
    "get_agent",
    "get_entry",
    "list_agents",
    "register_policy_factory",
    "register_variant",
    "resolve_variant_for_env_type",
]


def create_policy(cfg: AgentConfig, *, env_type: str | None = None) -> Any:

    if cfg.type == UNIFIED_AGENT_NAME:
        if not env_type:
            raise ValueError(
                "agent type 'unified' needs an env_type to resolve the concrete "
                "variant; pass create_policy(cfg, env_type=...) "
                "(the eval driver supplies rollout.env_type)."
            )
        overrides = (cfg.runtime_conf or {}).get("unified_variant_map")
        variant = resolve_variant_for_env_type(env_type, overrides=overrides)
        return _build_variant_policy(variant, cfg)

    entry = get_entry(cfg.type)
    if isinstance(entry, AgentVariant):
        return _build_variant_policy(entry, cfg)
    if entry is not None:
        return entry(cfg)

    raise ValueError(f"Unknown agent type {cfg.type!r}; available: {sorted(list_agents())}")


def _ctor_accepts(cls: type, name: str) -> bool:

    for klass in cls.__mro__:
        init = klass.__dict__.get("__init__")
        if init is None:
            continue
        params = inspect.signature(init).parameters
        if name in params:
            return True
        if not any(p.kind is p.VAR_KEYWORD for p in params.values()):
            return False
    return False


def _build_variant_policy(variant: AgentVariant, cfg: AgentConfig) -> Any:

    from mai_gym.agents.backends.http_chat_model import HttpChatModel

    rc = cfg.runtime_conf or {}
    extra = variant.extra or {}

    def _knob(key: str) -> Any:
        return rc.get(key, extra.get(key))

    server_pixels = bool(extra.get("release_server_pixels"))

    chat_model = HttpChatModel(
        url=cfg.base_url,
        model=cfg.model_name,
        tool_parser_name=variant.parser_name,
        api_key=cfg.api_key,
        timeout=rc.get("request_timeout", 600.0),
        preserve_thinking=variant.preserve_thinking,
        enable_thinking=_knob("enable_thinking"),
        max_pixels=_knob("max_pixels") if server_pixels else None,
        min_pixels=_knob("min_pixels") if server_pixels else None,
    )

    sampling_params: dict[str, Any] = {
        "max_new_tokens": rc.get("max_new_tokens", rc.get("max_response_len", 2048)),
        "temperature": rc.get("temperature", 0.0),
        "top_p": rc.get("top_p", 1.0),
    }

    if variant.stop_sequences:
        sampling_params["stop"] = list(variant.stop_sequences)
        sampling_params["no_stop_trim"] = True

    kwargs: dict[str, Any] = dict(
        chat_model=chat_model,
        sampling_params=sampling_params,
        original_image_width=rc.get("original_image_width"),
        original_image_height=rc.get("original_image_height"),
        resize_width=rc.get("resize_width"),
        resize_height=rc.get("resize_height"),
        system_prompt=rc.get("system_prompt", variant.system_prompt),
        image_retention=rc.get("image_retention", variant.default_image_retention),
    )
    for key in ("keep_n", "img_interval"):
        value = _knob(key)
        if value is not None and _ctor_accepts(variant.policy_cls, key):
            kwargs[key] = value
    if extra.get("windowed"):
        for key in ("window_size", "overlap"):
            value = _knob(key)
            if value is not None and _ctor_accepts(variant.policy_cls, key):
                kwargs[key] = value

    if not server_pixels:
        for key in ("resize_longest_edge", "max_pixels", "min_pixels", "image_factor"):
            val = rc.get(key)
            if val is not None and _ctor_accepts(variant.policy_cls, key):
                kwargs[key] = val
    return variant.policy_cls(**kwargs)


load_builtin_agents()
