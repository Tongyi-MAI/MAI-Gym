from __future__ import annotations

import logging
import os
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, cast

import yaml

from mai_gym.utils.config_env import resolve_environment

logger = logging.getLogger(__name__)


@dataclass
class OSSCreds:
    endpoint: str = ""
    access_key_id: str = ""
    access_key_secret: str = ""
    region: str | None = None
    connection_pool_size: int = 20
    bucket: str | None = None


@dataclass
class SlimeAdapterConfig:
    manager_url: str = "http://localhost:8080"
    num_sessions: int = 32
    max_turns: int = 20
    image_retention: int = 3

    window_size: int = 5
    overlap: int = 1

    windowwise_train: bool = False

    token_native_history: bool = False
    reward_allocation: str = "uniform"
    action_format: str = "xml"

    resize_height: int | None = None
    resize_width: int | None = None

    resize_longest_edge: int | None = None

    max_pixels: int | None = None
    min_pixels: int | None = None

    image_factor: int | None = None

    original_image_width: int | None = None
    original_image_height: int | None = None

    num_backup_sessions: int = 0

    tool_parser: str = "mobile"

    policy_variant: Literal["mai_ui", "qwen35"] = "mai_ui"

    agent_variant: str = ""

    unified_variant_map: dict[str, str] | None = None

    env_type: str = "real_device"

    env_pools: dict[str, dict] | None = None

    max_total_length: int = 0

    messages_log_root: str | None = None

    oss: OSSCreds | None = None

    screenshot_transport: Literal["inline", "oss"] = "inline"

    train_id: str = ""

    fleet_request_mode: Literal["auto", "off", "required"] = "auto"
    rollout_agent: Literal["gui", "sdk"] = "gui"
    sdk_coordinate_space: Literal["pixels", "normalized", "normalized_1000"] = "normalized_1000"
    sdk_device: str | None = None

    sdk_reasoning_end_token: str | None = None

    @property
    def controller_url(self) -> str:
        return self.manager_url

    @controller_url.setter
    def controller_url(self, value: str) -> None:
        self.manager_url = value

    def max_turns_for_env(self, env_type: str | None = None) -> int:
        pool = (self.env_pools or {}).get(env_type or self.env_type, {})
        value = pool.get("max_turns", self.max_turns)
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError("Environment max_turns must be a positive integer")
        return value

    def validate_sdk(self) -> None:
        if self.rollout_agent not in {"gui", "sdk"}:
            raise ValueError(f"Unknown rollout_agent: {self.rollout_agent!r}")
        if self.rollout_agent != "sdk":
            return
        if self.windowwise_train or self.token_native_history:
            raise ValueError(
                "SDK rollout uses one sample per model call; disable windowwise/token-native history"
            )
        if self.max_turns < 1 or self.max_turns > 500 or self.image_retention < 0:
            raise ValueError("SDK requires 1..500 turns and nonnegative image_retention")
        if self.sdk_coordinate_space not in {"pixels", "normalized", "normalized_1000"}:
            raise ValueError("Unsupported SDK coordinate space")
        if self.env_type not in {"real_device", "mobileworld", "mobileworld_k8s"}:
            raise ValueError("SDK training v1 requires MobileWorld")
        if any(
            getattr(self, key) is not None
            for key in (
                "resize_height",
                "resize_width",
                "resize_longest_edge",
                "max_pixels",
                "min_pixels",
                "image_factor",
            )
        ):
            raise ValueError(
                "SDK v1 uses checkpoint processor image sizing; remove GUI policy resize overrides"
            )

    @classmethod
    def from_yaml(cls, path: str) -> SlimeAdapterConfig:
        with open(path) as f:
            data = resolve_environment(yaml.safe_load(f) or {})
        if not isinstance(data, dict):
            raise ValueError("Slime adapter config must be a YAML mapping")
        data = dict(data)
        for canonical, legacy in (
            ("manager_url", "controller_url"),
            ("max_total_length", "max_total_tokens"),
        ):
            if legacy in data:
                legacy_val = data.pop(legacy)
                if canonical in data:
                    if data[canonical] != legacy_val:
                        raise ValueError(f"Conflicting {canonical} and legacy {legacy}")
                else:
                    data[canonical] = legacy_val
        raw_pools = data.get("env_pools")
        if isinstance(raw_pools, dict):
            normalized_pools: dict[str, dict] = {}
            for env_name, pool in raw_pools.items():
                if isinstance(pool, dict) and "controller_url" in pool:
                    pool = dict(pool)
                    legacy_url = pool.pop("controller_url")
                    if "manager_url" in pool:
                        if pool["manager_url"] != legacy_url:
                            raise ValueError(
                                f"Conflicting manager_url and legacy controller_url in env_pools.{env_name}"
                            )
                    else:
                        pool["manager_url"] = legacy_url
                normalized_pools[env_name] = pool
            data["env_pools"] = normalized_pools
        known = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in data.items() if k in known}

        oss_data = filtered.pop("oss", None)
        if oss_data:
            filtered["oss"] = OSSCreds(**oss_data)
        return cls(**filtered)

    @classmethod
    def load(cls) -> SlimeAdapterConfig:

        path = os.environ.get("SLIME_ADAPTER_CONFIG")
        cfg = cls.from_yaml(path) if path and os.path.isfile(path) else cls()

        env_variant = os.environ.get("AGENT_VARIANT", "").strip()
        if env_variant:
            cfg.agent_variant = env_variant

        env_num_sessions = os.environ.get("NUM_SESSIONS", "").strip()
        if env_num_sessions:
            try:
                cfg.num_sessions = int(env_num_sessions)
            except ValueError:
                logger.warning(
                    "NUM_SESSIONS=%r is not an int; keeping YAML value %d",
                    env_num_sessions,
                    cfg.num_sessions,
                )
        env_max_turns = os.environ.get("MAX_TURNS", "").strip()
        if env_max_turns:
            try:
                cfg.max_turns = int(env_max_turns)
            except ValueError:
                logger.warning(
                    "MAX_TURNS=%r is not an int; keeping YAML value %d",
                    env_max_turns,
                    cfg.max_turns,
                )
        env_transport = os.environ.get("MAI_GYM_SCREENSHOT_TRANSPORT", "").strip()
        if env_transport in ("inline", "oss"):
            cfg.screenshot_transport = cast(Literal["inline", "oss"], env_transport)
        elif env_transport:
            logger.warning(
                "MAI_GYM_SCREENSHOT_TRANSPORT=%r is invalid; keeping %s",
                env_transport,
                cfg.screenshot_transport,
            )

        env_windowwise = os.environ.get("WINDOWWISE_TRAIN", "").strip().lower()
        if env_windowwise in ("1", "true", "yes", "on"):
            cfg.windowwise_train = True
        elif env_windowwise in ("0", "false", "no", "off"):
            cfg.windowwise_train = False
        elif env_windowwise:
            logger.warning(
                "WINDOWWISE_TRAIN=%r is invalid; keeping %s",
                env_windowwise,
                cfg.windowwise_train,
            )

        env_train_id = os.environ.get("MAI_GYM_TRAIN_ID", "").strip()
        if env_train_id:
            cfg.train_id = env_train_id
        elif not cfg.train_id:
            hint = cfg.agent_variant or (Path(path).stem if path else "run")
            cfg.train_id = f"{time.strftime('%Y%m%d_%H%M')}_{hint}_{uuid.uuid4().hex[:6]}"
        cfg.validate_sdk()
        logger.info("slime adapter train_id=%s", cfg.train_id)
        return cfg
