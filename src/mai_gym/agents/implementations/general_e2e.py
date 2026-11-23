from __future__ import annotations

import logging

from mai_gym.agents._base import AgentVariant
from mai_gym.agents.implementations.qwen35_policy import Qwen35SLIMEPolicy
from mai_gym.agents.prompts import render
from mai_gym.agents.variants import register_variant

logger = logging.getLogger(__name__)


SYSTEM_PROMPT = render("qwen35_mobile_mw.j2").strip()

register_variant(
    AgentVariant(
        name="general_e2e",
        policy_cls=Qwen35SLIMEPolicy,
        parser_name="general_e2e",
        system_prompt=SYSTEM_PROMPT,
        default_image_retention=3,
        description="Qwen3.5 XML tool-call agent (original)",
    )
)
