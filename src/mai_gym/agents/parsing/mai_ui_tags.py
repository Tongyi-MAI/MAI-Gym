from __future__ import annotations

import json
import re
from typing import Any


def parse_tagged_text(text: str) -> dict[str, Any]:

    if "</think>" in text and "</thinking>" not in text:
        text = text.replace("</think>", "</thinking>")
        text = "<thinking>" + text

    pattern = r"<thinking>(.*?)</thinking>.*?<tool_call>(.*?)</tool_call>"

    result: dict[str, Any] = {
        "thinking": None,
        "tool_call": None,
    }

    match = re.search(pattern, text, re.DOTALL)
    if match:
        result = {
            "thinking": match.group(1).strip().strip('"'),
            "tool_call": match.group(2).strip().strip('"'),
        }

    if result["tool_call"]:
        try:
            result["tool_call"] = json.loads(result["tool_call"])
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in tool_call: {e}") from e

    return result
