from __future__ import annotations

from typing import Any

from pydantic import BaseModel


class Observation(BaseModel):
    screenshot: Any
    accessibility_tree: Any = None
    ask_user_response: str | None = None
    tool_call: Any | None = None
