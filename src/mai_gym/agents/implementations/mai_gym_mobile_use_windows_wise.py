from __future__ import annotations

from typing import Any

from .mai_gym_mobile_use import (
    HISTORY_IMAGE_PLACEHOLDER,
    MAIGymMobileUsePolicy,
    _is_image_turn,
    render_prompt_template,
)

WINDOW_SIZE = 10
OVERLAP = 2

ROLLOUT_APPS: list[str] = [
    "Contacts",
    "Settings",
    "Clock",
    "Maps",
    "Chrome",
    "Calendar",
    "files",
    "Gallery",
    "Taodian",
    "Mattermost",
    "Mastodon",
    "Mail",
    "SMS",
    "Camera",
]


def render_system_prompt(
    apps: list[str] | None = None,
    *,
    tools: str | list[dict[str, Any]] | None = None,
) -> str:
    return render_prompt_template(
        "mai_gym_mobile_use_windows_wise_system.j2",
        apps=apps if apps is not None else ROLLOUT_APPS,
        tools=tools,
    )


SYSTEM_PROMPT = render_system_prompt()


def window_start(
    step_idx: int,
    window_size: int = WINDOW_SIZE,
    overlap: int = OVERLAP,
) -> int:
    stride = window_size - overlap
    if stride <= 0:
        raise ValueError(
            f"window_size - overlap must be > 0 (W={window_size}, OV={overlap})"
        )
    return max(0, stride * ((step_idx - overlap) // stride))


def collapse_to_window(
    messages: list[dict[str, Any]],
    step_idx: int,
    window_size: int = WINDOW_SIZE,
    overlap: int = OVERLAP,
) -> list[dict[str, Any]]:
    start = window_start(step_idx, window_size, overlap)
    image_ordinal = -1
    out: list[dict[str, Any]] = []
    for message in messages:
        if _is_image_turn(message):
            image_ordinal += 1
            if image_ordinal < start:
                out.append({"role": "user", "content": HISTORY_IMAGE_PLACEHOLDER})
                continue
        out.append(message)
    return out


class MAIGymMobileUseWindowsWisePolicy(MAIGymMobileUsePolicy):
    """Mobile-use agent with a W=10, overlap=2 sawtooth history window."""

    def __init__(
        self,
        *args: Any,
        window_size: int = WINDOW_SIZE,
        overlap: int = OVERLAP,
        system_prompt: str | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            *args,
            system_prompt=SYSTEM_PROMPT if system_prompt is None else system_prompt,
            **kwargs,
        )
        self.window_size = window_size
        self.overlap = overlap

    def _prune_messages(self) -> list[dict[str, Any]]:
        return collapse_to_window(
            self._messages,
            step_idx=len(self.step_records),
            window_size=self.window_size,
            overlap=self.overlap,
        )


__all__ = [
    "MAIGymMobileUseWindowsWisePolicy",
    "OVERLAP",
    "ROLLOUT_APPS",
    "SYSTEM_PROMPT",
    "WINDOW_SIZE",
    "collapse_to_window",
    "render_system_prompt",
    "window_start",
]
