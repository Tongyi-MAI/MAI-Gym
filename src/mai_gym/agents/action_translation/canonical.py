from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

CanonicalKind = Literal[
    "click",
    "double_click",
    "right_click",
    "drag",
    "move",
    "type",
    "key",
    "hotkey",
    "scroll",
    "wait",
    "noop",
]


@dataclass(frozen=True)
class Canonical:
    kind: CanonicalKind
    x: int | None = None
    y: int | None = None
    end_x: int | None = None
    end_y: int | None = None
    text: str | None = None
    key: str | None = None
    keys: tuple[str, ...] = ()
    scroll_clicks: int = 0
    duration: float = 0.0
    button: str = "left"
    raw: str = ""

    @property
    def origin(self) -> tuple[int, int] | None:

        if self.x is None or self.y is None:
            return None
        return (self.x, self.y)


def noop(raw: str) -> Canonical:

    return Canonical(kind="noop", raw=raw)


__all__ = ["Canonical", "CanonicalKind", "noop"]
