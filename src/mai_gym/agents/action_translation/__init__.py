from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from mai_gym.agents.action_space import ActionSpace
from mai_gym.agents.action_translation.canonical import Canonical
from mai_gym.agents.action_translation.parser import PointerState, parse_pyautogui
from mai_gym.agents.action_translation.targets import to_mobile, to_web

logger = logging.getLogger(__name__)


_TERMINAL_PASSTHROUGH: frozenset[str] = frozenset(
    {
        "wait",
        "finished",
        "fail",
        "done",
        "terminate",
        "answer",
    }
)


_PASSTHROUGH_FIELDS: tuple[str, ...] = (
    "prediction",
    "reasoning_content",
    "status",
    "text",
)


@dataclass
class TranslationCounters:
    translated: int = 0
    noop_unmappable: int = 0
    noop_unrecognised: int = 0
    multi_action_dropped: int = 0
    failed_parse: int = 0
    by_kind: dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "translated": self.translated,
            "noop_unmappable": self.noop_unmappable,
            "noop_unrecognised": self.noop_unrecognised,
            "multi_action_dropped": self.multi_action_dropped,
            "failed_parse": self.failed_parse,
            "by_kind": dict(self.by_kind),
        }


def _wait_action() -> dict[str, Any]:
    return {"action_type": "wait"}


def _carry_passthrough(src: dict[str, Any], dst: dict[str, Any]) -> dict[str, Any]:

    for key in _PASSTHROUGH_FIELDS:
        if key in src and key not in dst:
            dst[key] = src[key]
    return dst


def translate(
    action: dict[str, Any],
    src: ActionSpace | str,
    dst: ActionSpace | str,
    *,
    screen_size: tuple[int, int],
    counters: TranslationCounters | None = None,
    mobile_pixel_coords: bool = False,
    preserve_sequence: bool = False,
    pointer: PointerState | None = None,
    mobile_keyboard: bool = False,
) -> dict[str, Any]:

    src, dst = ActionSpace(src), ActionSpace(dst)

    if counters is None:
        counters = TranslationCounters()

    if src == dst:
        return action

    if preserve_sequence and dst is ActionSpace.MOBILE_USE and pointer is None:
        pointer = PointerState()

    action_type = action.get("action_type", "")
    if preserve_sequence and action_type == "action_sequence":
        source_children: list[dict[str, Any]] = []
        for child in action.get("actions", []):
            if (
                src is ActionSpace.COMPUTER_USE
                and child.get("action_type") == "pyautogui_code"
                and source_children
                and source_children[-1].get("action_type") == "pyautogui_code"
            ):
                previous = source_children[-1]
                previous["code"] = previous["code"] + "\n" + child["code"]
            else:
                source_children.append(dict(child))
        children = [
            translate(
                child,
                src,
                dst,
                screen_size=screen_size,
                counters=counters,
                mobile_pixel_coords=mobile_pixel_coords,
                preserve_sequence=True,
                pointer=pointer,
                mobile_keyboard=mobile_keyboard,
            )
            for child in source_children
        ]
        return _carry_passthrough(action, {"action_type": "action_sequence", "actions": children})
    if action_type in _TERMINAL_PASSTHROUGH:
        return action

    if src is ActionSpace.COMPUTER_USE:
        return _translate_cua(
            action,
            dst,
            screen_size,
            counters,
            mobile_pixel_coords=mobile_pixel_coords,
            preserve_sequence=preserve_sequence,
            pointer=pointer,
            mobile_keyboard=mobile_keyboard,
        )

    if src is ActionSpace.MOBILE_USE and dst is ActionSpace.WEB_USE:
        counters.translated += 1
        return action

    logger.warning(
        "translate(): no translator implemented for %s -> %s; falling back to wait",
        src,
        dst,
    )
    counters.noop_unmappable += 1
    return _carry_passthrough(action, _wait_action())


def _translate_cua(
    action: dict[str, Any],
    dst: ActionSpace,
    screen_size: tuple[int, int],
    counters: TranslationCounters,
    *,
    mobile_pixel_coords: bool = False,
    preserve_sequence: bool = False,
    pointer: PointerState | None = None,
    mobile_keyboard: bool = False,
) -> dict[str, Any]:

    action_type = action.get("action_type", "")

    if action_type != "pyautogui_code":
        logger.warning(
            "translate(): COMPUTER_USE action_type=%r is not pyautogui_code; forwarding unchanged",
            action_type,
        )
        return action

    code = action.get("code", "")
    if not isinstance(code, str) or not code.strip():
        counters.failed_parse += 1
        return _carry_passthrough(action, _wait_action())

    try:
        canonicals = parse_pyautogui(
            code,
            pointer=pointer,
            preserve_clicks=preserve_sequence,
            preserve_moves=preserve_sequence and dst is ActionSpace.WEB_USE,
        )
    except SyntaxError:
        logger.warning("translate(): pyautogui code failed to parse: %r", code[:120])
        counters.failed_parse += 1
        return _carry_passthrough(action, _wait_action())

    if not canonicals:
        counters.noop_unrecognised += 1
        return _carry_passthrough(action, _wait_action())

    for c in canonicals:
        counters.by_kind[c.kind] = counters.by_kind.get(c.kind, 0) + 1

    if preserve_sequence:
        translated_actions = []
        for canonical in canonicals:
            translated = (
                to_mobile(canonical, screen_size, pixel_coords=mobile_pixel_coords)
                if dst is ActionSpace.MOBILE_USE
                and not (mobile_keyboard and canonical.kind in {"key", "hotkey"})
                else to_web(canonical, screen_size, extended=True)
            )
            if translated is None:
                counters.noop_unmappable += 1
                raise ValueError(f"Native action cannot be translated: {canonical.raw}")
            translated_actions.append(translated)
            counters.translated += 1
        result = (
            translated_actions[0]
            if len(translated_actions) == 1
            else {"action_type": "action_sequence", "actions": translated_actions}
        )
        return _carry_passthrough(action, result)

    primary, *rest = canonicals
    if rest:
        counters.multi_action_dropped += len(rest)
        logger.debug(
            "translate(): %d extra Canonical(s) dropped after %s",
            len(rest),
            primary.kind,
        )

    if dst is ActionSpace.MOBILE_USE:
        translated = to_mobile(
            primary,
            screen_size,
            pixel_coords=mobile_pixel_coords,
        )
    else:
        translated = to_web(primary, screen_size)
    if translated is None:
        logger.info(
            "translate(): %s -> %s has no mapping for Canonical(kind=%s, raw=%r); emitting wait",
            ActionSpace.COMPUTER_USE.value,
            dst.value,
            primary.kind,
            primary.raw,
        )
        counters.noop_unmappable += 1
        return _carry_passthrough(action, _wait_action())

    counters.translated += 1
    return _carry_passthrough(action, translated)


__all__ = [
    "ActionSpace",
    "Canonical",
    "TranslationCounters",
    "parse_pyautogui",
    "to_mobile",
    "to_web",
    "translate",
]
