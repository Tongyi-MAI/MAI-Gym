from __future__ import annotations

import logging
import os
import re
from collections.abc import Callable
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

logger = logging.getLogger(__name__)

_ENV_FLAG = "MAI_GYM_VLC_PLAYING_FALLBACK"
_FALSEY = {"0", "false", "no", "off"}


def vlc_playing_fallback_enabled() -> bool:

    return os.environ.get(_ENV_FLAG, "1").strip().lower() not in _FALSEY


def make_vlc_playing_metric(
    original_metric: Callable[..., Any],
    env: Any,
    vm_command_getter: Callable[[Any, dict[str, Any]], Any],
) -> Callable[..., float]:

    def _metric(actual_status_path: str | None, rule: dict[str, Any]) -> float:
        original_score = 0.0
        try:
            original_score = float(original_metric(actual_status_path, rule))
        except Exception as exc:
            logger.info("upstream is_vlc_playing failed; considering fallback: %s", exc)
        if original_score == 1.0 or not vlc_playing_fallback_enabled():
            return original_score

        try:
            evidence = vm_command_getter(
                env, {"type": "vm_command_line", "command": _probe_command()}
            )
        except Exception as exc:
            logger.warning("VLC fallback probe failed: %s", exc)
            return original_score
        return 1.0 if _probe_matches(evidence or "", rule) else original_score

    return _metric


def _probe_command() -> list[str]:
    return [
        "/bin/bash",
        "-lc",
        (
            "echo __PGREP__; "
            "pgrep -af '[v]lc' || true; "
            "echo __WINDOWS__; "
            "wmctrl -lpx 2>/dev/null | grep -i vlc || true; "
            "echo __MPRIS__; "
            "if command -v playerctl >/dev/null 2>&1; then "
            "  playerctl -l 2>/dev/null | grep -i vlc | while read -r p; do "
            "    echo player=$p; "
            '    playerctl -p "$p" status 2>/dev/null || true; '
            '    playerctl -p "$p" metadata xesam:url 2>/dev/null || true; '
            '    playerctl -p "$p" metadata xesam:title 2>/dev/null || true; '
            "  done; "
            "fi; "
            "echo __LSOF__; "
            "lsof -Fn -c vlc 2>/dev/null || true"
        ),
    ]


def _probe_matches(evidence: str, rule: dict[str, Any]) -> bool:
    if "vlc" not in evidence.lower():
        return False
    rule_type = str(rule.get("type", "")).strip()
    if rule_type == "file_name":
        return _file_rule_matches(evidence, str(rule.get("file_name", "")).strip())
    if rule_type == "url":
        return _url_rule_matches(evidence, str(rule.get("url", "")).strip())
    return False


def _file_rule_matches(evidence: str, expected: str) -> bool:
    if not expected:
        return False
    expected_path = f"/home/user/Desktop/{expected}"
    lines = [line.strip() for line in evidence.splitlines() if line.strip()]
    process_lines = [line for line in lines if "vlc" in line.lower() or line.startswith("n/")]
    joined = "\n".join(process_lines)
    return expected_path in joined or expected in joined


def _url_rule_matches(evidence: str, expected_url: str) -> bool:
    if not expected_url:
        return False
    if expected_url in evidence:
        return True

    parsed = urlparse(expected_url)
    expected_filename = Path(parsed.path).name
    if expected_filename and expected_filename in evidence:
        return True

    expected_host = (parsed.hostname or "").lower()
    expected_path = parsed.path
    for candidate in re.findall(r"https?://\S+", evidence):
        parsed_candidate = urlparse(candidate.rstrip("\"'<>),;"))
        if (parsed_candidate.hostname or "").lower() != expected_host:
            continue
        if expected_path and expected_path in parsed_candidate.path:
            return True
    return False


__all__ = [
    "make_vlc_playing_metric",
    "vlc_playing_fallback_enabled",
]
