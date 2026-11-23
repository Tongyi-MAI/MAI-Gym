from __future__ import annotations

import json
import logging
from typing import Any

logger = logging.getLogger(__name__)


MOCK_LOCAL_STORAGE_KEYS: dict[str, tuple[str, str]] = {
    "gmail_mock": ("xmail-clone-state", "xmail-clone-initialState"),
    "notion_mock": ("xotion-clone-state", "xotion-clone-initialState"),
}


_READ_JS = """
([curKey, initKey]) => {
    const _read = (key) => {
        try {
            const raw = localStorage.getItem(key);
            return raw == null ? null : JSON.parse(raw);
        } catch (e) {
            return null;
        }
    };
    return {
        current: _read(curKey),
        initial: _read(initKey),
    };
}
"""


async def read_state_from_browser(
    page: Any,
    app_slug: str | None,
    sid: str | None,
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:

    if not app_slug or not isinstance(app_slug, str):
        return None, None
    prefixes = MOCK_LOCAL_STORAGE_KEYS.get(app_slug)
    if prefixes is None:
        logger.warning(
            "[cua_hub] no localStorage key map for app_slug=%r — "
            "reward will see the broken-sync /go state; add an entry to "
            "MOCK_LOCAL_STORAGE_KEYS in state_extraction.py.",
            app_slug,
        )
        return None, None
    if not hasattr(page, "evaluate"):
        return None, None

    cur_base, init_base = prefixes
    suffix = f"_{sid}" if sid else ""
    cur_key = f"{cur_base}{suffix}"
    init_key = f"{init_base}{suffix}"

    try:
        result = await page.evaluate(_READ_JS, [cur_key, init_key])
    except Exception as exc:
        logger.warning(
            "[cua_hub] page.evaluate failed for app_slug=%s sid=%s: %s",
            app_slug,
            sid,
            exc,
        )
        return None, None

    if not isinstance(result, dict):
        return None, None
    current = result.get("current")
    initial = result.get("initial")
    if not isinstance(current, dict) and not isinstance(initial, dict):
        return None, None

    if initial is None and isinstance(current, dict):
        initial = current
    return (
        initial if isinstance(initial, dict) else None,
        current if isinstance(current, dict) else None,
    )


def compute_state_diff(
    initial: dict[str, Any] | None,
    current: dict[str, Any] | None,
) -> dict[str, Any]:

    if not isinstance(current, dict):
        return {}
    init = initial if isinstance(initial, dict) else {}
    diff: dict[str, Any] = {}
    for key, cur_val in current.items():
        init_val = init.get(key)
        if json.dumps(cur_val, sort_keys=True, default=str) != json.dumps(
            init_val,
            sort_keys=True,
            default=str,
        ):
            diff[key] = {"added": cur_val} if not init_val else {"modified": cur_val}
    return diff
