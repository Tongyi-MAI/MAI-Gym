import logging
from contextlib import suppress
from typing import Any

from playwright.async_api import Page

logger = logging.getLogger(__name__)


_SCROLL_DELTA = 200


_TOUCH_DRAG_STEPS = 10
_LONG_PRESS_MS = 500
_DOUBLE_TAP_GAP_MS = 100
_DEFAULT_WAIT_MS = 1000
_MAX_WAIT_MS = 10_000


_TERMINAL_ACTIONS = frozenset({"answer", "terminate", "finished", "status"})


def _to_css(val: int | float, scale: float) -> int:

    return int(val / scale)


async def _declarative_step(page: Page, action: dict[str, Any]) -> bool:

    result = await page.evaluate(
        """action => {
            const env = window.__ENV__;
            if (!env || typeof env.step !== "function") return null;
            return env.step(action);
        }""",
        action,
    )
    if result is None:
        return False
    if isinstance(result, dict) and result.get("ok") is False:
        raise RuntimeError(result.get("error") or f"declarative action failed: {action}")
    return True


def _wait_ms(action: dict[str, Any]) -> int:

    raw = action.get("waiting_time", action.get("duration", action.get("time", 1)))
    try:
        milliseconds = round(float(raw) * 1000)
    except (TypeError, ValueError):
        milliseconds = _DEFAULT_WAIT_MS
    return max(0, min(milliseconds, _MAX_WAIT_MS))


async def _touch_taps(
    page: Page,
    x: int,
    y: int,
    *,
    count: int = 1,
    hold_ms: int = 0,
) -> None:

    session = await page.context.new_cdp_session(page)
    touch_active = False
    try:
        for tap_index in range(count):
            await session.send(
                "Input.dispatchTouchEvent",
                {"type": "touchStart", "touchPoints": [{"x": x, "y": y}]},
            )
            touch_active = True
            if hold_ms:
                await page.wait_for_timeout(hold_ms)
            await session.send(
                "Input.dispatchTouchEvent",
                {"type": "touchEnd", "touchPoints": []},
            )
            touch_active = False
            if tap_index + 1 < count:
                await page.wait_for_timeout(_DOUBLE_TAP_GAP_MS)
    finally:
        if touch_active:
            with suppress(Exception):
                await session.send(
                    "Input.dispatchTouchEvent",
                    {"type": "touchCancel", "touchPoints": []},
                )
        with suppress(Exception):
            await session.detach()


async def _touch_drag(
    page: Page,
    start_x: int,
    start_y: int,
    end_x: int,
    end_y: int,
) -> None:

    session = await page.context.new_cdp_session(page)
    touch_active = False
    try:
        await session.send(
            "Input.dispatchTouchEvent",
            {
                "type": "touchStart",
                "touchPoints": [{"x": start_x, "y": start_y}],
            },
        )
        touch_active = True

        for step in range(1, _TOUCH_DRAG_STEPS + 1):
            progress = step / _TOUCH_DRAG_STEPS
            x = round(start_x + (end_x - start_x) * progress)
            y = round(start_y + (end_y - start_y) * progress)
            await session.send(
                "Input.dispatchTouchEvent",
                {
                    "type": "touchMove",
                    "touchPoints": [{"x": x, "y": y}],
                },
            )

        await session.send(
            "Input.dispatchTouchEvent",
            {"type": "touchEnd", "touchPoints": []},
        )
        touch_active = False
    finally:
        if touch_active:
            with suppress(Exception):
                await session.send(
                    "Input.dispatchTouchEvent",
                    {"type": "touchCancel", "touchPoints": []},
                )
        with suppress(Exception):
            await session.detach()


async def execute_action(
    page: Page,
    action: dict[str, Any],
    app_base_url: str,
    entry_page: str,
    *,
    device_scale_factor: float = 2.0,
) -> None:

    action_type = action.get("action_type", "")
    sf = device_scale_factor

    if action_type in _TERMINAL_ACTIONS:
        return

    if action_type == "click":
        x = _to_css(action["x"], sf)
        y = _to_css(action["y"], sf)
        await page.mouse.click(x, y, click_count=1)

    elif action_type == "move":
        await page.mouse.move(_to_css(action["x"], sf), _to_css(action["y"], sf))

    elif action_type == "double_tap":
        x = _to_css(action["x"], sf)
        y = _to_css(action["y"], sf)
        await _touch_taps(page, x, y, count=2)

    elif action_type == "long_press":
        x = _to_css(action["x"], sf)
        y = _to_css(action["y"], sf)
        await _touch_taps(page, x, y, hold_ms=_LONG_PRESS_MS)

    elif action_type in ("scroll", "swipe"):
        direction = action.get("direction", "down")
        dx, dy = 0, 0
        if action_type == "scroll":
            distance = action.get("pixels", _SCROLL_DELTA)
            if direction == "down":
                dy = distance
            elif direction == "up":
                dy = -distance
            elif direction == "right":
                dx = distance
            elif direction == "left":
                dx = -distance
        else:
            if direction == "down":
                dy = -_SCROLL_DELTA
            elif direction == "up":
                dy = _SCROLL_DELTA
            elif direction == "right":
                dx = -_SCROLL_DELTA
            elif direction == "left":
                dx = _SCROLL_DELTA
        x = _to_css(action.get("x", int(187 * sf)), sf)
        y = _to_css(action.get("y", int(406 * sf)), sf)
        await page.mouse.move(x, y)
        await page.mouse.wheel(dx, dy)

    elif action_type in ("type", "input_text"):
        text = action.get("text", "")
        await page.keyboard.type(text, delay=50)

    elif action_type == "navigate_home":
        if await _declarative_step(page, {"action": "home"}):
            return
        entry_url = f"{app_base_url}/{entry_page}"
        await page.goto(entry_url, wait_until="load")

    elif action_type == "navigate_back":
        if not await _declarative_step(page, {"action": "back"}):
            await page.go_back(wait_until="load")

    elif action_type == "keyboard_enter":
        await page.keyboard.press("Enter")

    elif action_type == "wait":
        if not await _declarative_step(page, {"action": "wait"}):
            await page.wait_for_timeout(_wait_ms(action))

    elif action_type == "key":
        key = action.get("key", "")
        if key:
            await page.keyboard.press(key)

    elif action_type == "drag":
        start_x = _to_css(action.get("start_x", action.get("x", 0)), sf)
        start_y = _to_css(action.get("start_y", action.get("y", 0)), sf)
        end_x = _to_css(action.get("end_x", start_x), sf)
        end_y = _to_css(action.get("end_y", start_y), sf)
        await _touch_drag(page, start_x, start_y, end_x, end_y)

    else:
        logger.warning("[ActionExecutor] Unknown action_type: %s", action_type)
