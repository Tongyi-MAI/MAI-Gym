from typing import Any

BASH_SENTINEL = "__EXECUTE_BASH__:"


_NOOP_ACTIONS = frozenset(
    {
        "terminate",
        "answer",
        "finished",
        "WAIT",
        "DONE",
        "FAIL",
        "wait",
        "ask_user",
        "navigate_back",
    }
)


def action_to_pyautogui(action: dict[str, Any]) -> str | None:

    action_type = action.get("action_type", "")

    if action_type in _NOOP_ACTIONS:
        return None

    if action_type == "execute_bash":
        cmd = action.get("command") or ""
        if not cmd:
            return None
        return BASH_SENTINEL + cmd

    if action_type == "pyautogui_code":
        return action.get("code") or None

    if action_type == "CLICK":
        x = int(action.get("x", 0))
        y = int(action.get("y", 0))
        button = action.get("button", "left")
        clicks = int(action.get("num_clicks", 1))
        return f"pyautogui.click(x={x}, y={y}, button={button!r}, clicks={clicks})"

    if action_type == "RIGHT_CLICK":
        x = int(action.get("x", 0))
        y = int(action.get("y", 0))
        return f"pyautogui.rightClick(x={x}, y={y})"

    if action_type == "DOUBLE_CLICK":
        x = int(action.get("x", 0))
        y = int(action.get("y", 0))
        return f"pyautogui.doubleClick(x={x}, y={y})"

    if action_type == "MOVE_TO":
        x = int(action.get("x", 0))
        y = int(action.get("y", 0))
        return f"pyautogui.moveTo(x={x}, y={y}, duration=0.5)"

    if action_type == "DRAG_TO":
        x = int(action.get("x", 0))
        y = int(action.get("y", 0))
        return f"pyautogui.moveTo(x={x}, y={y}, duration=1.0)"

    if action_type == "MOUSE_DOWN":
        x = int(action.get("x", 0))
        y = int(action.get("y", 0))
        button = action.get("button", "left")
        return f"pyautogui.mouseDown(x={x}, y={y}, button={button!r})"

    if action_type == "MOUSE_UP":
        x = int(action.get("x", 0))
        y = int(action.get("y", 0))
        button = action.get("button", "left")
        return f"pyautogui.mouseUp(x={x}, y={y}, button={button!r})"

    if action_type == "SCROLL":
        x = int(action.get("x", 0))
        y = int(action.get("y", 0))
        dy = int(action.get("dy", action.get("clicks", 0)))
        return f"pyautogui.scroll({dy}, x={x}, y={y})"

    if action_type == "TYPING":
        text = action.get("text", "")
        return f"pyautogui.typewrite({text!r})"

    if action_type == "PRESS":
        key = action.get("key", "")
        return f"pyautogui.press({key!r})"

    if action_type == "KEY_DOWN":
        key = action.get("key", "")
        return f"pyautogui.keyDown({key!r})"

    if action_type == "KEY_UP":
        key = action.get("key", "")
        return f"pyautogui.keyUp({key!r})"

    if action_type == "HOTKEY":
        keys = action.get("keys", [])
        keys_str = ", ".join(repr(k) for k in keys)
        return f"pyautogui.hotkey({keys_str})"

    if action_type == "click":
        x = int(action.get("x", 0))
        y = int(action.get("y", 0))
        return f"pyautogui.click(x={x}, y={y})"

    if action_type == "type":
        text = action.get("text", "")
        return f"pyautogui.typewrite({text!r})"

    if action_type == "scroll":
        x = int(action.get("x", 0))
        y = int(action.get("y", 0))
        direction = action.get("direction", "down")
        clicks = 3 if direction in ("down", "right") else -3
        return f"pyautogui.scroll({clicks}, x={x}, y={y})"

    if action_type == "navigate_home":
        return None

    if action_type == "input_text":
        text = action.get("text", "")
        return f"pyautogui.typewrite({text!r}, interval=0.05)"

    if action_type == "double_tap":
        x = int(action.get("x", 0))
        y = int(action.get("y", 0))
        return f"pyautogui.doubleClick(x={x}, y={y})"

    if action_type == "long_press":
        x = int(action.get("x", 0))
        y = int(action.get("y", 0))
        return (
            f"pyautogui.mouseDown(x={x}, y={y}, button='left'); "
            f"time.sleep(1.0); "
            f"pyautogui.mouseUp(x={x}, y={y}, button='left')"
        )

    if action_type == "drag":
        sx = int(action.get("start_x", 0))
        sy = int(action.get("start_y", 0))
        ex = int(action.get("end_x", 0))
        ey = int(action.get("end_y", 0))
        return (
            f"pyautogui.moveTo(x={sx}, y={sy}, duration=0.2); "
            f"pyautogui.dragTo(x={ex}, y={ey}, duration=0.5, button='left')"
        )

    if action_type == "keyboard_enter":
        return "pyautogui.press('enter')"

    return None
