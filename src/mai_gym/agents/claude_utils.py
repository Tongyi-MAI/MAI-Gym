from __future__ import annotations

import json
import re
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime


@dataclass(frozen=True)
class ClaudeRuntimeProfile:
    label: str
    think_mode: str
    default_max_tokens: int


_CLAUDE_45 = ClaudeRuntimeProfile("Claude 4.5 legacy", "legacy", 3000)
_CLAUDE_46 = ClaudeRuntimeProfile("Claude 4.6 legacy", "legacy", 3000)
_CLAUDE_47 = ClaudeRuntimeProfile("Claude 4.7 adaptive", "adaptive", 16000)
_CLAUDE_48 = ClaudeRuntimeProfile("Claude 4.8 adaptive", "adaptive", 16000)
_DEFAULT_PROFILE = _CLAUDE_48


def get_claude_runtime_profile(model_name: str) -> ClaudeRuntimeProfile:
    n = (model_name or "").lower().replace("_", "-").replace(".", "-")
    if "4-8" in n:
        return _CLAUDE_48
    if "4-7" in n:
        return _CLAUDE_47
    if "4-6" in n:
        return _CLAUDE_46
    if "4-5" in n:
        return _CLAUDE_45
    return _DEFAULT_PROFILE


SYSTEM_PROMPT = """<SYSTEM_CAPABILITY>
* You are utilising an Ubuntu virtual machine using x86_64 architecture with internet access.
* You can feel free to install Ubuntu applications with your bash action. Use curl instead of wget.
* To open the browser, please just click on the Chrome icon. Note, Chrome is what is installed on your system.
* When viewing a page it can be helpful to zoom out so that you can see everything on the page. Either that, or make sure you scroll down to see everything before deciding something isn't available.
* DO NOT ask users for clarification during task execution. DO NOT stop to request more information from users. Always take action using the available actions.
* When using your computer actions, they take a while to run and send back to you. Where possible/feasible, try to chain multiple of these actions into one batched request.
* TASK FEASIBILITY: You can declare a task infeasible at any point during execution — whether at the beginning after taking a screenshot, or later after attempting some actions and discovering barriers. If you determine that a task cannot be completed due to:
  - Missing required applications or dependencies that cannot be installed
  - Insufficient permissions or system limitations
  - Contradictory or impossible requirements
  - Any other fundamental barriers that make completion impossible
  Then you MUST output exactly "[INFEASIBLE]" (including the square brackets) anywhere in your response to trigger the fail action.
* The current date is {date}.
* Home directory of this Ubuntu system is '/home/user'.
* If you need a password for sudo, the password of the computer is '{password}'.
</SYSTEM_CAPABILITY>

<IMPORTANT>
* If the item you are looking at is a pdf, if after taking a single screenshot of the pdf it seems that you want to read the entire document, determine the URL, use curl to download the pdf, install and use pdftotext to convert it to a text file, and then read that text file directly.
</IMPORTANT>"""


_ACTION_SPACE_BASE = """<ACTION_SPACE>
The screen resolution is 1280x720. ALL coordinates you output MUST be in this
1280x720 pixel space ([0,0] = top-left, [1279,719] = bottom-right).

You interact with the computer by emitting one or more *actions*. Each action
is a JSON object with an ``action`` field plus that action's parameters:

- {"action": "left_click", "coordinate": [x, y]}  — left-click at (x,y). Optional "text" = modifier key(s) to hold, e.g. "ctrl".
- {"action": "right_click", "coordinate": [x, y]}
- {"action": "middle_click", "coordinate": [x, y]}
- {"action": "double_click", "coordinate": [x, y]}
- {"action": "triple_click", "coordinate": [x, y]}
- {"action": "mouse_move", "coordinate": [x, y]}  — move cursor to (x,y).
- {"action": "left_click_drag", "coordinate": [x, y], "start_coordinate": [sx, sy]}  — drag from start to (x,y).
- {"action": "left_mouse_down"} / {"action": "left_mouse_up"}  — press/release the left button (for fine-grained drags).
- {"action": "type", "text": "hello world"}  — type a literal string (no auto-Enter).
- {"action": "key", "text": "ctrl+s"}  — press a key combo. Optional "repeat": N to repeat.
- {"action": "hold_key", "text": "shift"}  — hold a key down (pair with later actions).
- {"action": "scroll", "coordinate": [x, y], "scroll_direction": "up|down|left|right", "scroll_amount": N}  — scroll N clicks at (x,y). Optional "text" = modifier to hold.
- {"action": "wait", "duration": S}  — wait S seconds for the UI to settle / load.
- {"action": "screenshot"}  — take a fresh screenshot (use only when you need to re-observe).

To END the task, emit ONE of:
- {"action": "done"}  — the task is complete and verified.
- {"action": "fail"}  — the task is infeasible / cannot be completed.
- {"action": "call_user"}  — you need the user (rarely; treated as a stop)."""


_ACTION_SPACE_ZOOM_REMAP = """
- {"action": "zoom", "region": [x0, y0, x1, y1]}  — magnify a rectangular region of the CURRENT screenshot. The "region" is given in the normal full-screen 1280x720 coordinates. This does NOT change the computer state; it returns a magnified crop. After a zoom, your VERY NEXT action's coordinate must be measured ON the magnified crop (its top-left = [0,0]); the system maps it back to the real screen automatically."""


_ACTION_SPACE_ZOOM_VIEW = """
- {"action": "zoom", "region": [x0, y0, x1, y1]}  — magnify a rectangular region of the CURRENT screenshot, given in the normal full-screen 1280x720 coordinates, so you can read small text / inspect detail. This is for LOOKING ONLY: it does NOT change the computer state, does NOT move the cursor, and does NOT change the coordinate system. ALL coordinates you give — including your very next action — stay in the normal full-screen 1280x720 space (do NOT measure them on the magnified crop)."""

_ACTION_SPACE_BASH = """
- {"action": "bash", "text": "ls -la /home/user"}  — run a shell command and receive its stdout/stderr back on the NEXT turn. PREFER bash for file operations, inspecting state, installing packages, and any task achievable from the command line. You MAY place a bash action in the same ``actions`` list as GUI actions — they execute strictly in the order you write them. IMPORTANT: a later action in the SAME batch CANNOT see an earlier bash's output (that output arrives next turn), so only batch a bash before actions that do NOT depend on its result; otherwise emit the bash alone and read its output next turn."""

_ACTION_SPACE_CLOSE = "\n</ACTION_SPACE>"


BATCHED_ACTION_PROMPT = """CRITICAL — BATCH ACTIONS: combine as many actions as possible into a SINGLE
response. Each round trip has significant latency, so batching is essential.
Do NOT take a screenshot after every action — only re-observe when the next
step genuinely depends on seeing the result. For form filling, menu navigation
and keyboard shortcuts you almost never need a screenshot between actions:
batch them all together (clicks, types, keypresses in one ``actions`` list)."""


_OUTPUT_FORMAT = """<OUTPUT_FORMAT>
First, briefly reason about the current screen and what to do next. Then output
your action(s) as a SINGLE JSON object inside a ```json fenced code block:

```json
{"actions": [
  {"action": "left_click", "coordinate": [640, 360]},
  {"action": "type", "text": "hello"}
]}
```

Rules:
- The ``actions`` array runs in order, in one batch.
- Emit the JSON block exactly once, as the LAST thing in your reply.
- Coordinates are in 1280x720 space.
- To finish, make the array a single {"action": "done"} (or "fail").
</OUTPUT_FORMAT>"""


def build_system_prompt(
    password: str,
    *,
    enable_zoom: bool,
    enable_bash: bool,
    max_steps: int,
    suffix: str = "",
    zoom_remap: bool = False,
) -> str:

    action_space = _ACTION_SPACE_BASE
    if enable_bash:
        action_space += _ACTION_SPACE_BASH
    if enable_zoom:
        action_space += _ACTION_SPACE_ZOOM_REMAP if zoom_remap else _ACTION_SPACE_ZOOM_VIEW
    action_space += _ACTION_SPACE_CLOSE

    parts = [
        SYSTEM_PROMPT.replace("{password}", password).replace(
            "{date}",
            datetime.now().strftime("%A, %B %d, %Y"),
        ),
        action_space,
        BATCHED_ACTION_PROMPT,
        _OUTPUT_FORMAT,
        f"<STEP_LIMIT>You have a maximum of {max_steps} steps. Use them wisely.</STEP_LIMIT>",
    ]
    if suffix:
        parts.append(suffix)
    return "\n\n".join(parts)


_JSON_FENCE_RE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL)


def extract_actions(text: str) -> list[dict] | None:

    candidates: list[str] = _JSON_FENCE_RE.findall(text or "")
    if not candidates:
        idx = (text or "").rfind('"actions"')
        if idx != -1:
            start = text.rfind("{", 0, idx)
            if start != -1:
                depth = 0
                for j in range(start, len(text)):
                    if text[j] == "{":
                        depth += 1
                    elif text[j] == "}":
                        depth -= 1
                        if depth == 0:
                            candidates = [text[start : j + 1]]
                            break
    for blob in reversed(candidates):
        try:
            obj = json.loads(blob)
        except (ValueError, TypeError):
            continue
        if isinstance(obj, dict) and isinstance(obj.get("actions"), list):
            return obj["actions"]
        if isinstance(obj, dict) and "action" in obj:
            return [obj]
    return None


_KEY_MAP = {
    "page_down": "pagedown",
    "page_up": "pageup",
    "super_l": "win",
    "super": "win",
    "escape": "esc",
    "return": "enter",
}


DONE = "DONE"
FAIL = "FAIL"
CALL_USER = "CALL_USER"


def action_to_pyautogui(
    act: dict,
    resize_factor: tuple[float, float],
    *,
    drag_duration: float = 0.5,
    move_duration: float = 0.2,
    coord_fn: Callable[[float, float], tuple[int, int]] | None = None,
) -> str:

    result = ""
    action = act.get("action", "")
    text = act.get("text")
    coordinate = act.get("coordinate")
    start_coordinate = act.get("start_coordinate")
    scroll_direction = act.get("scroll_direction")
    scroll_amount = act.get("scroll_amount")
    duration = act.get("duration")
    repeat = int(act.get("repeat") or 1)

    def _map(x: float, y: float) -> tuple[int, int]:
        if coord_fn is not None:
            return coord_fn(x, y)
        return int(x * resize_factor[0]), int(y * resize_factor[1])

    if coordinate:
        coordinate = _map(coordinate[0], coordinate[1])
    if start_coordinate:
        start_coordinate = _map(start_coordinate[0], start_coordinate[1])

    if action == "left_mouse_down":
        result += "pyautogui.mouseDown()\n"
    elif action == "left_mouse_up":
        result += "pyautogui.mouseUp()\n"
    elif action == "hold_key":
        if isinstance(text, str):
            for key in text.split("+"):
                result += f"pyautogui.keyDown('{key.strip().lower()}')\n"
    elif action in ("mouse_move", "left_click_drag"):
        if coordinate is None:
            return ""
        x, y = coordinate
        if action == "mouse_move":
            result += f"pyautogui.moveTo({x}, {y}, duration={duration or move_duration})\n"
        else:
            if start_coordinate:
                sx, sy = start_coordinate
                result += f"pyautogui.moveTo({sx}, {sy}, duration={duration or move_duration})\n"
            result += f"pyautogui.dragTo({x}, {y}, duration={duration or drag_duration})\n"
    elif action == "key":
        if not isinstance(text, str):
            return ""
        keys = text.split("+")
        for _ in range(repeat):
            for k in keys:
                kk = _KEY_MAP.get(k.strip().lower(), k.strip().lower())
                result += f"pyautogui.keyDown('{kk}')\n"
            for k in reversed(keys):
                kk = _KEY_MAP.get(k.strip().lower(), k.strip().lower())
                result += f"pyautogui.keyUp('{kk}')\n"
    elif action == "type":
        if not isinstance(text, str):
            return ""
        for char in text:
            if char == "\n":
                result += "pyautogui.press('enter')\n"
            elif char == "'":
                result += 'pyautogui.press("\'")\n'
            elif char == "\\":
                result += "pyautogui.press('\\\\')\n"
            elif char == '"':
                result += "pyautogui.press('\"')\n"
            else:
                result += f"pyautogui.press('{char}')\n"
    elif action == "scroll":
        if text is not None:
            result += f"pyautogui.keyDown('{str(text).lower()}')\n"
        amt_n = scroll_amount if scroll_amount is not None else 3
        if coordinate is None:
            if scroll_direction in ("up", "down"):
                result += f"pyautogui.scroll({amt_n if scroll_direction == 'up' else -amt_n})\n"
            elif scroll_direction in ("left", "right"):
                result += f"pyautogui.hscroll({amt_n if scroll_direction == 'right' else -amt_n})\n"
        else:
            x, y = coordinate
            if scroll_direction in ("up", "down"):
                result += (
                    f"pyautogui.scroll({amt_n if scroll_direction == 'up' else -amt_n}, {x}, {y})\n"
                )
            elif scroll_direction in ("left", "right"):
                result += f"pyautogui.hscroll({amt_n if scroll_direction == 'right' else -amt_n}, {x}, {y})\n"
        if text is not None:
            result += f"pyautogui.keyUp('{str(text).lower()}')\n"
    elif action in (
        "left_click",
        "right_click",
        "double_click",
        "middle_click",
        "left_press",
        "triple_click",
    ):
        if text:
            for k in str(text).split("+"):
                result += f"pyautogui.keyDown('{k.strip().lower()}')\n"
        coord_str = f"{coordinate[0]}, {coordinate[1]}" if coordinate is not None else ""
        fn = {
            "left_click": f"pyautogui.click({coord_str})\n",
            "right_click": f"pyautogui.rightClick({coord_str})\n",
            "double_click": f"pyautogui.doubleClick({coord_str})\n",
            "middle_click": f"pyautogui.middleClick({coord_str})\n",
            "triple_click": f"pyautogui.tripleClick({coord_str})\n",
            "left_press": (
                f"pyautogui.mouseDown({coord_str})\ntime.sleep(1)\npyautogui.mouseUp({coord_str})\n"
            ),
        }
        result += fn.get(action, "")
        if text:
            for k in reversed(str(text).split("+")):
                result += f"pyautogui.keyUp('{k.strip().lower()}')\n"
    elif action == "wait":
        result += f"pyautogui.sleep({duration or 0.5})\n"
    elif action == "fail":
        return FAIL
    elif action == "done":
        return DONE
    elif action == "call_user":
        return CALL_USER
    elif action in ("screenshot", "cursor_position"):
        result += "pyautogui.sleep(0.1)\n"

    return result
