import ast
import logging
import re
import time
from typing import Any

import httpx

from mai_gym.agents._action_parse import (
    parse_json_tool_call,
    parse_terminate_status,
)
from mai_gym.agents._action_parse import (
    project_coordinates as _project_coordinates,
)
from mai_gym.agents.action_space import ActionSpace
from mai_gym.agents.errors import AgentAPIError
from mai_gym.utils.images import b64_to_pil, pil_to_b64, resize_max_pixels
from mai_gym.utils.messages import elide_images

logger = logging.getLogger(__name__)

IMAGE_COLLAPSED = "<image collapsed>"


_DETAIL_CAPABILITY_NOTES = """
### Verification Discipline
- Do not claim success based on intention. Distinguish what you attempted from what the computer actually shows.
- Before `computer.terminate(status="success")`, have concrete evidence from the current or recent observation, bash output, or saved artifact.
- For SIMPLE tasks, one direct final verification is enough.
- For MEDIUM/COMPLEX tasks, verify each required output or transferred value before success.
- If the screen shows errors, warnings, red underlines, empty output files, unexpected dialogs, or unchanged state after an expected change, investigate before finishing.

### Recovery Discipline
- Use recovery behavior only when uncertain, blocked, or when repeated actions do not change state.
- If repeated actions do not change the state, do not repeat the same action. Try a different route.
- Prefer `computer.bash()` for inspection when GUI evidence is ambiguous.
- If a command prompt shows a continuation state such as `>`, `...`, or appears to wait after an unterminated quote/backslash, cancel it and issue a complete corrected command.
- If a typed command or config edit produced malformed syntax, read back the content and rewrite the smallest reliable block.

### Minimal Reliable Path
- Prefer the simplest reliable path that satisfies the observable requirement. Use GUI when the requested outcome depends on GUI state, but use direct file/config/CLI methods when they are more reliable and still satisfy the task.
- Avoid unnecessary long installs, slow downloads, complex build paths, and exploratory UI navigation when a direct configuration/file edit achieves the required final state.
"""


_SYSTEM_PROMPT = """
You are a computer agent capable of both GUI operations and command-line interactions. \
You are given a task instruction, the current screen screenshot, and (when applicable) \
the stdout/stderr output from your most recent bash command. You perform a series of \
actions to complete the task. The password of the computer is {password}.

You have access to three categories of actions:

## 1. GUI Actions (pyautogui)
Use these when you need to interact with graphical applications (browsers, office suites, \
file managers, settings panels, etc.).

Available GUI functions:
- `pyautogui.click(x, y)` — Left-click at coordinates (x, y).
- `pyautogui.rightClick(x, y)` — Right-click at coordinates (x, y).
- `pyautogui.doubleClick(x, y)` — Double-click at coordinates (x, y).
- `pyautogui.tripleClick(x, y)` — Triple-click at coordinates (x, y).
- `pyautogui.moveTo(x, y)` — Move the mouse cursor to (x, y).
- `pyautogui.dragTo(x, y, duration=0.5)` — Drag from current position to (x, y).
- `pyautogui.typewrite(text)` — Type a string of text (ASCII only, no CJK).
- `pyautogui.press(key)` — Press a single key (e.g. 'enter', 'tab', 'escape', 'backspace').
- `pyautogui.keyDown(key)` / `pyautogui.keyUp(key)` — Hold or release a key.
- `pyautogui.hotkey(key1, key2, ...)` — Press a key combination (e.g. hotkey('ctrl', 'c')).
- `pyautogui.scroll(clicks, x, y)` — Scroll up (positive) or down (negative) at (x, y).

Coordinates you output MUST be in the range [0, {range_scale}] on BOTH axes (this is \
the grounding range). They are linearly mapped to the real screen: x is scaled by \
screen_width / {range_scale} and y by screen_height / {range_scale}. So ({range_scale}, \
{range_scale}) is the bottom-right corner of the screen and ({range_half}, {range_half}) \
is the exact center.

## 2. Bash / CLI Actions
Use `computer.bash(command="...")` to execute shell commands directly and receive \
stdout/stderr output. This is the preferred method for ANY task that can be done via \
the command line.

IMPORTANT: NEVER type commands into a terminal window using pyautogui.typewrite(). \
Always use computer.bash() instead — it executes the command directly and returns the \
output for you to inspect.

## 3. Flow Control Actions
- `computer.wait()` — Pause for 20 seconds. Use when waiting for an application to \
load, a download to finish, or a long-running process to complete.
- `computer.terminate(status="success")` — End the task and report success.
- `computer.terminate(status="failure")` — End the task and report failure.

## Observation
After each action, you receive:
- A screenshot of the current screen state.
- If the previous action was computer.bash(), you also receive the command's \
stdout/stderr output as text under "## Bash Output:".

## Response format
For each step, first reason inside <thinking>...</thinking>, then emit ONE OR MORE \
actions, each wrapped in its OWN <tool_call>...</tool_call> tag. Actions execute \
strictly in the order you write them. You may freely MIX GUI (pyautogui) actions \
and computer.bash() / computer.wait() / computer.terminate() in a single response — \
each action goes in a separate <tool_call>.

<thinking>
{thought}
</thinking>
<tool_call>
{code}
</tool_call>

The content inside each <tool_call>...</tool_call> is one action to run — either pyautogui code or one of the following functions:
- {"name": "computer.wait", "description": "Make the computer wait for 20 seconds \
for installation, running code, etc.", "parameters": {"type": "object", "properties": \
{}, "required": []}}
- {"name": "computer.terminate", "description": "Terminate the current task and report \
its completion status", "parameters": {"type": "object", "properties": {"status": \
{"type": "string", "enum": ["success", "failure"], "description": "The status of the \
task"}, "answer": {"type": "string", "description": "The answer of the task"}}, \
"required": ["status"]}}
- {"name": "computer.bash", "description": "Execute a bash command on the computer \
and return stdout/stderr output. Use this for file operations, installing packages, \
running scripts, checking system state, etc.", "parameters": {"type": "object", \
"properties": {"command": {"type": "string", "description": "The bash command to run"}}, \
"required": ["command"]}}

### Ordering & atomicity (IMPORTANT)
- Put EXACTLY ONE action in each <tool_call> (one pyautogui call, or one \
computer.bash()/wait()/terminate()). Emit as many <tool_call> blocks as you need; \
they execute strictly top-to-bottom.
- Adjacent GUI (pyautogui) actions are merged and run together in ONE process, \
with a short 0.5s pause between actions, so \
key combinations split across consecutive <tool_call> blocks stay safe — e.g. \
keyDown('shift') then click then keyUp('shift') in three separate <tool_call> blocks \
still works.
- A later action in the SAME response CANNOT see the stdout of an earlier \
computer.bash() in that response — bash output is returned to you in the NEXT step's \
observation under "## Bash Output:". So only batch actions together when the later \
ones do NOT depend on an earlier bash result; otherwise emit the bash alone and read \
its output next step.
- A computer.terminate() ends the task; do not put actions after it.

### Efficiency (STRONGLY ENCOURAGED)
- PREFER emitting MULTIPLE <tool_call> blocks in one step. Whenever you are already \
confident about several consecutive actions, batch them into a SINGLE response. Do \
not waste a whole step on a single trivial click when the next few actions are obvious.
- The ONLY reason to stop and emit fewer actions: a later action depends on what the \
NEW screenshot or a bash command's stdout will reveal. In that case, emit up to that \
point and let the next observation guide you.

## Response Examples

- GUI action

<thinking>xxx</thinking>
<tool_call>pyautogui.click(835, 420)</tool_call>

- GUI action（multiple actions）

<thinking>xxx</thinking>
<tool_call>pyautogui.click(960, 52)</tool_call>
<tool_call>pyautogui.hotkey('ctrl', 'a')</tool_call>
<tool_call>pyautogui.typewrite('how to install python', interval=0.05)</tool_call>
<tool_call>pyautogui.press('enter')</tool_call>

- Bash action

<thinking>xxx</thinking>
<tool_call>computer.bash(command="ls -la ~ && cat ~/output.txt")</tool_call>


- Flow control

<thinking>xxx</thinking>
<tool_call>computer.wait()</tool_call>

- Flow control

<thinking>xxx</thinking>
<tool_call>computer.terminate(status="success")</tool_call>


## Note
### CLI first
- You should use bash to complete the task in a higher priority than GUI.

### Thoroughness (CRITICAL)
- **Do NOT rush to declare terminate.** Most tasks require 20-50 steps. If you are finishing in fewer than 15 steps on a multi-step task, you have likely missed critical steps.
- **After saving (Ctrl+S), stop and verify**: Does the file/document actually reflect all required changes? Are there additional configuration steps (data ranges, formatting, sheet references)?
- **Multi-step tasks**: Break the task into subtasks mentally. Verify each subtask is complete before moving to the next.

### Persistence (CRITICAL)
- **Do NOT give up easily.** If your first approach fails, try at least 3 alternative methods before considering infeasible().
- **Slow loading is NOT failure.** If a page or application is loading, use wait() and try again. Web pages may take 10-30 seconds.
- **Unexpected dialogs or errors**: Read them carefully, dismiss or handle appropriately, then continue with the task.
- **If a click doesn't work**: Try clicking a slightly different position, or try an alternative approach (keyboard shortcut, menu, terminal command).

### Accuracy
- **Read data from the screen.** NEVER guess, estimate, or hallucinate values. If the task requires reading a number, price, or text from the screen, zoom in or scroll to read it clearly.
- **Verify typed content**: After typing important data, glance at what appeared on screen to confirm it matches what you intended.
- **File paths and names**: Read them exactly from the screen or task description. Do not assume or abbreviate.

## Multi-Application Coordination

When tasks involve multiple applications (e.g., copy data from Chrome to Calc, or read email and create a file):

1. **Complete one application's subtask fully** before switching to another.
2. **After pasting or transferring data**: Switch focus to the target app and verify the data arrived correctly.
3. **Keep track of window positions**: Note which windows are open and use Alt+Tab or taskbar to switch.
4. **Save intermediate work**: Before switching away from an application, save your progress (Ctrl+S where applicable).
5. **If an application becomes unresponsive**: Wait 5-10 seconds, then try clicking on it again. Do not abandon the task.
6. **The coordinate range is [0, {range_scale}] on both axes.** This means the bottom-right corner of the screen is ({range_scale}, {range_scale}), and ({range_half}, {range_half}) is the center. Always emit click/move/drag coordinates within this range.

## Infeasibility Criteria (STRICT)

You may ONLY declare infeasible() when ALL of the following are true:
- You have attempted at least 3 different approaches to complete the task
- The blocker is fundamental and cannot be worked around:
  - Required application is genuinely not installed AND cannot be installed
  - Required file confirmed not to exist after searching
  - Required URL/service is down after multiple retry attempts (not just slow loading)
  - Task requires physical hardware not available (printer, scanner, camera)
  - Task contains contradictory or logically impossible requirements

**You must NOT declare infeasible when:**
- A page is loading slowly (use wait(), retry)
- A dialog you didn't expect appears (read it, handle it)
- Your first click didn't land correctly (try again)
- An application takes time to start (wait for it)
- You're unsure how to do something (try menu exploration, right-click menus, Help)

## Information Extraction

When the task asks you to find/read/report information:
1. Navigate to where the information is displayed
2. Read the EXACT values from the screen (do not round or approximate)
3. Pass the information as the answer parameter in `computer.terminate(status="success", answer="...")`
4. Format: use the exact format requested in the task (number, text, path, etc.)
""".strip()


_INSTRUCTION_TEMPLATE = (
    "# Task Instruction:\n{instruction}\n\n"
    "Please generate the next move according to the screenshot, "
    "task instruction and previous steps (if provided).\n"
)
_STEP_TEMPLATE = "# Step {step_num}:\n"


_CODE_BLOCK_RE = re.compile(r"```(?:code|python)?\s*(.*?)\s*```", re.DOTALL | re.IGNORECASE)


_THINKING_TAG_RE = re.compile(r"<thinking>(.*?)</thinking>", re.DOTALL | re.IGNORECASE)
_TOOL_CALL_RE = re.compile(r"<tool_call>(.*?)</tool_call>", re.DOTALL | re.IGNORECASE)
_ACTION_TAG_RE = re.compile(r"<action>(.*?)</action>", re.DOTALL | re.IGNORECASE)
_THINK_TAG_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL | re.IGNORECASE)


def _render_history_turn(step_num: int, cot: dict[str, Any]) -> str:

    parts: list[str] = []
    reasoning = (cot.get("reasoning_content") or "").strip()
    think = (cot.get("think") or "").strip()
    thought = (cot.get("thought") or "").strip()
    replay_reasoning = reasoning or think or thought
    if replay_reasoning:
        parts.append(f"<thinking>{replay_reasoning}</thinking>")
    action = (cot.get("action") or "").strip()
    if action:
        if "<tool_call" in action.lower():
            parts.append(action)
        else:
            parts.append(f"<tool_call>{action}</tool_call>")
    return _STEP_TEMPLATE.format(step_num=step_num) + "\n".join(parts) + "\n"


def _fmt_scale(n: float | None) -> str:

    if n is None:
        return "raw screen pixels"
    return str(int(n)) if float(n).is_integer() else str(n)


def _format_coordinate_prompt(
    prompt: str,
    *,
    screen_size: tuple[int, int],
    range_scale: float | None,
) -> str:

    width, height = screen_size
    if range_scale is not None:
        range_half = max(0.0, range_scale / 2) if range_scale > 0 else 0.0
        return (
            prompt.replace("{range_scale}", _fmt_scale(range_scale))
            .replace("{range_half}", _fmt_scale(range_half))
            .replace("{screen_width}", str(width))
            .replace("{screen_height}", str(height))
            .replace("{screen_center}", f"({width // 2}, {height // 2})")
        )

    raw_paragraph = (
        "Coordinates you output MUST be raw screen pixel coordinates. "
        f"The visible screen size is {width}x{height}: x ranges from 0 to {width}, "
        f"y ranges from 0 to {height}, and the screen center is ({width // 2}, {height // 2}). "
        "Do not normalize coordinates and do not use a 0-1 or 0-1000 coordinate range."
    )
    prompt = re.sub(
        r"Coordinates you output MUST be in the range \[0, \{range_scale\}\].*?is the exact center\.",
        raw_paragraph,
        prompt,
        flags=re.DOTALL,
    )
    prompt = re.sub(
        r"6\. \*\*The coordinate range is \[0, \{range_scale\}\].*?Always emit click/move/drag coordinates within this range\.",
        "6. **Use raw screen pixel coordinates.** "
        f"The visible screen size is {width}x{height}: x ranges from 0 to {width}, "
        f"y ranges from 0 to {height}, and the center is ({width // 2}, {height // 2}). "
        "Do not normalize coordinates.",
        prompt,
        flags=re.DOTALL,
    )
    prompt = re.sub(
        r"6\. The coordinate range is \[0, \{range_scale\}\].*?Always emit click/move/drag coordinates within this range\.",
        "6. Use raw screen pixel coordinates. "
        f"The visible screen size is {width}x{height}: x ranges from 0 to {width}, "
        f"y ranges from 0 to {height}, and the center is ({width // 2}, {height // 2}). "
        "Do not normalize coordinates.",
        prompt,
        flags=re.DOTALL,
    )
    return (
        prompt.replace("{range_scale}", f"{width} for x and {height} for y")
        .replace("{range_half}", f"{width // 2}, {height // 2}")
        .replace("{screen_width}", str(width))
        .replace("{screen_height}", str(height))
        .replace("{screen_center}", f"({width // 2}, {height // 2})")
    )


def _extract_bash_command(code_block: str) -> str | None:

    try:
        tree = ast.parse(code_block, mode="exec")
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                func = node.func
                if (
                    isinstance(func, ast.Attribute)
                    and isinstance(func.value, ast.Name)
                    and func.value.id == "computer"
                    and func.attr == "bash"
                ):
                    for kw in node.keywords:
                        if kw.arg == "command" and isinstance(kw.value, ast.Constant):
                            val = kw.value.value
                            if isinstance(val, str):
                                return val
                    if node.args and isinstance(node.args[0], ast.Constant):
                        val = node.args[0].value
                        if isinstance(val, str):
                            return val
    except SyntaxError:
        pass

    try:
        obj = ast.literal_eval(code_block)
        if isinstance(obj, dict):
            params = obj.get("parameters") or obj.get("arguments") or {}
            command = params.get("command") if isinstance(params, dict) else None
            if isinstance(command, str):
                return command
    except (ValueError, SyntaxError):
        pass

    m = re.search(r"""command\s*=\s*[rRbBfFuU]{0,2}('''|\"\"\"|['"])""", code_block)
    if m:
        opening = m.group(1)
        rest = code_block[m.end() :]
        if opening in ("'''", '"""'):
            idx = rest.find(opening)
        else:
            idx = -1
            for i, ch in enumerate(rest):
                if ch == opening and (i == 0 or rest[i - 1] != "\\"):
                    idx = i
                    break
        if idx >= 0:
            return rest[:idx]

        if rest.strip():
            return rest

    return None


def _extract_call_string_arg(code_block: str, arg_name: str) -> str | None:

    try:
        tree = ast.parse(code_block, mode="exec")
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            for kw in node.keywords:
                if kw.arg == arg_name and isinstance(kw.value, ast.Constant):
                    val = kw.value.value
                    if isinstance(val, str):
                        return val
    except (SyntaxError, ValueError):
        pass

    m = re.search(
        rf"""{re.escape(arg_name)}\s*=\s*[rRbBfFuU]{{0,2}}(['"])(.*?)\1""",
        code_block,
        flags=re.DOTALL,
    )
    if m:
        return m.group(2)
    return None


def _parse_response(
    response: dict[str, Any],
    screen_size: tuple[int, int],
    thinking: bool = False,
    range_scale: float | None = 1000,
) -> tuple[dict[str, Any], dict[str, Any]]:

    content: str = response.get("content", "") or ""
    reasoning: str = (response.get("reasoning_content") or "").strip()
    sections: dict[str, Any] = {"reasoning_content": reasoning}

    if "</think>" in content.lower() and "<think>" not in content.lower():
        content = "<think>" + content

    think_m = _THINK_TAG_RE.search(content)
    sections["think"] = think_m.group(1).strip() if think_m else ""
    content = _THINK_TAG_RE.sub("", content).strip()

    thinking_tag = _THINKING_TAG_RE.search(content)
    explicit_thought = thinking_tag.group(1).strip() if thinking_tag else ""
    sections["thought"] = explicit_thought
    content = _THINKING_TAG_RE.sub("", content).strip()

    blocks = _TOOL_CALL_RE.findall(content)
    if not blocks:
        action_m = _ACTION_TAG_RE.search(content)
        if action_m:
            blocks = [action_m.group(1)]
        else:
            fenced = _CODE_BLOCK_RE.findall(content)
            if fenced:
                blocks = [fenced[-1]]

    if not blocks:
        logger.warning("[GeneralE2ECodeAgentPolicy] No <tool_call> code found in response")
        return {"action_type": "FAIL"}, sections

    actions: list[dict[str, Any]] = []
    original_blocks: list[str] = []
    codes: list[str] = []
    for block in blocks:
        code_block = block.strip()
        inner_fence = _CODE_BLOCK_RE.search(code_block)
        if inner_fence:
            code_block = inner_fence.group(1).strip()
        if not code_block:
            continue
        original_blocks.append(f"<tool_call>\n{code_block}\n</tool_call>")
        code_lower = code_block.lower()

        json_action = parse_json_tool_call(code_block)
        if json_action is not None:
            actions.append(json_action)
            action_type = json_action.get("action_type")
            if action_type == "execute_bash":
                codes.append(f"bash: {json_action.get('command', '')}")
            elif action_type == "wait":
                codes.append("WAIT")
            elif action_type == "finished":
                status = json_action.get("status")
                codes.append("DONE" if status == "success" else "FAIL")
            else:
                codes.append(str(json_action))
            continue

        if "computer.bash" in code_lower:
            cmd = _extract_bash_command(code_block)
            if cmd:
                actions.append({"action_type": "execute_bash", "command": cmd})
                codes.append(f"bash: {cmd}")
                continue
            logger.warning(
                "[GeneralE2ECodeAgentPolicy] computer.bash detected but command "
                "extraction failed: %r",
                code_block[:200],
            )
            continue

        if "computer.wait" in code_lower:
            actions.append({"action_type": "wait"})
            codes.append("WAIT")
            continue

        if "computer.terminate" in code_lower:
            status = parse_terminate_status(code_block)
            action = {"action_type": "finished", "status": status}
            answer = _extract_call_string_arg(code_block, "answer")
            if answer:
                action["answer"] = answer
            actions.append(action)
            codes.append("DONE" if status == "success" else "FAIL")
            continue

        projected = _project_coordinates(
            code_block,
            screen_size[0],
            screen_size[1],
            range_scale,
        )
        if projected:
            actions.append({"action_type": "pyautogui_code", "code": projected})
            codes.append(projected)

    if not actions:
        return {"action_type": "FAIL"}, sections

    sections["action"] = "\n".join(original_blocks)
    sections["original_code"] = sections["action"]
    sections["code"] = " ; ".join(codes)

    if len(actions) == 1:
        return actions[0], sections
    return {"action_type": "action_sequence", "actions": actions}, sections


_PARSE_RETRY_FEEDBACK = """\
Your previous response could not be parsed into an executable action.

Re-emit the next action using exactly one of these valid formats inside <tool_call>:
- computer.bash(command="...")
- computer.wait()
- computer.terminate(status="success") or computer.terminate(status="failure")
- pyautogui.<function>(...)

Do not put JSON objects inside <tool_call> for multi-line bash commands. If you need
to run a multi-line script, use computer.bash(command="python3 - <<'PY'\\n...\\nPY").
Return only the corrected reasoning and <tool_call> blocks for the next action.
"""


def _append_parse_retry_feedback(
    messages: list[dict[str, Any]],
    response_text: str,
) -> list[dict[str, Any]]:

    excerpt = response_text.strip()
    if len(excerpt) > 2000:
        excerpt = excerpt[:2000] + "\n... [truncated]"
    feedback = _PARSE_RETRY_FEEDBACK
    if excerpt:
        feedback += f"\nUnparseable previous response excerpt:\n{excerpt}"
    return [*messages, {"role": "user", "content": feedback}]


def _format_bash_block(output: str, error: str, returncode: int) -> str:

    parts = [f"Previous bash output (returncode={returncode}):"]
    if output:
        parts.append("stdout:\n" + output.rstrip()[:4000])
    if error:
        parts.append("stderr:\n" + error.rstrip()[:2000])
    if not output and not error:
        parts.append("(no output)")
    return "\n\n".join(parts)


def _content_blocks_to_text_and_reasoning(content: Any) -> tuple[str, str]:

    if isinstance(content, str):
        return content, ""
    if not isinstance(content, list):
        return "", ""

    text_parts: list[str] = []
    reasoning_parts: list[str] = []
    for block in content:
        if not isinstance(block, dict):
            continue
        btype = block.get("type")
        if btype == "text":
            text_parts.append(block.get("text", ""))
        elif btype == "thinking":
            reasoning_parts.append(
                block.get("thinking") or block.get("text") or block.get("content") or ""
            )
    return "".join(text_parts), "".join(reasoning_parts)


class DashscopeBaseAgentPolicy:
    NATIVE_ACTION_SPACE = ActionSpace.COMPUTER_USE

    SYSTEM_PROMPT = _SYSTEM_PROMPT
    DETAIL_CAPABILITY_NOTES = _DETAIL_CAPABILITY_NOTES

    def __init__(
        self,
        model_name: str,
        llm_base_url: str,
        api_key: str = "EMPTY",
        *,
        history_n: int = 3,
        temperature: float = 1.0,
        top_p: float = 0.95,
        max_tokens: int = 4096,
        password: str = "password",
        screen_size: tuple[int, int] = (1920, 1080),
        range_scale: float | None = 1000,
        thinking: bool = True,
        max_retries: int = 5,
        retry_sleep: float = 3.0,
        parse_fail_giveup_n: int = 3,
        parse_retry_n: int = 3,
        max_image_pixels: int | None = None,
        max_steps: int = 100,
        thinking_budget_tokens: int = 10000,
        provider: str | None = None,
        anthropic_version: str | None = None,
        image_format: str = "png",
        jpeg_quality: int = 90,
        request_timeout: float = 1200.0,
        enable_thinking: bool | None = None,
        thinking_budget: int | None = None,
        preserve_thinking: bool | None = None,
    ) -> None:
        self._model = model_name
        self._api_key = api_key
        self._base_url = llm_base_url.rstrip("/") if llm_base_url else ""
        self._history_n = history_n
        self._temperature = temperature
        self._top_p = top_p
        self._max_tokens = max_tokens
        self._password = password
        self._screen_size = screen_size
        self._range_scale = range_scale
        self._thinking = thinking
        self._enable_thinking = enable_thinking
        self._thinking_budget = thinking_budget
        self._preserve_thinking = preserve_thinking
        self._max_retries = max_retries
        self._retry_sleep = retry_sleep
        self._parse_fail_giveup_n = parse_fail_giveup_n
        self._parse_retry_n = parse_retry_n
        self._consecutive_parse_failures = 0
        self._max_image_pixels = max_image_pixels
        self._max_steps = max_steps
        self._thinking_budget_tokens = thinking_budget_tokens

        prov = (provider or "").strip().lower()
        self._provider = prov
        if prov in ("anthropic", "dashscope_claude"):
            self._anthropic = True
        elif prov in ("openai", "vllm"):
            self._anthropic = False
        else:
            self._anthropic = "/anthropic" in self._base_url.lower()
        if prov == "dashscope_claude":
            self._anthropic_path = "/chat/completions"
            self._anthropic_auth_bearer = True
        else:
            self._anthropic_path = "/messages"
            self._anthropic_auth_bearer = False
        self._anthropic_version = anthropic_version

        self._image_format = (image_format or "png").lower()
        self._jpeg_quality = jpeg_quality
        self._image_media = "image/jpeg" if self._image_format in ("jpeg", "jpg") else "image/png"

        self._system_prompt = _format_coordinate_prompt(
            self.SYSTEM_PROMPT.replace("{password}", password),
            screen_size=screen_size,
            range_scale=range_scale,
        ).replace("## Response format", self.DETAIL_CAPABILITY_NOTES + "\n\n## Response format", 1)

        self._client = httpx.Client(timeout=request_timeout)

        self._observations: list[str] = []
        self._actions: list[str] = []
        self._cots: list[dict[str, Any]] = []

        self._bash_results: list[str] = []
        self._instruction: str = ""
        self.last_prediction: str | None = None
        self.last_step_log: dict[str, Any] | None = None

    @classmethod
    def from_agent_config(cls, cfg: Any) -> "DashscopeBaseAgentPolicy":
        rc = cfg.runtime_conf or {}
        screen = rc.get("screen_size", [1920, 1080])
        return cls(
            model_name=cfg.model_name,
            llm_base_url=cfg.base_url,
            api_key=cfg.api_key,
            history_n=rc.get("history_n", 3),
            temperature=rc.get("temperature", 1.0),
            top_p=rc.get("top_p", 0.95),
            max_tokens=rc.get("max_tokens", rc.get("max_new_tokens", 4096)),
            password=rc.get("password", "password"),
            screen_size=(screen[0], screen[1]),
            range_scale=(
                rc["range_scale"] if "range_scale" in rc else rc.get("coordinate_range", 1000)
            ),
            thinking=rc.get("thinking", True),
            max_retries=rc.get("max_retries", 5),
            retry_sleep=rc.get("retry_sleep", 3.0),
            parse_fail_giveup_n=rc.get("parse_fail_giveup_n", 3),
            parse_retry_n=rc.get("parse_retry_n", 3),
            max_image_pixels=rc.get("max_image_pixels", None),
            max_steps=cfg.max_steps,
            thinking_budget_tokens=rc.get("thinking_budget_tokens", 10000),
            provider=rc.get("provider"),
            anthropic_version=rc.get("anthropic_version"),
            image_format=rc.get("image_format", "png"),
            jpeg_quality=rc.get("jpeg_quality", 90),
            request_timeout=rc.get("request_timeout", 1200.0),
            enable_thinking=rc.get("enable_thinking"),
            thinking_budget=rc.get("thinking_budget"),
            preserve_thinking=rc.get("preserve_thinking"),
        )

    def initialize(self, instruction: str) -> None:
        self._instruction = instruction
        self._observations.clear()
        self._actions.clear()
        self._cots.clear()
        self._bash_results.clear()
        self._consecutive_parse_failures = 0

    def done(self) -> None:
        self._observations.clear()
        self._actions.clear()
        self._cots.clear()
        self._bash_results.clear()
        self._instruction = ""
        self._consecutive_parse_failures = 0

    def close(self) -> None:
        self._client.close()

    async def select_action(self, observation: dict[str, Any]) -> dict[str, Any]:
        screenshot_b64 = observation.get("image", "")

        current_bash_block = ""
        if "bash_returncode" in observation:
            current_bash_block = _format_bash_block(
                observation.get("bash_output", "") or "",
                observation.get("bash_error", "") or "",
                int(observation.get("bash_returncode", 0) or 0),
            )

        img = b64_to_pil(screenshot_b64)
        if self._max_image_pixels is not None:
            img = resize_max_pixels(img, self._max_image_pixels)
        resized_b64 = pil_to_b64(img, fmt=self._image_format, quality=self._jpeg_quality)

        messages = self._build_messages(resized_b64, current_bash_block)

        action: dict[str, Any] = {"action_type": "FAIL"}
        cot: dict[str, Any] = {}
        response: dict[str, Any] | None = None

        got_response = False

        parse_attempts = 0
        call_messages = messages

        for attempt in range(self._max_retries):
            temp = self._temperature if attempt == 0 else max(0.2, self._temperature)
            try:
                response = self._call_vlm(call_messages, temperature=temp)
            except Exception as exc:
                logger.error(
                    "[GeneralE2ECodeAgentPolicy] VLM call failed (attempt %d/%d): %s; sleeping %.1fs",
                    attempt + 1,
                    self._max_retries,
                    exc,
                    self._retry_sleep,
                )
                time.sleep(self._retry_sleep)
                continue

            if not response:
                logger.error(
                    "[GeneralE2ECodeAgentPolicy] Empty response (attempt %d/%d); sleeping %.1fs",
                    attempt + 1,
                    self._max_retries,
                    self._retry_sleep,
                )
                time.sleep(self._retry_sleep)
                continue

            got_response = True
            self.last_prediction = response.get("content", "")
            action, cot = _parse_response(
                response,
                self._screen_size,
                self._thinking,
                self._range_scale,
            )

            if action["action_type"] != "FAIL":
                break

            parse_attempts += 1
            if parse_attempts >= self._parse_retry_n:
                logger.warning(
                    "[GeneralE2ECodeAgentPolicy] Parse failed %d/%d times, giving up "
                    "this step (not an API failure — won't burn max_retries=%d)",
                    parse_attempts,
                    self._parse_retry_n,
                    self._max_retries,
                )
                break

            logger.warning(
                "[GeneralE2ECodeAgentPolicy] Parse failed (parse attempt %d/%d), retrying",
                parse_attempts,
                self._parse_retry_n,
            )
            call_messages = _append_parse_retry_feedback(
                messages,
                response.get("content", "") if response else "",
            )

        self._observations.append(resized_b64)
        self._actions.append(cot.get("action", ""))
        self._cots.append(cot)
        self._bash_results.append(current_bash_block)

        self.last_step_log = {
            "system_prompt": self._system_prompt,
            "instruction": self._instruction,
            "rendered_messages": elide_images(call_messages),
            "raw_response_text": self.last_prediction or "",
            "raw_reasoning_content": (response or {}).get("reasoning_content", ""),
            "parsed_action_dict": dict(action),
            "parsed_cot": dict(cot),
        }

        logger.info(
            "[GeneralE2ECodeAgentPolicy] step=%d action_type=%s",
            len(self._actions),
            action.get("action_type"),
        )

        if action["action_type"] == "FAIL":
            if not got_response:
                raise AgentAPIError(
                    f"[GeneralE2ECodeAgentPolicy] LLM call failed/empty after "
                    f"{self._max_retries} retries"
                )
            return self._on_parse_failure(reason="Parse failed after all retries")

        self._consecutive_parse_failures = 0

        is_finishing = action.get("action_type") == "finished" or (
            action.get("action_type") == "action_sequence"
            and any(child.get("action_type") == "finished" for child in action.get("actions", []))
        )
        current_step = len(self._actions)
        if current_step > self._max_steps and not is_finishing:
            logger.warning(
                "[GeneralE2ECodeAgentPolicy] Reached max_steps=%d, forcing FAIL",
                self._max_steps,
            )
            return {"action_type": "finished", "status": "failure"}

        return action

    def _on_parse_failure(self, *, reason: str) -> dict[str, Any]:

        self._consecutive_parse_failures += 1
        if self._consecutive_parse_failures >= self._parse_fail_giveup_n:
            logger.error(
                "[GeneralE2ECodeAgentPolicy] %s; %d consecutive parse failures "
                "(>= giveup_n=%d), terminating with FAIL.",
                reason,
                self._consecutive_parse_failures,
                self._parse_fail_giveup_n,
            )
            return {"action_type": "finished", "status": "failure"}
        logger.warning(
            "[GeneralE2ECodeAgentPolicy] %s; consecutive=%d/%d, emitting no-op.",
            reason,
            self._consecutive_parse_failures,
            self._parse_fail_giveup_n,
        )
        return {
            "action_type": "pyautogui_code",
            "code": "import time\ntime.sleep(1.0)",
        }

    def _build_messages(
        self,
        current_screenshot_b64: str,
        current_bash_block: str,
    ) -> list[dict[str, Any]]:

        instruction_prompt = (
            "# Task Instruction:\n"
            f"{self._instruction}\n"
            "Please complete the task by generating a sequence of actions. "
            "Refer to the system prompt for the response format and guidelines."
        )
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": self._system_prompt},
        ]

        n_steps = len(self._actions)
        img_window_start = max(0, n_steps - max(0, self._history_n - 1))

        if n_steps == 0:
            current_parts: list[dict[str, Any]] = [
                {"type": "text", "text": instruction_prompt},
            ]
            if current_bash_block:
                current_parts.append({"type": "text", "text": current_bash_block})
            current_parts.append(
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:{self._image_media};base64,{current_screenshot_b64}",
                    },
                }
            )
            messages.append({"role": "user", "content": current_parts})
            return messages

        for i in range(n_steps):
            user_parts: list[dict[str, Any]] = []
            if i == 0:
                user_parts.append({"type": "text", "text": instruction_prompt})
            if i < len(self._bash_results) and self._bash_results[i]:
                user_parts.append({"type": "text", "text": self._bash_results[i]})
            if i < img_window_start:
                user_parts.append({"type": "text", "text": IMAGE_COLLAPSED})
            else:
                user_parts.append(
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:{self._image_media};base64,{self._observations[i]}",
                        },
                    }
                )
            messages.append({"role": "user", "content": user_parts})
            step_text = _render_history_turn(i + 1, self._cots[i])
            messages.append({"role": "assistant", "content": step_text})

        current_parts = []
        if current_bash_block:
            current_parts.append({"type": "text", "text": current_bash_block})
        current_parts.append(
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:{self._image_media};base64,{current_screenshot_b64}",
                },
            }
        )
        messages.append({"role": "user", "content": current_parts})

        return messages

    def _call_vlm(
        self,
        messages: list[dict[str, Any]],
        *,
        temperature: float | None = None,
    ) -> dict[str, Any] | None:

        if self._anthropic:
            return self._call_anthropic(messages, temperature=temperature)

        url = f"{self._base_url}/chat/completions"
        payload: dict[str, Any] = {
            "model": self._model,
            "messages": messages,
            "max_tokens": self._max_tokens,
        }
        if "claude" not in self._model.lower():
            payload["top_p"] = self._top_p
            payload["temperature"] = temperature if temperature is not None else self._temperature
        if self._thinking_budget is not None:
            payload["thinking_budget"] = self._thinking_budget
        if self._preserve_thinking is not None:
            payload["preserve_thinking"] = self._preserve_thinking
        if self._enable_thinking is not None:
            payload["enable_thinking"] = self._enable_thinking
            if self._provider == "vllm":
                payload["chat_template_kwargs"] = {"enable_thinking": self._enable_thinking}
        elif self._thinking and self._provider == "vllm":
            payload["enable_thinking"] = True
            payload["preserve_thinking"] = True
            payload["chat_template_kwargs"] = {
                "enable_thinking": True,
                "preserve_thinking": True,
            }
        elif self._thinking:
            payload["thinking"] = {
                "type": "enabled",
                "budget_tokens": self._thinking_budget_tokens,
            }
        headers: dict[str, str] = {"Content-Type": "application/json"}
        if self._api_key and self._api_key != "EMPTY":
            headers["Authorization"] = f"Bearer {self._api_key}"

        for attempt in range(self._max_retries):
            try:
                resp = self._client.post(url, json=payload, headers=headers)
            except Exception as exc:
                logger.error(
                    "[GeneralE2ECodeAgentPolicy] HTTP error (attempt %d/%d): %s; sleeping %.1fs",
                    attempt + 1,
                    self._max_retries,
                    exc,
                    self._retry_sleep,
                )
                time.sleep(self._retry_sleep)
                continue

            if resp.status_code != 200:
                logger.error(
                    "[GeneralE2ECodeAgentPolicy] Non-200 (attempt %d/%d): %d %s; sleeping %.1fs",
                    attempt + 1,
                    self._max_retries,
                    resp.status_code,
                    resp.text[:500],
                    self._retry_sleep,
                )
                time.sleep(self._retry_sleep)
                continue

            try:
                data = resp.json()
            except Exception:
                logger.error(
                    "[GeneralE2ECodeAgentPolicy] non-JSON (attempt %d/%d): %s; sleeping %.1fs",
                    attempt + 1,
                    self._max_retries,
                    resp.text[:300],
                    self._retry_sleep,
                )
                time.sleep(self._retry_sleep)
                continue

            if "choices" in data:
                choices = data.get("choices") or []
                if not choices:
                    logger.warning(
                        "[GeneralE2ECodeAgentPolicy] empty choices (attempt %d/%d): %s; sleeping %.1fs",
                        attempt + 1,
                        self._max_retries,
                        str(data)[:300],
                        self._retry_sleep,
                    )
                    time.sleep(self._retry_sleep)
                    continue

                choice = choices[0]
                if choice.get("finish_reason") != "stop":
                    reason = choice.get("finish_reason")
                    logger.warning(
                        "[GeneralE2ECodeAgentPolicy] finish_reason=%s (attempt %d/%d); sleeping %.1fs",
                        reason,
                        attempt + 1,
                        self._max_retries,
                        self._retry_sleep,
                    )
                    time.sleep(self._retry_sleep)
                    continue

                msg = choice["message"]
                text, block_reasoning = _content_blocks_to_text_and_reasoning(
                    msg.get("content", "")
                )
                reasoning_content = (
                    msg.get("reasoning_content")
                    or msg.get("reasoning")
                    or choice.get("reasoning_content")
                    or choice.get("reasoning")
                    or msg.get("model_extra", {}).get("reasoning_content")
                    or msg.get("model_extra", {}).get("reasoning")
                    or block_reasoning
                    or ""
                )
                return {
                    "content": text,
                    "reasoning_content": reasoning_content,
                }

            if data.get("type") == "message" or isinstance(data.get("content"), list):
                stop_reason = data.get("stop_reason")
                if stop_reason not in (None, "end_turn", "stop_sequence"):
                    logger.warning(
                        "[GeneralE2ECodeAgentPolicy] stop_reason=%s (attempt %d/%d); sleeping %.1fs",
                        stop_reason,
                        attempt + 1,
                        self._max_retries,
                        self._retry_sleep,
                    )
                    time.sleep(self._retry_sleep)
                    continue

                text, block_reasoning = _content_blocks_to_text_and_reasoning(
                    data.get("content", "")
                )
                reasoning_content = (
                    data.get("reasoning_content") or data.get("reasoning") or block_reasoning or ""
                )
                if text or reasoning_content:
                    return {"content": text, "reasoning_content": reasoning_content}

                logger.warning(
                    "[GeneralE2ECodeAgentPolicy] empty message content (attempt %d/%d): %s; sleeping %.1fs",
                    attempt + 1,
                    self._max_retries,
                    str(data)[:300],
                    self._retry_sleep,
                )
                time.sleep(self._retry_sleep)
                continue

            logger.error(
                "[GeneralE2ECodeAgentPolicy] unexpected response schema (attempt %d/%d): %s; sleeping %.1fs",
                attempt + 1,
                self._max_retries,
                str(data)[:300],
                self._retry_sleep,
            )
            time.sleep(self._retry_sleep)

        return None

    def _call_anthropic(
        self,
        messages: list[dict[str, Any]],
        *,
        temperature: float | None = None,
    ) -> dict[str, Any] | None:

        url = f"{self._base_url}{self._anthropic_path}"
        system, amsgs = _openai_to_anthropic(messages)

        payload: dict[str, Any] = {
            "model": self._model,
            "messages": amsgs,
            "max_tokens": self._max_tokens,
            "stream": False,
        }
        if system:
            payload["system"] = system
        if self._thinking:
            if self._max_tokens <= self._thinking_budget_tokens:
                payload["max_tokens"] = self._thinking_budget_tokens + 4096
            payload["thinking"] = {
                "type": "enabled",
                "budget_tokens": self._thinking_budget_tokens,
            }
            payload["temperature"] = 1.0

        headers: dict[str, str] = {"Content-Type": "application/json"}
        if self._api_key and self._api_key != "EMPTY":
            headers["Authorization"] = (
                f"Bearer {self._api_key}" if self._anthropic_auth_bearer else self._api_key
            )
        if self._anthropic_version:
            headers["anthropic-version"] = self._anthropic_version

        for attempt in range(self._max_retries):
            try:
                resp = self._client.post(url, json=payload, headers=headers)
            except Exception as exc:
                logger.error(
                    "[GeneralE2ECodeAgentPolicy] anthropic HTTP error (attempt %d/%d): %s; sleeping %.1fs",
                    attempt + 1,
                    self._max_retries,
                    exc,
                    self._retry_sleep,
                )
                time.sleep(self._retry_sleep)
                continue

            if resp.status_code != 200:
                logger.error(
                    "[GeneralE2ECodeAgentPolicy] anthropic non-200 (attempt %d/%d): %d %s; sleeping %.1fs",
                    attempt + 1,
                    self._max_retries,
                    resp.status_code,
                    resp.text[:500],
                    self._retry_sleep,
                )
                time.sleep(self._retry_sleep)
                continue

            try:
                data = resp.json()
            except Exception:
                logger.error(
                    "[GeneralE2ECodeAgentPolicy] anthropic non-JSON (attempt %d/%d): %s; sleeping %.1fs",
                    attempt + 1,
                    self._max_retries,
                    resp.text[:300],
                    self._retry_sleep,
                )
                time.sleep(self._retry_sleep)
                continue

            if data.get("type") != "message":
                logger.error(
                    "[GeneralE2ECodeAgentPolicy] anthropic non-message (attempt %d/%d): %s; sleeping %.1fs",
                    attempt + 1,
                    self._max_retries,
                    str(data)[:300],
                    self._retry_sleep,
                )
                time.sleep(self._retry_sleep)
                continue

            blocks = data.get("content") or []
            text, reasoning = _content_blocks_to_text_and_reasoning(blocks)
            if not text and not reasoning:
                logger.warning(
                    "[GeneralE2ECodeAgentPolicy] anthropic empty content (attempt %d/%d); sleeping %.1fs",
                    attempt + 1,
                    self._max_retries,
                    self._retry_sleep,
                )
                time.sleep(self._retry_sleep)
                continue
            return {"content": text, "reasoning_content": reasoning}

        return None


def _openai_to_anthropic(
    messages: list[dict[str, Any]],
) -> tuple[str | None, list[dict[str, Any]]]:

    system: str | None = None
    out: list[dict[str, Any]] = []
    for m in messages:
        role = m.get("role")
        content = m.get("content")
        if role == "system":
            if isinstance(content, str):
                system = content
            elif isinstance(content, list):
                system = "".join(
                    p.get("text", "")
                    for p in content
                    if isinstance(p, dict) and p.get("type") == "text"
                )
            continue
        if isinstance(content, str):
            out.append({"role": role, "content": content})
            continue
        blocks: list[dict[str, Any]] = []
        for part in content or []:
            ptype = part.get("type")
            if ptype == "text":
                blocks.append({"type": "text", "text": part.get("text", "")})
            elif ptype == "image_url":
                url = (part.get("image_url") or {}).get("url", "")
                media = "image/png"
                b64 = url
                if url.startswith("data:"):
                    head, _, b64 = url.partition(",")
                    if ";" in head:
                        media = head[len("data:") : head.index(";")] or "image/png"
                blocks.append(
                    {
                        "type": "image",
                        "source": {"type": "base64", "media_type": media, "data": b64},
                    }
                )
        out.append({"role": role, "content": blocks})
    return system, out


from mai_gym.agents import register_policy_factory

register_policy_factory("dashscope_base_agent", DashscopeBaseAgentPolicy.from_agent_config)
