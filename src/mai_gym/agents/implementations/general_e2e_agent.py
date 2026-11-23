import ast
import logging
import time
from typing import Any

from mai_gym.agents._action_parse import parse_terminate_status
from mai_gym.agents._action_parse import project_coordinates as _project_coordinates
from mai_gym.agents.errors import AgentAPIError
from mai_gym.agents.implementations.dashscope_base_agent import (
    _ACTION_TAG_RE,
    _CODE_BLOCK_RE,
    _THINK_TAG_RE,
    _THINKING_TAG_RE,
    _TOOL_CALL_RE,
    DashscopeBaseAgentPolicy,
    _extract_call_string_arg,
)
from mai_gym.utils.images import b64_to_pil, pil_to_b64, resize_max_pixels
from mai_gym.utils.messages import elide_images

logger = logging.getLogger(__name__)


_GUI_DETAIL_CAPABILITY_NOTES = """
### Verification Discipline
- Do not claim success based on intention. Distinguish what you attempted from what the computer actually shows.
- Before `computer.terminate(status="success")`, have concrete evidence from the current or recent observation or saved artifact.
- For SIMPLE tasks, one direct final verification is enough.
- For MEDIUM/COMPLEX tasks, verify each required output or transferred value before success.
- If the screen shows errors, warnings, red underlines, empty output files, unexpected dialogs, or unchanged state after an expected change, investigate before finishing.

### Recovery Discipline
- Use recovery behavior only when uncertain, blocked, or when repeated actions do not change state.
- If repeated actions do not change the state, do not repeat the same action. Try a different visible route.
- Prefer application menus, dialogs, keyboard shortcuts, context menus, and visible controls when direct clicking is ambiguous.
- If typed text or a setting edit appears malformed, read back the visible content and correct the smallest reliable part.

### Minimal Reliable Path
- Prefer the simplest visible path that satisfies the observable requirement.
- Avoid unnecessary long navigation, slow downloads, complex setup paths, and exploratory UI wandering when a direct visible control achieves the required final state.
""".strip()


_SYSTEM_PROMPT = """
You are a GUI agent. You are given a task instruction and the current screen screenshot. You perform a series of visible actions to complete the task. The password of the computer is {password}.

You have access to two categories of actions:

## 1. GUI Actions (pyautogui)
Use these to interact with graphical applications such as browsers, office suites, file managers, settings panels, editors, and dialogs.

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

Coordinates you output MUST be in the range [0, {range_scale}] on BOTH axes (this is the grounding range). They are linearly mapped to the real screen: x is scaled by screen_width / {range_scale} and y by screen_height / {range_scale}. So ({range_scale}, {range_scale}) is the bottom-right corner of the screen and ({range_half}, {range_half}) is the exact center.

## 2. Flow Control Actions
- `computer.wait()` — Pause for 20 seconds. Use when waiting for an application to load, a page to finish, a dialog to appear, or a long visible operation to complete.
- `computer.terminate(status="success", answer="...")` — End the task and report success. Include the requested findings in `answer` for information-retrieval tasks.
- `computer.terminate(status="failure")` — End the task and report failure.

## Observation
After each action, you receive a screenshot of the current screen state.

## Response format
For each step, first reason inside <thinking>...</thinking>, then emit ONE OR MORE actions, each wrapped in its OWN <tool_call>...</tool_call> tag. Actions execute strictly in the order you write them.

<thinking>
{thought}
</thinking>
<tool_call>
{code}
</tool_call>

The content inside each <tool_call>...</tool_call> is one action to run: either pyautogui code or one of the following functions:
- {"name": "computer.wait", "description": "Make the computer wait for 20 seconds.", "parameters": {"type": "object", "properties": {}, "required": []}}
- {"name": "computer.terminate", "description": "Terminate the current task and report its completion status", "parameters": {"type": "object", "properties": {"status": {"type": "string", "enum": ["success", "failure"], "description": "The status of the task"}, "answer": {"type": "string", "description": "The answer of the task"}}, "required": ["status"]}}

### Ordering & atomicity (IMPORTANT)
- Put EXACTLY ONE action in each <tool_call>: one pyautogui call, `computer.wait()`, or `computer.terminate()`.
- Emit as many <tool_call> blocks as you need; they execute strictly top-to-bottom.
- Adjacent GUI actions are merged and run together in ONE process, so key combinations split across consecutive <tool_call> blocks stay safe.
- A later action in the SAME response cannot see the new screenshot caused by an earlier action in that response. Only batch actions when the later actions do not depend on seeing the next screen.
- A `computer.terminate()` ends the task; do not put actions after it.

### Efficiency
- Prefer emitting multiple <tool_call> blocks in one step when you are confident about several consecutive visible actions.
- Stop and wait for the next observation when the next action depends on a newly changed screen.

## Response Examples

- GUI action

<thinking>xxx</thinking>
<tool_call>pyautogui.click(835, 420)</tool_call>

- GUI action (multiple actions)

<thinking>xxx</thinking>
<tool_call>pyautogui.click(960, 52)</tool_call>
<tool_call>pyautogui.hotkey('ctrl', 'a')</tool_call>
<tool_call>pyautogui.typewrite('example text', interval=0.05)</tool_call>
<tool_call>pyautogui.press('enter')</tool_call>

- Flow control

<thinking>xxx</thinking>
<tool_call>computer.wait()</tool_call>

- Finish

<thinking>xxx</thinking>
<tool_call>computer.terminate(status="success")</tool_call>

## Note
### Thoroughness
- Do not rush to declare terminate. Multi-step tasks usually require careful visible checks.
- Multi-step tasks: break the task into subtasks mentally and keep track of unfinished requirements.

### Persistence
- Do not give up easily. If your first approach fails, try at least 3 alternative visible methods before considering infeasible.
- Slow loading is not failure. If a page or application is loading, use wait() and try again.
- Unexpected dialogs or errors: read them carefully, dismiss or handle appropriately, then continue with the task.
- If a click does not work, try a slightly different position, keyboard shortcut, menu, context menu, or another visible route.

### Accuracy
- Read data from the screen. Never guess, estimate, or hallucinate values.
- Verify typed content after entering important data.
- File paths and names: read them exactly from the screen or task description. Do not assume or abbreviate.

### Task Fidelity
- Preserve the exact task intent: verb, object, constraints, target file/location, and required final state. Do not silently change the task into a related but different task.
- Before acting, identify the required operation type: edit, remove, add, count, report, transform, navigate, submit, or verify. Keep that operation type unchanged unless the task itself changes.
- Do not replace a required exact value, option, source, destination, or format with a near match. If the exact requested target is absent after careful checking, treat that as a blocker to investigate, not permission to approximate.
- Track required subgoals explicitly in your reasoning. If a task has multiple requirements, completing only the most visible one is not enough.

### Output Ownership
- Default to modifying the task-specified, currently open, downloaded, or original artifact. Do not create a substitute file, merged artifact, copied folder, or local stand-in unless the task explicitly asks for a new artifact or the original is confirmed unavailable.
- When an app is already open on a document, presentation, image, project, or settings page, assume that open artifact is task-relevant until you verify otherwise.
- If multiple possible source files, images, folders, or windows are visible, identify the task-specified source and destination before editing.

### Commit Discipline
- GUI text or setting edits are not complete until the application has committed them. After typing, selecting, toggling, or dragging, use the appropriate commit action when needed: Enter, Tab, blur/focus change, Save, Apply, OK, Done, or equivalent.
- Do not terminate while the only evidence is text still sitting in a focused input field, an unsaved dialog, a pending form, or an unconfirmed editor buffer.
- For sliders, color pickers, dropdowns, and dialogs, correct wrong-control clicks or accidental extreme values before continuing.

### Feasibility Honesty
- If a required option, time, feature, target, service, or file is absent after checking the relevant visible UI carefully, do not fabricate it and do not choose a near match without instruction.
- Do not invent application capabilities, themes, accounts, cloud locations, remote files, or services. Use only what is actually visible or what you can reliably create through the visible UI within the task requirements.
- If the exact request cannot be completed, use the infeasibility criteria instead of constructing a workaround that no longer satisfies the task.

## Multi-Application Coordination
When tasks involve multiple applications:
1. Complete one application's subtask fully before switching to another.
2. After pasting or transferring data, switch focus to the target app and verify the data arrived correctly.
3. Keep track of window positions and use Alt+Tab or the taskbar to switch.
4. Save intermediate work before switching away from an application when saving is relevant.
5. If an application becomes unresponsive, wait 5-10 seconds, then try clicking on it again.
6. The coordinate range is [0, {range_scale}] on both axes. Always emit click/move/drag coordinates within this range.

## Infeasibility Criteria
You may only declare failure for infeasibility when all of the following are true:
- You have attempted at least 3 different visible approaches to complete the task.
- The blocker is fundamental and cannot be worked around through the visible UI.
- The required application, file, option, target, service, or capability is confirmed unavailable from the visible evidence.
- The task contains contradictory or logically impossible requirements.

You must not declare infeasible when:
- A page is loading slowly.
- A dialog you did not expect appears.
- Your first click did not land correctly.
- An application takes time to start.
- You are unsure how to do something and have not tried menus, context menus, keyboard shortcuts, or visible help.

## Information Extraction
When the task asks you to find/read/report information:
1. Navigate to where the information is displayed.
2. Read the exact values from the screen.
3. Pass the information as the answer parameter in `computer.terminate(status="success", answer="...")` when an answer is required.
4. Use the exact format requested in the task.
""".strip()


def _expand_swipes(code: str) -> str:
    try:
        module = ast.parse(code)
    except SyntaxError:
        return code
    statements = []
    changed = False
    for statement in module.body:
        call = statement.value if isinstance(statement, ast.Expr) else None
        if (
            isinstance(call, ast.Call)
            and isinstance(call.func, ast.Attribute)
            and isinstance(call.func.value, ast.Name)
            and call.func.value.id == "pyautogui"
            and call.func.attr == "swipe"
            and len(call.args) in (4, 5)
            and all(keyword.arg in {"duration", "button"} for keyword in call.keywords)
        ):
            statements.extend(
                ast.parse(
                    f"pyautogui.moveTo({ast.unparse(call.args[0])}, {ast.unparse(call.args[1])})"
                ).body
            )
            drag = ast.Call(
                func=ast.Attribute(
                    value=ast.Name(id="pyautogui", ctx=ast.Load()), attr="dragTo", ctx=ast.Load()
                ),
                args=call.args[2:],
                keywords=call.keywords,
            )
            statements.append(ast.Expr(value=drag))
            changed = True
        else:
            statements.append(statement)
    if not changed:
        return code
    module.body = statements
    return ast.unparse(ast.fix_missing_locations(module))


def _parse_gui_response(
    response: dict[str, Any],
    screen_size: tuple[int, int],
    range_scale: float | None = 1000,
) -> tuple[dict[str, Any], dict[str, Any]]:

    content: str = response.get("content", "") or ""
    reasoning: str = (response.get("reasoning_content") or "").strip()
    sections: dict[str, Any] = {"reasoning_content": reasoning}

    if "</think>" in content.lower() and "<think>" not in content.lower():
        content = "<think>" + content

    if not reasoning:
        think_m = _THINK_TAG_RE.search(content)
        if think_m:
            reasoning = think_m.group(1).strip()
            sections["reasoning_content"] = reasoning
    content = _THINK_TAG_RE.sub("", content).strip()

    thinking_tag = _THINKING_TAG_RE.search(content)
    explicit_thought = thinking_tag.group(1).strip() if thinking_tag else ""
    sections["thought"] = explicit_thought or reasoning
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
        logger.warning("[GeneralE2EAgentPolicy] No GUI action code found in response")
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

        if "computer.wait" in code_lower:
            actions.append({"action_type": "wait"})
            codes.append("WAIT")
            continue

        if "computer.terminate" in code_lower:
            status = parse_terminate_status(code_block)
            terminal = {"action_type": "finished", "status": status}
            answer = _extract_call_string_arg(code_block, "answer")
            if answer is not None:
                terminal["answer"] = answer
            actions.append(terminal)
            codes.append("DONE" if status == "success" else "FAIL")
            continue

        if "computer." in code_lower:
            logger.warning(
                "[GeneralE2EAgentPolicy] Unsupported non-GUI computer action ignored: %r",
                code_block[:200],
            )
            continue

        projected = _project_coordinates(
            _expand_swipes(code_block),
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


class GeneralE2EAgentPolicy(DashscopeBaseAgentPolicy):
    SYSTEM_PROMPT = _SYSTEM_PROMPT
    DETAIL_CAPABILITY_NOTES = _GUI_DETAIL_CAPABILITY_NOTES

    @classmethod
    def from_agent_config(cls, cfg: Any) -> DashscopeBaseAgentPolicy:
        policy = super().from_agent_config(cfg)
        if (cfg.runtime_conf or {}).get("device_type") == "mobile":
            policy._system_prompt += (
                "\n\nTouchscreen input: every drag or swipe requires an explicit starting point. "
                "Emit pyautogui.moveTo(start_x, start_y) before pyautogui.dragTo(end_x, end_y) "
                "in the same response. The touchscreen has no persistent mouse cursor."
            )
        return policy

    async def select_action(self, observation: dict[str, Any]) -> dict[str, Any]:
        screenshot_b64 = observation.get("image", "")

        img = b64_to_pil(screenshot_b64)
        if self._max_image_pixels is not None:
            img = resize_max_pixels(img, self._max_image_pixels)
        resized_b64 = pil_to_b64(img, fmt=self._image_format, quality=self._jpeg_quality)

        messages = self._build_messages(resized_b64, "")

        action: dict[str, Any] = {"action_type": "FAIL"}
        cot: dict[str, Any] = {}
        response: dict[str, Any] | None = None
        got_response = False
        parse_attempts = 0

        for attempt in range(self._max_retries):
            temp = self._temperature if attempt == 0 else max(0.2, self._temperature)
            try:
                response = self._call_vlm(messages, temperature=temp)
            except Exception as exc:
                logger.error(
                    "[GeneralE2EAgentPolicy] VLM call failed (attempt %d/%d): %s; sleeping %.1fs",
                    attempt + 1,
                    self._max_retries,
                    exc,
                    self._retry_sleep,
                )
                time.sleep(self._retry_sleep)
                continue

            if not response:
                logger.error(
                    "[GeneralE2EAgentPolicy] Empty response (attempt %d/%d); sleeping %.1fs",
                    attempt + 1,
                    self._max_retries,
                    self._retry_sleep,
                )
                time.sleep(self._retry_sleep)
                continue

            got_response = True
            self.last_prediction = response.get("content", "")
            action, cot = _parse_gui_response(
                response,
                self._screen_size,
                self._range_scale,
            )

            if action["action_type"] != "FAIL":
                break

            parse_attempts += 1
            if parse_attempts >= self._parse_retry_n:
                logger.warning(
                    "[GeneralE2EAgentPolicy] Parse failed %d/%d times, giving up this step",
                    parse_attempts,
                    self._parse_retry_n,
                )
                break

            logger.warning(
                "[GeneralE2EAgentPolicy] Parse failed (parse attempt %d/%d), retrying",
                parse_attempts,
                self._parse_retry_n,
            )

        self._observations.append(resized_b64)
        self._actions.append(cot.get("action", ""))
        self._cots.append(cot)

        self.last_step_log = {
            "system_prompt": self._system_prompt,
            "instruction": self._instruction,
            "rendered_messages": elide_images(messages),
            "raw_response_text": self.last_prediction or "",
            "raw_reasoning_content": (response or {}).get("reasoning_content", ""),
            "parsed_action_dict": dict(action),
            "parsed_cot": dict(cot),
        }

        logger.info(
            "[GeneralE2EAgentPolicy] step=%d action_type=%s",
            len(self._actions),
            action.get("action_type"),
        )

        if action["action_type"] == "FAIL":
            if not got_response:
                raise AgentAPIError(
                    f"[GeneralE2EAgentPolicy] LLM call failed/empty after {self._max_retries} retries"
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
                "[GeneralE2EAgentPolicy] Reached max_steps=%d, forcing FAIL",
                self._max_steps,
            )
            return {"action_type": "finished", "status": "failure"}

        return action


GeneralE2ECodeAgentPolicy = GeneralE2EAgentPolicy


from mai_gym.agents import register_policy_factory

register_policy_factory("general_e2e_agent", GeneralE2EAgentPolicy.from_agent_config)
