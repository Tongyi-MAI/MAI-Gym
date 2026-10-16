from __future__ import annotations

import base64
import copy
import struct
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from mai_gym.agents.backends.http_chat_model import HttpChatModel
from mai_gym.agents.parsing.mai_gym_computer_use import (
    ActionParseResult,
    parse_action_response,
)

IMAGE_COLLAPSED = "<image_collapsed>"
MAX_ACTION_PARSE_RETRIES = 3

_ASK_USER_FLOW_ACTION = (
    '- `computer.ask_user(question="...")` asks for information that is unavailable '
    "from the task or environment."
)
_ASK_USER_TOOL_DESCRIPTION = '- `computer.ask_user(question="...")`'


@dataclass
class StepRecord:
    response_text: str
    observation_screenshot: str
    action_dict: dict[str, Any]
    format_reward: float
    is_terminal: bool
    rendered_messages: list[dict[str, Any]] = field(default_factory=list)
    finish_reason: str = ""
    response_token_ids: list[int] = field(default_factory=list)
    response_logprobs: list[float] = field(default_factory=list)
    prompt_token_ids: list[int] = field(default_factory=list)
    multimodal_train_inputs: dict[str, Any] | None = None
    cached_tokens: int = 0
    prompt_tokens: int = 0


def _decode_png_size(image_b64: str) -> tuple[int, int] | None:
    try:
        raw = base64.b64decode(image_b64, validate=True)
    except (ValueError, TypeError):
        return None
    if len(raw) < 24 or raw[:8] != b"\x89PNG\r\n\x1a\n" or raw[12:16] != b"IHDR":
        return None
    return struct.unpack(">II", raw[16:24])


def _window_start(step_idx: int, window_size: int, overlap: int) -> int:
    stride = window_size - overlap
    if stride <= 0:
        raise ValueError(
            f"window_size - overlap must be > 0 (W={window_size}, OV={overlap})"
        )
    return max(0, stride * ((step_idx - overlap) // stride))


def _format_bash_output(observation: dict[str, Any]) -> str:
    if not any(
        key in observation for key in ("bash_output", "bash_error", "bash_returncode")
    ):
        return ""
    output = str(observation.get("bash_output", "") or "")
    error = str(observation.get("bash_error", "") or "")
    returncode = int(observation.get("bash_returncode", 0) or 0)
    parts = [f"## Bash Output\nreturncode={returncode}"]
    if output:
        parts.append("stdout:\n" + output.rstrip()[:4000])
    if error:
        parts.append("stderr:\n" + error.rstrip()[:2000])
    if not output and not error:
        parts.append("(no output)")
    return "\n\n".join(parts)


def _render_history_turn(reasoning: str, history_action: str) -> str:
    parts: list[str] = []
    if reasoning.strip():
        parts.append(f"<think>{reasoning.strip()}</think>")
    if history_action.strip():
        parts.append(history_action.strip())
    return "\n".join(parts)


def _render_system_prompt(
    *,
    password: str,
    screen_size: tuple[int, int],
    range_scale: float | None,
    enable_ask_user: bool,
) -> str:
    prompt_path = (
        Path(__file__).resolve().parents[1]
        / "prompts"
        / "mai_gym_computer_use_system.j2"
    )
    prompt = prompt_path.read_text(encoding="utf-8")
    if range_scale is None:
        coordinate_scale = f"raw pixels: x=0..{screen_size[0]}, y=0..{screen_size[1]}"
        coordinate_half = f"{screen_size[0] // 2}, {screen_size[1] // 2}"
    else:
        coordinate_scale = str(int(range_scale) if float(range_scale).is_integer() else range_scale)
        half = range_scale / 2
        coordinate_half = str(int(half) if float(half).is_integer() else half)
    return (
        prompt.replace("{password}", password or "(not configured)")
        .replace("{range_scale}", coordinate_scale)
        .replace("{range_half}", coordinate_half)
        .replace(
            "{ask_user_flow_action}",
            _ASK_USER_FLOW_ACTION if enable_ask_user else "",
        )
        .replace(
            "{ask_user_tool_description}",
            _ASK_USER_TOOL_DESCRIPTION if enable_ask_user else "",
        )
        .strip()
    )


class MAIGymComputerUsePolicy:
    """Dependency-light MAI-Gym computer-use policy.

    The policy owns prompt rendering, history/window state, parse retries and
    action records. HTTP transport and action parsing are separate modules.
    """

    NATIVE_ACTION_SPACE = "computer_use"

    def __init__(
        self,
        chat_model: HttpChatModel,
        sampling_params: dict[str, Any] | None = None,
        *,
        password: str = "",
        screen_size: tuple[int, int] = (1920, 1080),
        range_scale: float | None = 1000,
        keep_n: int = 3,
        img_interval: int = 10,
        image_retention: int | None = None,
        enable_ask_user: bool = False,
        window_size: int | None = None,
        overlap: int = 1,
        max_history_turns: int = 0,
        max_parse_retries: int = MAX_ACTION_PARSE_RETRIES,
        **_: Any,
    ) -> None:
        self.chat_model = chat_model
        self.sampling_params = sampling_params or {}
        self._password = password
        self._screen_size = screen_size
        self._range_scale = range_scale
        self._img_keep_n = max(0, int(image_retention if image_retention is not None else keep_n))
        self._img_interval = max(0, int(img_interval))
        self._enable_ask_user = bool(enable_ask_user)
        self._window_size = int(window_size) if window_size else None
        self._overlap = max(0, int(overlap))
        if self._window_size is not None and self._window_size - self._overlap <= 0:
            raise ValueError(
                f"window_size - overlap must be > 0 "
                f"(W={self._window_size}, OV={self._overlap})"
            )
        self._max_history_turns = max(0, int(max_history_turns))
        self.max_parse_retries = max(0, int(max_parse_retries))
        self.system_prompt = _render_system_prompt(
            password=self._password,
            screen_size=self._screen_size,
            range_scale=self._range_scale,
            enable_ask_user=self._enable_ask_user,
        )
        self.timing: Any = None
        self.last_prediction: str | None = None
        self.step_records: list[StepRecord] = []
        self._instruction = ""
        self._actions: list[dict[str, Any]] = []
        self._reasoning: list[str] = []
        self._history_actions: list[str] = []
        self._observations: list[str] = []
        self._bash_results: list[str] = []

    def initialize(self, instruction: str) -> None:
        if instruction is None:
            raise ValueError("instruction must not be None")
        self.chat_model.reset()
        self._instruction = instruction
        self.last_prediction = None
        self.step_records.clear()
        self._actions.clear()
        self._reasoning.clear()
        self._history_actions.clear()
        self._observations.clear()
        self._bash_results.clear()

    def done(self) -> None:
        return None

    async def close(self) -> None:
        await self.chat_model.aclose()

    def _current_image_keep_count(self) -> int | None:
        if self._img_keep_n <= 0 or self._img_interval <= 0:
            return None
        total = len(self._actions) + 1
        capacity = self._img_keep_n + self._img_interval
        if total <= capacity:
            return total
        trims = 1 + (total - capacity - 1) // (self._img_interval + 1)
        return total - trims * (self._img_interval + 1)

    def _visible_image_start(self, n_steps: int) -> int:
        if self._window_size is not None:
            return _window_start(n_steps, self._window_size, self._overlap)
        keep_count = self._current_image_keep_count()
        if keep_count is None:
            return 0
        return max(0, n_steps + 1 - keep_count)

    def _build_messages(
        self,
        current_screenshot_b64: str,
        current_bash_block: str,
    ) -> list[dict[str, Any]]:
        n_steps = len(self._actions)
        history_start = (
            max(0, n_steps - self._max_history_turns)
            if self._max_history_turns
            else 0
        )
        image_start = self._visible_image_start(n_steps)
        instruction = (
            "# Task Instruction:\n"
            f"{self._instruction}\n"
            "Complete the task with actions in the system-prompt format."
        )
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": self.system_prompt}
        ]

        for index in range(history_start, n_steps):
            user_parts: list[dict[str, Any]] = []
            if index == history_start:
                user_parts.append({"type": "text", "text": instruction})
            if self._bash_results[index]:
                user_parts.append({"type": "text", "text": self._bash_results[index]})
            if index < image_start:
                user_parts.append({"type": "text", "text": IMAGE_COLLAPSED})
            else:
                user_parts.append(
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{self._observations[index]}"
                        },
                    }
                )
            messages.append({"role": "user", "content": user_parts})
            messages.append(
                {
                    "role": "assistant",
                    "content": _render_history_turn(
                        self._reasoning[index], self._history_actions[index]
                    ),
                    "_output_ids": self.step_records[index].response_token_ids,
                }
            )

        current_parts: list[dict[str, Any]] = []
        if n_steps == 0:
            current_parts.append({"type": "text", "text": instruction})
        if current_bash_block:
            current_parts.append({"type": "text", "text": current_bash_block})
        current_parts.append(
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{current_screenshot_b64}"},
            }
        )
        messages.append({"role": "user", "content": current_parts})
        return messages

    def _parse_model_response(
        self,
        response: dict[str, Any],
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        parsed = parse_action_response(
            str(response.get("content", "")),
            screen_size=self._screen_size,
            range_scale=self._range_scale,
            enable_ask_user=self._enable_ask_user,
            finish_reason=str(response.get("finish_reason", "")),
        )
        cot = {
            "reasoning_content": str(response.get("reasoning_content", "")),
            "think": parsed.reasoning,
            "action": parsed.history_action,
        }
        if parsed.error:
            cot["error"] = parsed.error
        return parsed.action_dict, cot

    @staticmethod
    def _retry_feedback(response_text: str) -> str:
        excerpt = response_text.strip()
        if len(excerpt) > 2000:
            excerpt = excerpt[:2000] + "\n... [truncated]"
        return (
            "Your previous response could not be parsed into an executable action.\n"
            "Use raw <bash>...</bash>, or one supported pyautogui/wait/terminate "
            "call inside <tool_call>...</tool_call>. Never call computer.bash.\n\n"
            f"Unparseable previous response excerpt:\n{excerpt}"
        )

    async def select_action(self, observation: dict[str, Any]) -> dict[str, Any]:
        screenshot = str(observation.get("image", "") or "")
        if not screenshot:
            if self._observations:
                screenshot = self._observations[-1]
            else:
                return {"action_type": "wait"}
        size = _decode_png_size(screenshot)
        if size is not None and not self._actions:
            self._screen_size = size
            self.system_prompt = _render_system_prompt(
                password=self._password,
                screen_size=self._screen_size,
                range_scale=self._range_scale,
                enable_ask_user=self._enable_ask_user,
            )

        bash_block = _format_bash_output(observation)
        base_messages = self._build_messages(screenshot, bash_block)
        messages = base_messages
        result: Any = None
        parsed: ActionParseResult | None = None
        for _attempt in range(self.max_parse_retries + 1):
            result = await self.chat_model.generate(
                messages,
                self.sampling_params,
                timing=self.timing,
            )
            parsed = parse_action_response(
                str(result.content),
                screen_size=self._screen_size,
                range_scale=self._range_scale,
                enable_ask_user=self._enable_ask_user,
                finish_reason=str(getattr(result, "finish_reason", "")),
            )
            if parsed.is_valid:
                break
            messages = [
                *base_messages,
                {"role": "assistant", "content": str(result.content)},
                {"role": "user", "content": self._retry_feedback(str(result.content))},
            ]

        assert result is not None and parsed is not None
        if parsed.is_valid:
            action = parsed.action_dict
        else:
            action = {"action_type": "terminate", "status": "failure"}

        record = StepRecord(
            response_text=str(result.content),
            observation_screenshot=screenshot,
            action_dict=action,
            format_reward=parsed.format_reward if parsed.is_valid else 0.0,
            is_terminal=parsed.is_done if parsed.is_valid else True,
            rendered_messages=copy.deepcopy(base_messages),
            finish_reason=str(getattr(result, "finish_reason", "")),
            response_token_ids=list(getattr(result, "output_ids", [])),
            response_logprobs=list(getattr(result, "output_logprobs", [])),
            prompt_token_ids=list(getattr(result, "prompt_token_ids", [])),
            multimodal_train_inputs=getattr(result, "multimodal_train_inputs", None),
            cached_tokens=int(getattr(result, "cached_tokens", 0)),
            prompt_tokens=int(getattr(result, "prompt_tokens", 0)),
        )
        self.step_records.append(record)
        self.last_prediction = str(result.content)
        self._observations.append(screenshot)
        self._bash_results.append(bash_block)
        self._actions.append(action)
        self._reasoning.append(parsed.reasoning)
        self._history_actions.append(parsed.history_action)
        return action


__all__ = [
    "IMAGE_COLLAPSED",
    "MAIGymComputerUsePolicy",
    "MAX_ACTION_PARSE_RETRIES",
    "StepRecord",
]
