from __future__ import annotations

import os
import re
from typing import Any

_JUDGE_RE = re.compile(
    r"reward_judge|call_llm_judge|llm_judge|dashscope|\bopenai\b|\banthropic\b"
    r"|qwen-?vl|gpt-4|gpt-3|/tmp/reward_judge|vision_judge|judge_prompt"
    r"|OpenAI\(|\.chat\.completions|generativeai|claude-",
    re.I,
)


def uses_llm_judge(reward_source: str) -> bool:

    return bool(_JUDGE_RE.search(reward_source or ""))


def resolve_mock_url(app: str, kind: str, ports: dict[str, int]) -> str | None:

    for slug in (f"{app.lower()}_mock", app.lower()):
        port = ports.get(slug)
        if port is not None:
            return f"localhost:{port}" if kind == "host" else f"http://localhost:{port}"
    return None


_PLACEHOLDER_RE = re.compile(r"__CUA_GYM_([A-Z0-9_]+?)_(URL|HOST)__")


def materialize_placeholders(
    text: str,
    ports: dict[str, int],
) -> tuple[str, set[str]]:

    unresolved: set[str] = set()

    def _repl(m: re.Match[str]) -> str:
        value = resolve_mock_url(m.group(1), m.group(2).lower(), ports)
        if value is None:
            unresolved.add(m.group(0))
            return m.group(0)
        return value

    return _PLACEHOLDER_RE.sub(_repl, text), unresolved


def _is_remote(url: str) -> bool:
    return "://" in (url or "")


def extract_stage_files(
    steps: list[dict[str, Any]],
    bundle_dir: str,
) -> tuple[list[tuple[str, str]], list[dict[str, Any]]]:

    stage: list[tuple[str, str]] = []
    new_steps: list[dict[str, Any]] = []
    for step in steps:
        if step.get("type") != "download":
            new_steps.append(step)
            continue
        files = step.get("parameters", {}).get("files", []) or []
        remote_files: list[dict[str, Any]] = []
        for f in files:
            url = f.get("url", "")
            if _is_remote(url):
                remote_files.append(f)
            else:
                local_path = os.path.normpath(os.path.join(bundle_dir, url))
                stage.append((local_path, f.get("path")))
        if remote_files:
            new_step = dict(step)
            new_step["parameters"] = {**step.get("parameters", {}), "files": remote_files}
            new_steps.append(new_step)

    return stage, new_steps


_OFFICE_SUFFIXES = (
    ".xlsx",
    ".xls",
    ".ods",
    ".csv",
    ".docx",
    ".doc",
    ".odt",
    ".rtf",
    ".pptx",
    ".ppt",
    ".odp",
)

_SOFFICE_BINS = ("soffice", "libreoffice", "localc", "lowriter", "loimpress")


_LO_RESET_MARKER = "mai-gym:lo-reset"


_RECOVERY_ITEM_PATTERN = r'<item oor:path="/org.openoffice.Office.Recovery[^>]*>.*?</item>'


_LO_RESET_SRC = f'''"{_LO_RESET_MARKER}"
import os, re, subprocess
PROFILE = os.environ.get("MAI_GYM_LO_PROFILE", "/home/user/.config/libreoffice/4")
try:
    if subprocess.run(["pgrep", "-x", "soffice.bin"],
                      capture_output=True).returncode == 0:
        raise SystemExit(0)
    try:
        os.remove(PROFILE + "/.lock")
    except OSError:
        pass
    reg = PROFILE + "/user/registrymodifications.xcu"
    with open(reg, encoding="utf-8") as fh:
        before = fh.read()
    after = re.sub({_RECOVERY_ITEM_PATTERN!r}, "", before, flags=re.S)
    if after != before:
        with open(reg, "w", encoding="utf-8") as fh:
            fh.write(after)
except SystemExit:
    raise
except Exception:
    pass
'''


def _step_opens_office_doc(step: dict[str, Any]) -> bool:
    if step.get("type") != "open":
        return False
    path = str(step.get("parameters", {}).get("path", "")).lower()
    return path.endswith(_OFFICE_SUFFIXES)


def _step_launches_soffice(step: dict[str, Any]) -> bool:
    if step.get("type") not in ("launch", "execute"):
        return False
    command = step.get("parameters", {}).get("command") or []
    if isinstance(command, str):
        command = command.split()
    if not command:
        return False
    argv0 = os.path.basename(str(command[0])).lower()
    return argv0 in _SOFFICE_BINS


def needs_libreoffice_reset(steps: list[dict[str, Any]]) -> bool:

    return any(_step_opens_office_doc(s) or _step_launches_soffice(s) for s in steps)


def prepend_libreoffice_reset(
    steps: list[dict[str, Any]],
) -> list[dict[str, Any]]:

    if not needs_libreoffice_reset(steps) or _already_prepended(steps):
        return steps
    reset_step = {
        "type": "execute",
        "parameters": {"command": ["python3", "-c", _LO_RESET_SRC]},
    }
    return [reset_step, *steps]


def _already_prepended(steps: list[dict[str, Any]]) -> bool:
    for step in steps:
        command = step.get("parameters", {}).get("command") or []
        if any(_LO_RESET_MARKER in str(part) for part in command):
            return True
    return False
