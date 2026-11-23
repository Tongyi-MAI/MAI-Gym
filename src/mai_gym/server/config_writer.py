from __future__ import annotations

import contextlib
import logging
import os
import re
import stat
from pathlib import Path

logger = logging.getLogger(__name__)

_MAX_SESSIONS_RE = re.compile(r"^(\s*max_sessions:\s*)(\d+)(.*)$")
_POOL_SIZE_RE = re.compile(r"^(\s*pool_size:\s*)(\d+)(.*)$")
_NUM_ENVS_RE = re.compile(r"^(\s*num_envs:\s*)(\d+)(.*)$")
_LIST_ITEM_RE = re.compile(r"^(\s*)-\s")


def _name_re(device_name: str) -> re.Pattern[str]:
    return re.compile(r'^(\s*)-\s+name:\s*["\']?' + re.escape(device_name) + r'["\']?\s*(#.*)?$')


def persist_max_sessions(config_path: str | None, device_name: str, new_max: int) -> bool:

    return persist_scale_settings(config_path, device_name, new_max)


def persist_scale_settings(
    config_path: str | None,
    device_name: str,
    new_max: int,
    *,
    pool_size: int | None = None,
    num_envs: int | None = None,
) -> bool:

    if not config_path:
        return False
    path = Path(config_path)
    try:
        text = path.read_text()
        original_mode = stat.S_IMODE(path.stat().st_mode)
    except OSError as exc:
        logger.warning("persist_scale_settings: cannot read %s (%s)", config_path, exc)
        return False

    lines = text.splitlines(keepends=True)
    changes: list[tuple[int, re.Pattern[str], int, str]] = []
    max_idx = _locate_device_key_line(lines, device_name, _MAX_SESSIONS_RE)
    if max_idx is None:
        logger.warning(
            "persist_scale_settings: no max_sessions line for device %r in %s",
            device_name,
            config_path,
        )
        return False
    changes.append((max_idx, _MAX_SESSIONS_RE, new_max, "max_sessions"))

    if pool_size is not None:
        pool_idx = _locate_device_key_line(lines, device_name, _POOL_SIZE_RE)
        if pool_idx is None:
            logger.warning(
                "persist_scale_settings: no pool_size line for device %r in %s",
                device_name,
                config_path,
            )
            return False
        changes.append((pool_idx, _POOL_SIZE_RE, pool_size, "pool_size"))

    if num_envs is not None:
        num_idxs = [i for i, line in enumerate(lines) if _NUM_ENVS_RE.match(line.rstrip("\n"))]
        if len(num_idxs) != 1:
            logger.warning(
                "persist_scale_settings: expected one num_envs line in %s, got %d",
                config_path,
                len(num_idxs),
            )
            return False
        changes.append((num_idxs[0], _NUM_ENVS_RE, num_envs, "num_envs"))

    for idx, pattern, value, _ in changes:
        line = lines[idx]
        match = pattern.match(line.rstrip("\n"))
        if match is None:
            return False
        newline = "\n" if line.endswith("\n") else ""
        lines[idx] = f"{match.group(1)}{value}{match.group(3)}{newline}"

    tmp = f"{config_path}.tmp.{os.getpid()}"
    try:
        with open(tmp, "w") as f:
            f.write("".join(lines))
        os.chmod(tmp, original_mode)
        os.replace(tmp, config_path)
    except OSError as exc:
        logger.warning("persist_scale_settings: write failed for %s (%s)", config_path, exc)
        with contextlib.suppress(OSError):
            os.unlink(tmp)
        return False
    logger.info(
        "persist_scale_settings: %s device=%s values=%s",
        config_path,
        device_name,
        {name: value for _, _, value, name in changes},
    )
    return True


def _locate_max_sessions_line(lines: list[str], device_name: str) -> int | None:

    return _locate_device_key_line(lines, device_name, _MAX_SESSIONS_RE)


def _locate_device_key_line(
    lines: list[str], device_name: str, key_re: re.Pattern[str]
) -> int | None:

    name_re = _name_re(device_name)
    start = indent = None
    for i, ln in enumerate(lines):
        m = name_re.match(ln.rstrip("\n"))
        if m:
            start, indent = i, len(m.group(1))
            break

    if start is not None:
        for j in range(start + 1, len(lines)):
            mm = _LIST_ITEM_RE.match(lines[j].rstrip("\n"))
            if mm and len(mm.group(1)) <= indent:
                break
            if key_re.match(lines[j].rstrip("\n")):
                return j
        return None

    idxs = [i for i, ln in enumerate(lines) if key_re.match(ln.rstrip("\n"))]
    return idxs[0] if len(idxs) == 1 else None
