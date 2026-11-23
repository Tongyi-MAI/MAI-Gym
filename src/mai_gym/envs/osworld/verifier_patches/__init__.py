from __future__ import annotations

import copy
import logging
import os
from collections.abc import Callable
from typing import Any

logger = logging.getLogger(__name__)

_ENV_FLAG = "MAI_GYM_USE_FIXED_VERIFIER"
_TRUTHY = {"1", "true", "yes", "on"}


def fixed_verifier_enabled() -> bool:

    override = _force_state[0]
    if override is not None:
        return override
    return os.environ.get(_ENV_FLAG, "").strip().lower() in _TRUTHY


_force_state: list[bool | None] = [None]


def set_fixed_verifier(enabled: bool | None) -> None:

    _force_state[0] = enabled


def _bare_uuid(task_id: str | None) -> str | None:

    if not task_id:
        return task_id
    return task_id.split("/")[-1]


def apply_config_patch(task_id: str | None, evaluator: dict[str, Any]) -> dict[str, Any]:

    if not fixed_verifier_enabled():
        return evaluator
    uid = _bare_uuid(task_id)
    fn = CONFIG_PATCHES.get(uid)
    if fn is None:
        return evaluator
    if PATCH_META.get(uid, {}).get("status") != "active":
        logger.info(
            "[verifier_patches] %s registered but status!=active; using ORIGINAL",
            uid,
        )
        return evaluator
    try:
        patched = fn(copy.deepcopy(evaluator))
    except Exception:
        logger.exception(
            "[verifier_patches] config patch for %s raised; using ORIGINAL",
            task_id,
        )
        return evaluator
    logger.info("[verifier_patches] applied config patch -> %s", task_id)
    return patched


def metric_override_for(task_id: str | None, func_name: str) -> Callable[..., Any] | None:

    if not fixed_verifier_enabled():
        return None
    uid = _bare_uuid(task_id)
    if PATCH_META.get(uid, {}).get("status") != "active":
        return None
    return METRIC_OVERRIDES.get(uid, {}).get(func_name)


def _as_list(x: Any) -> list[Any]:
    if x is None:
        return []
    return x if isinstance(x, list) else [x]


def drop_metric_indices(ev: dict[str, Any], indices: list[int]) -> dict[str, Any]:

    drop = set(indices)
    funcs = _as_list(ev.get("func"))
    results = _as_list(ev.get("result"))
    expected = _as_list(ev.get("expected"))
    options = _as_list(ev.get("options"))

    def keep(seq: list[Any]) -> list[Any]:
        return [v for i, v in enumerate(seq) if i not in drop]

    new_funcs = keep(funcs)
    ev["func"] = new_funcs if len(new_funcs) != 1 else new_funcs[0]
    if results:
        nr = keep(results)
        ev["result"] = nr if len(nr) != 1 else nr[0]
    if expected:
        ne = keep(expected)
        ev["expected"] = ne if len(ne) != 1 else ne[0]
    if options:
        no = keep(options)
        if no:
            ev["options"] = no if len(no) != 1 else no[0]
        else:
            ev.pop("options", None)

    return ev


def set_all_options(ev: dict[str, Any], **kv: Any) -> dict[str, Any]:

    n = len(_as_list(ev.get("func"))) or 1
    opts = _as_list(ev.get("options"))
    if not opts:
        opts = [{} for _ in range(n)]
    opts = [dict(o or {}) for o in opts]
    for o in opts:
        o.update(kv)
    ev["options"] = opts if len(opts) != 1 else opts[0]
    return ev


def set_option_at(ev: dict[str, Any], idx: int, **kv: Any) -> dict[str, Any]:

    n = len(_as_list(ev.get("func"))) or 1
    opts = _as_list(ev.get("options")) or [{} for _ in range(n)]
    opts = [dict(o or {}) for o in opts]
    while len(opts) <= idx:
        opts.append({})
    opts[idx].update(kv)
    ev["options"] = opts if len(opts) != 1 else opts[0]
    return ev


def bash_history_metric_indices(ev: dict[str, Any]) -> list[int]:

    other_probe = (
        " ps ",
        "ps aux",
        "pgrep",
        "pidof",
        "ps -",
        " ls ",
        "ls -",
        " find ",
        "stat ",
        " test -",
        "[ -e",
        "[ -f",
        "[ -d",
        "[ -s",
        " wc ",
        "md5sum",
        "sha1",
        "sha256",
        "curl",
        "wget",
        "http://",
        "https://",
        "python ",
        "python3 ",
        "nc ",
        "ss ",
        "netstat",
        "xdotool",
        "wmctrl",
    )
    out = []
    for i, r in enumerate(_as_list(ev.get("result"))):
        if not isinstance(r, dict) or r.get("type") != "vm_command_line":
            continue
        cmd = r.get("command")
        cmd_s = " ".join(cmd) if isinstance(cmd, list) else str(cmd or "")
        if ".bash_history" not in cmd_s:
            continue
        if any(tok in cmd_s for tok in other_probe):
            continue
        out.append(i)
    return out


def drop_bash_history_metrics(ev: dict[str, Any]) -> dict[str, Any]:

    idx = bash_history_metric_indices(ev)
    return drop_metric_indices(ev, idx) if idx else ev


from .patches import CONFIG_PATCHES, METRIC_OVERRIDES, PATCH_META

__all__ = [
    "fixed_verifier_enabled",
    "set_fixed_verifier",
    "apply_config_patch",
    "metric_override_for",
    "drop_metric_indices",
    "drop_bash_history_metrics",
    "bash_history_metric_indices",
    "set_all_options",
    "set_option_at",
    "CONFIG_PATCHES",
    "METRIC_OVERRIDES",
    "PATCH_META",
]
