from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

_BOOTSTRAP = (
    "import importlib.util, json, sys\n"
    "spec = importlib.util.spec_from_file_location('cua_hub_reward', sys.argv[1])\n"
    "mod = importlib.util.module_from_spec(spec)\n"
    "spec.loader.exec_module(mod)\n"
    "payload = json.loads(sys.stdin.read())\n"
    "fn = getattr(mod, 'reward', None)\n"
    "if fn is None:\n"
    "    print(json.dumps({'score': 0.0, 'reason': 'no reward() in reward.py'}))\n"
    "    sys.exit(0)\n"
    "try:\n"
    "    res = fn(payload['initial_state'],\n"
    "             payload['current_state'],\n"
    "             payload['state_diff'])\n"
    "except TypeError:\n"
    "    # Fallback: some reward.py only take (initial, current).\n"
    "    res = fn(payload['initial_state'], payload['current_state'])\n"
    "if isinstance(res, (int, float)):\n"
    "    print(json.dumps({'score': float(res), 'reason': None}))\n"
    "elif isinstance(res, dict):\n"
    "    print(json.dumps({'score': float(res.get('score', 0.0)),\n"
    "                      'reason': res.get('reason')}))\n"
    "else:\n"
    "    print(json.dumps({'score': 0.0,\n"
    "                      'reason': f'unexpected reward type {type(res).__name__}'}))\n"
)


def run_reward(
    task_dir: Path,
    initial_state: dict[str, Any] | None,
    current_state: dict[str, Any] | None,
    state_diff: dict[str, Any] | None,
    *,
    timeout: float = 30.0,
) -> tuple[float | None, str | None]:

    reward_path = task_dir / "reward.py"
    if not reward_path.is_file():
        return None, f"reward.py missing under {task_dir}"

    payload = json.dumps(
        {
            "initial_state": initial_state or {},
            "current_state": current_state or {},
            "state_diff": state_diff or {},
        }
    )
    try:
        proc = subprocess.run(
            ["python3", "-c", _BOOTSTRAP, str(reward_path)],
            input=payload,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=str(task_dir),
        )
    except subprocess.TimeoutExpired:
        return 0.0, "reward.py timed out"
    if proc.returncode != 0:
        return 0.0, f"reward.py exit {proc.returncode}: {proc.stderr[-200:]}"
    try:
        body = json.loads(proc.stdout.strip().splitlines()[-1])
        return float(body.get("score", 0.0)), body.get("reason")
    except Exception as exc:
        return 0.0, f"reward.py output parse failed: {exc}"
