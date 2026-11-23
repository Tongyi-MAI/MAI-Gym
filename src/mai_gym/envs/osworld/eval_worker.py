from __future__ import annotations

import argparse
import json
import traceback
from pathlib import Path
from typing import Any

from mai_gym.envs.osworld._eval import osworld_task_eval
from mai_gym.envs.osworld.backend import OSWorldBackend
from mai_gym.models.session import RunningBackend


def _write_response(path: Path, payload: dict[str, Any]) -> None:
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(payload), encoding="utf-8")
    temp.chmod(0o600)
    temp.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--response", type=Path, required=True)
    args = parser.parse_args()

    try:
        request = json.loads(args.request.read_text(encoding="utf-8"))

        args.request.unlink(missing_ok=True)
        running = RunningBackend(**request["backend"])
        backend = OSWorldBackend(running)
        backend._cache_root = Path(request["cache_root"]).resolve()
        result = osworld_task_eval(
            backend,
            request["task_config"],
            request.get("action_history"),
            request.get("cache_key"),
            request.get("client_password", "password"),
        )
        _write_response(args.response, {"ok": True, "result": result})
        return 0
    except BaseException as exc:
        _write_response(
            args.response,
            {
                "ok": False,
                "error_type": type(exc).__name__,
                "error": str(exc),
                "traceback": traceback.format_exc(limit=20),
            },
        )
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
