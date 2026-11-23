from __future__ import annotations

import logging
import os
import re
from typing import Any

from ..helpers.coding_sandbox import (
    DEFAULT_TIMEOUT_S,
    extract_code_from_response,
    run_tests_sync,
)

logger = logging.getLogger(__name__)

CODING_TIMEOUT = float(os.environ.get("CODING_TIMEOUT_S", str(DEFAULT_TIMEOUT_S)))


def _wrap_as_solve_function(code: str) -> str:

    if re.search(r"^def solve\s*\(", code, re.MULTILINE):
        code = re.sub(r"""(['"])(/home/user/[^'"]+\.xlsx)\1""", "path", code)
        return code

    code = re.sub(r"""(['"])(/home/user/[^'"]+\.xlsx)\1""", "FILE", code)
    lines = code.strip().split("\n")
    indented = ["    " + line for line in lines]
    return "def solve(path):\n    FILE = path\n" + "\n".join(indented)


def compute_coding_reward(
    response: str,
    test_cases: str,
    *,
    entry_point: str | None = None,
    timeout: float = CODING_TIMEOUT,
) -> tuple[float, dict[str, Any]]:

    solution_code = extract_code_from_response(response)

    if not solution_code.strip():
        return 0.0, {
            "error": "No code found in response",
            "solution_code": "",
            "format_reward": 0.0,
        }

    format_reward = 1.0

    if "from solution import solve" in test_cases:
        solution_code = _wrap_as_solve_function(solution_code)

    reward, result = run_tests_sync(
        solution_code,
        test_cases,
        entry_point=entry_point,
        timeout=timeout,
    )

    info = {
        "solution_code": solution_code[:2000],
        "stdout": result.stdout[:2000],
        "stderr": result.stderr[:2000],
        "exit_code": result.exit_code,
        "timed_out": result.timed_out,
        "format_reward": format_reward,
    }
    if result.error:
        info["error"] = result.error

    return reward, info


async def async_rm(args: Any, sample: Any) -> float:

    import asyncio

    metadata = sample.metadata or {}
    if isinstance(metadata, str):
        try:
            metadata = __import__("json").loads(metadata)
        except Exception:
            metadata = {}

    test_cases = metadata.get("test_cases", "")
    response = sample.response or ""

    if not test_cases:
        logger.warning("No test_cases in sample metadata, returning 0.0")
        return 0.0

    loop = asyncio.get_running_loop()
    reward, info = await loop.run_in_executor(
        None,
        lambda: compute_coding_reward(response, test_cases),
    )

    logger.info(
        "[CodingRM] task=%s reward=%.2f exit=%s timeout=%s",
        metadata.get("task_id", "?"),
        reward,
        info.get("exit_code", "?"),
        info.get("timed_out", False),
    )

    return reward
