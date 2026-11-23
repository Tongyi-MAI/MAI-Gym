from __future__ import annotations

import contextlib
import logging
import os
import re
import resource
import shutil
import signal
import subprocess
import tempfile
from dataclasses import dataclass

logger = logging.getLogger(__name__)


DEFAULT_TIMEOUT_S = 30
DEFAULT_MAX_MEMORY_MB = 512
DEFAULT_MAX_OUTPUT_CHARS = 50000


@dataclass
class ExecutionResult:
    stdout: str
    stderr: str
    exit_code: int
    timed_out: bool = False
    error: str = ""

    @property
    def success(self) -> bool:
        return self.exit_code == 0 and not self.timed_out


def _set_resource_limits() -> None:

    try:
        max_mem = int(os.environ.get("SANDBOX_MAX_MEMORY_MB", str(DEFAULT_MAX_MEMORY_MB)))
        mem_bytes = max_mem * 1024 * 1024
        resource.setrlimit(resource.RLIMIT_AS, (mem_bytes, mem_bytes))
    except (ValueError, OSError):
        pass
    with contextlib.suppress(ValueError, OSError):
        resource.setrlimit(resource.RLIMIT_CORE, (0, 0))


def _run_file_in_dir(
    file_path: str,
    cwd: str,
    *,
    timeout: float = DEFAULT_TIMEOUT_S,
    max_output_chars: int = DEFAULT_MAX_OUTPUT_CHARS,
) -> ExecutionResult:

    proc_env = os.environ.copy()
    proc_env["PYTHONDONTWRITEBYTECODE"] = "1"
    proc_env["PYTHONIOENCODING"] = "utf-8"

    proc = subprocess.Popen(
        ["python3", "-u", file_path],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        cwd=cwd,
        env=proc_env,
        preexec_fn=_set_resource_limits,
        start_new_session=True,
    )

    try:
        stdout_bytes, stderr_bytes = proc.communicate(timeout=timeout)
        stdout = stdout_bytes.decode("utf-8", errors="replace")[:max_output_chars]
        stderr = stderr_bytes.decode("utf-8", errors="replace")[:max_output_chars]
        return ExecutionResult(stdout=stdout, stderr=stderr, exit_code=proc.returncode)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except (OSError, ProcessLookupError):
            proc.kill()
        proc.wait()
        return ExecutionResult(
            stdout="",
            stderr="",
            exit_code=-1,
            timed_out=True,
            error=f"Execution timed out after {timeout}s",
        )


def extract_code_from_response(response: str) -> str:

    cleaned = response

    cleaned = re.sub(r"<think>.*?</think>", "", cleaned, flags=re.DOTALL)

    cleaned = re.sub(r"</think>\s*", "", cleaned)

    cleaned = re.sub(r"<think>\s*", "", cleaned)

    seed_match = re.search(
        r"<function=code_execute>\s*<parameter=code>(.*?)</parameter>",
        cleaned,
        re.DOTALL,
    )
    if seed_match:
        return seed_match.group(1).strip()

    seed_match2 = re.search(
        r"<parameter=code>(.*?)</parameter>",
        cleaned,
        re.DOTALL,
    )
    if seed_match2 and "code_execute" in cleaned:
        return seed_match2.group(1).strip()

    for text in (cleaned, response):
        python_blocks = re.findall(r"```python\s*\n(.*?)```", text, re.DOTALL)
        if python_blocks:
            for block in python_blocks:
                if "def solve" in block:
                    return block.strip()
            return python_blocks[-1].strip()

    for text in (cleaned, response):
        code_blocks = re.findall(r"```\s*\n(.*?)```", text, re.DOTALL)
        if code_blocks:
            for block in code_blocks:
                if "def solve" in block or "import openpyxl" in block:
                    return block.strip()
            return code_blocks[-1].strip()

    for text in (cleaned, response):
        match = re.search(r"(def solve\s*\(.*?\).*?)(?:\n\n[A-Z]|\Z)", text, re.DOTALL)
        if match:
            return match.group(1).strip()

    lines = cleaned.strip().split("\n")
    code_start = 0
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith(("def ", "class ", "import ", "from ", "#")):
            code_start = i
            break
        if stripped and not stripped[0].isalpha():
            code_start = i
            break

    return "\n".join(lines[code_start:]).strip()


def _build_kodcode_test_runner(test_code: str) -> str:

    test_fns = re.findall(r"def (test_\w+)", test_code)
    if not test_fns:
        return test_code

    runner_lines = [
        test_code,
        "",
        "# --- Run all test functions ---",
    ]
    for fn in test_fns:
        runner_lines.append(f"{fn}()")
    runner_lines.append('print("All tests passed!")')

    return "\n".join(runner_lines)


def run_kodcode_tests_sync(
    solution_code: str,
    test_code: str,
    *,
    timeout: float = DEFAULT_TIMEOUT_S,
    max_output_chars: int = DEFAULT_MAX_OUTPUT_CHARS,
) -> tuple[float, ExecutionResult]:

    tmpdir = None
    try:
        tmpdir = tempfile.mkdtemp(prefix="kodcode_")

        solution_path = os.path.join(tmpdir, "solution.py")
        with open(solution_path, "w") as f:
            f.write(solution_code)

        isolated_file = os.path.join(tmpdir, "input.xlsx")
        test_runner = _build_kodcode_test_runner(test_code)
        test_runner = test_runner.replace(
            "FILE = '/tmp/input.xlsx'",
            f"FILE = '{isolated_file}'",
        )
        test_path = os.path.join(tmpdir, "test_solution.py")
        with open(test_path, "w") as f:
            f.write(test_runner)

        result = _run_file_in_dir(
            test_path,
            tmpdir,
            timeout=timeout,
            max_output_chars=max_output_chars,
        )

        reward = 1.0 if result.success else 0.0
        return reward, result

    except Exception as exc:
        return 0.0, ExecutionResult(
            stdout="",
            stderr="",
            exit_code=-1,
            error=f"Sandbox error: {exc}",
        )
    finally:
        if tmpdir:
            shutil.rmtree(tmpdir, ignore_errors=True)


def run_tests_sync(
    solution_code: str,
    test_cases: str,
    *,
    entry_point: str | None = None,
    timeout: float = DEFAULT_TIMEOUT_S,
    max_output_chars: int = DEFAULT_MAX_OUTPUT_CHARS,
) -> tuple[float, ExecutionResult]:

    if "from solution import" in test_cases:
        return run_kodcode_tests_sync(
            solution_code,
            test_cases,
            timeout=timeout,
            max_output_chars=max_output_chars,
        )

    parts = [solution_code.strip(), "", test_cases.strip(), ""]
    test_fns = re.findall(r"def (test_\w+)", test_cases)
    if test_fns:
        for fn in test_fns:
            parts.append(f"{fn}()")
        parts.append('print("All tests passed!")')

    combined = "\n".join(parts)
    tmpdir = None
    try:
        tmpdir = tempfile.mkdtemp(prefix="coding_sandbox_")
        code_path = os.path.join(tmpdir, "_run.py")
        with open(code_path, "w") as f:
            f.write(combined)

        result = _run_file_in_dir(
            code_path,
            tmpdir,
            timeout=timeout,
            max_output_chars=max_output_chars,
        )
        return (1.0 if result.success else 0.0), result
    except Exception as exc:
        return 0.0, ExecutionResult(
            stdout="",
            stderr="",
            exit_code=-1,
            error=f"Sandbox error: {exc}",
        )
    finally:
        if tmpdir:
            shutil.rmtree(tmpdir, ignore_errors=True)
