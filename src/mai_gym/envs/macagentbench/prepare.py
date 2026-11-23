import ast
import logging
import time
from pathlib import Path
from typing import Any

from mai_gym.envs.shared.prepare import prepare_native
from mai_gym.envs.shared.provenance import PreparationError


def native_preparation_hooks(source: Path, env_class: type) -> dict[str, Any]:
    path = source / "batch_run.py"
    names = {"wait_for_ssh", "_fix_screen_recording_permission"}
    functions = [
        node
        for node in ast.parse(path.read_text()).body
        if isinstance(node, ast.FunctionDef) and node.name in names
    ]
    if {node.name for node in functions} != names:
        raise PreparationError("Pinned MacAgentBench source is missing native boot helpers")
    namespace: dict[str, Any] = {
        "MacOSEnv": env_class,
        "time": time,
        "logger": logging.getLogger(__name__),
    }
    body: list[ast.stmt] = list(functions)
    exec(compile(ast.Module(body=body, type_ignores=[]), str(path), "exec"), namespace)
    return {name: namespace[name] for name in names}


def prepare(settings: dict[str, Any]) -> dict[str, Any]:
    profile = prepare_native("macagentbench", settings)
    vm = settings.get("vm", {})
    files = {asset.get("path") for asset in profile["assets"] if asset.get("kind") != "docker"}
    images = {asset.get("image") for asset in profile["assets"] if asset.get("kind") == "docker"}
    if vm.get("image") not in images:
        raise PreparationError("MacAgentBench vm.image must match a verified Docker profile asset")
    for key in ("disk", "base_system"):
        path = str(Path(vm.get(key, "")).expanduser().resolve())
        if path not in files:
            raise PreparationError(f"MacAgentBench vm.{key} must match a verified profile asset")
    return profile
