from __future__ import annotations

import logging
import re
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from mai_gym.envs.osworld.backend import OSWorldBackend

logger = logging.getLogger(__name__)

_REWARD_RE = re.compile(r"REWARD:\s*(-?\d+(?:\.\d+)?)")


def _parse_reward_output(output: str) -> tuple[float, str]:

    matches = list(_REWARD_RE.finditer(output or ""))
    if not matches:
        tail = (output or "").strip()[-200:]
        return 0.0, f"no REWARD line in reward.py output: {tail!r}"
    score = float(matches[-1].group(1))
    score = max(0.0, min(1.0, score))
    score_line = ""
    for line in reversed((output or "").splitlines()):
        if line.strip().startswith("Score:"):
            score_line = line.strip()
            break
    reason = score_line or f"REWARD: {score}"
    return score, reason


def run_python_evaluator(
    reward_source: str,
    execute: Callable[[str], str],
) -> tuple[float, str]:

    try:
        output = execute(reward_source)
    except Exception as exc:
        logger.warning("[cua-eval] reward.py execution failed: %s", exc)
        return 0.0, f"reward.py execution failed: {exc}"
    return _parse_reward_output(output)


def _run_python_in_vm(vm_ip: str, server_port: int, source: str) -> str:

    import base64

    import httpx

    b64 = base64.b64encode(source.encode("utf-8")).decode("ascii")
    remote = "/tmp/_cua_reward.py"
    cmd = ["bash", "-lc", f"echo {b64} | base64 -d > {remote} && python3 {remote}"]
    r = httpx.post(
        f"http://{vm_ip}:{server_port}/execute",
        json={"command": cmd, "shell": False},
        timeout=180,
    )
    body = r.json()
    if not isinstance(body, dict):
        return ""
    return (body.get("output") or "") + (body.get("error") or "")


def _run_cua_python_evaluator(
    backend: OSWorldBackend,
    task_config: dict[str, Any],
    evaluator: dict[str, Any],
) -> tuple[float, str]:

    bundle_dir = task_config.get("_dir")
    if not bundle_dir:
        return 0.0, "cua python evaluator: task_config missing '_dir'"
    reward_path = Path(bundle_dir) / (evaluator.get("url") or "./reward.py")
    if not reward_path.is_file():
        return 0.0, f"cua python evaluator: reward.py not found: {reward_path}"
    source = reward_path.read_text(encoding="utf-8", errors="ignore")
    vm_ip, server_port = backend._parse_vm_endpoint()

    def execute(src: str) -> str:
        return _run_python_in_vm(vm_ip, server_port, src)

    return run_python_evaluator(source, execute)


def _resolve_local_eval_file(task_config: dict[str, Any], config: dict[str, Any]) -> str:

    raw_path = str(config["path"])
    path = Path(raw_path)
    if path.is_absolute():
        resolved = path
    else:
        examples_dir = task_config.get("__examples_dir")
        if not examples_dir:
            raise FileNotFoundError(
                f"local_file {raw_path!r} requires task_config['__examples_dir']"
            )
        resolved = Path(str(examples_dir)) / path
    if not resolved.is_file():
        raise FileNotFoundError(f"local_file not found: {resolved}")
    return str(resolved)


def osworld_task_eval(
    backend: OSWorldBackend,
    task_config: dict[str, Any],
    action_history: list[str] | None = None,
    cache_key: str | None = None,
    client_password: str = "password",
) -> tuple[float, str] | None:

    evaluator = task_config.get("evaluator")
    if not evaluator:
        return None

    if evaluator.get("type") == "python":
        return _run_cua_python_evaluator(backend, task_config, evaluator)

    from mai_gym.envs.osworld.verifier_patches import (
        apply_config_patch,
        metric_override_for,
    )

    evaluator = apply_config_patch(task_config.get("id"), evaluator)

    funcs = evaluator.get("func")
    if funcs == "infeasible":
        history = action_history or []
        if history and history[-1] == "FAIL":
            return 1.0, "infeasible: agent terminated with FAIL"
        return 0.0, (
            f"infeasible: agent did not terminate with FAIL "
            f"(last action={history[-1] if history else 'none'})"
        )

    try:
        from desktop_env.controllers.python import PythonController
        from desktop_env.controllers.setup import SetupController
        from desktop_env.evaluators import getters as getters_pkg
        from desktop_env.evaluators import metrics as metrics_pkg
    except ImportError as exc:
        raise RuntimeError(
            "OSWorldBackend.task_eval requires the 'osworld' extra and the desktop_env package."
        ) from exc

    from mai_gym.envs.osworld.ocr_pool import install_ocr_pool

    install_ocr_pool()

    def _get_local_file(env: Any, config: dict[str, Any]) -> str:
        return _resolve_local_eval_file(task_config, config)

    getters_pkg.get_local_file = _get_local_file

    vm_ip, server_port = backend._parse_vm_endpoint()
    cache_dir = backend._task_cache_dir(task_config.get("id"), cache_key=cache_key)
    meta = backend.backend.metadata
    chromium_port = int(meta.get("chromium_port", 9222))
    vlc_port = int(meta.get("vlc_port", 8080))
    vnc_port = int(meta.get("vnc_port", 8006))

    de = meta.get("desktop_env")
    if de is not None:
        setup_controller = de.setup_controller
        if hasattr(setup_controller, "cache_dir"):
            setup_controller.cache_dir = cache_dir
        if hasattr(setup_controller, "client_password"):
            setup_controller.client_password = client_password
    else:
        setup_controller = SetupController(
            vm_ip=vm_ip,
            server_port=server_port,
            chromium_port=chromium_port,
            vlc_port=vlc_port,
            cache_dir=cache_dir,
            client_password=client_password,
        )

    postconfig = evaluator.get("postconfig") or []
    if postconfig:
        logger.info(
            "[OSWorldBackend] task_eval: running %d postconfig step(s)",
            len(postconfig),
        )
        setup_controller.setup(postconfig)

    env = _OSWorldEvalEnv(
        controller=PythonController(vm_ip, server_port),
        setup_controller=setup_controller,
        cache_dir=cache_dir,
        evaluator=evaluator,
        vm_ip=vm_ip,
        server_port=server_port,
        chromium_port=chromium_port,
        vlc_port=vlc_port,
        vnc_port=vnc_port,
    )

    funcs_list = funcs if isinstance(funcs, list) else [funcs]

    _task_id = task_config.get("id")
    metric_callables = []
    for f in funcs_list:
        metric_fn = metric_override_for(_task_id, f)
        if metric_fn is None:
            metric_fn = getattr(metrics_pkg, f)
            if f == "is_vlc_playing":
                from mai_gym.envs.osworld.vlc_playing import make_vlc_playing_metric

                metric_fn = make_vlc_playing_metric(
                    metric_fn,
                    env,
                    getters_pkg.get_vm_command_line,
                )
        metric_callables.append(metric_fn)

    result_cfgs = evaluator.get("result", [])
    if not isinstance(result_cfgs, list):
        result_cfgs = [result_cfgs]
    result_getters = [getattr(getters_pkg, f"get_{r['type']}") for r in result_cfgs]

    expected_cfgs_raw = evaluator.get("expected")
    if expected_cfgs_raw is None:
        expected_cfgs: list[Any] = [None] * len(metric_callables)
        expected_getters: list[Any] = [None] * len(metric_callables)
    else:
        expected_cfgs = (
            expected_cfgs_raw if isinstance(expected_cfgs_raw, list) else [expected_cfgs_raw]
        )
        expected_getters = [
            getattr(getters_pkg, f"get_{e['type']}") if e else None for e in expected_cfgs
        ]

    options_raw = evaluator.get("options", [])
    n_metrics = len(metric_callables)
    options_list = list(options_raw) if isinstance(options_raw, list) else [options_raw] * n_metrics
    if not options_list:
        options_list = [{}] * n_metrics

    while len(options_list) < n_metrics:
        options_list.append(options_list[-1] if options_list else {})

    conj = evaluator.get("conj", "and")
    return _evaluate_metrics(
        metric_callables=metric_callables,
        result_getters=result_getters,
        result_cfgs=result_cfgs,
        expected_getters=expected_getters,
        expected_cfgs=expected_cfgs,
        options_list=options_list,
        funcs_list=funcs_list,
        conj=conj,
        env=env,
    )


def _evaluate_metrics(
    *,
    metric_callables: list[Any],
    result_getters: list[Any],
    result_cfgs: list[Any],
    expected_getters: list[Any],
    expected_cfgs: list[Any],
    options_list: list[Any],
    funcs_list: list[Any],
    conj: str,
    env: Any,
) -> tuple[float, str]:

    results: list[float] = []
    for idx, metric_fn in enumerate(metric_callables):
        opts = options_list[idx] or {}
        try:
            result_state = result_getters[idx](env, result_cfgs[idx])
        except Exception as exc:
            logger.warning(
                "[OSWorldBackend] task_eval result getter %r failed: %s",
                funcs_list[idx],
                exc,
            )
            if conj == "and":
                return 0.0, f"result getter {funcs_list[idx]!r} raised: {exc}"
            results.append(0.0)
            continue

        if expected_getters[idx] is not None:
            try:
                expected_state = expected_getters[idx](
                    env,
                    expected_cfgs[idx],
                )
            except Exception as exc:
                logger.warning(
                    "[OSWorldBackend] task_eval expected getter %r failed: %s",
                    funcs_list[idx],
                    exc,
                )
                if conj == "and":
                    return (
                        0.0,
                        f"expected getter {funcs_list[idx]!r} raised: {exc}",
                    )
                results.append(0.0)
                continue
            metric_args: tuple[Any, ...] = (result_state, expected_state)
        else:
            metric_args = (result_state,)

        try:
            score = float(metric_fn(*metric_args, **opts))
        except Exception as exc:
            logger.warning(
                "[OSWorldBackend] task_eval metric %r raised: %s",
                funcs_list[idx],
                exc,
            )
            if conj == "and":
                return 0.0, f"metric {funcs_list[idx]!r} raised: {exc}"
            results.append(0.0)
            continue

        results.append(score)
        if conj == "and" and score == 0.0:
            return 0.0, f"metric {funcs_list[idx]!r} returned 0"
        if conj == "or" and score == 1.0:
            return 1.0, f"metric {funcs_list[idx]!r} returned 1"

    if not results:
        return 0.0, "no metrics produced a result"
    final = sum(results) / len(results) if conj == "and" else max(results)
    reason = "; ".join(f"{f}={s}" for f, s in zip(funcs_list, results, strict=False))
    return final, f"{reason} (conj={conj})"


class _OSWorldEvalEnv:
    def __init__(
        self,
        controller: Any,
        setup_controller: Any,
        cache_dir: str,
        evaluator: dict[str, Any],
        vm_ip: str,
        server_port: int,
        chromium_port: int = 9222,
        vlc_port: int = 8080,
        vnc_port: int = 8006,
    ) -> None:
        self.controller = controller
        self.setup_controller = setup_controller
        self.cache_dir = cache_dir
        self.evaluator = evaluator
        self.vm_ip = vm_ip
        self.server_port = server_port
        self.chromium_port = chromium_port
        self.vlc_port = vlc_port
        self.vnc_port = vnc_port
        self.current_use_proxy = False

        self.action_history: list[Any] = []

    @property
    def vm_platform(self) -> str:
        return self.controller.get_vm_platform()

    @property
    def vm_machine(self) -> str:

        return self.controller.get_vm_machine()
