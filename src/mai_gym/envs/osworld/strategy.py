import asyncio
import base64
import json
import logging
from pathlib import Path
from typing import Any

from mai_gym.envs.shared.backend import NativeBackend
from mai_gym.envs.shared.strategy_native import NativeStrategy
from mai_gym.infra.backend import Backend
from mai_gym.models.session import SessionState, SessionTask

logger = logging.getLogger(__name__)


def _extract_vscode_install_extension(step: Any) -> str | None:

    if not isinstance(step, dict) or step.get("type") != "command":
        return None
    params = step.get("parameters") or {}
    cmd = params.get("command")
    if not isinstance(cmd, list) or len(cmd) != 3:
        return None
    if cmd[0] != "code" or cmd[1] != "--install-extension":
        return None
    ext_id = cmd[2]
    if not isinstance(ext_id, str) or "." not in ext_id or "/" in ext_id:
        return None
    return ext_id


class OSWorldStrategy:
    def __init__(
        self,
        examples_dir: str | None = None,
        *,
        vscode_ext_mirror: dict | None = None,
    ) -> None:
        self._examples_dir = Path(examples_dir) if examples_dir else None

        base_url = (vscode_ext_mirror or {}).get("base_url") or None
        extensions = list((vscode_ext_mirror or {}).get("extensions") or [])
        self._vscode_ext_mirror_base_url: str | None = (
            base_url.rstrip("/") + "/" if base_url else None
        )
        self._vscode_ext_mirror_ids: set[str] = set(extensions)

    def _load_task_config(self, task_id: str) -> dict | None:
        if not self._examples_dir or not task_id:
            return None

        candidate = self._examples_dir / f"{task_id}.json"
        if not candidate.is_file():
            matches = list(self._examples_dir.glob(f"*/{task_id}.json")) or list(
                self._examples_dir.glob(f"**/{task_id}.json")
            )
            if matches:
                candidate = matches[0]
        if not candidate.is_file():
            logger.warning(
                "[OSWorldStrategy] task json not found for id=%s under %s",
                task_id,
                self._examples_dir,
            )
            return None
        try:
            cfg = json.loads(candidate.read_text())
        except Exception as exc:
            logger.warning(
                "[OSWorldStrategy] failed to parse task json %s: %s",
                candidate,
                exc,
            )
            return None
        self._rewrite_vscode_ext_installs(cfg)

        cfg["__examples_dir"] = str(self._examples_dir)
        return cfg

    def _rewrite_vscode_ext_installs(self, cfg: dict) -> None:

        if not self._vscode_ext_mirror_base_url or not self._vscode_ext_mirror_ids:
            return
        steps = cfg.get("config")
        if not isinstance(steps, list) or not steps:
            return
        new_steps: list[dict] = []
        rewrote: list[str] = []
        for step in steps:
            ext_id = _extract_vscode_install_extension(step)
            if ext_id is None or ext_id not in self._vscode_ext_mirror_ids:
                new_steps.append(step)
                continue
            local_path = f"/tmp/{ext_id}.vsix"
            new_steps.append(
                {
                    "type": "download",
                    "parameters": {
                        "files": [
                            {
                                "url": f"{self._vscode_ext_mirror_base_url}{ext_id}.vsix",
                                "path": local_path,
                            }
                        ],
                    },
                }
            )
            new_steps.append(
                {
                    "type": "command",
                    "parameters": {
                        "command": ["code", "--install-extension", local_path],
                    },
                }
            )
            rewrote.append(ext_id)
        if rewrote:
            cfg["config"] = new_steps
            logger.info(
                "[OSWorldStrategy] rewrote vscode ext install(s) for task %s: "
                "%s → download+local-vsix (mirror=%s)",
                cfg.get("id"),
                rewrote,
                self._vscode_ext_mirror_base_url,
            )

    _EPHEMERAL_TIMEOUT = 900.0

    async def _run_script_in_vm(
        self,
        client: Backend,
        src: str,
        dest: str,
    ) -> dict[str, Any]:

        b64 = base64.b64encode(src.encode("utf-8")).decode("ascii")
        cmd = ["bash", "-lc", f"echo {b64} | base64 -d > {dest} && python3 {dest}"]
        return await client.run_command(cmd, timeout=self._EPHEMERAL_TIMEOUT)

    async def _ephemeral_reset(
        self,
        client: Backend,
        task: SessionTask,
        phase: str | None,
    ) -> str | None:

        is_golden = phase == "golden"
        src = task.golden_patch if is_golden else task.initial_setup
        label = "golden_patch" if is_golden else "initial_setup"
        if not src:
            raise RuntimeError(f"ephemeral OSWorld reset: no {label} script")
        dest = f"/tmp/_eb_{label}.py"
        res = await self._run_script_in_vm(client, src, dest)
        rc = res.get("returncode", 0)
        if rc not in (0, None):
            tail = (res.get("error") or res.get("output") or "")[-400:]
            raise RuntimeError(f"ephemeral OSWorld {label} exited rc={rc}: {tail!r}")
        return None

    async def _ephemeral_evaluate(
        self,
        client: Backend,
        task: SessionTask,
    ) -> tuple[float | None, str | None]:

        from mai_gym.envs.osworld._eval import _parse_reward_output

        if not task.reward:
            return None, "ephemeral OSWorld eval: no reward script"
        res = await self._run_script_in_vm(client, task.reward, "/tmp/_eb_reward.py")
        out = (res.get("output") or "") + (res.get("error") or "")
        return _parse_reward_output(out)

    _WORLD_EXTRACT_SH = (
        "set -euo pipefail\n"
        'test "$(sha256sum {tmp} | cut -d\' \' -f1)" = "{sha}"\n'
        "mkdir -p {dst}\n"
        "tar -xzf {tmp} -C {dst}\n"
        "if [ -f {dst}/.synthetic_workspace/SHA256SUMS ]; then "
        "( cd {dst} && sha256sum -c .synthetic_workspace/SHA256SUMS ) && "
        "rm -rf {dst}/.synthetic_workspace; fi\n"
        "chown -R user:user {dst} 2>/dev/null || true\n"
    )
    _WORLD_DEFAULT_EXTRACT_TO = "/home/user"

    async def on_attach_task(
        self,
        client: Backend,
        state: SessionState,
        *,
        task: SessionTask,
        package: bytes | None = None,
        world_path: str | None = None,
    ) -> None:

        if isinstance(client, NativeBackend):
            await NativeStrategy().on_attach_task(
                client, state, task=task, package=package, world_path=world_path
            )
            return
        if not task.world_oss_uri:
            return

        if world_path is None:
            raise RuntimeError(
                f"OSWorld world attach: controller passed no local archive for "
                f"{task.world_oss_uri} (OSS download/cache failed upstream)"
            )
        upload = getattr(client, "upload_file", None)
        if upload is None:
            raise RuntimeError("OSWorld world attach: backend lacks upload_file (/setup/upload)")
        dst = task.world_extract_to or self._WORLD_DEFAULT_EXTRACT_TO
        tmp = "/tmp/_eb_world.tar.gz"
        await upload(world_path, tmp)
        sh = self._WORLD_EXTRACT_SH.format(
            sha=task.world_sha256 or "",
            tmp=tmp,
            dst=dst,
        )
        res = await client.run_command(
            ["bash", "-lc", sh],
            timeout=self._EPHEMERAL_TIMEOUT,
        )
        rc = res.get("returncode", 0)
        if rc not in (0, None):
            tail = (res.get("error") or res.get("output") or "")[-400:]
            raise RuntimeError(
                f"OSWorld world extract exited rc={rc} (uri={task.world_oss_uri}): {tail!r}"
            )
        logger.info(
            "[OSWorldStrategy] provisioned world %s -> %s on backend %s",
            task.world_oss_uri,
            dst,
            getattr(state.backend, "id", "?"),
        )

    async def on_detach_task(
        self,
        client: Backend,
        state: SessionState,
        *,
        task: SessionTask,
    ) -> None:
        return

    async def on_reset(
        self,
        client: Backend,
        state: SessionState,
        **kwargs: Any,
    ) -> str | None:

        if isinstance(client, NativeBackend):
            return await NativeStrategy().on_reset(client, state, **kwargs)
        if state.session_task is not None:
            return await self._ephemeral_reset(
                client,
                state.session_task,
                kwargs.get("task_phase"),
            )

        seed_task_id = kwargs.get("seed_task_id")
        task_id = kwargs.get("task_id")
        goal = kwargs.get("goal")

        state.current_task_config = None
        config_lookup_id = seed_task_id or task_id
        if not config_lookup_id:
            return None

        task_cfg = self._load_task_config(config_lookup_id)
        if task_cfg is None:
            return None

        state.current_task_config = task_cfg
        if hasattr(client, "reset_environment"):
            await client.reset_environment()

        from mai_gym.envs.osworld.cua_bundle import prepend_libreoffice_reset

        init_cfg = {
            **task_cfg,
            "config": prepend_libreoffice_reset(task_cfg.get("config") or []),
        }

        await asyncio.wait_for(client.task_init(init_cfg), timeout=600)

        task_instruction = task_cfg.get("instruction")
        if task_instruction and goal is None:
            return task_instruction
        return None

    async def on_evaluate(
        self,
        client: Backend,
        state: SessionState,
        **kwargs: Any,
    ) -> tuple[float | None, str | None]:

        if isinstance(client, NativeBackend):
            return await NativeStrategy().on_evaluate(client, state, **kwargs)
        if state.session_task is not None:
            return await self._ephemeral_evaluate(client, state.session_task)
        if state.current_task_config is None:
            return None, None
        result = await client.task_eval(
            state.current_task_config,
            list(state.action_history),
        )
        if result is not None:
            return result
        return None, None

    async def on_tear_down(
        self,
        client: Backend,
        state: SessionState,
        **kwargs: Any,
    ) -> None:
        if isinstance(client, NativeBackend):
            await NativeStrategy().on_tear_down(client, state, **kwargs)
        return


def create_strategy(*, requested_type: str, **settings: Any) -> OSWorldStrategy:
    from mai_gym.models.environment import identity

    if identity(requested_type).profile == "cua":
        from mai_gym.envs.osworld.cua_strategy import CUAOSWorldStrategy

        return CUAOSWorldStrategy(cua_osworld_tasks_dir=settings.get("cua_osworld_tasks_dir"))
    return OSWorldStrategy(
        examples_dir=settings.get("osworld_examples_dir"),
        vscode_ext_mirror=settings.get("vscode_ext_mirror"),
    )
