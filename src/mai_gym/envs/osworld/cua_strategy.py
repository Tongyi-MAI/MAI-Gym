import asyncio
import base64
import json
import logging
from pathlib import Path
from typing import Any

from mai_gym.envs.osworld.strategy import OSWorldStrategy
from mai_gym.infra.backend import Backend
from mai_gym.models.session import SessionState

logger = logging.getLogger(__name__)


class CUAOSWorldStrategy(OSWorldStrategy):
    def __init__(self, cua_osworld_tasks_dir: str | None = None) -> None:
        super().__init__()
        self._tasks_dir = Path(cua_osworld_tasks_dir) if cua_osworld_tasks_dir else None

    def _load_task_config(self, task_id: str | None) -> dict[str, Any] | None:
        if not self._tasks_dir or not task_id:
            return None
        meta_path = self._tasks_dir / task_id / "task.json"
        if not meta_path.is_file():
            logger.warning("[CUAOSWorldStrategy] missing task.json: %s", meta_path)
            return None
        try:
            cfg = json.loads(meta_path.read_text())
        except Exception as exc:
            logger.warning(
                "[CUAOSWorldStrategy] failed to parse %s: %s",
                meta_path,
                exc,
            )
            return None

        cfg["_dir"] = str(meta_path.parent)
        return cfg

    async def _stage_files(
        self,
        client: Backend,
        stage: list[tuple[str, str]],
    ) -> None:

        run_command = getattr(client, "run_command", None)
        if run_command is None:
            logger.warning(
                "[CUAOSWorldStrategy] backend lacks run_command — cannot stage "
                "bundle files; setup will likely fail",
            )
            return
        for local_path, vm_path in stage:
            p = Path(local_path)
            if not p.is_file():
                logger.warning("[CUAOSWorldStrategy] stage file missing: %s", p)
                continue
            encoded = base64.b64encode(p.read_bytes()).decode("ascii")
            inline = (
                "import base64, os\n"
                f"os.makedirs(os.path.dirname({vm_path!r}) or '.', exist_ok=True)\n"
                f"open({vm_path!r}, 'wb').write(base64.b64decode({encoded!r}))\n"
            )
            await run_command(["python3", "-c", inline])

    async def on_reset(
        self,
        client: Backend,
        state: SessionState,
        **kwargs: Any,
    ) -> str | None:

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
        lookup_id = seed_task_id or task_id
        if not lookup_id:
            return None
        task = self._load_task_config(lookup_id)
        if task is None:
            return None
        state.current_task_config = task

        if hasattr(client, "reset_environment"):
            await client.reset_environment()

        from mai_gym.envs.osworld.cua_bundle import (
            extract_stage_files,
            prepend_libreoffice_reset,
        )

        steps = task.get("config") or []
        stage, stripped = extract_stage_files(steps, task["_dir"])
        await self._stage_files(client, stage)

        stripped = prepend_libreoffice_reset(stripped)
        staged_cfg = {**task, "config": stripped}
        await asyncio.wait_for(client.task_init(staged_cfg), timeout=600)

        instruction = task.get("instruction")
        if instruction and goal is None:
            return instruction
        return None
