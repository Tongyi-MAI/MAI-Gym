from __future__ import annotations

import asyncio
import logging
import math
import time
import uuid

from mai_gym.agents.generation import RecordingChatModel
from mai_gym.client.sdk_client import SDKClient, SDKInfrastructureError
from mai_gym.runner.base import TaskConfig, TaskRunner, Trajectory

logger = logging.getLogger(__name__)


def sdk_conversation_type():
    try:
        from mai_ui_mobileworld.conversation import SDKConversation
    except ImportError as exc:
        raise RuntimeError(
            "SDK rollout requires mai-ui-mobileworld>=0.2.1 on rollout workers"
        ) from exc
    return SDKConversation


class SDKRolloutAgent(TaskRunner):
    def __init__(
        self,
        env_client,
        model: RecordingChatModel,
        sampling_params: dict,
        *,
        coordinate_space="normalized_1000",
        device=None,
        image_retention=3,
        timing=None,
    ):
        super().__init__(env_client)
        self.conversation_type = sdk_conversation_type()
        self.model, self.sampling_params = model, dict(sampling_params)
        self.coordinate_space, self.device = coordinate_space, device
        self.image_retention, self.timing = image_retention, timing
        self.runtime = {}

    async def _operation(self, transport, kind, **kwargs):
        started = time.monotonic()
        try:
            return await transport.operation(kind, **kwargs)
        finally:
            if self.timing is not None:
                field = {
                    "reset": "env_reset",
                    "open": "env_reset",
                    "execute": "env_step",
                    "finish": "env_eval",
                    "cancel": "env_eval",
                }[kind]
                setattr(
                    self.timing, field, getattr(self.timing, field) + time.monotonic() - started
                )

    async def run(self, session_id: str, task: TaskConfig) -> Trajectory:
        if task.use_judge or task.evaluation_mode != "built_in":
            raise ValueError("SDK training v1 requires the MobileWorld built-in evaluator")
        if task.max_steps < 1:
            raise ValueError("SDK max_steps must be positive")
        transport = SDKClient(self.env_client, session_id)
        finished = False
        self.model.reset()
        try:
            reset = await self._operation(
                transport,
                "reset",
                reset={
                    "category": task.category,
                    "task_id": task.task_id,
                    "goal": task.goal,
                    "target_apps": task.target_apps,
                    "seed_task_id": task.seed_task_id,
                    "attempt_id": task.attempt_id,
                    "train_id": task.train_id,
                    "rollout_id": task.rollout_id,
                    "evaluation": task.is_eval,
                },
            )
            observations = reset.get("obs_messages") or []
            instruction = (
                observations[0].get("instruction") if observations else None
            ) or task.goal
            if not instruction:
                raise SDKInfrastructureError("MobileWorld reset returned no task instruction")
            observation = await self._operation(
                transport, "open", device=self.device, coordinate_space=self.coordinate_space
            )
            self.runtime = observation["runtime"]
            conversation = self.conversation_type(
                instruction, self.coordinate_space, self.image_retention
            )
            steps = []
            for _ in range(task.max_steps):
                conversation.observe(observation)
                result = await self.model.generate(
                    conversation.outgoing(), self.sampling_params, timing=self.timing
                )
                cell = conversation.accept(result.content)
                cell_id = uuid.uuid4().hex
                if cell.format_error:
                    observation = {
                        "result": {
                            "cell_id": cell_id,
                            "status": "code_error",
                            "outputs": [],
                            "error": {"type": "FormatError", "message": cell.format_error},
                        }
                    }
                else:
                    observation = await self._operation(
                        transport, "execute", cell_id=cell_id, code=cell.code
                    )
                outcome = observation["result"]
                steps.append(
                    {
                        "cell_id": cell_id,
                        "status": outcome["status"],
                        "terminal": outcome.get("terminal"),
                        "format_error": cell.format_error,
                    }
                )
                if outcome["status"] not in {"ok", "code_error"}:
                    raise SDKInfrastructureError("SDK cell did not produce a valid outcome")
                if outcome.get("terminal") in {"finish", "answer"}:
                    break

            result = await self._operation(transport, "finish")
            reward = float(result["evaluation"]["reward"])
            if not math.isfinite(reward):
                raise SDKInfrastructureError("Evaluator returned a non-finite reward")
            finished = True
            return Trajectory(task.task_id, steps, reward)
        finally:
            if not finished:
                try:
                    await self._operation(transport, "cancel", timeout=60)
                except (Exception, asyncio.CancelledError) as exc:
                    logger.warning("SDK cleanup requires backend retirement: %s", exc)
