from __future__ import annotations

import asyncio
import base64
import logging
import os
import random
import threading
import time
import uuid
from io import BytesIO
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

import httpx

from mai_gym.client.env_client import APIError, EnvClient, EnvTransportError, PollTimeoutError
from mai_gym.runner.base import TaskConfig, TaskRunner, Trajectory

if TYPE_CHECKING:
    from mai_gym.agents.action_space import ActionSpace

logger = logging.getLogger(__name__)


_PRE_EVAL_SETTLE_SECS = float(os.environ.get("MAI_GYM_PRE_EVAL_SETTLE_SECS", "30"))
_SCREENSHOT_FETCH_MAX_ATTEMPTS = 10
_SCREENSHOT_FETCH_TIMEOUT_S = 30.0
_SCREENSHOT_FETCH_BACKOFF_MIN_S = 0.25
_SCREENSHOT_FETCH_BACKOFF_MAX_S = 10.0


class ScreenshotTransportError(RuntimeError):
    pass


def _is_retryable_step_error(exc: BaseException) -> bool:
    if isinstance(exc, (PollTimeoutError, EnvTransportError, httpx.TransportError)):
        return True
    return isinstance(exc, APIError) and exc.status_code >= 500


class _LeaseKeepalive:
    def __init__(
        self, client: EnvClient, session_id: str, interval_s: float, max_s: float = 3600.0
    ) -> None:
        self._client = client
        self._session_id = session_id
        self._interval_s = interval_s
        self._max_s = max_s
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, name="lease-keepalive", daemon=True)

    def _loop(self) -> None:
        deadline = time.monotonic() + self._max_s
        while not self._stop.wait(self._interval_s):
            if time.monotonic() > deadline:
                break
            self._client.keepalive_session(self._session_id)

    def __enter__(self) -> _LeaseKeepalive:
        self._thread.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self._stop.set()
        self._thread.join(timeout=0.2)


class Policy(Protocol):
    NATIVE_ACTION_SPACE: ActionSpace

    async def select_action(self, observation: dict[str, Any]) -> dict[str, Any]: ...


@runtime_checkable
class LifecyclePolicy(Policy, Protocol):
    def initialize(self, instruction: str) -> None: ...
    def done(self) -> None: ...


def _log_step(
    traj_logger: Any,
    task_name: str,
    task_goal: str,
    step: int,
    prediction: str | None,
    action: dict[str, Any],
    screenshot_b64: str,
) -> None:

    try:
        from PIL import Image

        from mai_gym.models.observation import Observation

        img = Image.open(BytesIO(base64.b64decode(screenshot_b64))) if screenshot_b64 else None
        obs = Observation(screenshot=img)
        traj_logger.log_traj(
            task_name=task_name,
            task_goal=task_goal,
            step=step,
            prediction=prediction or "",
            action=action,
            obs=obs,
        )
    except Exception as exc:
        logger.warning("Failed to log trajectory step: %s", exc)


class RolloutAgent(TaskRunner):
    def __init__(
        self,
        env_client: EnvClient,
        policy: Policy,
        *,
        env_type: str | None = None,
        oss_creds: Any = None,
        model_name: str | None = None,
        step_retries: int = 0,
        lease_keepalive_s: float = 0.0,
        preserve_action_sequences: bool = False,
    ) -> None:

        from mai_gym.agents.action_space import env_action_space
        from mai_gym.agents.action_translation import TranslationCounters
        from mai_gym.agents.action_translation.parser import PointerState

        super().__init__(env_client)
        self.policy = policy
        self.env_type = env_type

        self._mobile_pixel_coords = env_type in {
            "real_device",
            "mobileworld",
            "mobileworld_k8s",
            "androidworld",
        }

        self._model_name = model_name

        self._oss_creds = oss_creds
        self._step_retries = max(0, int(step_retries))
        self._lease_keepalive_s = max(0.0, float(lease_keepalive_s))
        self._preserve_action_sequences = preserve_action_sequences
        self.translation_counters = TranslationCounters()
        self._translation_pointer = PointerState()

        self._dst_space: ActionSpace | None = None
        self._src_space: ActionSpace | None = getattr(
            policy,
            "NATIVE_ACTION_SPACE",
            None,
        )

        if hasattr(policy, "initialize") and self._src_space is None:
            raise TypeError(
                f"{type(policy).__name__} defines initialize() (a lifecycle "
                "policy) but is missing the required NATIVE_ACTION_SPACE class "
                "attribute. Without it, isinstance(policy, LifecyclePolicy) is "
                "False and initialize(instruction) is silently skipped — the "
                "agent never receives its task goal. Set "
                "`NATIVE_ACTION_SPACE = ActionSpace.<...>` on the policy class."
            )
        if env_type is not None:
            try:
                self._dst_space = env_action_space(env_type)
            except KeyError as exc:
                logger.warning(
                    "RolloutAgent: %s; action translation disabled",
                    exc.args[0] if exc.args else exc,
                )
        if (
            self._src_space is not None
            and self._dst_space is not None
            and self._src_space != self._dst_space
        ):
            logger.info(
                "RolloutAgent: translating actions %s -> %s",
                self._src_space,
                self._dst_space,
            )

    def _screen_size(self) -> tuple[int, int]:

        for attr in ("screen_size", "_screen_size"):
            value = getattr(self.policy, attr, None)
            if value is not None and len(value) >= 2:
                return int(value[0]), int(value[1])
        logger.warning(
            "RolloutAgent: policy %s has no screen_size; using (1280, 720)",
            type(self.policy).__name__,
        )
        return 1280, 720

    def _maybe_translate(self, action: dict[str, Any]) -> dict[str, Any]:

        from mai_gym.agents.action_space import ActionSpace

        if self._src_space is None or self._dst_space is None or self._src_space == self._dst_space:
            return action
        from mai_gym.agents.action_translation import translate

        return translate(
            action,
            self._src_space,
            self._dst_space,
            screen_size=self._screen_size(),
            counters=self.translation_counters,
            mobile_pixel_coords=self._mobile_pixel_coords,
            mobile_keyboard=self.env_type == "mobilegym",
            preserve_sequence=self._preserve_action_sequences,
            pointer=(
                self._translation_pointer
                if self._preserve_action_sequences and self._dst_space is ActionSpace.MOBILE_USE
                else None
            ),
        )

    async def _rehydrate_image(self, obs: dict[str, Any]) -> None:

        if obs.get("image"):
            return
        uri = obs.get("image_uri")
        if not uri:
            raise ScreenshotTransportError(
                "manager returned an observation without inline image bytes or image_uri"
            )
        if self._oss_creds is None:
            raise ScreenshotTransportError(
                f"manager returned {uri!r}, but rollout OSS credentials are not configured"
            )

        from mai_gym.io.oss_io import fetch_image_b64, get_async_fs, is_oss_path

        if not is_oss_path(uri):
            raise ScreenshotTransportError(f"unsupported screenshot image_uri: {uri!r}")

        last_exc: Exception | None = None
        for attempt in range(1, _SCREENSHOT_FETCH_MAX_ATTEMPTS + 1):
            try:
                image = await fetch_image_b64(
                    uri,
                    fs=get_async_fs(self._oss_creds),
                    timeout_s=_SCREENSHOT_FETCH_TIMEOUT_S,
                )
                if not image:
                    raise ValueError("OSS screenshot object was empty")
                obs["image"] = image
                return
            except Exception as exc:
                last_exc = exc
                if attempt >= _SCREENSHOT_FETCH_MAX_ATTEMPTS:
                    break
                delay = random.uniform(
                    _SCREENSHOT_FETCH_BACKOFF_MIN_S, _SCREENSHOT_FETCH_BACKOFF_MAX_S
                )
                logger.warning(
                    "Screenshot OSS GET failed (attempt=%d/%d uri=%s); retrying in %.2fs: %s",
                    attempt,
                    _SCREENSHOT_FETCH_MAX_ATTEMPTS,
                    uri,
                    delay,
                    exc,
                )
                await asyncio.sleep(delay)

        raise ScreenshotTransportError(
            f"failed to fetch screenshot {uri!r} after "
            f"{_SCREENSHOT_FETCH_MAX_ATTEMPTS} attempts: {last_exc}"
        ) from last_exc

    async def _step_with_retry(
        self, session_id: str, wire_action: dict[str, Any], step_no: int
    ) -> Any:
        if self._step_retries <= 0:
            return await self.env_client.astep(session_id, wire_action)
        action_id = uuid.uuid4().hex
        for attempt in range(self._step_retries + 1):
            try:
                return await self.env_client.astep(session_id, wire_action, action_id=action_id)
            except Exception as exc:
                if attempt >= self._step_retries or not _is_retryable_step_error(exc):
                    raise
                delay = min(2.0**attempt, 8.0)
                logger.warning(
                    "Step %d transport failure; retry %d/%d in %.0fs (action_id=%s): %s",
                    step_no,
                    attempt + 1,
                    self._step_retries,
                    delay,
                    action_id,
                    exc,
                )
                await asyncio.sleep(delay)
        raise AssertionError("unreachable")

    async def run(
        self,
        session_id: str,
        task: TaskConfig,
        *,
        tear_down: bool = True,
        traj_logger: Any = None,
    ) -> Trajectory:

        reset_resp = await self.env_client.areset(
            session_id,
            task.category,
            task_id=task.task_id,
            goal=task.goal,
            target_apps=task.target_apps,
            seed_task_id=task.seed_task_id,
            attempt_id=task.attempt_id,
            train_id=task.train_id,
            rollout_id=task.rollout_id,
            evaluation=task.is_eval,
            model_name=self._model_name,
        )

        obs = reset_resp.obs_messages[0] if reset_resp.obs_messages else {}
        await self._rehydrate_image(obs)

        instruction = obs.get("instruction") or task.goal
        if not instruction:
            raise ValueError(
                "[RolloutAgent] empty instruction after reset — refusing to roll "
                "out a goal-less episode. "
                f"session={session_id} task_id={task.task_id!r} "
                f"category={task.category!r} goal={task.goal!r} "
                f"obs_keys={list(obs.keys())} "
                f"obs_instruction={obs.get('instruction')!r}"
            )
        task_name = task.task_id or task.category
        self._translation_pointer.position = None

        task_name_for_log = task_name.replace("/", "_") if task_name else task_name

        if isinstance(self.policy, LifecyclePolicy):
            self.policy.initialize(instruction)

        steps: list[dict[str, Any]] = []
        is_done = False

        for step_num in range(task.max_steps):
            if is_done:
                break

            pre_action_b64 = obs.get("image", "")
            if self._lease_keepalive_s > 0:
                with _LeaseKeepalive(self.env_client, session_id, self._lease_keepalive_s):
                    action = await self.policy.select_action(obs)
            else:
                action = await self.policy.select_action(obs)

            prediction = getattr(self.policy, "last_prediction", None)

            if prediction:
                action["prediction"] = prediction

            step_log_payload = getattr(self.policy, "last_step_log", None)
            if step_log_payload:
                reasoning = step_log_payload.get("raw_reasoning_content")
                if reasoning:
                    action["reasoning_content"] = reasoning
                try:
                    await self.env_client.alog_messages(
                        session_id,
                        step_num + 1,
                        step_log_payload,
                    )
                except Exception as exc:
                    logger.warning("log_messages failed at step %d: %s", step_num + 1, exc)

            try:
                wire_action = self._maybe_translate(action)
            except ValueError as exc:
                if not task.require_score:
                    raise
                logger.warning("Ending episode after unsupported model action: %s", exc)
                wire_action = {
                    "action_type": "terminate",
                    "status": "failure",
                    "source": "harness",
                    "reason": "unsupported_model_action",
                    "error": str(exc),
                    "requested_action": action,
                }
                action = wire_action
            step_resp = await self._step_with_retry(session_id, wire_action, step_num + 1)

            if traj_logger:
                _log_step(
                    traj_logger,
                    task_name_for_log,
                    instruction,
                    step_num + 1,
                    prediction,
                    action,
                    pre_action_b64,
                )

            steps.append(
                {
                    "step": step_num + 1,
                    "action": action,
                    "prediction": prediction,
                }
            )
            obs = step_resp.obs_messages[0] if step_resp.obs_messages else {}
            await self._rehydrate_image(obs)
            is_done = step_resp.is_done

        if _PRE_EVAL_SETTLE_SECS > 0:
            logger.info(
                "Settling %.0fs before evaluate (task=%s)",
                _PRE_EVAL_SETTLE_SECS,
                task.task_id,
            )
            await asyncio.sleep(_PRE_EVAL_SETTLE_SECS)

        evaluation_options: dict[str, Any] = {"require_score": True} if task.require_score else {}
        eval_resp = await self.env_client.aeval(
            session_id,
            task.category,
            task_id=task.task_id,
            use_judge=task.use_judge,
            evaluation_mode=task.evaluation_mode,
            **evaluation_options,
        )

        if traj_logger:
            try:
                traj_logger.log_score(eval_resp.reward)
            except Exception as exc:
                logger.warning("Failed to log score: %s", exc)

        if tear_down:
            await self.env_client.atear_down(
                session_id,
                task.category,
                task_id=task.task_id,
            )

        if isinstance(self.policy, LifecyclePolicy):
            self.policy.done()

        return Trajectory(
            task_id=task.task_id,
            steps=steps,
            reward=eval_resp.reward,
            success=getattr(eval_resp, "success", None),
        )
