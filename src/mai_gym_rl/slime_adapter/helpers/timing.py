from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

from mai_gym.client.env_client import EnvClient


@dataclass
class TrajectoryTiming:
    concurrency_wait: float = 0.0
    env_acquire: float = 0.0
    env_reset: float = 0.0
    env_step: float = 0.0
    env_eval: float = 0.0
    session_release: float = 0.0
    total: float = 0.0

    image_preprocess: float = 0.0
    tokenize: float = 0.0
    chat_template: float = 0.0
    vision_extract: float = 0.0
    image_encode: float = 0.0
    hf_processor: float = 0.0
    fast_mm_token_plan: float = 0.0
    text_encode: float = 0.0
    token_native_splice: float = 0.0
    preprocess_unattributed: float = 0.0
    request_prepare: float = 0.0
    sglang_http_post: float = 0.0
    postprocess: float = 0.0

    sglang_server_e2e: float = 0.0
    sglang_prompt_tokens: int = 0
    sglang_completion_tokens: int = 0
    n_model_calls: int = 0
    n_images: int = 0

    n_steps: int = 0
    n_env_retries: int = 0

    @property
    def model_total(self) -> float:

        return (
            self.image_preprocess
            + self.tokenize
            + self.request_prepare
            + self.sglang_http_post
            + self.postprocess
        )

    @property
    def cpu_multimodal_total(self) -> float:

        return self.image_preprocess + self.tokenize

    def to_dict(self) -> dict[str, float]:

        n = max(self.n_steps, 1)
        return {
            "timing/concurrency_wait": self.concurrency_wait,
            "timing/env_session_acquire": self.env_acquire,
            "timing/env_reset": self.env_reset,
            "timing/env_step_action_and_screenshot": self.env_step,
            "timing/env_eval": self.env_eval,
            "timing/session_release": self.session_release,
            "timing/total": self.total,
            "timing/model_image_preprocess": self.image_preprocess,
            "timing/model_tokenize": self.tokenize,
            "timing/model_cpu_multimodal_total": self.cpu_multimodal_total,
            "timing/model_chat_template": self.chat_template,
            "timing/model_vision_extract": self.vision_extract,
            "timing/model_image_encode": self.image_encode,
            "timing/model_hf_multimodal_processor": self.hf_processor,
            "timing/model_fast_mm_token_plan": self.fast_mm_token_plan,
            "timing/model_text_encode": self.text_encode,
            "timing/model_token_native_splice": self.token_native_splice,
            "timing/model_preprocess_unattributed": self.preprocess_unattributed,
            "timing/model_request_prepare": self.request_prepare,
            "timing/model_sglang_http_and_gpu": self.sglang_http_post,
            "timing/model_postprocess": self.postprocess,
            "timing/model_total": self.model_total,
            "timing/model_sglang_server_compute": self.sglang_server_e2e,
            "timing/model_sglang_network_transport": max(
                0.0,
                self.sglang_http_post - self.sglang_server_e2e,
            ),
            "timing/per_step_env_action_and_screenshot": self.env_step / n,
            "timing/per_step_model_image_preprocess": self.image_preprocess / n,
            "timing/per_step_model_tokenize": self.tokenize / n,
            "timing/per_step_model_cpu_multimodal_total": self.cpu_multimodal_total / n,
            "timing/per_step_model_sglang_http_and_gpu": self.sglang_http_post / n,
            "timing/per_step_model_sglang_server_compute": self.sglang_server_e2e / n,
            "timing/per_step_model_sglang_network_transport": max(
                0.0,
                self.sglang_http_post - self.sglang_server_e2e,
            )
            / n,
            "timing/per_step_model_postprocess": self.postprocess / n,
            "timing/per_step_model_total": self.model_total / n,
            "timing/sglang_prompt_tokens_per_step": self.sglang_prompt_tokens / n,
            "timing/sglang_completion_tokens_per_step": self.sglang_completion_tokens / n,
            "timing/sglang_decode_tokens_per_sec": (
                self.sglang_completion_tokens / self.sglang_server_e2e
                if self.sglang_server_e2e > 0
                else 0.0
            ),
            "timing/model_calls": float(self.n_model_calls),
            "timing/images": float(self.n_images),
            "timing/n_steps": float(self.n_steps),
            "timing/n_env_retries": float(self.n_env_retries),
            "timing/overhead": max(
                0.0,
                self.total
                - (
                    self.concurrency_wait
                    + self.env_acquire
                    + self.env_reset
                    + self.env_step
                    + self.env_eval
                    + self.session_release
                    + self.model_total
                ),
            ),
        }


class TimedEnvClient:
    def __init__(self, client: EnvClient, timing: TrajectoryTiming) -> None:
        self._client = client
        self._timing = timing

    def reset(self, *args: Any, **kwargs: Any) -> Any:
        t0 = time.monotonic()
        try:
            return self._client.reset(*args, **kwargs)
        finally:
            self._timing.env_reset += time.monotonic() - t0

    def step(self, *args: Any, **kwargs: Any) -> Any:
        t0 = time.monotonic()
        try:
            return self._client.step(*args, **kwargs)
        finally:
            elapsed = time.monotonic() - t0
            self._timing.env_step += elapsed
            self._timing.n_steps += 1

    def eval(self, *args: Any, **kwargs: Any) -> Any:
        t0 = time.monotonic()
        try:
            return self._client.eval(*args, **kwargs)
        finally:
            self._timing.env_eval += time.monotonic() - t0

    async def areset(self, *args: Any, **kwargs: Any) -> Any:
        t0 = time.monotonic()
        try:
            return await self._client.areset(*args, **kwargs)
        finally:
            self._timing.env_reset += time.monotonic() - t0

    async def astep(self, *args: Any, **kwargs: Any) -> Any:
        t0 = time.monotonic()
        try:
            return await self._client.astep(*args, **kwargs)
        finally:
            elapsed = time.monotonic() - t0
            self._timing.env_step += elapsed
            self._timing.n_steps += 1

    async def aeval(self, *args: Any, **kwargs: Any) -> Any:
        t0 = time.monotonic()
        try:
            return await self._client.aeval(*args, **kwargs)
        finally:
            self._timing.env_eval += time.monotonic() - t0

    def acquire(self, *args: Any, **kwargs: Any) -> Any:
        t0 = time.monotonic()
        try:
            return self._client.acquire(*args, **kwargs)
        finally:
            self._timing.env_acquire += time.monotonic() - t0

    def release(self, *args: Any, **kwargs: Any) -> Any:
        t0 = time.monotonic()
        try:
            return self._client.release(*args, **kwargs)
        finally:
            self._timing.session_release += time.monotonic() - t0

    def discard(self, *args: Any, **kwargs: Any) -> Any:
        t0 = time.monotonic()
        try:
            return self._client.discard(*args, **kwargs)
        finally:
            self._timing.session_release += time.monotonic() - t0

    def log_messages(self, *args: Any, **kwargs: Any) -> Any:
        return self._client.log_messages(*args, **kwargs)

    def list_sessions(self, *args: Any, **kwargs: Any) -> Any:
        return self._client.list_sessions(*args, **kwargs)

    def ensure_pool(self, *args: Any, **kwargs: Any) -> Any:
        return self._client.ensure_pool(*args, **kwargs)

    def close(self) -> None:
        self._client.close()

    def asession(self, *args: Any, **kwargs: Any):
        return self._client.asession(*args, **kwargs)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._client, name)
