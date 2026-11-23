from __future__ import annotations

import asyncio
import os
from collections.abc import Callable, Coroutine
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING, Any, TypeVar

if TYPE_CHECKING:
    from openai import OpenAI

T = TypeVar("T")


def run_async_judge(factory: Callable[[], Coroutine[Any, Any, T]]) -> T:  # noqa: UP047
    with ThreadPoolExecutor(max_workers=1, thread_name_prefix="native-judge") as executor:
        return executor.submit(lambda: asyncio.run(factory())).result()


def judge_client(settings: dict[str, Any]) -> OpenAI:
    from openai import OpenAI

    judge = settings["judge"]
    if not judge.get("label"):
        raise ValueError("Native judge configuration requires an explicit label")
    return OpenAI(
        api_key=os.environ[judge["api_key_env"]],
        base_url=judge["base_url"],
        timeout=judge.get("timeout", 120),
        max_retries=2,
    )


def judge_identity(settings: dict[str, Any]) -> dict[str, Any]:
    judge = settings.get("judge", {})
    return {
        key: judge[key]
        for key in ("label", "model", "base_url", "api_key_env", "score_threshold")
        if key in judge
    }
