import asyncio
import functools
import json
from collections.abc import AsyncIterator, Callable
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, cast

import click
from fastapi import FastAPI, HTTPException

from mai_gym.envs.registry import get_environment, load_component
from mai_gym.envs.shared.runtime import NativeRuntime


def create_worker_app(
    environment: str,
    settings: dict[str, Any],
    *,
    runtime_factory: Callable[[], NativeRuntime] | None = None,
) -> FastAPI:
    state: dict[str, Any] = {
        "leased": False,
        "task_id": None,
        "goal": None,
        "display_id": 0,
        "faulted": False,
    }
    executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="native-runtime")
    completed_steps: dict[str, dict[str, Any]] = {}
    cached_verdict: dict[str, Any] | None = None

    async def call(function: Any, *args: Any) -> Any:
        return await asyncio.get_running_loop().run_in_executor(
            executor, functools.partial(function, *args)
        )

    def build() -> NativeRuntime:
        if runtime_factory is not None:
            return runtime_factory()
        env = get_environment(environment)
        profile = load_component(env.preparation)(settings)
        tasks = load_component(env.task_loader)(profile)
        factory = load_component(f"mai_gym.envs.{env.name}.runtime:create_runtime")
        created = factory(profile, settings, tasks)
        if not isinstance(created, NativeRuntime):
            raise TypeError("Native runtime factory returned an incompatible runtime")
        return created

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        state["runtime"] = await call(build)
        try:
            yield
        finally:
            await call(state["runtime"].close)
            executor.shutdown(wait=True)

    app = FastAPI(title=f"MAI-Gym native {environment} worker", lifespan=lifespan)
    lock = asyncio.Lock()

    def runtime() -> NativeRuntime:
        if "runtime" not in state:
            raise HTTPException(503, "Native runtime is not ready")
        return cast(NativeRuntime, state["runtime"])

    def require_task(task_name: str | None = None) -> None:
        if state["faulted"]:
            raise HTTPException(503, "Native reset failed; replace this worker before retrying")
        if state["task_id"] is None or (task_name is not None and task_name != state["task_id"]):
            raise HTTPException(409, "No matching initialized native episode")

    def require_display(display_id: int | None) -> None:
        if not state["leased"] or display_id != state["display_id"]:
            raise HTTPException(409, "Stale or missing native episode lease")

    @app.get("/health")
    async def health() -> dict[str, Any]:
        return {"ok": "runtime" in state and not state["faulted"], "env_type": environment}

    @app.get("/devices")
    async def devices() -> dict[str, Any]:
        return {"devices": ["native"]}

    @app.get("/state")
    async def diagnostics() -> dict[str, Any]:
        async with lock:
            return {
                **await call(runtime().diagnostics),
                "faulted": state["faulted"],
                "leased": state["leased"],
                "display_id": state["display_id"],
            }

    @app.post("/init")
    async def init() -> dict[str, Any]:
        runtime()
        return {"ok": True}

    @app.post("/display/acquire")
    async def acquire() -> dict[str, Any]:
        async with lock:
            if state["faulted"]:
                raise HTTPException(503, "Native reset failed; replace this worker before retrying")
            if state["leased"]:
                raise HTTPException(
                    409, "This native worker already has an exclusive episode lease"
                )
            state["leased"] = True
            state["display_id"] += 1
            return {"logical_id": state["display_id"], "sf_display_id": state["display_id"]}

    @app.post("/display/release")
    async def release(display_id: int | None = None) -> dict[str, Any]:
        async with lock:
            require_display(display_id)
            if state["task_id"] is not None:
                await call(runtime().tear_down)
            state.update(leased=False, task_id=None, goal=None)
            return {"ok": True}

    @app.post("/display/claim_app")
    @app.post("/display/release_app")
    async def claim() -> dict[str, Any]:
        return {"ok": True}

    @app.get("/screenshot")
    async def screenshot(display_id: int | None = None) -> dict[str, Any]:
        async with lock:
            require_display(display_id)
            require_task()
            return {"b64_png": await call(runtime().screenshot)}

    @app.get("/task/list")
    async def tasks() -> dict[str, Any]:
        return {"tasks": await call(runtime().task_metadata)}

    @app.post("/task/init")
    async def initialize(body: dict[str, Any]) -> dict[str, Any]:
        nonlocal cached_verdict
        async with lock:
            require_display(body.get("display_id"))
            if state["faulted"]:
                raise HTTPException(503, "Native reset failed; replace this worker before retrying")
            if runtime().task is not None:
                state["faulted"] = True
                await call(runtime().tear_down)
                state["faulted"] = False
            state.update(task_id=None, goal=None)
            completed_steps.clear()
            cached_verdict = None
            task_id = body["task_name"]
            try:
                goal = await call(runtime().initialize, task_id)
            except Exception:
                state["faulted"] = True
                raise
            state.update(task_id=task_id, goal=goal)
            return {"ok": True, "evidence_dir": str(runtime().episode_dir)}

    @app.get("/task/goal")
    async def goal(task_name: str, display_id: int | None = None) -> str:
        require_display(display_id)
        require_task(task_name)
        return str(state["goal"])

    @app.post("/step")
    async def step(body: dict[str, Any]) -> dict[str, Any]:
        nonlocal cached_verdict
        async with lock:
            require_display(body.get("display_id"))
            require_task()
            action = body["action"]
            action_id = body.get("action_id")
            if action_id and action_id in completed_steps:
                return completed_steps[action_id]
            try:
                result = await call(runtime().execute_decision, action)
            except Exception:
                state["faulted"] = True
                raise
            if action_id:
                completed_steps[action_id] = result
            cached_verdict = None
            return cast(dict[str, Any], result)

    @app.post("/episode/terminal")
    async def terminal(body: dict[str, Any]) -> dict[str, Any]:
        nonlocal cached_verdict
        async with lock:
            require_display(body.get("display_id"))
            require_task()
            await call(runtime().execute_decision, body["action"])
            cached_verdict = None
            return {"ok": True}

    @app.get("/task/eval")
    async def evaluate(task_name: str, display_id: int | None = None) -> dict[str, Any]:
        nonlocal cached_verdict
        async with lock:
            require_display(display_id)
            require_task(task_name)
            if cached_verdict is None:
                cached_verdict = cast(dict[str, Any], await call(runtime().evaluate))
            return cached_verdict

    @app.post("/task/tear_down")
    async def tear_down(body: dict[str, Any]) -> dict[str, Any]:
        async with lock:
            require_display(body.get("display_id"))
            require_task(body["task_name"])
            state["faulted"] = True
            await call(runtime().tear_down)
            state.update(task_id=None, goal=None, faulted=False)
            return {"ok": True}

    return app


@click.command()
@click.option("--env", "environment", required=True)
@click.option("--config", type=click.Path(exists=True, dir_okay=False), required=True)
@click.option("--host", default="127.0.0.1")
@click.option("--port", type=int, default=6800)
def main(environment: str, config: str, host: str, port: int) -> None:
    import uvicorn

    settings = json.loads(Path(config).read_text())
    uvicorn.run(create_worker_app(environment, settings), host=host, port=port)


if __name__ == "__main__":
    main()
