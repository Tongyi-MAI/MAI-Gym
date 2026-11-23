import base64
import json
import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
from starlette.staticfiles import StaticFiles

from mai_gym.envs.webenv.action_executor import execute_action
from mai_gym.envs.webenv.context_manager import _DEVICE_SCALE_FACTOR, ContextManager
from mai_gym.envs.webenv.declarative_evaluator import run_declarative_evaluator
from mai_gym.envs.webenv.evaluator import run_evaluator

logger = logging.getLogger(__name__)

DEVICE_ID = "web-env-0"


class StepBody(BaseModel):
    device: str | None = None
    action: dict[str, Any]
    display_id: int | None = None


class InitBody(BaseModel):
    device: str | None = None


class EvalBody(BaseModel):
    task_id: str | None = None
    evaluator_js: str | None = None


class TaskBody(BaseModel):
    task_name: str
    req_device: str | None = None
    display_id: int | None = None


def _load_tasks(tasks_file: str | None) -> list[dict[str, Any]]:

    if tasks_file is None:
        return []
    p = Path(tasks_file)
    if not p.exists():
        logger.warning("[WebEnvServer] Tasks file not found: %s", tasks_file)
        return []
    data = json.loads(p.read_text(encoding="utf-8"))
    tasks = data.get("tasks", [])
    if not isinstance(tasks, list) or not all(isinstance(task, dict) for task in tasks):
        raise ValueError(f"Expected a list of task records in {p}")
    return tasks


def _load_task_suites(task_suites_dir: str | None) -> list[dict[str, Any]]:

    if task_suites_dir is None:
        return []
    root = Path(task_suites_dir)
    if not root.is_dir():
        logger.warning("[WebEnvServer] Task suites directory not found: %s", root)
        return []

    tasks: list[dict[str, Any]] = []
    seen: set[str] = set()
    for tasks_path in sorted(root.glob("*/tasks.json")):
        env_dir = tasks_path.parent
        env_id = env_dir.name
        suite = json.loads(tasks_path.read_text(encoding="utf-8"))
        start_page = f"{env_id}/index.html?bare=1"

        for raw in suite.get("tasks", []):
            local_id = raw.get("id")
            if not local_id:
                continue
            task_id = f"{env_id}/{local_id}"
            if task_id in seen:
                raise ValueError(f"duplicate declarative task id: {task_id}")
            seen.add(task_id)
            level = str(raw.get("level") or "L2")
            tasks.append(
                {
                    "task_id": task_id,
                    "instruction": raw.get("instruction") or "",
                    "description": raw.get("name") or local_id,
                    "tags": [env_id, level, *(raw.get("tags") or [])],
                    "apps": [env_id],
                    "env_id": env_id,
                    "local_task_id": local_id,
                    "level": level,
                    "max_steps": int(raw.get("max_steps", 40)),
                    "start_page": start_page,
                    "reset": {
                        "seed": int(raw.get("seed", 0)),
                        "patch": ((raw.get("init") or {}).get("patch") or []),
                    },
                    "goal": raw.get("goal") or [],
                }
            )
    return tasks


async def _execute_step_action(
    page: Any,
    action: dict[str, Any],
    app_base_url: str,
    entry_page: str,
    device_scale_factor: float,
) -> None:

    if action.get("action_type") == "answer":
        has_declarative_api = await page.evaluate("() => !!(window.__ENV__ && window.__ENV__.step)")
        if has_declarative_api:
            await page.evaluate(
                "answer => window.__ENV__.step({action: 'finish', answer})",
                action.get("text", action.get("answer")),
            )
            return
    await execute_action(
        page,
        action,
        app_base_url,
        entry_page,
        device_scale_factor=device_scale_factor,
    )


def create_webenv_app(
    static_dir: str,
    *,
    evaluators_dir: str | None = None,
    tasks_file: str | None = None,
    task_suites_dir: str | None = None,
    entry_page: str = "homepage.html",
    server_port: int = 6800,
    viewport_width: int = 375,
    viewport_height: int = 812,
    device_scale_factor: float = _DEVICE_SCALE_FACTOR,
) -> FastAPI:

    ctx_mgr: ContextManager | None = None
    app_base_url = f"http://localhost:{server_port}/app"
    tasks_data = [*_load_tasks(tasks_file), *_load_task_suites(task_suites_dir)]
    task_by_id = {t["task_id"]: t for t in tasks_data if t.get("task_id")}

    def find_task(task_id: str | None) -> dict[str, Any]:
        task = task_by_id.get(task_id or "")
        if task is None and task_suites_dir and not tasks_file:
            raise HTTPException(404, f"Unknown task: {task_id}")
        return task or {}

    @asynccontextmanager
    async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
        nonlocal ctx_mgr
        from playwright.async_api import async_playwright

        pw = await async_playwright().start()
        browser = await pw.chromium.launch(headless=True)
        ctx_mgr = ContextManager(
            browser,
            app_base_url,
            entry_page=entry_page,
            viewport_width=viewport_width,
            viewport_height=viewport_height,
            device_scale_factor=device_scale_factor,
        )
        logger.info("[WebEnvServer] Chromium ready — serving %s", static_dir)
        yield
        await browser.close()
        await pw.stop()
        logger.info("[WebEnvServer] Chromium shut down")

    app = FastAPI(title="WebEnvServer", lifespan=lifespan)

    app.mount("/app", StaticFiles(directory=static_dir, html=True), name="webapp")

    @app.get("/health")
    async def health() -> dict[str, Any]:
        return {"ok": True}

    @app.get("/devices")
    async def devices() -> dict[str, Any]:
        return {"devices": [DEVICE_ID]}

    @app.get("/tasks")
    async def list_tasks() -> dict[str, Any]:
        return {"tasks": tasks_data}

    @app.post("/init")
    async def init(body: InitBody | None = None) -> dict[str, Any]:
        return {"ok": True}

    @app.post("/display/acquire")
    async def acquire_display(
        device: str = Query(DEVICE_ID),
    ) -> dict[str, Any]:
        assert ctx_mgr is not None
        display_id = await ctx_mgr.acquire()
        logger.info("[WebEnvServer] Acquired display %d", display_id)
        return {"logical_id": display_id, "sf_display_id": display_id}

    @app.post("/display/release")
    async def release_display(
        device: str = Query(DEVICE_ID),
        display_id: int = Query(...),
    ) -> dict[str, Any]:
        assert ctx_mgr is not None
        await ctx_mgr.release(display_id)
        logger.info("[WebEnvServer] Released display %d", display_id)
        return {"ok": True}

    @app.get("/screenshot")
    async def screenshot(
        device: str = Query(DEVICE_ID),
        return_b64: str = Query("true"),
        display_id: int | None = Query(None),
    ) -> dict[str, Any]:
        assert ctx_mgr is not None
        page = ctx_mgr.get_page(display_id)
        buf = await page.screenshot(type="png")
        b64 = base64.b64encode(buf).decode("ascii")
        return {"b64_png": b64}

    @app.post("/step")
    async def step(body: StepBody) -> dict[str, Any]:
        assert ctx_mgr is not None
        page = ctx_mgr.get_page(body.display_id)
        await _execute_step_action(
            page,
            body.action,
            ctx_mgr.app_base_url,
            ctx_mgr.entry_page,
            ctx_mgr.device_scale_factor,
        )
        return {"ok": True}

    @app.post("/display/claim_app")
    async def claim_app(
        device: str = Query(DEVICE_ID),
        display_id: int = Query(...),
        package_name: str = Query(...),
    ) -> dict[str, Any]:
        assert ctx_mgr is not None
        page = ctx_mgr.get_page(display_id)
        entry_url = f"{ctx_mgr.app_base_url}/{ctx_mgr.entry_page}"
        await page.goto(entry_url, wait_until="load")
        return {"ok": True}

    @app.post("/display/release_app")
    async def release_app(
        device: str = Query(DEVICE_ID),
        package_name: str = Query(...),
    ) -> dict[str, Any]:
        return {"ok": True}

    @app.post("/evaluate")
    async def evaluate(
        device: str = Query(DEVICE_ID),
        display_id: int = Query(...),
        body: EvalBody | None = None,
    ) -> dict[str, Any]:
        assert ctx_mgr is not None
        page = ctx_mgr.get_page(display_id)
        task_id = body.task_id if body else None
        evaluator_js = body.evaluator_js if body else None
        task = find_task(task_id) if task_id else {}
        if task.get("goal"):
            result = await run_declarative_evaluator(page, task)
            return {"reward": result["score"], "success": result["success"]}
        score = await run_evaluator(
            page, task_id=task_id, evaluator_js=evaluator_js, evaluators_dir=evaluators_dir
        )
        return {"reward": score}

    @app.post("/task/init")
    async def task_init(body: TaskBody) -> dict[str, Any]:
        assert ctx_mgr is not None
        task = find_task(body.task_name)
        start_page = task.get("start_page") or entry_page
        await ctx_mgr.open_task(
            body.display_id,
            start_page,
            body.task_name,
            reset_options=task.get("reset"),
        )
        if any(
            goal.get("source") == "trajectory" and goal.get("path", "").startswith("accepted_")
            for goal in task.get("goal", [])
        ):
            await ctx_mgr.get_page(body.display_id).evaluate(
                Path(__file__).with_name("trajectory.js").read_text(encoding="utf-8")
            )
        return {"ok": True}

    @app.get("/task/goal")
    async def task_goal(
        task_name: str = Query(...),
        display_id: int | None = Query(None),
    ) -> str:
        task = task_by_id.get(task_name)
        if task is None:
            logger.warning("[WebEnvServer] task_goal: unknown task %s", task_name)
            return ""
        return task.get("instruction") or ""

    @app.get("/task/eval")
    async def task_eval(
        task_name: str = Query(...),
        display_id: int | None = Query(None),
    ) -> dict[str, Any]:
        assert ctx_mgr is not None
        page = ctx_mgr.get_page(display_id)
        task = find_task(task_name)
        if task.get("goal"):
            result = await run_declarative_evaluator(page, task)
            return {
                "score": result["score"],
                "success": result["success"],
                "reason": json.dumps(result, ensure_ascii=False),
            }
        score = await run_evaluator(page, task_id=task_name, evaluators_dir=evaluators_dir)
        return {"score": score, "reason": ""}

    @app.post("/task/tear_down")
    async def task_tear_down(body: TaskBody) -> dict[str, Any]:
        return {"ok": True}

    return app
