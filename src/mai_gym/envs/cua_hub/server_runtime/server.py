from __future__ import annotations

import asyncio
import base64
import json
import logging
import uuid
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel

from . import state_extraction as _state_extraction

logger = logging.getLogger(__name__)

DEVICE_ID = "cua-hub-0"


class StepBody(BaseModel):
    device: str = DEVICE_ID
    action: dict[str, Any]
    display_id: int | None = None


class InitBody(BaseModel):
    device: str | None = None


class TaskInitBody(BaseModel):
    task_name: str
    req_device: str | None = None
    display_id: int | None = None


class TaskTearDownBody(BaseModel):
    task_name: str
    req_device: str | None = None
    display_id: int | None = None


class _PagePool:
    def __init__(self, num_pages: int) -> None:
        self._num = num_pages
        self._free: list[int] = list(range(num_pages))
        self._pages: dict[int, Any] = {}
        self._state: dict[int, dict[str, Any]] = {}
        self._lock = asyncio.Lock()

    async def acquire(self) -> int:
        async with self._lock:
            if not self._free:
                raise HTTPException(503, "CUAHubPool: no free pages")
            idx = self._free.pop(0)
            return idx + 1

    async def release(self, display_id: int) -> None:
        async with self._lock:
            idx = display_id - 1
            if 0 <= idx < self._num and idx not in self._free:
                self._free.append(idx)
                self._state.pop(display_id, None)

    def bind_page(self, display_id: int, page: Any) -> None:
        self._pages[display_id] = page

    def page(self, display_id: int) -> Any:
        if display_id not in self._pages:
            raise HTTPException(404, f"unknown display_id={display_id}")
        return self._pages[display_id]

    def set_state(self, display_id: int, **fields: Any) -> None:
        st = self._state.setdefault(display_id, {})
        st.update(fields)

    def get_state(self, display_id: int) -> dict[str, Any]:
        return self._state.get(display_id, {})

    def clear_state(self, display_id: int) -> None:
        self._state.pop(display_id, None)


def _load_task_bundle(tasks_root: Path, task_name: str) -> dict[str, Any]:

    task_dir = tasks_root / task_name
    meta_path = task_dir / "task.json"
    if not meta_path.is_file():
        raise HTTPException(404, f"task.json missing: {meta_path}")
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise HTTPException(500, f"task.json parse error: {exc}") from exc
    meta["_dir"] = str(task_dir)
    return meta


def create_cua_hub_app(
    *,
    tasks_root: str,
    num_pages: int = 8,
    server_port: int = 6800,
    nginx_origin: str = "http://127.0.0.1:7000",
    viewport_size: tuple[int, int] = (1280, 800),
    device_scale_factor: float = 1.0,
    browser_factory: Callable[..., Awaitable[Any]] | None = None,
    state_client_factory: Callable[..., Any] | None = None,
    reward_runner: Callable[..., tuple[float | None, str | None]] | None = None,
) -> FastAPI:

    from . import reward_runner as _rr

    tasks_root_path = Path(tasks_root)
    state: dict[str, Any] = {
        "pool": _PagePool(num_pages),
        "browser": None,
        "context": None,
        "playwright": None,
        "http": None,
    }
    _reward = reward_runner or _rr.run_reward

    vw, vh = int(viewport_size[0]), int(viewport_size[1])
    context_kwargs: dict[str, Any] = {
        "viewport": {"width": vw, "height": vh},
        "device_scale_factor": float(device_scale_factor),
    }

    @asynccontextmanager
    async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
        if browser_factory is not None:
            browser = await browser_factory()
            if hasattr(browser, "new_context"):
                try:
                    context = await browser.new_context(**context_kwargs)
                except TypeError:
                    context = await browser.new_context()
            else:
                context = browser
        else:
            from playwright.async_api import async_playwright

            pw = await async_playwright().start()
            state["playwright"] = pw
            browser = await pw.chromium.launch(headless=True)
            context = await browser.new_context(**context_kwargs)
        state["browser"] = browser
        state["context"] = context

        if state_client_factory is not None:
            state["http"] = state_client_factory()
        else:
            import httpx

            state["http"] = httpx.AsyncClient(base_url=nginx_origin, timeout=30.0)

        logger.info(
            "[CUAHubServer] ready: nginx=%s tasks=%s n_pages=%d",
            nginx_origin,
            tasks_root_path,
            num_pages,
        )
        try:
            yield
        finally:
            try:
                http = state.get("http")
                if http is not None and hasattr(http, "aclose"):
                    await http.aclose()
                if browser is not None and hasattr(browser, "close"):
                    await browser.close()
                pw = state.get("playwright")
                if pw is not None:
                    await pw.stop()
            except Exception as exc:
                logger.warning("[CUAHubServer] shutdown error: %s", exc)

    app = FastAPI(title="CUAHubServer", lifespan=lifespan)

    def _pool() -> _PagePool:
        return state["pool"]

    def _http() -> Any:
        if state["http"] is None:
            raise HTTPException(503, "http client not initialized")
        return state["http"]

    async def _new_page() -> Any:
        ctx = state["context"]
        if ctx is None:
            raise HTTPException(503, "browser not initialized")
        if hasattr(ctx, "new_page"):
            return await ctx.new_page()
        return ctx

    _register_backend_routes(
        app,
        state,
        _pool,
        _new_page,
        nginx_origin,
        device_scale_factor,
    )
    _register_task_routes(
        app,
        _pool,
        _http,
        _reward,
        tasks_root_path,
        nginx_origin,
    )

    return app


def _register_backend_routes(
    app: FastAPI,
    state: dict[str, Any],
    _pool: Callable[[], _PagePool],
    _new_page: Callable[[], Awaitable[Any]],
    nginx_origin: str,
    device_scale_factor: float,
) -> None:

    @app.get("/health")
    async def health() -> dict[str, Any]:
        return {"ok": state["http"] is not None}

    @app.get("/devices")
    async def devices() -> dict[str, Any]:
        return {"devices": [DEVICE_ID]}

    @app.post("/init")
    async def init(body: InitBody | None = None) -> dict[str, Any]:
        return {"ok": True}

    @app.post("/display/acquire")
    async def acquire_display(device: str = Query(DEVICE_ID)) -> dict[str, Any]:
        display_id = await _pool().acquire()
        page = await _new_page()
        _pool().bind_page(display_id, page)
        sid = uuid.uuid4().hex
        _pool().set_state(display_id, sid=sid)
        return {"logical_id": display_id, "sf_display_id": display_id, "sid": sid}

    @app.post("/display/release")
    async def release_display(
        device: str = Query(DEVICE_ID),
        display_id: int = Query(...),
    ) -> dict[str, Any]:
        page = _pool()._pages.pop(display_id, None)
        if page is not None and hasattr(page, "close"):
            try:
                await page.close()
            except Exception as exc:
                logger.warning("[CUAHubServer] page.close error: %s", exc)
        await _pool().release(display_id)
        return {"ok": True}

    @app.post("/display/claim_app")
    async def claim_app(
        device: str = Query(DEVICE_ID),
        display_id: int = Query(...),
        package_name: str = Query(...),
    ) -> dict[str, Any]:

        page = _pool().page(display_id)
        st = _pool().get_state(display_id)
        sid = st.get("sid", "default")
        target = f"{nginx_origin}/{package_name}/?sid={sid}"
        if hasattr(page, "goto"):
            await page.goto(target)
        _pool().set_state(display_id, app_slug=package_name)
        return {"ok": True, "url": target}

    @app.post("/display/release_app")
    async def release_app(
        device: str = Query(DEVICE_ID),
        package_name: str = Query(...),
    ) -> dict[str, Any]:
        return {"ok": True}

    @app.get("/screenshot")
    async def screenshot(
        device: str = Query(DEVICE_ID),
        display_id: int | None = Query(None),
        return_b64: str = Query("true"),
    ) -> dict[str, Any]:
        if display_id is None:
            raise HTTPException(400, "display_id required")
        page = _pool().page(display_id)
        buf = await page.screenshot(type="png") if hasattr(page, "screenshot") else b""
        b64 = base64.b64encode(buf).decode("ascii") if isinstance(buf, bytes) else buf
        return {"b64_png": b64}

    @app.post("/step")
    async def step(body: StepBody) -> dict[str, Any]:

        try:
            from .action_executor import execute_action
        except ImportError:
            from mai_gym.envs.webenv.action_executor import execute_action

        if body.display_id is None:
            raise HTTPException(400, "display_id required")
        page = _pool().page(body.display_id)
        if hasattr(page, "goto"):
            await execute_action(
                page,
                body.action,
                app_base_url=nginx_origin,
                entry_page="index.html",
                device_scale_factor=float(device_scale_factor),
            )
        return {"ok": True}


def _register_task_routes(
    app: FastAPI,
    _pool: Callable[[], _PagePool],
    _http: Callable[[], Any],
    _reward: Callable[..., tuple[float | None, str | None]],
    tasks_root_path: Path,
    nginx_origin: str,
) -> None:

    @app.post("/task/init")
    async def task_init(body: TaskInitBody) -> dict[str, Any]:
        if body.display_id is None:
            raise HTTPException(400, "display_id required")
        task = _load_task_bundle(tasks_root_path, body.task_name)

        sid = _pool().get_state(body.display_id).get("sid", "default")
        app_slug = task.get("app_slug") or task.get("apps", [None])[0]
        if not app_slug:
            raise HTTPException(400, f"task {body.task_name}: missing app_slug")

        initial_state = task.get("initial_state") or {}
        http = _http()

        await http.post(
            f"/{app_slug}/post",
            params={"sid": sid},
            json={"action": "set", "state": initial_state},
        )

        page = _pool().page(body.display_id)
        if hasattr(page, "goto"):
            await page.goto(f"{nginx_origin}/{app_slug}/?sid={sid}")

        _pool().set_state(
            body.display_id,
            task=task,
            task_name=body.task_name,
            app_slug=app_slug,
            initial_state=initial_state,
        )
        return {"ok": True, "task_name": body.task_name, "app_slug": app_slug}

    @app.get("/task/goal")
    async def task_goal(
        task_name: str = Query(...),
        display_id: int | None = Query(None),
    ) -> Any:
        if display_id is None:
            raise HTTPException(400, "display_id required")
        st = _pool().get_state(display_id)
        task = st.get("task")
        if task is None:
            raise HTTPException(404, "no active task")
        return task.get("instruction") or ""

    @app.api_route("/task/eval", methods=["GET", "POST"])
    async def task_eval(
        task_name: str = Query(...),
        display_id: int | None = Query(None),
    ) -> dict[str, Any]:
        if display_id is None:
            raise HTTPException(400, "display_id required")
        st = _pool().get_state(display_id)
        task = st.get("task")
        if task is None:
            raise HTTPException(404, "no active task")
        sid = st.get("sid", "default")
        app_slug = st.get("app_slug")

        page = _pool()._pages.get(display_id)
        initial, current = await _state_extraction.read_state_from_browser(
            page,
            app_slug,
            sid,
        )

        if current is None:
            http = _http()
            r = await http.get(f"/{app_slug}/go", params={"sid": sid})
            go_body = r.json() if hasattr(r, "json") else r
            initial = go_body.get("initial_state")
            current = go_body.get("current_state")
            diff = go_body.get("state_diff")
        else:
            diff = _state_extraction.compute_state_diff(initial, current)

        score, reason = await asyncio.to_thread(
            _reward,
            Path(task["_dir"]),
            initial,
            current,
            diff,
        )
        return {"score": score or 0.0, "reason": reason}

    @app.post("/task/tear_down")
    async def task_tear_down(body: TaskTearDownBody) -> dict[str, Any]:
        if body.display_id is None:
            return {"ok": True}
        st = _pool().get_state(body.display_id)
        sid = st.get("sid", "default")
        app_slug = st.get("app_slug")
        if app_slug is not None:
            try:
                await _http().post(
                    f"/{app_slug}/post",
                    params={"sid": sid},
                    json={"action": "reset"},
                )
            except Exception as exc:
                logger.warning("[CUAHubServer] reset post failed: %s", exc)
        _pool().clear_state(body.display_id)
        return {"ok": True}

    @app.get("/task/list")
    async def task_list(verbose: int = 0) -> list[dict[str, Any]]:

        items: list[dict[str, Any]] = []
        if not tasks_root_path.is_dir():
            return items

        for tjson in sorted(tasks_root_path.rglob("task.json")):
            try:
                meta = json.loads(tjson.read_text(encoding="utf-8"))
            except Exception:
                continue
            rel = tjson.parent.relative_to(tasks_root_path)
            entry: dict[str, Any] = {
                "name": str(rel),
                "tags": meta.get("tags") or [],
                "apps": [meta.get("app_slug")]
                if meta.get("app_slug")
                else (meta.get("apps") or []),
            }
            if verbose:
                task_dir = tjson.parent

                has_reward_py = (task_dir / "reward.py").is_file()
                has_initial_setup = (
                    any(
                        (task_dir / f"initial_setup{ext}").exists()
                        for ext in (".py", ".sh", ".xlsx", ".docx", ".pptx")
                    )
                    or "initial_state" in meta
                )
                entry.update(
                    {
                        "difficulty": meta.get("difficulty"),
                        "instruction": meta.get("instruction"),
                        "judge": "reward_py" if has_reward_py else "state_diff",
                        "has_reward_py": has_reward_py,
                        "has_initial_setup": has_initial_setup,
                    }
                )

                entry = {k: v for k, v in entry.items() if v not in (None, [], {})}
            items.append(entry)
        return items

    @app.post("/evaluate")
    async def evaluate_unsupported() -> dict[str, Any]:
        raise HTTPException(404, "CUAHub uses /task/eval — set env_type=cua_hub")
