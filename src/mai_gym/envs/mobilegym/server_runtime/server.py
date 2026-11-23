from __future__ import annotations

import asyncio
import base64
import contextlib
import inspect
import json
import logging
import socket
import sys
import uuid
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, cast

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
from starlette.staticfiles import StaticFiles

from mai_gym.envs.mobilegym.server_runtime.static_server import resolve_media_directory

logger = logging.getLogger(__name__)

DEVICE_ID = "mobilegym-0"
LOGICAL_VIEWPORT = (360, 800)
PHYSICAL_SCREEN = (1080, 2400)
DEVICE_SCALE_FACTOR = 3


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
    params: dict[str, Any] | None = None


class TaskTearDownBody(BaseModel):
    task_name: str
    req_device: str | None = None
    display_id: int | None = None


class _MobileGymPool:
    def __init__(self, pool: Any, num_pages: int, *, strict: bool = False) -> None:
        self._pool = pool
        self._num = num_pages
        self._free: list[int] = list(range(num_pages))
        self._task_state: dict[int, dict[str, Any]] = {}
        self._lock = asyncio.Lock()
        self.strict = strict
        self._leases: dict[int, int] = {}
        self._next_lease = 0
        self.bad_pages: set[int] = set()

    async def acquire(self) -> int:
        async with self._lock:
            if not self._free:
                raise HTTPException(503, "MobileGymPool: no free pages")
            idx = self._free.pop(0)
            self._next_lease += 1
            lease = self._next_lease if self.strict else idx + 1
            self._leases[lease] = idx
            return lease

    async def release(self, display_id: int) -> None:
        async with self._lock:
            idx = self._leases.get(display_id, display_id - 1 if not self.strict else -1)
            if self.strict and idx == -1:
                raise HTTPException(409, "Stale or missing MobileGym episode lease")
            if 0 <= idx < self._num and idx not in self._free:
                if self.strict and idx not in self.bad_pages:
                    await self.tear_down(display_id)
                if idx not in self.bad_pages:
                    self._free.append(idx)
                self._leases.pop(display_id, None)
                self._task_state.pop(display_id, None)

    def env(self, display_id: int | None) -> Any:
        if display_id is None:
            raise HTTPException(400, "display_id required")
        idx = self._leases.get(display_id, display_id - 1 if not self.strict else -1)
        if self.strict and idx == -1:
            raise HTTPException(409, "Stale or missing MobileGym episode lease")
        if not 0 <= idx < self._num:
            raise HTTPException(404, f"unknown display_id={display_id}")
        if idx in self.bad_pages:
            raise HTTPException(503, "Native MobileGym reset failed; replace this worker")
        return self._pool[idx]

    def invalidate(self, display_id: int) -> None:
        if self.strict and display_id in self._leases:
            self.bad_pages.add(self._leases[display_id])

    async def tear_down(self, display_id: int) -> None:
        task = self.get_state(display_id).get("task")
        function = getattr(task, "teardown", None)
        if callable(function):
            try:
                result = function(self.env(display_id))
                if inspect.isawaitable(result):
                    await result
            except Exception:
                self.invalidate(display_id)
                raise
        self.clear_state(display_id)

    def set_state(self, display_id: int, **fields: Any) -> None:
        st = self._task_state.setdefault(display_id, {})
        st.update(fields)

    def get_state(self, display_id: int) -> dict[str, Any]:
        return self._task_state.get(display_id, {})

    def clear_state(self, display_id: int) -> None:
        self._task_state.pop(display_id, None)


def _verbose_task_fields(cls: Any) -> dict[str, Any]:

    fields: dict[str, Any] = {
        "difficulty": getattr(cls, "difficulty", None),
        "scope": getattr(cls, "scope", None),
        "objective": getattr(cls, "objective", None),
        "composition": getattr(cls, "composition", None),
        "max_steps": getattr(cls, "max_steps", None),
        "templates": list(getattr(cls, "templates", []) or []),
        "judge": _infer_judge(cls),
        "parameters": _summarise_parameters(getattr(cls, "parameters", {}) or {}),
    }

    return {k: v for k, v in fields.items() if v not in (None, [], {})}


def _infer_judge(cls: Any) -> str:

    del cls
    return "state_diff"


def _summarise_parameters(parameters: dict[str, Any]) -> dict[str, Any]:

    summary: dict[str, Any] = {}
    for key, spec in parameters.items():
        if not isinstance(spec, dict):
            summary[key] = {"type": "unknown"}
            continue
        entry: dict[str, Any] = {
            "type": spec.get("type", "unknown"),
        }
        if "default" in spec:
            entry["has_default"] = True
        if "sampler" in spec or "source" in spec:
            entry["has_sampler"] = True
        if "values" in spec and isinstance(spec["values"], (list, tuple)):
            entry["values"] = list(spec["values"])
        elif "values" in spec and isinstance(spec["values"], dict):
            entry["values"] = list(spec["values"].keys())
        if "description" in spec:
            entry["description"] = str(spec["description"])
        summary[key] = entry
    return summary


_ACTION_TYPE_ALIAS = {
    "tap": "click",
    "press": "click",
    "input": "type",
    "input_text": "type",
    "swipe": "swipe",
    "scroll": "swipe",
    "back": "back",
    "navigate_back": "back",
    "home": "home",
    "navigate_home": "home",
    "wait": "wait",
    "answer": "answer",
    "terminate": "complete",
    "finished": "complete",
    "keyboard_enter": "enter",
    "enter": "enter",
    "open_app": "awake",
    "open": "awake",
    "launch_app": "awake",
    "awake": "awake",
}


def _maybe_normalize_xy(x: Any, y: Any, physical_size: tuple[int, int]) -> tuple[float, float]:

    fx, fy = float(x), float(y)
    if fx > 1.0 or fy > 1.0:
        w, h = physical_size
        return fx / float(w), fy / float(h)
    return fx, fy


def _canonicalize_data(
    action_type: str,
    raw: dict[str, Any],
    physical_size: tuple[int, int] = (1080, 2400),
) -> dict[str, Any]:

    data: dict[str, Any] = {
        k: v
        for k, v in raw.items()
        if k
        not in {
            "action_type",
            "type",
            "x",
            "y",
            "start_x",
            "start_y",
            "end_x",
            "end_y",
            "text",
            "value",
            "app_name",
            "app",
            "direction",
            "prediction",
            "reasoning_content",
            "status",
        }
    }

    if "x" in raw and "y" in raw:
        with contextlib.suppress(TypeError, ValueError):
            nx, ny = _maybe_normalize_xy(raw["x"], raw["y"], physical_size)
            data["point"] = [nx, ny]

    if "start_x" in raw and "start_y" in raw:
        with contextlib.suppress(TypeError, ValueError):
            nx, ny = _maybe_normalize_xy(
                raw["start_x"],
                raw["start_y"],
                physical_size,
            )
            data["point1"] = [nx, ny]
    if "end_x" in raw and "end_y" in raw:
        with contextlib.suppress(TypeError, ValueError):
            nx, ny = _maybe_normalize_xy(
                raw["end_x"],
                raw["end_y"],
                physical_size,
            )
            data["point2"] = [nx, ny]

    if "text" in raw and "value" not in raw:
        data["value"] = raw["text"]
    elif action_type == "awake" and ("app_name" in raw or "app" in raw):
        data["value"] = raw.get("app_name") or raw.get("app")
    elif "value" in raw:
        data["value"] = raw["value"]

    if action_type == "swipe" and "direction" in raw and "point1" not in data:
        direction = str(raw["direction"]).lower()
        if direction in ("up", "down", "left", "right"):
            x, y = data.get("point", (0.5, 0.5))
            dx, dy = {"up": (0, -0.25), "down": (0, 0.25), "left": (-0.25, 0), "right": (0.25, 0)}[
                direction
            ]

            def clamp(value: float) -> float:
                return min(0.95, max(0.05, value))

            data["point1"] = [clamp(x - dx), clamp(y - dy)]
            data["point2"] = [clamp(x + dx), clamp(y + dy)]

    return data


def _translate_action(
    raw: dict[str, Any],
    physical_size: tuple[int, int] = (1080, 2400),
) -> Any:

    from bench_env.env.base import Action, ActionType

    raw_type = (raw.get("action_type") or raw.get("type") or "noop").lower()
    canonical = _ACTION_TYPE_ALIAS.get(raw_type, raw_type)
    try:
        atype = ActionType(canonical.upper())
    except ValueError:
        atype = ActionType.NOOP

    data = _canonicalize_data(canonical, raw, physical_size=physical_size)
    return Action(
        action_type=atype,
        data=data,
        thought=raw.get("reasoning_content"),
    )


def create_mobilegym_app(
    static_dir: str,
    *,
    media_dir: str | None = None,
    num_pages: int = 8,
    server_port: int = 6800,
    static_port: int | None = None,
    viewport_size: tuple[int, int] = LOGICAL_VIEWPORT,
    physical_size: tuple[int, int] = PHYSICAL_SCREEN,
    delay_after_action: float = 0.8,
    pool_factory: Callable[..., Any] | None = None,
    task_manifest: list[dict[str, Any]] | None = None,
    evidence_dir: str | None = None,
    profile_identity: dict[str, Any] | None = None,
    framework_source_digest: str | None = None,
) -> FastAPI:

    media_directory = resolve_media_directory(static_dir, media_dir)
    if task_manifest is not None and (
        viewport_size != LOGICAL_VIEWPORT or physical_size != PHYSICAL_SCREEN
    ):
        raise ValueError(
            "Profiled MobileGym requires a 360x800 logical viewport and 1080x2400 physical "
            "screenshots at 3x device scale; physical dimensions are not viewport dimensions"
        )

    state: dict[str, Any] = {"pool": None, "static_proc": None}
    manifest = (
        {entry["task_id"]: entry for entry in task_manifest} if task_manifest is not None else None
    )

    if static_port is None:
        with socket.socket() as listener:
            listener.bind(("127.0.0.1", 0))
            static_port = listener.getsockname()[1]
    app_url = f"http://localhost:{static_port}/"

    def retain(display_id: int | None, name: str, data: Any) -> None:
        entry = _pool().get_state(display_id) if display_id is not None else {}
        directory = entry.get("evidence_dir")
        if directory:
            (Path(directory) / name).write_text(json.dumps(data, indent=2, default=str) + "\n")

    @asynccontextmanager
    async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
        if pool_factory is None:
            import subprocess

            command = [
                sys.executable,
                "-m",
                "mai_gym.envs.mobilegym.server_runtime.static_server",
                "--port",
                str(static_port),
                "--static-dir",
                static_dir,
            ]
            if media_directory is not None:
                command.extend(["--media-dir", str(media_directory)])
            proc = subprocess.Popen(
                command,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            state["static_proc"] = proc

            def stop_static() -> None:
                if proc.poll() is None:
                    proc.terminate()
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait()

            import urllib.request

            for _ in range(30):
                try:
                    urllib.request.urlopen(app_url, timeout=0.5).close()
                    break
                except Exception:
                    if proc.poll() is not None:
                        stop_static()
                        raise RuntimeError("MobileGym static frontend failed to start") from None
                    await asyncio.sleep(0.1)
            else:
                stop_static()
                state["static_proc"] = None
                raise RuntimeError("MobileGym static frontend did not become ready")

        underlying = None
        try:
            if pool_factory is not None:
                underlying = await pool_factory(app_url, num_pages)
            else:
                from bench_env.env.pool import EnvPool

                underlying = EnvPool(
                    url=app_url,
                    n=num_pages,
                    isolation="contexts" if manifest is not None else "pages",
                    headless=True,
                    viewport_size=viewport_size,
                    physical_size=physical_size,
                    device_scale_factor=DEVICE_SCALE_FACTOR,
                    coord_space="norm_0_1",
                    delay_after_action=delay_after_action,
                )
                await underlying.__aenter__()
        except BaseException:
            if underlying is not None and hasattr(underlying, "__aexit__"):
                with contextlib.suppress(Exception):
                    await underlying.__aexit__(None, None, None)
            if state["static_proc"] is not None:
                stop_static()
            state["static_proc"] = None
            raise
        state["pool"] = _MobileGymPool(underlying, num_pages=num_pages, strict=manifest is not None)
        logger.info("[MobileGymServer] pool ready (n=%d) %s", num_pages, app_url)
        try:
            yield
        finally:
            try:
                if hasattr(underlying, "__aexit__"):
                    await underlying.__aexit__(None, None, None)
            except Exception as exc:
                logger.warning("[MobileGymServer] pool shutdown error: %s", exc)
            static_process = state.get("static_proc")
            if static_process is not None:
                try:
                    stop_static()
                except Exception as exc:
                    logger.warning("[MobileGymServer] static shutdown error: %s", exc)
            state["pool"] = None
            state["static_proc"] = None
            logger.info("[MobileGymServer] pool closed")

    app = FastAPI(title="MobileGymServer", lifespan=lifespan)
    app.mount("/app", StaticFiles(directory=static_dir, html=True), name="webapp")
    if media_directory is not None:
        app.mount("/cdn", StaticFiles(directory=str(media_directory)), name="cdn")

    def _pool() -> _MobileGymPool:
        if state["pool"] is None:
            raise HTTPException(503, "pool not initialized")
        return cast(_MobileGymPool, state["pool"])

    @app.get("/health")
    async def health() -> dict[str, Any]:
        return {"ok": state["pool"] is not None and not state["pool"].bad_pages}

    @app.get("/state")
    async def diagnostics(display_id: int | None = None) -> dict[str, Any]:
        pool = _pool()
        if display_id is None:
            return {
                "env_type": "mobilegym",
                "action_space": "mobile_use",
                "viewport_size": list(viewport_size),
                "physical_size": list(physical_size),
                "device_scale_factor": DEVICE_SCALE_FACTOR,
                "media_available": media_directory is not None,
                "leased_count": len(pool._leases),
                "faulted": bool(pool.bad_pages),
            }
        pool.env(display_id)
        entry = pool.get_state(display_id)
        return {
            "task_id": entry.get("task_name"),
            "episode_dir": entry.get("evidence_dir"),
            "params": getattr(entry.get("task"), "params", {}),
            "seed": getattr(entry.get("task"), "_seed", None),
            "answer_empty": not entry.get("answer"),
        }

    @app.get("/devices")
    async def devices() -> dict[str, Any]:
        return {"devices": [DEVICE_ID]}

    @app.post("/init")
    async def init(body: InitBody | None = None) -> dict[str, Any]:
        return {"ok": True}

    @app.post("/display/acquire")
    async def acquire_display(device: str = Query(DEVICE_ID)) -> dict[str, Any]:
        display_id = await _pool().acquire()
        return {"logical_id": display_id, "sf_display_id": display_id}

    @app.post("/display/release")
    async def release_display(
        device: str = Query(DEVICE_ID),
        display_id: int = Query(...),
    ) -> dict[str, Any]:
        await _pool().release(display_id)
        return {"ok": True}

    @app.post("/display/claim_app")
    async def claim_app(
        device: str = Query(DEVICE_ID),
        display_id: int = Query(...),
        package_name: str = Query(...),
    ) -> dict[str, Any]:

        return {"ok": True}

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
        env = _pool().env(display_id)

        if hasattr(env, "screenshot"):
            buf = await env.screenshot()
        else:
            buf = await env.page.screenshot(type="png")
        b64 = base64.b64encode(buf).decode("ascii") if isinstance(buf, bytes) else buf
        entry = _pool().get_state(display_id) if display_id is not None else {}
        if entry.get("evidence_dir"):
            index = entry.get("screenshot_count", 0)
            (Path(entry["evidence_dir"]) / f"screenshot{index}.png").write_bytes(
                base64.b64decode(b64)
            )
            entry["screenshot_count"] = index + 1
        return {"b64_png": b64}

    @app.post("/step")
    async def step(body: StepBody) -> dict[str, Any]:
        if body.display_id is None:
            raise HTTPException(400, "display_id required")
        env = _pool().env(body.display_id)
        try:
            if body.action.get("action_type") == "key":
                await env.page.keyboard.press(body.action["key"])
                result = None
            else:
                action = _translate_action(body.action, physical_size=physical_size)
                result = await env.step(action)
        except Exception:
            if body.display_id is not None:
                _pool().invalidate(body.display_id)
            raise
        entry = _pool().get_state(body.display_id) if body.display_id is not None else {}
        if entry.get("evidence_dir"):
            factual = {
                key: value
                for key, value in body.action.items()
                if key not in {"prediction", "reasoning_content"}
            }
            with (Path(entry["evidence_dir"]) / "actions.jsonl").open("a") as stream:
                stream.write(json.dumps(factual) + "\n")
        raw_type = (body.action.get("action_type") or body.action.get("type") or "").lower()
        if raw_type == "answer":
            answer_text = body.action.get("text")
            if answer_text is None:
                answer_text = body.action.get("value", "")
            _pool().set_state(body.display_id, answer=str(answer_text))
        return {
            "ok": True,
            "done": bool(getattr(result, "done", False)),
            "info": getattr(result, "info", {}) or {},
        }

    @app.post("/episode/terminal")
    async def terminal(body: StepBody) -> dict[str, Any]:
        pool = _pool()
        pool.env(body.display_id)
        if body.display_id is None or not pool.get_state(body.display_id).get("task"):
            raise HTTPException(409, "No initialized MobileGym episode")
        answer = body.action.get("answer", body.action.get("text"))
        pool.set_state(body.display_id, answer=str(answer) if answer is not None else None)
        retain(
            body.display_id,
            "terminal.json",
            {
                key: value
                for key, value in body.action.items()
                if key not in {"prediction", "reasoning_content"}
            },
        )
        return {"ok": True}

    @app.post("/task/init")
    async def task_init(body: TaskInitBody) -> dict[str, Any]:
        pool = _pool()
        if body.display_id is None:
            raise HTTPException(400, "display_id required")
        env = pool.env(body.display_id)

        from bench_env.task import TaskRegistry

        reg = TaskRegistry()
        if manifest is not None and body.task_name not in manifest:
            raise HTTPException(404, "task is absent from the pinned MobileGym manifest")
        parameters = body.params or {}
        if manifest is not None:
            recorded = manifest[body.task_name].get("params", {})
            if parameters and parameters != recorded:
                raise HTTPException(422, "task parameters conflict with the pinned manifest")
            parameters = recorded
        try:
            if manifest is not None and "seed" in manifest[body.task_name]:
                parameters = {**parameters, "_seed": manifest[body.task_name]["seed"]}
            task = reg.create_task(body.task_name, **parameters)
        except Exception as exc:
            raise HTTPException(404, f"task not found: {body.task_name}: {exc}") from exc

        if pool.strict:
            await pool.tear_down(body.display_id)
        try:
            init_obs = await task.setup(env)
        except Exception:
            pool.invalidate(body.display_id)
            raise
        pool.set_state(
            body.display_id,
            task=task,
            task_name=body.task_name,
            init_obs=init_obs,
            last_obs=init_obs,
            answer=None,
        )
        if evidence_dir:
            episode_path = Path(evidence_dir) / uuid.uuid4().hex
            episode_path.mkdir(parents=True, exist_ok=False)
            pool.set_state(body.display_id, evidence_dir=str(episode_path), screenshot_count=0)
            retain(
                body.display_id,
                "identity.json",
                {
                    "profile": profile_identity,
                    "framework_source_digest": framework_source_digest,
                    "viewport_size": list(viewport_size),
                    "physical_size": list(physical_size),
                    "device_scale_factor": DEVICE_SCALE_FACTOR,
                    "action_space": "mobile_use",
                    "task_id": body.task_name,
                    "params": getattr(task, "params", {}),
                    "seed": getattr(task, "_seed", None),
                    "app_url": app_url,
                },
            )
            retain(body.display_id, "initial-state.json", getattr(init_obs, "state", init_obs))
        result = {"ok": True, "task_name": body.task_name}
        episode = pool.get_state(body.display_id).get("evidence_dir")
        if episode:
            result["evidence_dir"] = episode
        return result

    @app.get("/task/goal")
    async def task_goal(
        task_name: str = Query(...),
        display_id: int | None = Query(None),
    ) -> Any:
        if display_id is None:
            raise HTTPException(400, "display_id required")
        _pool().env(display_id)
        st = _pool().get_state(display_id)
        task = st.get("task")
        if task is None:
            raise HTTPException(404, "no active task; call /task/init first")
        if _pool().strict and st.get("task_name") != task_name:
            raise HTTPException(409, "Task does not match the active MobileGym episode")

        if hasattr(task, "description"):
            instruction = task.description
        elif hasattr(task, "instruction_text"):
            instruction = task.instruction_text()
        else:
            instruction = task.templates[0] if getattr(task, "templates", None) else ""
        return instruction

    @app.api_route("/task/eval", methods=["GET", "POST"])
    async def task_eval(
        task_name: str = Query(...),
        display_id: int | None = Query(None),
    ) -> dict[str, Any]:
        if display_id is None:
            raise HTTPException(400, "display_id required")
        pool = _pool()
        env = pool.env(display_id)
        st = pool.get_state(display_id)
        task = st.get("task")
        if task is None:
            raise HTTPException(404, "no active task")
        if st.get("task_name", task_name) != task_name:
            raise HTTPException(409, "task does not match the active episode")

        last_obs = None
        for method_name in ("get_observation", "observe"):
            fn = getattr(env, method_name, None)
            if fn is None:
                continue
            try:
                last_obs = await fn()
                break
            except Exception as exc:
                logger.warning(
                    "[MobileGymServer] %s.%s failed: %s — falling through",
                    type(env).__name__,
                    method_name,
                    exc,
                )
        if last_obs is None:
            if pool.strict:
                raise HTTPException(
                    500, "Native MobileGym state is unavailable; episode is ungraded"
                )
            last_obs = st.get("last_obs")

        from bench_env.task.judge import JudgeInput

        judge_in = JudgeInput(
            init_obs=st.get("init_obs"),
            last_obs=last_obs,
            answer=st.get("answer"),
        )
        if callable(getattr(task, "evaluate", None)):
            try:
                native = task.evaluate(judge_in)
                if native.judge_error:
                    raise ValueError(native.judge_error)
                result = {
                    "score": float(native.progress),
                    "success": bool(native.passed),
                    "native_success": bool(native.success),
                    "clean": bool(native.clean),
                    "reason": json.dumps(native.to_dict()),
                    "native_result": native.to_dict(),
                }
                retain(display_id, "final-state.json", getattr(last_obs, "state", last_obs))
                retain(display_id, "verifier.json", result)
                return result
            except Exception as exc:
                raise HTTPException(
                    500, "Native MobileGym evaluator failed; episode is ungraded"
                ) from exc
        if pool.strict:
            raise HTTPException(500, "Pinned MobileGym tasks require the native evaluate method")
        try:
            checks = task.check_goals(judge_in) or []
        except Exception as exc:
            logger.exception("[MobileGymServer] check_goals failed")
            raise HTTPException(
                500, "native MobileGym verifier failed; episode is ungraded"
            ) from exc
        passed = [bool(c.get("passed")) for c in checks if isinstance(c, dict)]
        if not passed:
            raise HTTPException(500, "native MobileGym verifier returned no checks")
        score = float(sum(passed)) / float(len(passed))
        reason = "all checks passed" if all(passed) else json.dumps(checks)[:512]
        return {"score": score, "success": all(passed), "reason": reason}

    @app.post("/task/tear_down")
    async def task_tear_down(body: TaskTearDownBody) -> dict[str, Any]:
        if body.display_id is not None:
            if _pool().strict:
                _pool().env(body.display_id)
                await _pool().tear_down(body.display_id)
            else:
                _pool().clear_state(body.display_id)
        return {"ok": True}

    @app.get("/task/list")
    async def task_list(verbose: int = 0) -> list[dict[str, Any]]:

        if manifest is not None:
            return [{**entry, "name": task_id} for task_id, entry in manifest.items()]

        from bench_env.task import TaskRegistry

        registered: list[dict[str, Any]] = []
        reg = TaskRegistry()
        for suite in reg.list_suites():
            try:
                names = reg.list_tasks(suite)
            except Exception as exc:
                logger.warning("[MobileGymServer] list_tasks(%s): %s", suite, exc)
                continue
            for name in names:
                try:
                    cls = reg.get(suite, name)
                except Exception:
                    continue
                entry: dict[str, Any] = {
                    "name": f"{suite}.{name}",
                    "tags": list(getattr(cls, "capabilities", []) or []),
                    "apps": list(getattr(cls, "apps", []) or []),
                }
                if verbose:
                    entry.update(_verbose_task_fields(cls))
                registered.append(entry)
        return registered

    @app.post("/evaluate")
    async def evaluate_unsupported() -> dict[str, Any]:
        raise HTTPException(404, "MobileGym uses /task/eval — set env_type=mobilegym")

    return app
