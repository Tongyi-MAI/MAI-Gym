import asyncio
import base64
import contextlib
import logging
import re
import shutil
import time
from pathlib import Path
from typing import Any

import httpx

from mai_gym.envs.shared.computer_actions import BASH_SENTINEL, action_to_pyautogui
from mai_gym.models.session import RunningBackend

logger = logging.getLogger(__name__)


def create_backend(backend: RunningBackend) -> Any:
    if backend.metadata.get("runtime_protocol") == "native":
        from mai_gym.envs.shared.backend import NativeBackend

        return NativeBackend(backend)
    return OSWorldBackend(backend)


_OSW_PROXY_SETTINGS: dict[str, Any] | None = None


def configure_osw_proxy(settings: dict[str, Any] | None) -> None:

    global _OSW_PROXY_SETTINGS

    if not settings or not settings.get("enabled"):
        _OSW_PROXY_SETTINGS = None
        return

    try:
        from desktop_env.providers.aws.proxy_pool import get_global_proxy_pool
    except ImportError as exc:
        logger.warning(
            "[OSWorldBackend] configure_osw_proxy: desktop_env not importable "
            "(%s) — proxy stays off. Install the 'osworld' extra to enable.",
            exc,
        )
        _OSW_PROXY_SETTINGS = None
        return

    host = settings.get("host", "")
    port = int(settings.get("port") or 0)
    if not host or not port:
        logger.warning(
            "[OSWorldBackend] configure_osw_proxy: host/port missing "
            "(host=%r port=%r); proxy stays off",
            host,
            port,
        )
        _OSW_PROXY_SETTINGS = None
        return

    pool = get_global_proxy_pool()
    already = any(
        getattr(p, "host", None) == host and getattr(p, "port", None) == port
        for p in getattr(pool, "proxies", [])
    )
    if not already:
        pool.add_proxy(
            host=host,
            port=port,
            username=settings.get("username") or None,
            password=settings.get("password") or None,
            protocol=settings.get("protocol", "http"),
        )
        logger.info(
            "[OSWorldBackend] registered proxy %s:%d in upstream ProxyPool",
            host,
            port,
        )
    _OSW_PROXY_SETTINGS = dict(settings)


_CALL_TIMEOUT = 240.0
_HEALTH_TIMEOUT = 10.0
_SCREENSHOT_TIMEOUT = 60.0

_INIT_READINESS_TIMEOUT = 240.0
_INIT_READINESS_INTERVAL = 1.0


_POST_INIT_WAIT_TIME = 60.0


_RESTART_READINESS_TIMEOUT = 240.0

_PYAUTOGUI_PREAMBLE = "import pyautogui; import time; pyautogui.FAILSAFE = False; "

_BASH_HISTORY_FLAG = "MAI_GYM_BASH_WRITE_HISTORY"
_TRUTHY = {"1", "true", "yes", "on"}


def _maybe_record_bash_history(cmd: str) -> str:

    import os

    if os.environ.get(_BASH_HISTORY_FLAG, "").strip().lower() not in _TRUTHY:
        return cmd
    import base64

    b64 = base64.b64encode(cmd.encode("utf-8", "surrogateescape")).decode("ascii")

    record = f"printf '%s\\n' \"$(echo '{b64}' | base64 -d)\" >> ~/.bash_history 2>/dev/null\n"
    return record + cmd


_PRESS_LT_RE = re.compile(r"""pyautogui\.press\(["'](?:<|\\u003c)["']\)""")
_TYPEWRITE_RE = re.compile(r"""pyautogui\.typewrite\((["'])(.*?)\1(?:,\s*interval=([\d.]+))?\)""")


def _fix_pyautogui_less_than_bug(command: str) -> str:

    command = _PRESS_LT_RE.sub('pyautogui.hotkey("shift", ",")', command)

    def _expand_typewrite(match: re.Match[str]) -> str:
        quote = match.group(1)
        content = match.group(2)
        interval = match.group(3)

        with contextlib.suppress(UnicodeDecodeError):
            content = content.encode("utf-8").decode("unicode_escape")
        if "<" not in content:
            return match.group(0)
        suffix = f", interval={interval}" if interval else ""
        parts = content.split("<")
        pieces: list[str] = []
        for idx, part in enumerate(parts):
            if idx == 0:
                if part:
                    pieces.append(f"pyautogui.typewrite({quote}{part}{quote}{suffix})")
            else:
                pieces.append('pyautogui.hotkey("shift", ",")')
                if part:
                    pieces.append(f"pyautogui.typewrite({quote}{part}{quote}{suffix})")
        return "; ".join(pieces)

    return _TYPEWRITE_RE.sub(_expand_typewrite, command)


def _safe_cache_component(value: str) -> str:

    return re.sub(r"[^A-Za-z0-9_.=-]+", "_", str(value)).strip("_") or "unknown"


class OSWorldBackend:
    def __init__(self, backend: RunningBackend) -> None:
        self.backend = backend

        self._cache_root = Path.cwd().resolve() / "cache" / "osworld"

        self._eval_isolation_override: bool | None = None

        self._client: httpx.AsyncClient | None = None
        self._init_timeout = _INIT_READINESS_TIMEOUT
        self._init_interval = _INIT_READINESS_INTERVAL
        self._post_init_wait_time = _POST_INIT_WAIT_TIME

        self._restart_lock = asyncio.Lock()

        self._proxy_bootstrapped = False

    def _url(self, path: str) -> str:
        return self.backend.url.rstrip("/") + path

    def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=_CALL_TIMEOUT)
        return self._client

    async def health_check(self) -> bool:
        try:
            r = await self._get_client().get(self._url("/screenshot"), timeout=_HEALTH_TIMEOUT)
            return r.status_code == 200
        except Exception:
            return False

    async def init_device(self) -> None:

        deadline = time.monotonic() + self._init_timeout
        last_exc: Exception | None = None
        while time.monotonic() < deadline:
            try:
                r = await self._get_client().get(
                    self._url("/screenshot"),
                    timeout=_HEALTH_TIMEOUT,
                )
                if r.status_code == 200:
                    logger.info("[OSWorldBackend] %s ready", self.backend.url)
                    return
                last_exc = RuntimeError(f"status={r.status_code}")
            except Exception as exc:
                last_exc = exc
            if self._init_interval > 0:
                await asyncio.sleep(self._init_interval)
        raise RuntimeError(
            f"OSWorld VM at {self.backend.url} not ready after "
            f"{self._init_timeout:.0f}s: {last_exc}"
        )

    async def screenshot(self, display_id: int | None) -> str:

        r = await self._get_client().get(self._url("/screenshot"), timeout=_SCREENSHOT_TIMEOUT)
        r.raise_for_status()
        return base64.b64encode(r.content).decode("ascii")

    async def do_step(
        self,
        action: dict[str, Any],
        display_id: int | None,
        inter_action_wait: float = 0.0,
    ) -> dict[str, Any] | None:

        if action.get("action_type") == "action_sequence":
            return await self._do_action_sequence(
                action.get("actions", []),
                display_id,
                inter_action_wait,
            )

        pyautogui_code = action_to_pyautogui(action)
        if pyautogui_code is None:
            return None

        if pyautogui_code.startswith(BASH_SENTINEL):
            cmd = pyautogui_code[len(BASH_SENTINEL) :]

            payload_command = ["bash", "-c", _maybe_record_bash_history(cmd)]
            try:
                r = await self._get_client().post(
                    self._url("/execute"),
                    json={"command": payload_command, "shell": False},
                    timeout=_CALL_TIMEOUT,
                )
            except Exception as exc:
                logger.warning("[OSWorldBackend] /execute bash failed: %s", exc)
                return {
                    "output": "",
                    "error": f"transport error: {exc}",
                    "returncode": -1,
                }
            body = self._safe_json(r) or {}
            output = (body.get("output") or "") if isinstance(body, dict) else ""
            error = (body.get("error") or "") if isinstance(body, dict) else ""
            returncode = body.get("returncode") if isinstance(body, dict) else None
            if returncode is None and r.status_code != 200:
                returncode = -1
            elif returncode is None:
                returncode = 0
            logger.info(
                "[OSWorldBackend] bash returncode=%s output[:200]=%r error[:200]=%r",
                returncode,
                output[:200],
                error[:200],
            )
            return {
                "output": output,
                "error": error,
                "returncode": int(returncode),
            }

        full_code = _fix_pyautogui_less_than_bug(pyautogui_code)
        if not full_code.strip():
            return None
        full_source = _PYAUTOGUI_PREAMBLE + full_code
        payload_command = ["python", "-c", full_source]
        try:
            r = await self._get_client().post(
                self._url("/execute"),
                json={"command": payload_command, "shell": False},
                timeout=_CALL_TIMEOUT,
            )
        except Exception as exc:
            logger.warning("[OSWorldBackend] execute failed: %s", exc)
            raise

        if r.status_code != 200:
            body = self._safe_json(r)
            logger.warning(
                "[OSWorldBackend] /execute HTTP %d body=%r action_type=%r",
                r.status_code,
                body,
                action.get("action_type"),
            )
        else:
            body = self._safe_json(r)
            if isinstance(body, dict):
                rc = body.get("returncode")
                err = body.get("error") or ""
                status = body.get("status")
                if status == "error" or (rc is not None and rc != 0) or err:
                    logger.warning(
                        "[OSWorldBackend] /execute status=%s returncode=%s "
                        "error=%r output=%r action_type=%r",
                        status,
                        rc,
                        err[:200],
                        (body.get("output") or "")[:200],
                        action.get("action_type"),
                    )

        return None

    async def _do_action_sequence(
        self,
        actions: list[dict[str, Any]],
        display_id: int | None,
        inter_action_wait: float = 0.0,
    ) -> dict[str, Any] | None:

        outputs: list[str] = []
        errors: list[str] = []
        returncode = 0
        pending_gui: list[str] = []

        async def flush_gui() -> None:
            if not pending_gui:
                return
            merged = "\ntime.sleep(0.5)\n".join(pending_gui)
            pending_gui.clear()
            await self.do_step(
                {"action_type": "pyautogui_code", "code": merged},
                display_id,
                inter_action_wait=inter_action_wait,
            )
            if inter_action_wait > 0:
                await asyncio.sleep(inter_action_wait)

        for sub in actions:
            sub_type = sub.get("action_type", "")

            if sub_type in ("finished", "terminate", "answer", "DONE", "FAIL"):
                break
            if sub_type == "pyautogui_code":
                code = sub.get("code")
                if code:
                    pending_gui.append(code)
                continue

            await flush_gui()
            res = await self.do_step(
                sub,
                display_id,
                inter_action_wait=inter_action_wait,
            )
            if isinstance(res, dict):
                out = str(res.get("output") or "")
                err = str(res.get("error") or "")
                if out:
                    outputs.append(out)
                if err:
                    errors.append(err)
                rc = res.get("returncode", 0)
                if rc:
                    returncode = int(rc)
            if inter_action_wait > 0:
                await asyncio.sleep(inter_action_wait)

        await flush_gui()

        if outputs or errors:
            return {
                "output": "\n".join(outputs),
                "error": "\n".join(errors),
                "returncode": returncode,
            }
        return None

    @staticmethod
    def _safe_json(r: httpx.Response) -> Any:
        try:
            return r.json()
        except Exception:
            return None

    async def run_command(
        self,
        command: list[str],
        shell: bool = False,
        timeout: float = _CALL_TIMEOUT,
    ) -> dict[str, Any]:

        try:
            r = await self._get_client().post(
                self._url("/execute"),
                json={"command": command, "shell": shell},
                timeout=timeout,
            )
        except Exception as exc:
            logger.warning("[OSWorldBackend] run_command transport error: %s", exc)
            return {"output": "", "error": f"transport error: {exc}", "returncode": -1}
        body = self._safe_json(r) or {}
        if not isinstance(body, dict):
            body = {}
        rc = body.get("returncode")
        if rc is None:
            rc = 0 if r.status_code == 200 else -1
        return {
            "output": body.get("output") or "",
            "error": body.get("error") or "",
            "returncode": int(rc),
        }

    _UPLOAD_TIMEOUT = 600.0

    async def upload_file(
        self,
        local_path: str,
        remote_path: str,
        *,
        timeout: float | None = None,
    ) -> None:

        p = Path(local_path)
        data = await asyncio.to_thread(p.read_bytes)
        r = await self._get_client().post(
            self._url("/setup/upload"),
            data={"file_path": remote_path},
            files={"file_data": (p.name, data, "application/octet-stream")},
            timeout=timeout or self._UPLOAD_TIMEOUT,
        )
        r.raise_for_status()

    async def evaluate(self, display_id: int | None, task_id: str | None) -> float | None:

        return None

    async def task_init(self, task_config: dict[str, Any]) -> None:

        config_steps = task_config.get("config") or []
        if not config_steps:
            return

        use_proxy = bool(
            task_config.get("proxy", False) and _OSW_PROXY_SETTINGS,
        )

        de = self.backend.metadata.get("desktop_env")
        if de is not None:
            logger.info(
                "[OSWorldBackend] task_init (delegated): %d setup step(s) for task %s use_proxy=%s",
                len(config_steps),
                task_config.get("id"),
                use_proxy,
            )
            if use_proxy:
                await self._maybe_run_proxy_setup(de.setup_controller)
            await asyncio.to_thread(
                de.setup_controller.setup,
                config_steps,
                use_proxy,
            )
            await self._settle_after_task_init(task_config.get("id"))
            return

        try:
            from desktop_env.controllers.setup import SetupController
        except ImportError as exc:
            raise RuntimeError(
                "OSWorldBackend.task_init requires the 'osworld' extra and "
                "the desktop_env package. Run 'uv sync --extra osworld' then "
                f"'uv pip install -e /opt/OSWorld --no-deps'. "
                f"Underlying ImportError: {exc}"
            ) from exc

        vm_ip, server_port = self._parse_vm_endpoint()
        cache_dir = self._task_cache_dir(task_config.get("id"))
        chromium_port = self.backend.metadata.get("chromium_port")
        if chromium_port is None:
            logger.warning(
                "[OSWorldBackend] backend %s has no chromium_port in metadata; "
                "falling back to 9222. In multi-container launches this "
                "cascade-fails — check the launcher's metadata wiring.",
                self.backend.id,
            )
        controller = SetupController(
            vm_ip=vm_ip,
            server_port=server_port,
            chromium_port=int(chromium_port or 9222),
            vlc_port=int(self.backend.metadata.get("vlc_port", 8080)),
            cache_dir=cache_dir,
        )
        logger.info(
            "[OSWorldBackend] task_init: running %d setup step(s) for task %s use_proxy=%s",
            len(config_steps),
            task_config.get("id"),
            use_proxy,
        )
        if use_proxy:
            await self._maybe_run_proxy_setup(controller)
        await asyncio.to_thread(controller.setup, config_steps, use_proxy)
        await self._settle_after_task_init(task_config.get("id"))

    async def _maybe_run_proxy_setup(self, controller: Any) -> None:

        settings = _OSW_PROXY_SETTINGS
        if self._proxy_bootstrapped or settings is None:
            return

        client_password = str(settings.get("client_password", "password"))
        username = str(settings.get("username") or "")
        password = str(settings.get("password") or "")
        credentials = f"{username}:{password}@" if username or password else ""
        protocol = str(settings.get("protocol") or "http")
        upstream = f"{protocol} {credentials}{settings['host']}:{int(settings['port'])}"
        sudo = f"echo '{client_password}' | sudo -S bash -c"
        proxy_commands = [
            f'{sudo} "apt-get update"',
            f'{sudo} "apt-get install -y tinyproxy"',
            f"{sudo} \"echo 'Port 18888' > /tmp/tinyproxy.conf\"",
            f"{sudo} \"echo 'Allow 127.0.0.1' >> /tmp/tinyproxy.conf\"",
        ]
        logger.info(
            "[OSWorldBackend] proxy bootstrap on backend %s: tinyproxy install"
            " + chromium-only (no ~/.bashrc env writes)",
            self.backend.id,
        )
        try:
            for cmd in proxy_commands:
                await asyncio.to_thread(
                    controller._execute_setup,
                    [cmd],
                    shell=True,
                )

            upstream_command = f"{sudo} \"echo 'Upstream {upstream}' >> /tmp/tinyproxy.conf\""
            response = await self._get_client().post(
                self._url("/execute"),
                json={
                    "command": ["bash", "-c", upstream_command],
                    "shell": False,
                },
                timeout=_CALL_TIMEOUT,
            )
            body = self._safe_json(response) or {}
            returncode = body.get("returncode") if isinstance(body, dict) else None
            if response.status_code != 200 or returncode not in (None, 0, "0"):
                raise RuntimeError(
                    "credential-bearing tinyproxy configuration failed "
                    f"(HTTP {response.status_code}, rc={returncode!r})"
                )
            await asyncio.to_thread(
                controller._launch_setup,
                ["tinyproxy -c /tmp/tinyproxy.conf -d"],
                shell=True,
            )
        except Exception as exc:
            logger.warning(
                "[OSWorldBackend] proxy bootstrap on backend %s failed: %s "
                "— continuing without proxy for this task",
                self.backend.id,
                exc,
            )
            return
        self._proxy_bootstrapped = True

    async def _settle_after_task_init(self, task_id: str | None) -> None:

        if self._post_init_wait_time <= 0:
            return
        logger.info(
            "[OSWorldBackend] task_init: settling %.0fs for apps to stabilize (task %s)",
            self._post_init_wait_time,
            task_id,
        )
        await asyncio.sleep(self._post_init_wait_time)

    def _parse_vm_endpoint(self) -> tuple[str, int]:

        from urllib.parse import urlparse

        parsed = urlparse(self.backend.url)
        host = parsed.hostname or "localhost"
        port = parsed.port or 5000
        return host, port

    def _task_cache_dir(
        self,
        task_id: str | None,
        cache_key: str | None = None,
        *,
        clean: bool = False,
    ) -> str:

        safe_id = _safe_cache_component(task_id or "unknown")
        safe_key = _safe_cache_component(
            cache_key or self.backend.id or self.backend.device_name or "default"
        )
        path = self._cache_root / safe_id / safe_key
        if clean and path.is_dir():
            shutil.rmtree(path)
        path.mkdir(parents=True, exist_ok=True)
        return str(path)

    async def list_tasks(self) -> list[dict[str, Any]]:
        return []

    async def reset_environment(self) -> None:

        import shutil
        import subprocess

        container_name = self.backend.metadata.get("container_name")
        if not container_name:
            logger.debug(
                "[OSWorldBackend] reset_environment skipped: no container_name "
                "in metadata (backend %s)",
                self.backend.id,
            )
            return

        docker_bin = shutil.which("docker")
        if docker_bin is None:
            logger.warning(
                "[OSWorldBackend] reset_environment: docker CLI not on PATH; "
                "skipping reset for container %s",
                container_name,
            )
            return

        logger.info(
            "[OSWorldBackend] reset_environment: docker restart %s",
            container_name,
        )
        try:
            await asyncio.to_thread(
                subprocess.run,
                [docker_bin, "restart", container_name],
                check=True,
                capture_output=True,
                timeout=120,
            )
        except subprocess.CalledProcessError as exc:
            logger.error(
                "[OSWorldBackend] docker restart %s failed (rc=%d): %s",
                container_name,
                exc.returncode,
                exc.stderr.decode(errors="replace"),
            )
            raise
        except subprocess.TimeoutExpired:
            logger.error(
                "[OSWorldBackend] docker restart %s timed out after 120s",
                container_name,
            )
            raise

        prev_timeout = self._init_timeout
        self._init_timeout = _RESTART_READINESS_TIMEOUT
        try:
            try:
                await self.init_device()
            except RuntimeError as init_exc:
                await self._escalate_to_full_restart(init_exc)
        finally:
            self._init_timeout = prev_timeout

    async def _escalate_to_full_restart(self, init_exc: RuntimeError) -> None:

        de = self.backend.metadata.get("desktop_env")
        if de is None:
            logger.error(
                "[OSWorldBackend] reset_environment: in-place restart of "
                "container %s failed and no desktop_env in metadata "
                "(reused/attach mode) — cannot escalate. Backend %s is dead.",
                self.backend.metadata.get("container_name"),
                self.backend.id,
            )
            raise init_exc

        old_url = self.backend.url
        old_chromium = self.backend.metadata.get("chromium_port")
        old_vlc = self.backend.metadata.get("vlc_port")
        old_vnc = self.backend.metadata.get("vnc_port")
        old_container = self.backend.metadata.get("container_name")

        async with self._restart_lock:
            try:
                await self.init_device()
                logger.info(
                    "[OSWorldBackend] reset_environment: backend %s healed "
                    "by concurrent restart (now at %s); skipping escalation",
                    self.backend.id,
                    self.backend.url,
                )
                return
            except RuntimeError:
                pass

            path_to_vm = getattr(de, "path_to_vm", None)
            headless = getattr(de, "headless", True)
            os_type = getattr(de, "os_type", "Ubuntu")
            if not path_to_vm:
                logger.error(
                    "[OSWorldBackend] reset_environment: desktop_env has no "
                    "path_to_vm; cannot relaunch container for backend %s",
                    self.backend.id,
                )
                raise init_exc

            logger.warning(
                "[OSWorldBackend] reset_environment: ESCALATING to full "
                "restart for backend %s (container=%s, old url=%s) — "
                "in-place docker restart failed: %s",
                self.backend.id,
                old_container,
                old_url,
                init_exc,
            )

            try:
                await asyncio.to_thread(de.provider.stop_emulator, path_to_vm)
            except Exception as exc:
                logger.warning(
                    "[OSWorldBackend] stop_emulator raised during escalation "
                    "(continuing to start_emulator): %s",
                    exc,
                )

            try:
                await asyncio.to_thread(
                    de.provider.start_emulator,
                    path_to_vm,
                    headless=headless,
                    os_type=os_type,
                )
            except Exception as exc:
                logger.error(
                    "[OSWorldBackend] start_emulator failed during escalation for backend %s: %s",
                    self.backend.id,
                    exc,
                )
                raise

            new_server_port = int(de.provider.server_port)
            new_chromium_port = int(de.provider.chromium_port)
            new_vlc_port = int(de.provider.vlc_port)
            new_vnc_port = int(de.provider.vnc_port)
            new_container = de.provider.container
            new_container_name = new_container.name if new_container is not None else ""

            de.server_port = new_server_port
            de.chromium_port = new_chromium_port
            de.vlc_port = new_vlc_port
            de.vnc_port = new_vnc_port
            de.vm_ip = "localhost"

            self.backend.url = f"http://localhost:{new_server_port}"
            self.backend.metadata["chromium_port"] = new_chromium_port
            self.backend.metadata["vlc_port"] = new_vlc_port
            self.backend.metadata["vnc_port"] = new_vnc_port
            self.backend.metadata["container_name"] = new_container_name

            await self.init_device()

            logger.info(
                "[OSWorldBackend] reset_environment: full restart healed "
                "backend %s — container %s -> %s, url %s -> %s, "
                "chrome %s -> %d, vlc %s -> %d, vnc %s -> %d",
                self.backend.id,
                old_container,
                new_container_name,
                old_url,
                self.backend.url,
                old_chromium,
                new_chromium_port,
                old_vlc,
                new_vlc_port,
                old_vnc,
                new_vnc_port,
            )

    async def close(self) -> None:
        if self._client is not None:
            with contextlib.suppress(Exception):
                await self._client.aclose()
            self._client = None

    async def task_eval(
        self,
        task_config: dict[str, Any],
        action_history: list[str] | None = None,
        cache_key: str | None = None,
        client_password: str = "password",
    ) -> tuple[float, str] | None:

        from mai_gym.envs.osworld.eval_isolation import (
            eval_isolation_settings,
            evaluate_in_subprocess,
        )

        isolation_enabled = (
            eval_isolation_settings().enabled
            if self._eval_isolation_override is None
            else self._eval_isolation_override
        )
        if isolation_enabled:
            return await evaluate_in_subprocess(
                self,
                task_config,
                action_history,
                cache_key,
                client_password,
            )

        from mai_gym.envs.osworld._eval import osworld_task_eval

        return await asyncio.to_thread(
            osworld_task_eval,
            self,
            task_config,
            action_history,
            cache_key,
            client_password,
        )
