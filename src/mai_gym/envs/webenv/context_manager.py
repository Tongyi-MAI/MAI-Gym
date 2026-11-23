import json
import logging
import threading
from typing import Any

from playwright.async_api import Browser, BrowserContext, Page

logger = logging.getLogger(__name__)


def _trace_context_script(task_id: str, *, start_filename: str = "", trainable: bool = True) -> str:

    context = {
        "task_id": task_id,
        "taskId": task_id,
        "reference_trace_id": task_id,
        "trace_id": task_id,
        "trainable": bool(trainable),
        "source": "webenv_server",
        "start_filename": start_filename,
    }
    context_json = json.dumps(context, ensure_ascii=False)
    return f"""
    (() => {{
      const context = {context_json};
      try {{
        const text = JSON.stringify(context);
        window.GhostwebActiveTraceId = context.task_id;
        window.__ghostwebActiveTraceId = context.task_id;
        window.localStorage.setItem('ghostweb.execution_context', text);
        window.localStorage.setItem('ghostweb.current_task_id', context.task_id);
        window.localStorage.setItem('ghostweb.trainable_trace_id', context.task_id);
        window.sessionStorage.setItem('ghostweb.execution_context', text);
        window.sessionStorage.setItem('ghostweb.current_task_id', context.task_id);
        window.sessionStorage.setItem('ghostweb.trainable_trace_id', context.task_id);
      }} catch (e) {{
        console.error('Failed to initialize Ghostweb task context', e);
      }}
    }})()
    """


_FRAME_STRIP_JS = """
(() => {
    const style = document.createElement('style');
    style.textContent = `
        html {
            background: #fff !important;
            overflow: hidden !important;
            height: 100% !important;
        }
        body {
            max-width: none !important;
            margin: 0 !important;
            border-radius: 0 !important;
            box-shadow: none !important;
            width: 100vw !important;
            height: 100vh !important;
            overflow: hidden !important;
        }
        body::before, body::after {
            display: none !important;
        }
    `;
    if (document.head) {
        document.head.appendChild(style);
    } else {
        document.addEventListener('DOMContentLoaded', () => {
            document.head.appendChild(style);
        });
    }
})();
"""


_VIEWPORT_WIDTH = 375
_VIEWPORT_HEIGHT = 812
_DEVICE_SCALE_FACTOR = 2


class ContextManager:
    def __init__(
        self,
        browser: Browser,
        app_base_url: str,
        *,
        entry_page: str = "homepage.html",
        viewport_width: int = _VIEWPORT_WIDTH,
        viewport_height: int = _VIEWPORT_HEIGHT,
        device_scale_factor: float = _DEVICE_SCALE_FACTOR,
    ) -> None:
        self._browser = browser
        self.app_base_url = app_base_url
        self.entry_page = entry_page
        self.viewport_width = viewport_width
        self.viewport_height = viewport_height
        self.device_scale_factor = device_scale_factor
        self._contexts: dict[int, BrowserContext] = {}
        self._pages: dict[int, Page] = {}
        self._next_id = 1
        self._lock = threading.Lock()

    async def acquire(self) -> int:

        ctx = await self._browser.new_context(
            viewport={"width": self.viewport_width, "height": self.viewport_height},
            device_scale_factor=self.device_scale_factor,
        )

        await ctx.add_init_script(_FRAME_STRIP_JS)

        page = await ctx.new_page()
        entry_url = f"{self.app_base_url}/{self.entry_page}"
        await page.goto(entry_url, wait_until="load")

        with self._lock:
            display_id = self._next_id
            self._next_id += 1
            self._contexts[display_id] = ctx
            self._pages[display_id] = page

        logger.debug("[ContextManager] Created context display_id=%d", display_id)
        return display_id

    async def release(self, display_id: int) -> None:

        with self._lock:
            ctx = self._contexts.pop(display_id, None)
            self._pages.pop(display_id, None)

        if ctx is not None:
            await ctx.close()
            logger.debug("[ContextManager] Closed context display_id=%d", display_id)

    def get_page(self, display_id: int | None) -> Page:

        if display_id is None:
            with self._lock:
                if len(self._pages) == 1:
                    return next(iter(self._pages.values()))
            raise KeyError("display_id is required")
        with self._lock:
            try:
                return self._pages[display_id]
            except KeyError:
                raise KeyError(f"Unknown display_id: {display_id}") from None

    async def open_task(
        self,
        display_id: int | None,
        start_page: str,
        task_id: str,
        *,
        trainable: bool = True,
        reset_options: dict[str, Any] | None = None,
    ) -> None:

        page = self.get_page(display_id)
        script = _trace_context_script(task_id, start_filename=start_page, trainable=trainable)
        await page.context.add_init_script(script)
        url = f"{self.app_base_url}/{start_page}"
        await page.goto(url, wait_until="load")

        await page.evaluate(script)
        if reset_options is not None:
            await page.wait_for_function("window.__ENV_READY__ === true", timeout=15_000)
            await page.evaluate("options => window.__ENV__.reset(options)", reset_options)
        logger.debug(
            "[ContextManager] open_task display_id=%s task=%s → %s",
            display_id,
            task_id,
            start_page,
        )
