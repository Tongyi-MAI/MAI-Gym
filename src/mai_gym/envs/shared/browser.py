import time
import uuid
from typing import Any

from mai_gym.envs.shared.runtime import NativeRuntime


def factual_action(action: dict[str, Any]) -> dict[str, Any]:
    fields = {
        "action_type",
        "x",
        "y",
        "text",
        "direction",
        "key",
        "url",
        "duration",
        "start_x",
        "start_y",
        "end_x",
        "end_y",
        "button",
        "pixels",
    }
    return {key: value for key, value in action.items() if key in fields}


def execute_browser_action(page: Any, action: dict[str, Any]) -> None:
    kind = action["action_type"]
    if kind == "move":
        page.mouse.move(action["x"], action["y"])
    elif kind in {"click", "double_tap", "long_press"}:
        page.mouse.click(
            action["x"],
            action["y"],
            click_count=2 if kind == "double_tap" else 1,
            button="right" if kind == "long_press" else action.get("button", "left"),
        )
    elif kind in {"type", "input_text"}:
        page.keyboard.insert_text(action["text"])
    elif kind == "keyboard_enter":
        page.keyboard.press("Enter")
    elif kind == "key":
        page.keyboard.press(action["key"])
    elif kind == "scroll":
        if "x" in action and "y" in action:
            page.mouse.move(action["x"], action["y"])
        distance = action.get("pixels", 600)
        direction = action.get("direction", "down")
        if direction not in {"up", "down", "left", "right"}:
            raise ValueError(f"Unsupported scroll direction: {direction}")
        page.mouse.wheel(
            distance * ({"left": -1, "right": 1}.get(direction, 0)),
            distance * ({"up": -1, "down": 1}.get(direction, 0)),
        )
    elif kind == "navigate_back":
        page.go_back()
    elif kind in {"open_url", "goto"}:
        page.goto(action["url"], wait_until="domcontentloaded")
    elif kind == "drag":
        page.mouse.move(action["start_x"], action["start_y"])
        page.mouse.down()
        try:
            page.mouse.move(action["end_x"], action["end_y"], steps=10)
        finally:
            page.mouse.up()
    elif kind == "wait":
        time.sleep(min(max(float(action.get("duration", 1)), 0), 10))
    else:
        raise ValueError(f"Unsupported browser action: {kind}")


class BrowserRuntime(NativeRuntime):
    def __init__(
        self, profile: dict[str, Any], settings: dict[str, Any], tasks: list[dict[str, Any]]
    ) -> None:
        super().__init__(profile, settings, tasks)
        from playwright.sync_api import sync_playwright

        self.playwright = sync_playwright().start()
        options: dict[str, Any] = {"headless": settings.get("headless", True)}
        if settings.get("browser_executable"):
            options["executable_path"] = settings["browser_executable"]
        self.browser = self.playwright.chromium.launch(**options)
        self.context: Any = None
        self.page: Any = None
        self.context_id: str | None = None
        self.last_closed_context_id: str | None = None

    def begin_browser(self, task_id: str, start_url: str) -> dict[str, Any]:
        if self.context is not None:
            self.context.close()
            self.last_closed_context_id = self.context_id
        task = self.begin(task_id)
        self.context = self.browser.new_context(
            viewport=self.settings.get("viewport", {"width": 1280, "height": 720})
        )
        self.context_id = uuid.uuid4().hex
        self.page = self.context.new_page()
        self.page.goto(
            start_url,
            wait_until="domcontentloaded",
            timeout=self.settings.get("navigation_timeout_ms", 60000),
        )
        self.screenshot()
        self.write_json(
            "browser.json",
            {
                "version": self.browser.version,
                "executable": self.settings.get("browser_executable"),
            },
        )
        self.write_json(
            "initial_state.json",
            {"url": self.page.url, "cookies": self.context.cookies(), "task_id": task_id},
        )
        return task

    def screenshot(self) -> str:
        if self.page is None:
            self.context = self.browser.new_context()
            self.page = self.context.new_page()
        return self.retain_screenshot(self.page.screenshot(type="png"))

    def act(self, action: dict[str, Any]) -> dict[str, Any]:
        if action.get("action_type") == "action_sequence":
            return {"actions": [self.act(item) for item in action["actions"]]}
        executed = factual_action(action)
        execute_browser_action(self.page, executed)
        self.screenshot()
        self.record_action(executed, url=self.page.url)
        return {"url": self.page.url}

    def tear_down(self) -> None:
        if self.context is not None:
            self.context.close()
            self.last_closed_context_id = self.context_id
        self.context = None
        self.page = None
        self.context_id = None
        super().tear_down()

    def diagnostics(self) -> dict[str, Any]:
        return {
            **super().diagnostics(),
            "context_id": self.context_id,
            "last_closed_context_id": self.last_closed_context_id,
            "url": self.page.url if self.page is not None else None,
            "browser_version": self.browser.version,
        }

    def close(self) -> None:
        self.tear_down()
        self.browser.close()
        self.playwright.stop()
