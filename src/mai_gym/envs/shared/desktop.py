from typing import Any

from mai_gym.envs.shared.computer_actions import BASH_SENTINEL, action_to_pyautogui
from mai_gym.envs.shared.runtime import NativeRuntime
from mai_gym.infra.docker_images import PinnedDockerClient


def create_desktop_env(env_class: Any, profile: dict[str, Any], settings: dict[str, Any]) -> Any:
    if settings["desktop"].get("provider_name", "vmware") != "docker":
        return env_class(**settings["desktop"])
    image = settings["docker_image"]
    image_id = next(asset["id"] for asset in profile["assets"] if asset.get("image") == image)

    class ProfileDesktopEnv(env_class):  # type: ignore[misc]
        def _start_emulator(self) -> None:
            if not isinstance(self.provider.client, PinnedDockerClient):
                self.provider.client = PinnedDockerClient(self.provider.client, image, image_id)
            super()._start_emulator()

    return ProfileDesktopEnv(**settings["desktop"])


class DesktopRuntime(NativeRuntime):
    env: Any

    def screenshot(self) -> str:
        data = self.env._get_obs()["screenshot"]
        if not data:
            raise RuntimeError("Native desktop controller returned no screenshot")
        return self.retain_screenshot(data)

    def act(self, action: dict[str, Any]) -> dict[str, Any]:
        if action.get("action_type") == "action_sequence":
            return {"actions": [self.act(item) for item in action["actions"]]}
        code = action_to_pyautogui(action)
        if code and code.startswith(BASH_SENTINEL):
            raise ValueError("Native GUI profile does not permit shell actions")
        if code is None and action.get("action_type") != "wait":
            raise ValueError(f"Unsupported native desktop action: {action.get('action_type')}")
        _, reward, done, info = self.env.step(code or "WAIT")
        self.screenshot()
        result = {"reward": reward, "done": done, "info": info}
        self.record_action(action, result)
        return result
