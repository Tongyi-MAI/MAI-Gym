from typing import Any

from mai_gym.envs.shared.computer_actions import BASH_SENTINEL, action_to_pyautogui
from mai_gym.envs.shared.desktop import create_desktop_env
from mai_gym.envs.shared.runtime import NativeRuntime
from mai_gym.envs.shared.tasks import task_file


class OSWorld2Runtime(NativeRuntime):
    def __init__(
        self, profile: dict[str, Any], settings: dict[str, Any], tasks: list[dict[str, Any]]
    ) -> None:
        super().__init__(profile, settings, tasks)
        from desktop_env.desktop_env import DesktopEnv

        self.env = create_desktop_env(DesktopEnv, profile, settings)

    def initialize(self, task_id: str) -> str:
        from task_loader import load_task_from_file

        task = self.begin(task_id)
        native_task = load_task_from_file(str(task_file(self.profile, task, suffix=".py")))
        self.env.reset(task_config=native_task)
        self.write_json(
            "vm.json",
            {
                "provider": self.settings["desktop"].get("provider_name"),
                "container_id": getattr(getattr(self.env.provider, "container", None), "id", None),
                "server_port": self.env.server_port,
                "base_image": self.settings["desktop"].get("path_to_vm"),
            },
        )
        return str(self.env.instruction)

    def screenshot(self) -> str:
        return self.retain_screenshot(self.env._get_obs()["screenshot"])

    def act(self, action: dict[str, Any]) -> dict[str, Any]:
        if action.get("action_type") == "action_sequence":
            return {"actions": [self.act(item) for item in action["actions"]]}
        code = action_to_pyautogui(action)
        if code and code.startswith(BASH_SENTINEL):
            raise ValueError("OSWorld 2 paper GUI profile does not permit shell actions")
        if code is None and action.get("action_type") != "wait":
            raise ValueError(f"Unsupported native desktop action: {action.get('action_type')}")
        _, reward, done, info = self.env.step(code or "WAIT")
        result = {"reward": reward, "done": done, "info": info}
        self.record_action(action, result)
        return result

    def evaluate(self) -> dict[str, Any]:
        native = self.env.evaluate()
        if isinstance(native, dict):
            score = native.get("score", native.get("reward"))
            if score is None:
                raise ValueError("OSWorld 2 native evaluator returned no score")
            return self.verdict(float(score), success=native.get("success"), detail=native)
        return self.verdict(float(native))

    def close(self) -> None:
        self.env.close()


def create_runtime(
    profile: dict[str, Any], settings: dict[str, Any], tasks: list[dict[str, Any]]
) -> OSWorld2Runtime:
    return OSWorld2Runtime(profile, settings, tasks)
