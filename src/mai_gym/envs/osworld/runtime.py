import json
from typing import Any

from mai_gym.envs.shared.desktop import DesktopRuntime, create_desktop_env
from mai_gym.envs.shared.tasks import task_file


class OSWorldRuntime(DesktopRuntime):
    def __init__(
        self, profile: dict[str, Any], settings: dict[str, Any], tasks: list[dict[str, Any]]
    ) -> None:
        super().__init__(profile, settings, tasks)
        from desktop_env.desktop_env import DesktopEnv

        self.env = create_desktop_env(DesktopEnv, profile, settings)

    def initialize(self, task_id: str) -> str:
        task = self.begin(task_id)
        native = json.loads(task_file(self.profile, task).read_text())
        self.env.reset(task_config=native)
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

    def evaluate(self) -> dict[str, Any]:
        return self.verdict(float(self.env.evaluate()))

    def close(self) -> None:
        self.env.close()


def create_runtime(
    profile: dict[str, Any], settings: dict[str, Any], tasks: list[dict[str, Any]]
) -> OSWorldRuntime:
    return OSWorldRuntime(profile, settings, tasks)
