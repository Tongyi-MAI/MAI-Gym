import json
import subprocess
import uuid
from pathlib import Path
from typing import Any

from mai_gym.envs.shared.browser import execute_browser_action, factual_action
from mai_gym.envs.shared.provenance import PreparationError
from mai_gym.envs.shared.runtime import NativeRuntime
from mai_gym.envs.shared.tasks import task_file


def validate_site_compose(
    config: dict[str, Any], images: set[str], project: str, allowed_bind_files: set[str]
) -> None:
    for service in config.get("services", {}).values():
        if service.get("container_name") or service.get("network_mode") == "host":
            raise PreparationError(
                "WebArena site containers require project-owned names and isolated networks"
            )
        if service.get("image") not in images:
            raise PreparationError("WebArena site image is absent from the verified profile assets")
        for volume in service.get("volumes", []):
            if (
                isinstance(volume, dict)
                and volume.get("type") == "bind"
                and (not volume.get("read_only") or volume.get("source") not in allowed_bind_files)
            ):
                raise PreparationError(
                    "WebArena bind mounts require verified read-only assets; writable data uses project-owned volumes"
                )
    for section in ("volumes", "networks"):
        for resource in config.get(section, {}).values():
            if isinstance(resource, dict) and resource.get("external"):
                raise PreparationError(
                    "WebArena site reset cannot use external volumes or networks"
                )
            if (
                isinstance(resource, dict)
                and resource.get("name")
                and not resource["name"].startswith(project + "_")
            ):
                raise PreparationError("WebArena volumes and networks require project-owned names")


def native_config(task: dict[str, Any], urls: dict[str, str]) -> dict[str, Any]:
    serialized = json.dumps(task)
    for name, url in urls.items():
        serialized = serialized.replace(f"__{name}__", url)
    if "__" in serialized and any(
        f"__{name}__" in serialized
        for name in (
            "REDDIT",
            "SHOPPING",
            "SHOPPING_ADMIN",
            "GITLAB",
            "WIKIPEDIA",
            "MAP",
            "HOMEPAGE",
        )
    ):
        raise PreparationError("WebArena task has unresolved native site placeholders")
    resolved: dict[str, Any] = json.loads(serialized)
    return resolved


class WebArenaRuntime(NativeRuntime):
    def __init__(
        self, profile: dict[str, Any], settings: dict[str, Any], tasks: list[dict[str, Any]]
    ) -> None:
        super().__init__(profile, settings, tasks)
        from browser_env import ScriptBrowserEnv

        self.env = ScriptBrowserEnv(
            headless=True,
            observation_type="image",
            viewport_size=settings.get("viewport", {"width": 1280, "height": 720}),
        )
        self.project = f"mai-gym-web-{uuid.uuid4().hex}"
        self.trajectory: list[Any] = []
        self.config_file: Any = None
        result = subprocess.run(
            [
                "docker",
                "compose",
                "--project-name",
                self.project,
                "--file",
                settings["sites"]["compose_file"],
                "config",
                "--format",
                "json",
            ],
            capture_output=True,
            text=True,
            check=True,
            timeout=30,
        )
        validate_site_compose(
            json.loads(result.stdout),
            {asset["image"] for asset in profile["assets"] if asset.get("kind") == "docker"},
            self.project,
            {asset["path"] for asset in profile["assets"] if asset.get("kind") != "docker"},
        )

    def _sites(self, *args: str) -> None:
        compose_file = self.settings["sites"]["compose_file"]
        subprocess.run(
            ["docker", "compose", "--project-name", self.project, "--file", compose_file, *args],
            check=True,
            capture_output=True,
            text=True,
            timeout=600,
        )

    def initialize(self, task_id: str) -> str:
        task = self.begin(task_id)
        self._sites("down", "--volumes", "--remove-orphans")
        self._sites("up", "--detach", "--wait")
        original = task.get("native_config") or json.loads(
            task_file(self.profile, task).read_text()
        )
        config = native_config(original, self.profile["site_urls"])
        self.config_file = self.episode_path / "task.json"
        auth_dir = self.episode_path / "auth"
        auth_dir.mkdir()
        if config.get("storage_state"):
            config["storage_state"] = str(auth_dir / Path(config["storage_state"]).name)
        bootstrap = self.settings["sites"].get("auth_command")
        if bootstrap:
            subprocess.run(
                [arg.replace("{auth_dir}", str(auth_dir)) for arg in bootstrap],
                check=True,
                capture_output=True,
                timeout=300,
            )
        if config.get("storage_state") and not Path(config["storage_state"]).is_file():
            raise PreparationError(
                "WebArena native authentication did not create this episode's storage state"
            )
        self.config_file.write_text(json.dumps(config))
        observation, info = self.env.reset(options={"config_file": str(self.config_file)})
        self.trajectory = [{"observation": observation, "info": info}]
        self.screenshot()
        return str(config["intent"])

    def screenshot(self) -> str:
        return self.retain_screenshot(self.env.page.screenshot(type="png"))

    def act(self, action: dict[str, Any]) -> dict[str, Any]:
        from browser_env.actions import create_none_action

        if action.get("action_type") == "action_sequence":
            return {"actions": [self.act(item) for item in action["actions"]]}
        executed = factual_action(action)
        execute_browser_action(self.env.page, executed)
        native = create_none_action()
        native["raw_prediction"] = json.dumps(executed)
        self.trajectory.append(native)
        observation, _, _, _, info = self.env.step(create_none_action())
        self.trajectory.append({"observation": observation, "info": info})
        self.screenshot()
        self.record_action(executed, url=self.env.page.url)
        return {"url": self.env.page.url}

    def evaluate(self) -> dict[str, Any]:
        from browser_env.actions import create_stop_action
        from evaluation_harness.evaluators import evaluator_router

        trajectory = [*self.trajectory, create_stop_action(self.final_answer)]
        evaluator = evaluator_router(str(self.config_file))
        score = evaluator(
            trajectory=trajectory,
            config_file=str(self.config_file),
            page=self.env.page,
            client=self.env.get_page_client(self.env.page),
        )
        return self.verdict(float(score))

    def close(self) -> None:
        try:
            self.env.close()
        finally:
            self._sites("down", "--volumes", "--remove-orphans")


def create_runtime(
    profile: dict[str, Any], settings: dict[str, Any], tasks: list[dict[str, Any]]
) -> WebArenaRuntime:
    return WebArenaRuntime(profile, settings, tasks)
