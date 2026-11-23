import base64
import json
import os
import shlex
import socket
import sys
import uuid
from contextlib import ExitStack
from pathlib import Path
from typing import Any

from mai_gym.envs.shared.desktop import DesktopRuntime
from mai_gym.envs.shared.provenance import PreparationError
from mai_gym.envs.shared.tasks import task_file


def isolated_ports() -> dict[str, int]:
    names = ["API", "VNC", "SSH", *[f"APP_{port}" for port in range(3001, 3019)]]
    result = {}
    with ExitStack() as stack:
        for name in names:
            listener = stack.enter_context(socket.socket())
            listener.bind(("0.0.0.0", 0))
            result[name] = listener.getsockname()[1]
    return result


class MyPCBenchRuntime(DesktopRuntime):
    def __init__(
        self, profile: dict[str, Any], settings: dict[str, Any], tasks: list[dict[str, Any]]
    ) -> None:
        super().__init__(profile, settings, tasks)
        if any(
            os.environ.get(key)
            for key in ("MYPCBENCH_RUBRIC_JUDGE_MOCK_SCORE", "MYPCBENCH_RUBRIC_JUDGE_MOCK_RESULT")
        ):
            raise PreparationError(
                "Native MyPCBench evaluation requires a real judge; unset mock judge variables"
            )
        self.ports = isolated_ports()
        os.environ["MYPCBENCH_REUSE_CONTAINER"] = "0"
        for key, value in self.ports.items():
            os.environ[
                f"MYPCBENCH_HOST_{key}_PORT"
                if key in {"API", "VNC", "SSH"}
                else f"MYPCBENCH_HOST_APP_PORT_{key.removeprefix('APP_')}"
            ] = str(value)
        sys.path.insert(0, str(Path(profile["sources"]["runtime"]["path"]) / "agent-harness"))
        from env import MyPCBenchEnv

        options = dict(settings["desktop"])
        options["container_name"] = f"mai-gym-pc-{uuid.uuid4().hex}"
        self.env = MyPCBenchEnv(**options)
        self.native_task: dict[str, Any] | None = None

    def initialize(self, task_id: str) -> str:
        task = self.begin(task_id)
        self.native_task = (
            dict(task)
            if task.get("instruction") and task.get("grading")
            else json.loads(task_file(self.profile, task).read_text())
        )
        observation = self.env.reset(task_config=self.native_task, soft=False)
        self.write_json(
            "vm.json", {"ports": self.ports, "overlay": getattr(self.env, "_overlay_path", None)}
        )
        return str(observation.get("instruction") or self.env.instruction)

    def record_action(self, action: dict[str, Any], result: Any = None, *, url: str = "") -> None:
        super().record_action(action, result, url=url)
        row = {
            "step_num": len(self.actions),
            "action": action,
            "response": "",
            "reward": (result or {}).get("reward"),
            "done": (result or {}).get("done"),
            "info": (result or {}).get("info"),
            "screenshot_file": self.actions[-1]["screenshot"],
        }
        with (self.episode_path / "traj.jsonl").open("a") as stream:
            stream.write(json.dumps(row, default=str) + "\n")

    def evaluate(self) -> dict[str, Any]:
        if self.native_task is None:
            raise RuntimeError("MyPCBench has no initialized native task")
        from judge_results import judge_one
        from utils.rubric_judge import build_rubric_bundle

        grading = self.native_task.get("grading", {})
        rubrics = grading.get("rubrics")
        if not rubrics or grading.get("checks"):
            raise ValueError("MyPCBench native rubric profile requires rubric-only task grading")
        encoded = base64.b64encode(self.final_answer.encode()).decode()
        self.env._execute_shell(
            f"printf %s {encoded} | base64 -d > /tmp/mypcbench_agent_response.txt"
        )
        self.write_json("final_answer.json", {"answer": self.final_answer})
        manifest = {
            "task_id": self.native_task.get("id"),
            "category": self.native_task.get("category"),
            "grading_type": grading.get("type"),
            "has_programmatic_checks": False,
            "has_rubrics": True,
            "checks": [],
            "rubrics": rubrics,
        }
        build_rubric_bundle(
            save_dir=str(self.episode_path),
            task_dict=self.native_task,
            grading_manifest=manifest,
            task_language="en",
            env_language="en",
            snapshot_name=None,
            persona=self.env.persona,
            world=self.env.world,
            vm_id=self.env.persona,
            actor_id=None,
        )
        judge = self.settings["judge"]
        native_script = (
            Path(self.profile["sources"]["evaluator"]["path"])
            / "agent-harness/utils/osworld_full_traj_judge.py"
        )
        command = shlex.join([sys.executable, str(native_script)])
        updates = {
            "OPENAI_API_KEY": os.environ[judge["api_key_env"]],
            "OPENAI_BASE_URL": judge["base_url"],
            "MYPCBENCH_RUBRIC_JUDGE_MODEL": judge["model"],
        }
        previous = {key: os.environ.get(key) for key in updates}
        try:
            os.environ.update(updates)
            result = judge_one(self.episode_path, command, judge.get("timeout", 300), True)
        finally:
            for key, value in previous.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value
        if result.get("status") != "judged" or result.get("score") is None:
            raise RuntimeError("MyPCBench rubric judge failed; native task remains ungraded")
        return self.verdict(result["score"], detail=result)

    def terminal(self, action: dict[str, Any]) -> None:
        super().terminal(action)
        row = {
            "step_num": len(self.actions) + 1,
            "action": {"action_type": "answer", "text": self.final_answer},
            "response": self.final_answer,
            "done": True,
            "screenshot_file": f"screenshot{max(0, self._screenshot_index - 1)}.png",
        }
        with (self.episode_path / "traj.jsonl").open("a") as stream:
            stream.write(json.dumps(row) + "\n")

    def close(self) -> None:
        self.env.close()


def create_runtime(
    profile: dict[str, Any], settings: dict[str, Any], tasks: list[dict[str, Any]]
) -> MyPCBenchRuntime:
    return MyPCBenchRuntime(profile, settings, tasks)
