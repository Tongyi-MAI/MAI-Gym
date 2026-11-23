import base64
import json
import sys
import time
import uuid
from pathlib import Path
from typing import Any

from mai_gym.envs.shared.judge import judge_identity
from mai_gym.models.evaluation import EvaluationScore, task_succeeded


class NativeRuntime:
    @property
    def episode_path(self) -> Path:
        if self.episode_dir is None:
            raise RuntimeError("No native episode initialized")
        return self.episode_dir

    @property
    def active_task(self) -> dict[str, Any]:
        if self.task is None:
            raise RuntimeError("No native task initialized")
        return self.task

    def __init__(
        self, profile: dict[str, Any], settings: dict[str, Any], tasks: list[dict[str, Any]]
    ) -> None:
        self.profile = profile
        self.settings = settings
        self.tasks = {task["task_id"]: task for task in tasks}
        self.task: dict[str, Any] | None = None
        self.episode_dir: Path | None = None
        self.actions: list[dict[str, Any]] = []
        self.final_answer = ""
        self.terminal_action: dict[str, Any] | None = None
        self.decision_count = 0
        self.verdicts: list[dict[str, Any]] = []
        self._screenshot_index = 0

    def task_metadata(self) -> list[dict[str, Any]]:
        fields = {
            "task_id",
            "goal",
            "instruction",
            "description",
            "max_steps",
            "category",
            "seed",
            "task_class",
        }
        return [
            {key: value for key, value in task.items() if key in fields}
            for task in self.tasks.values()
        ]

    def begin(self, task_id: str) -> dict[str, Any]:
        if task_id not in self.tasks:
            raise ValueError(f"Task {task_id!r} is absent from the pinned manifest")
        self.task = self.tasks[task_id]
        self.actions = []
        self.final_answer = ""
        self.terminal_action = None
        self.decision_count = 0
        self.verdicts = []
        self._screenshot_index = 0
        root = Path(self.settings.get("evidence_dir", "artifacts/native"))
        self.episode_dir = root / uuid.uuid4().hex
        self.episode_dir.mkdir(parents=True, exist_ok=False)
        self.write_json(
            "identity.json",
            {
                "task_id": task_id,
                "profile": self.profile,
                "judge": judge_identity(self.settings),
                "framework_source_digest": self.settings.get("framework_source_digest"),
                "worker_python": sys.version,
                "runtime": type(self).__module__,
            },
        )
        return self.task

    def write_json(self, name: str, data: Any) -> None:
        if self.episode_dir is None:
            raise RuntimeError("No native episode initialized")
        (self.episode_dir / name).write_text(json.dumps(data, indent=2, default=str) + "\n")

    def retain_screenshot(self, data: bytes) -> str:
        if self.episode_dir is not None:
            (self.episode_dir / f"screenshot{self._screenshot_index}.png").write_bytes(data)
            self._screenshot_index += 1
        return base64.b64encode(data).decode()

    def record_action(self, action: dict[str, Any], result: Any = None, *, url: str = "") -> None:
        entry = {
            "decision": self.decision_count,
            "step": len(self.actions),
            "time": time.time(),
            "action": action,
            "result": result,
            "url": url,
            "screenshot": f"screenshot{max(0, self._screenshot_index - 1)}.png",
        }
        self.actions.append(entry)
        if self.episode_dir is not None:
            with (self.episode_dir / "actions.jsonl").open("a") as stream:
                stream.write(json.dumps(entry, default=str) + "\n")

    def terminal(self, action: dict[str, Any]) -> None:
        self.terminal_action = action
        self.final_answer = str(action.get("text") or action.get("answer") or "")
        self.write_json("terminal.json", {"action": action, "final_answer": self.final_answer})

    def finish_exhausted_episode(self) -> None:
        budget = self.active_task.get("max_steps")
        if (
            self.terminal_action is None
            and budget is not None
            and self.decision_count >= int(budget)
        ):
            self.terminal(
                {
                    "action_type": "terminate",
                    "status": "failure",
                    "source": "harness",
                    "reason": "decision_budget_exhausted",
                }
            )

    def execute_decision(self, action: dict[str, Any]) -> dict[str, Any]:
        budget = self.active_task.get("max_steps")
        if budget is not None and self.decision_count >= int(budget):
            raise ValueError("Native task decision budget exhausted")
        self.decision_count += 1

        def execute(item: dict[str, Any]) -> dict[str, Any]:
            kind = item.get("action_type")
            if kind == "action_sequence":
                results = []
                for child in item["actions"]:
                    result = execute(child)
                    results.append(result)
                    if result.get("done"):
                        return {"done": True, "results": results}
                return {"done": False, "results": results}
            if kind in {"answer", "finished", "terminate"}:
                self.terminal(item)
                return {"done": True}
            return {"done": False, "result": self.act(item)}

        return execute(action)

    def verdict(
        self, reward: float, *, success: bool | None = None, detail: Any = None
    ) -> dict[str, Any]:
        if not 0 <= reward <= 1:
            raise ValueError("Native benchmark reward must be in [0, 1]")
        score = EvaluationScore(reward, success)
        result = {
            "score": float(score),
            "success": task_succeeded(score),
            "reason": json.dumps(detail, default=str),
            "evidence_dir": str(self.episode_dir),
        }
        self.write_json(f"verifier{len(self.verdicts)}.json", result)
        self.verdicts.append(result)
        self.write_json("verifier.json", result)
        return result

    def initialize(self, task_id: str) -> str:
        raise NotImplementedError

    def screenshot(self) -> str:
        raise NotImplementedError

    def act(self, action: dict[str, Any]) -> Any:
        raise NotImplementedError

    def evaluate(self) -> dict[str, Any]:
        raise NotImplementedError

    def tear_down(self) -> None:
        self.task = None

    def close(self) -> None:
        self.tear_down()

    def diagnostics(self) -> dict[str, Any]:
        return {
            "task_id": self.task.get("task_id") if self.task else None,
            "episode_dir": str(self.episode_dir) if self.episode_dir else None,
            "decision_count": self.decision_count,
            "executed_actions": len(self.actions),
            "final_answer_empty": not self.final_answer,
            "profile_label": self.profile.get("label"),
            "profile_sha256": self.profile.get("lock_sha256"),
        }
