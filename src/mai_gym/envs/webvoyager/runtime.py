import importlib.util
from pathlib import Path
from typing import Any

from mai_gym.envs.shared.browser import BrowserRuntime
from mai_gym.envs.shared.judge import judge_client


class WebVoyagerRuntime(BrowserRuntime):
    def initialize(self, task_id: str) -> str:
        task = self.tasks[task_id]
        self.begin_browser(task_id, task["web"])
        return str(task["ques"])

    def evaluate(self) -> dict[str, Any]:
        self.finish_exhausted_episode()
        if self.terminal_action is None:
            raise ValueError("WebVoyager requires a recorded terminal decision before grading")
        self.write_json(
            "interact_messages.json",
            [
                {"role": "system", "content": ""},
                {
                    "role": "user",
                    "content": f"Now given a task: {self.active_task['ques']} Please interact with the web page.",
                },
                {
                    "role": "assistant",
                    "content": (
                        f"Action: ANSWER; [{self.final_answer}]"
                        if self.final_answer
                        else f"Action: FAIL; {self.terminal_action.get('reason', 'harness_abort')}"
                        if self.terminal_action.get("source") == "harness"
                        else "Action: DONE"
                    ),
                },
            ],
        )
        path = Path(self.profile["sources"]["evaluator"]["path"]) / "evaluation/auto_eval.py"
        spec = importlib.util.spec_from_file_location("webvoyager_native_evaluator", path)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"Cannot load the pinned WebVoyager evaluator: {path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        result = module.auto_eval_by_gpt4v(
            str(self.episode_path),
            judge_client(self.settings),
            self.settings["judge"]["model"],
            self.settings["judge"].get("image_count", 1),
        )
        if result is None:
            raise RuntimeError("WebVoyager native judge returned no verdict")
        return self.verdict(float(result))


def create_runtime(
    profile: dict[str, Any], settings: dict[str, Any], tasks: list[dict[str, Any]]
) -> WebVoyagerRuntime:
    return WebVoyagerRuntime(profile, settings, tasks)
