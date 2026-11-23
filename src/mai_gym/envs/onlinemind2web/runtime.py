import shutil
import sys
from pathlib import Path
from typing import Any

from mai_gym.envs.onlinemind2web.trajectory import action_text
from mai_gym.envs.shared.browser import BrowserRuntime
from mai_gym.envs.shared.judge import judge_client, run_async_judge


class OnlineMind2WebRuntime(BrowserRuntime):
    def initialize(self, task_id: str) -> str:
        task = self.tasks[task_id]
        self.begin_browser(task_id, task["website"])
        return str(task["confirmed_task"])

    def evaluate(self) -> dict[str, Any]:
        self.finish_exhausted_episode()
        if self.terminal_action is None:
            raise ValueError(
                "OnlineMind2Web requires a recorded terminal decision for native trajectory grading"
            )
        sys.path.insert(0, str(Path(self.profile["sources"]["evaluator"]["path"]) / "src"))
        from methods.webjudge_online_mind2web import WebJudge_Online_Mind2Web_eval
        from utils import OpenaiEngine, extract_predication

        model = OpenaiEngine(api_key="runtime-client", model=self.settings["judge"]["model"])
        model.client = judge_client(self.settings)
        trajectory = self.episode_path / "trajectory"
        trajectory.mkdir(exist_ok=True)
        steps = []
        for index, entry in enumerate(self.actions):
            filename = f"{index:04d}.png"
            shutil.copyfile(self.episode_path / entry["screenshot"], trajectory / filename)
            steps.append(
                {
                    "step": index,
                    "action": action_text(entry["action"]),
                    "thought": None,
                    "screenshot": filename,
                    "url": entry["url"],
                }
            )
        if not steps:
            raise ValueError("WebJudge needs factual executed actions and screenshots")
        filename = f"{len(steps):04d}.png"
        shutil.copyfile(self.episode_path / self.actions[-1]["screenshot"], trajectory / filename)
        steps.append(
            {
                "step": len(steps),
                "action": (
                    f"TASK_ABORT -> REASON: {self.terminal_action.get('reason', 'harness_abort')}"
                    if self.terminal_action.get("source") == "harness"
                    else "TASK_COMPLETE"
                )
                + (f" -> ANSWER: {self.final_answer}" if self.final_answer else ""),
                "thought": None,
                "screenshot": filename,
                "url": self.page.url,
            }
        )
        self.write_json(
            "result.json",
            {
                "schema_version": "online-mind2web-v2",
                "task_id": self.active_task["task_id"],
                "task": self.active_task["confirmed_task"],
                "agent_final_answer": self.final_answer,
                "reference_length": self.active_task.get("reference_length"),
                "action_history": steps,
            },
        )
        schema_path = (
            Path(self.profile["sources"]["evaluator"]["path"]) / "data/schema_v2/schema_v2.json"
        )
        if self.profile.get("label") == "official-latest":
            import json

            import jsonschema

            jsonschema.validate(
                json.loads((self.episode_path / "result.json").read_text()),
                json.loads(schema_path.read_text()),
            )
        paths = [str(trajectory / step["screenshot"]) for step in steps]
        actions = [step["action"] for step in steps]
        messages, text, system, record, points = run_async_judge(
            lambda: WebJudge_Online_Mind2Web_eval(
                self.active_task["confirmed_task"],
                actions,
                paths,
                model,
                self.settings["judge"].get("score_threshold", 3),
            )
        )
        self.write_json(
            "webjudge-input.json",
            {
                "input_text": text,
                "system": system,
                "key_points": points,
                "image_judge_record": record,
            },
        )
        response = model.generate(messages)[0]
        if "status:" not in response.lower():
            raise RuntimeError("WebJudge returned no native status")
        score = extract_predication(response, "WebJudge_Online_Mind2Web_eval")
        self.write_json(
            "webjudge.json",
            {
                "input_text": text,
                "system": system,
                "key_points": points,
                "image_judge_record": record,
                "response": response,
                "score": score,
            },
        )
        return self.verdict(float(score))


def create_runtime(
    profile: dict[str, Any], settings: dict[str, Any], tasks: list[dict[str, Any]]
) -> OnlineMind2WebRuntime:
    return OnlineMind2WebRuntime(profile, settings, tasks)
