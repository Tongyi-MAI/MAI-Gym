from io import BytesIO
from pathlib import Path
from typing import Any

from mai_gym.envs.shared.runtime import NativeRuntime


class AndroidWorldRuntime(NativeRuntime):
    def __init__(
        self, profile: dict[str, Any], settings: dict[str, Any], tasks: list[dict[str, Any]]
    ) -> None:
        super().__init__(profile, settings, tasks)
        from android_world.env import env_launcher

        if settings.get("a11y_forwarder_apk"):
            from android_env.wrappers import a11y_grpc_wrapper

            apk = Path(settings["a11y_forwarder_apk"]).expanduser().resolve()

            def forwarder_bytes() -> bytes:
                return apk.read_bytes()

            a11y_grpc_wrapper._get_accessibility_forwarder_apk = forwarder_bytes

        self.env = env_launcher.load_and_setup_env(
            console_port=settings.get("console_port", 5554),
            grpc_port=settings.get("grpc_port", 8554),
            adb_path=settings["adb_path"],
            emulator_setup=settings.get("emulator_setup", False),
            freeze_datetime=True,
        )
        self.native_task: Any = None

    def initialize(self, task_id: str) -> str:
        from android_world import registry

        task = self.begin(task_id)
        task_class = registry.TaskRegistry().get_registry(family="android_world")[
            task.get("task_class", task_id)
        ]
        self.native_task = task_class(task["params"])
        self.native_task.initialize_task(self.env)
        self.write_json("parameters.json", self.native_task.params)
        return str(self.native_task.goal)

    def screenshot(self) -> str:
        from PIL import Image

        buffer = BytesIO()
        Image.fromarray(self.env.get_state(wait_to_stabilize=True).pixels).save(
            buffer, format="PNG"
        )
        return self.retain_screenshot(buffer.getvalue())

    def act(self, action: dict[str, Any]) -> None:
        from android_world.env.json_action import JSONAction

        if action.get("action_type") == "drag":
            from android_env.proto import adb_pb2
            from android_world.env import adb_utils

            coordinates = [action[key] for key in ("start_x", "start_y", "end_x", "end_y")]
            if self.settings.get("coordinate_space", "pixels") == "normalized":
                width, height = self.env.logical_screen_size
                coordinates = [
                    round(float(value) * (size - 1))
                    for value, size in zip(coordinates, (width, height, width, height), strict=True)
                ]
            coordinates = [int(value) for value in coordinates]
            command = adb_utils.generate_swipe_command(*coordinates, duration_ms=600)
            response = adb_utils.issue_generic_request(command, self.env.controller)
            if response.status != adb_pb2.AdbResponse.Status.OK:
                raise RuntimeError("AndroidWorld native controller rejected the drag gesture")
            self.record_action(
                {
                    "action_type": "drag",
                    **dict(zip(("start_x", "start_y", "end_x", "end_y"), coordinates, strict=True)),
                }
            )
            return

        allowed = {
            "action_type",
            "index",
            "x",
            "y",
            "text",
            "direction",
            "goal_status",
            "app_name",
            "keycode",
            "clear_text",
        }
        native = {key: value for key, value in action.items() if key in allowed}
        if native.get("action_type") == "type":
            native["action_type"] = "input_text"
        if self.settings.get("coordinate_space", "pixels") == "normalized":
            pixels = self.env.get_state().pixels
            for key, size in (("x", pixels.shape[1]), ("y", pixels.shape[0])):
                if key in native:
                    native[key] = round(float(native[key]) * (size - 1))
        self.env.execute_action(JSONAction(**native))
        self.record_action(native)

    def terminal(self, action: dict[str, Any]) -> None:
        from android_world.env.json_action import JSONAction

        super().terminal(action)
        if self.final_answer:
            native = {"action_type": "answer", "text": self.final_answer}
            self.env.execute_action(JSONAction(**native))
            self.record_action(native)

    def evaluate(self) -> dict[str, Any]:
        return self.verdict(float(self.native_task.is_successful(self.env)))

    def tear_down(self) -> None:
        if self.native_task is not None:
            self.native_task.tear_down(self.env)
            self.native_task = None
        super().tear_down()

    def close(self) -> None:
        self.tear_down()
        self.env.close()


def create_runtime(
    profile: dict[str, Any], settings: dict[str, Any], tasks: list[dict[str, Any]]
) -> AndroidWorldRuntime:
    return AndroidWorldRuntime(profile, settings, tasks)
