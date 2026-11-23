from pathlib import Path
from typing import Any

from mai_gym.envs.macagentbench.prepare import native_preparation_hooks
from mai_gym.envs.shared.desktop import DesktopRuntime
from mai_gym.envs.shared.provenance import PreparationError
from mai_gym.envs.shared.tasks import task_file
from mai_gym.infra.launcher.docker import DockerLauncher
from mai_gym.models.config import DeviceConfig, DockerLaunchConfig


class MacAgentBenchRuntime(DesktopRuntime):
    def __init__(
        self, profile: dict[str, Any], settings: dict[str, Any], tasks: list[dict[str, Any]]
    ) -> None:
        super().__init__(profile, settings, tasks)
        from controllers.env import MacOSEnv

        self.env = MacOSEnv(config_file=settings["macos_config"])
        self.hooks = native_preparation_hooks(Path(profile["sources"]["runtime"]["path"]), MacOSEnv)
        self.launcher = DockerLauncher()
        self.container_name: str | None = None
        vm = settings.get("vm", {})
        image = vm.get("image", "")
        if "@sha256:" not in image:
            raise PreparationError("MacAgentBench vm.image requires a pinned Docker digest")
        self.vm_image = image
        source = Path(profile["sources"]["runtime"]["path"])
        self.device = DeviceConfig(
            name="macos-vm",
            type="macagentbench",
            env_type="macagentbench",
            device_type="computer",
            infra_type="docker",
            connection="local",
            launch=DockerLaunchConfig(
                image=image,
                container_port=10022,
                reuse_existing=False,
                env_vars={
                    "EXTRA": "-display none -vnc 0.0.0.0:1,password=off",
                    "CPU": "Haswell-noTSX",
                    "CPUID_FLAGS": "kvm=on,vendor=GenuineIntel,+invtsc,vmware-cpuid-freq=on",
                    "SHORTNAME": "sonoma",
                    "USERNAME": self.env.username,
                    "PASSWORD": self.env.password,
                },
                volumes=[
                    f"{vm['disk']}:/home/arch/OSX-KVM/mac_hdd_ng_src.img:ro",
                    f"{vm['base_system']}:/home/arch/OSX-KVM/BaseSystem_src.img:ro",
                    f"{source}/launcher/docker/custom_entrypoint.sh:/home/arch/OSX-KVM/entrypoint.sh:ro",
                ],
            ),
        )

    def initialize(self, task_id: str) -> str:
        task = self.begin(task_id)
        self.env.close_connection()
        if self.container_name is not None:
            self.launcher.stop_owned_container(self.device, self.container_name)
        self.container_name = self.launcher.start_owned_container(
            self.device, self.env.port, extra_options=("--device", "/dev/kvm")
        )
        self.env.docker_name = self.container_name
        self.hooks["wait_for_ssh"](self.env)
        self.hooks["_fix_screen_recording_permission"](self.env)
        self.write_json(
            "vm.json",
            {
                "container_name": self.container_name,
                "image": self.vm_image,
                "ssh_port": self.env.port,
            },
        )
        self.env.init_task(str(task_file(self.profile, task)))
        return str(self.env.task.instruction)

    def evaluate(self) -> dict[str, Any]:
        result = self.env.evaluate_task()
        if isinstance(result, bool):
            return self.verdict(float(result), success=result)
        return self.verdict(
            float(result),
            success=bool(result),
            detail=result.to_result_txt(steps_used=self.decision_count, max_steps=50),
        )

    def close(self) -> None:
        self.env.close_connection()
        if self.container_name is not None:
            self.launcher.stop_owned_container(self.device, self.container_name)
            self.container_name = None


def create_runtime(
    profile: dict[str, Any], settings: dict[str, Any], tasks: list[dict[str, Any]]
) -> MacAgentBenchRuntime:
    return MacAgentBenchRuntime(profile, settings, tasks)
