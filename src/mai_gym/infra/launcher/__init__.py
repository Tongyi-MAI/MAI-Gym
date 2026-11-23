from typing import cast

from mai_gym.envs.registry import get_environment, load_component
from mai_gym.infra.launcher._base import Launcher
from mai_gym.infra.launcher.attach import AttachLauncher
from mai_gym.infra.launcher.docker import DockerLauncher
from mai_gym.infra.launcher.local import LocalLauncher
from mai_gym.infra.launcher.ssh_command import SshCommandLauncher
from mai_gym.models.config import DeviceConfig

SUPPORTED_PAIRS = {
    (domain, infra)
    for domain in ("mobile", "computer", "web")
    for infra in ("local", "docker", "ssh", "attach")
} | {("mobile", "k8s"), ("computer", "k8s"), ("mobile", "agent")}


def create_launcher(device: DeviceConfig) -> Launcher:
    pair = (device.device_type, device.infra_type)
    if pair not in SUPPORTED_PAIRS:
        raise ValueError(f"Unsupported device/infrastructure combination: {pair}")
    environment = get_environment(device.env_type)
    profile = load_component(environment.launch_profile)(device=device)
    if (
        profile.max_sessions_per_backend is not None
        and device.effective_sessions_per_backend > profile.max_sessions_per_backend
    ):
        raise ValueError(
            f"{device.env_type} requires sessions_per_backend <= {profile.max_sessions_per_backend}; increase max_sessions to provision separate workers"
        )
    if device.infra_type == "attach":
        assert device.attach is not None
        return AttachLauncher(backends=device.attach.backends)
    if device.infra_type in {"agent", "k8s"}:
        factory = getattr(profile, f"{device.infra_type}_factory")
        if factory is None:
            raise ValueError(f"{device.env_type} has no {device.infra_type} launch profile")
        return cast(Launcher, load_component(factory)())
    if device.infra_type == "docker":
        return (
            cast(Launcher, load_component(profile.docker_factory)())
            if device.is_local
            else DockerLauncher()
        )
    if device.infra_type == "ssh":
        return SshCommandLauncher()
    return LocalLauncher()


__all__ = ["Launcher", "create_launcher", "SUPPORTED_PAIRS"]
