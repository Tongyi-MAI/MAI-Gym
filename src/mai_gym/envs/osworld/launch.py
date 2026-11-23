from mai_gym.envs.shared.launch import LaunchProfile
from mai_gym.models.config import DeviceConfig


def launch_profile(*, device: DeviceConfig | None = None) -> LaunchProfile:
    protocol = device.env_config.get("runtime_protocol") if device is not None else None
    if protocol not in (None, "legacy", "native"):
        raise ValueError("OSWorld runtime_protocol must be native or legacy")
    if protocol == "native":
        if device is not None and device.env_config.get("profile") == "cua":
            raise ValueError("The OSWorld native worker cannot use the CUA bundle profile")
        return LaunchProfile(max_sessions_per_backend=1)
    return LaunchProfile(
        docker_factory="mai_gym.envs.osworld.launch_docker:OSWorldDockerProfile",
        k8s_factory="mai_gym.envs.osworld.launch_k8s:OSWorldK8sProfile",
        readiness_path="/screenshot",
    )
