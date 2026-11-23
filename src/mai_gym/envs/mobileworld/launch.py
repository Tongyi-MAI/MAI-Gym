from mai_gym.envs.shared.launch import LaunchProfile


def launch_profile(*, device: object | None = None) -> LaunchProfile:
    return LaunchProfile(
        docker_factory="mai_gym.envs.mobileworld.launch_docker:MobileWorldDockerProfile",
        k8s_factory="mai_gym.envs.mobileworld.launch_k8s:MobileWorldK8sProfile",
        agent_factory="mai_gym.infra.launcher.agent:MobileAgentLauncher",
    )
