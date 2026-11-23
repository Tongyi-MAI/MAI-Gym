from dataclasses import dataclass


@dataclass(frozen=True)
class LaunchProfile:
    docker_factory: str = "mai_gym.infra.launcher.docker:DockerLauncher"
    k8s_factory: str | None = None
    agent_factory: str | None = None
    readiness_path: str = "/health"
    max_sessions_per_backend: int | None = None


def service_profile() -> LaunchProfile:
    return LaunchProfile()
