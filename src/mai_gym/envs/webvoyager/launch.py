from mai_gym.envs.shared.launch import LaunchProfile


def launch_profile(*, device: object | None = None) -> LaunchProfile:
    return LaunchProfile(max_sessions_per_backend=1)
