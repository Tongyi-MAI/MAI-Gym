__all__ = ["EnvController", "create_app"]


def __getattr__(name: str):
    if name == "create_app":
        from mai_gym.server.app import create_app

        return create_app
    if name == "EnvController":
        from mai_gym.server.controller import EnvController

        return EnvController
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
