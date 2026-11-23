from __future__ import annotations

import os


def referenced_environment(value: object) -> dict[str, str]:
    result = {}
    if isinstance(value, dict):
        for item in value.values():
            result.update(referenced_environment(item))
    elif isinstance(value, list):
        for item in value:
            result.update(referenced_environment(item))
    elif isinstance(value, str) and value.startswith("env/"):
        name = value.removeprefix("env/").removeprefix("int/")
        if not name or not os.environ.get(name):
            raise ValueError(f"Adapter configuration requires environment variable {name}")
        result[name] = os.environ[name]
    return result


def resolve_environment(value: object):
    if isinstance(value, dict):
        return {key: resolve_environment(item) for key, item in value.items()}
    if isinstance(value, list):
        return [resolve_environment(item) for item in value]
    if isinstance(value, str) and value.startswith("env/"):
        text = next(iter(referenced_environment(value).values()))
        if value.startswith("env/int/"):
            try:
                return int(text)
            except ValueError:
                raise ValueError(f"{value[8:]} must be an integer") from None
        return text
    return value
