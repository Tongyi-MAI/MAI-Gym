from __future__ import annotations

from pathlib import Path

from jinja2 import Environment, FileSystemLoader, StrictUndefined

_ENV = Environment(
    loader=FileSystemLoader(str(Path(__file__).parent)),
    autoescape=False,
    undefined=StrictUndefined,
    keep_trailing_newline=True,
)


def render(template_name: str, /, **context: object) -> str:

    return _ENV.get_template(template_name).render(**context)


__all__ = ["render"]
