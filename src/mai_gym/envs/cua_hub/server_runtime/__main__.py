from __future__ import annotations

import contextlib
import json
import logging
import os
import subprocess
import time
from pathlib import Path

import click
import uvicorn

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


_NGINX_TEMPLATE = """\
worker_processes  1;
events { worker_connections 256; }
http {
    sendfile on;
    proxy_http_version 1.1;
    server {
        listen 7000 default_server;
        server_name _;
__LOCATIONS__
    }
}
"""

_LOCATION_TEMPLATE = """\
        location /__SLUG__/ {
            proxy_pass http://127.0.0.1:__PORT__/;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
        }
"""


def _resolve_enabled(
    enabled_csv: str | None,
    env_csv: str | None,
    ports: dict[str, int],
) -> dict[str, int]:
    raw = enabled_csv or env_csv or ""
    if not raw.strip():
        return dict(ports)
    requested = [s.strip() for s in raw.split(",") if s.strip()]
    out: dict[str, int] = {}
    for slug in requested:
        if slug in ports:
            out[slug] = ports[slug]
        else:
            alt = f"{slug}_mock"
            if alt in ports:
                out[alt] = ports[alt]
            else:
                logger.warning("unknown mock slug %r — skipping", slug)
    if not out:
        logger.warning("no enabled mocks matched; falling back to all")
        return dict(ports)
    return out


def _render_nginx(enabled: dict[str, int], dest: Path) -> None:
    locations = "".join(
        _LOCATION_TEMPLATE.replace("__SLUG__", slug).replace("__PORT__", str(port))
        for slug, port in enabled.items()
    )
    dest.write_text(_NGINX_TEMPLATE.replace("__LOCATIONS__", locations))


def _start_vite_previews(
    enabled: dict[str, int],
    hub_root: Path,
) -> list[subprocess.Popen]:
    procs: list[subprocess.Popen] = []
    for slug, port in enabled.items():
        cwd = hub_root / "websites" / slug
        if not (cwd / "dist").is_dir():
            logger.warning("dist/ missing for %s — preview will 404", slug)
        cmd = [
            "npx",
            "vite",
            "preview",
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
            "--outDir",
            "dist",
        ]
        logger.info("starting %s on :%d", slug, port)
        procs.append(subprocess.Popen(cmd, cwd=str(cwd)))
    return procs


def _start_nginx(conf_path: Path) -> subprocess.Popen:
    return subprocess.Popen(
        ["nginx", "-c", str(conf_path), "-g", "daemon off;"],
    )


@click.command()
@click.option("--port", default=6800, type=int, help="Public FastAPI port.")
@click.option("--host", default="0.0.0.0", help="Bind address.")
@click.option("--hub-root", required=True, type=click.Path(exists=True, file_okay=False))
@click.option("--ports-json", required=True, type=click.Path(exists=True, dir_okay=False))
@click.option("--tasks-root", default="/app/hub_tasks", type=str)
@click.option("--num-pages", default=8, type=int)
@click.option(
    "--enabled-apps",
    default=None,
    help="Comma-separated mock slugs to start; overrides CUA_HUB_APPS env var.",
)
@click.option(
    "--nginx-conf",
    default="/etc/nginx/cua_hub.conf",
    type=click.Path(),
    help="Where to write the rendered nginx config.",
)
@click.option(
    "--viewport-width",
    default=1280,
    type=int,
    help="Playwright viewport width (CSS px). Match the agent's screen_size.",
)
@click.option(
    "--viewport-height",
    default=800,
    type=int,
    help="Playwright viewport height (CSS px). Match the agent's screen_size.",
)
@click.option(
    "--device-scale-factor",
    default=1.0,
    type=float,
    help="Playwright DPR; screenshots come back at viewport × DSF pixels.",
)
def main(
    port: int,
    host: str,
    hub_root: str,
    ports_json: str,
    tasks_root: str,
    num_pages: int,
    enabled_apps: str | None,
    nginx_conf: str,
    viewport_width: int,
    viewport_height: int,
    device_scale_factor: float,
) -> None:

    ports_data = json.loads(Path(ports_json).read_text())
    ports: dict[str, int] = ports_data.get("mocks", ports_data)
    enabled = _resolve_enabled(
        enabled_apps,
        os.environ.get("CUA_HUB_APPS"),
        ports,
    )
    logger.info("enabled mocks: %s", sorted(enabled.keys())[:5] + ["..."])

    nginx_conf_path = Path(nginx_conf)
    nginx_conf_path.parent.mkdir(parents=True, exist_ok=True)
    _render_nginx(enabled, nginx_conf_path)

    vite_procs = _start_vite_previews(enabled, Path(hub_root))
    nginx_proc = _start_nginx(nginx_conf_path)

    time.sleep(2.0)

    from .server import create_cua_hub_app

    app = create_cua_hub_app(
        tasks_root=tasks_root,
        num_pages=num_pages,
        server_port=port,
        viewport_size=(viewport_width, viewport_height),
        device_scale_factor=device_scale_factor,
    )

    try:
        uvicorn.run(app, host=host, port=port)
    finally:
        logger.info("shutting down")
        for p in [nginx_proc, *vite_procs]:
            with contextlib.suppress(Exception):
                p.terminate()


if __name__ == "__main__":
    main()
