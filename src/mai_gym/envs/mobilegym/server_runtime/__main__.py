from __future__ import annotations

import json
from pathlib import Path

import click
import uvicorn


@click.command()
@click.option("--port", default=6800, type=int, help="Port to bind.")
@click.option("--host", default="0.0.0.0", help="Bind address.")
@click.option("--profile-config", type=click.Path(exists=True, dir_okay=False), default=None)
@click.option("--static-port", type=int, default=None)
@click.option(
    "--static-dir",
    required=True,
    type=click.Path(exists=True, file_okay=False),
    help="Directory containing the built MobileGym SPA (Vite dist/).",
)
@click.option(
    "--media-dir",
    default=None,
    type=click.Path(exists=True, file_okay=False, resolve_path=True),
    help="Extracted companion media, served at /cdn (profile: verified media tree).",
)
@click.option(
    "--num-pages",
    default=8,
    type=int,
    help="Number of Playwright pages in the EnvPool — equals sessions_per_backend.",
)
@click.option(
    "--viewport-w",
    default=360,
    type=int,
    help="Logical viewport width.",
)
@click.option(
    "--viewport-h",
    default=800,
    type=int,
    help="Logical viewport height.",
)
@click.option(
    "--delay-after-action",
    default=0.8,
    type=float,
    help="Seconds to sleep after each action before screenshot.",
)
def main(
    port: int,
    host: str,
    static_dir: str,
    media_dir: str | None,
    num_pages: int,
    viewport_w: int,
    viewport_h: int,
    delay_after_action: float,
    profile_config: str | None,
    static_port: int | None,
) -> None:

    from mai_gym.envs.mobilegym.server_runtime.server import create_mobilegym_app

    manifest = None
    settings = {}
    profile = None
    if profile_config:
        if (viewport_w, viewport_h) != (360, 800):
            raise click.ClickException(
                "Profiled MobileGym uses --viewport-w 360 --viewport-h 800 (logical pixels); "
                "its native 3x scale produces 1080x2400 screenshots"
            )
        from mai_gym.envs.mobilegym.prepare import prepare
        from mai_gym.envs.mobilegym.tasks import load_tasks

        settings = json.loads(Path(profile_config).read_text())
        if media_dir is not None:
            configured = settings.get("media_dir")
            if configured is not None:
                lock_root = Path(settings["profile_lock"]).expanduser().resolve().parent
                if (lock_root / Path(configured).expanduser()).resolve() != Path(media_dir):
                    raise click.ClickException("--media-dir conflicts with profile media_dir")
            settings["media_dir"] = media_dir
        profile = prepare(settings)
        media_dir = profile["media_dir"]
        if not any(
            asset.get("kind") == "tree"
            and Path(asset["path"]).resolve() == Path(static_dir).resolve()
            for asset in profile["assets"]
        ):
            raise click.ClickException(
                "MobileGym --static-dir must match a verified frontend tree asset in the profile lock"
            )
        manifest = [
            {**entry, "seed": entry.get("seed", settings.get("seed", 0))}
            for entry in load_tasks(profile)
        ]

    app = create_mobilegym_app(
        static_dir=static_dir,
        media_dir=media_dir,
        num_pages=num_pages,
        server_port=port,
        viewport_size=(viewport_w, viewport_h),
        delay_after_action=delay_after_action,
        task_manifest=manifest,
        static_port=static_port,
        evidence_dir=settings.get("evidence_dir"),
        profile_identity=profile,
        framework_source_digest=settings.get("framework_source_digest"),
    )
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
