from pathlib import Path

import click
import uvicorn
from fastapi import FastAPI
from starlette.staticfiles import StaticFiles


def resolve_media_directory(static_dir: str, media_dir: str | None) -> Path | None:
    directory = Path(media_dir).expanduser() if media_dir is not None else Path(static_dir) / "cdn"
    if directory.is_dir():
        return directory.resolve()
    if media_dir is not None:
        raise ValueError(f"MobileGym companion media directory is missing: {directory}")
    return None


def create_static_app(static_dir: str, media_dir: str | None = None) -> FastAPI:
    app = FastAPI()
    directory = resolve_media_directory(static_dir, media_dir)
    if directory is not None:
        app.mount("/cdn", StaticFiles(directory=str(directory)), name="cdn")
    app.mount("/", StaticFiles(directory=static_dir, html=True), name="frontend")
    return app


@click.command()
@click.option("--port", required=True, type=int)
@click.option("--static-dir", required=True, type=click.Path(exists=True, file_okay=False))
@click.option("--media-dir", default=None, type=click.Path(exists=True, file_okay=False))
def main(port: int, static_dir: str, media_dir: str | None) -> None:
    uvicorn.run(create_static_app(static_dir, media_dir), host="127.0.0.1", port=port)


if __name__ == "__main__":
    main()
