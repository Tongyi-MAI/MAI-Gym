import click
import uvicorn


@click.command()
@click.option("--port", default=6800, type=int, help="Port to serve on.")
@click.option("--host", default="0.0.0.0", help="Bind address.")
@click.option(
    "--static-dir",
    required=True,
    type=click.Path(exists=True, file_okay=False),
    help="Directory containing the web app static files.",
)
@click.option(
    "--evaluators-dir",
    default=None,
    type=click.Path(exists=True, file_okay=False),
    help="Directory containing JS evaluator scripts.",
)
@click.option(
    "--tasks-file",
    default=None,
    type=click.Path(exists=True, dir_okay=False),
    help="JSON file listing available tasks.",
)
@click.option(
    "--task-suites-dir",
    default=None,
    type=click.Path(exists=True, file_okay=False),
    help="Root containing declarative <env>/tasks.json suites served from <env>/index.html.",
)
@click.option(
    "--entry-page",
    default="homepage.html",
    help="Entry HTML page filename (default: homepage.html).",
)
@click.option("--viewport-width", default=375, type=click.IntRange(min=1))
@click.option("--viewport-height", default=812, type=click.IntRange(min=1))
@click.option("--device-scale-factor", default=2.0, type=click.FloatRange(min=0.1))
def main(
    port: int,
    host: str,
    static_dir: str,
    evaluators_dir: str | None,
    tasks_file: str | None,
    task_suites_dir: str | None,
    entry_page: str,
    viewport_width: int,
    viewport_height: int,
    device_scale_factor: float,
) -> None:

    from mai_gym.envs.webenv.server import create_webenv_app

    app = create_webenv_app(
        static_dir=static_dir,
        evaluators_dir=evaluators_dir,
        tasks_file=tasks_file,
        task_suites_dir=task_suites_dir,
        entry_page=entry_page,
        server_port=port,
        viewport_width=viewport_width,
        viewport_height=viewport_height,
        device_scale_factor=device_scale_factor,
    )
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
