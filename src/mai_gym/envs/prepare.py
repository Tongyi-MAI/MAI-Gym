import json
from pathlib import Path

import click

from mai_gym.envs.registry import get_environment, load_component


@click.command()
@click.option("--env", "environment", required=True)
@click.option("--config", type=click.Path(exists=True, dir_okay=False), required=True)
def main(environment: str, config: str) -> None:
    settings = json.loads(Path(config).read_text())
    env = get_environment(environment)
    profile = load_component(env.preparation)(settings)
    tasks = load_component(env.task_loader)(profile)
    click.echo(
        json.dumps(
            {"environment": env.name, "profile_sha256": profile["lock_sha256"], "tasks": len(tasks)}
        )
    )


if __name__ == "__main__":
    main()
