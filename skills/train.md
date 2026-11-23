# Training setup for coding agents

[Setup index](README.md) · [Training documentation](../site/docs/raw/training.md) · [Environment guides](envs.md)

1. Use the prebuilt Slime image `ghcr.io/tongyi-mai/mai_gym_slime:v1`. The image supplies the training runtime; do not assemble a separate GPU dependency stack.
2. Follow the selected [environment guides](envs.md) to prepare managers, native verifiers, and registered task pools. Complete a single-task evaluation and reset verification for each environment.
3. Follow the [container quickstart](../site/docs/raw/training.md#start-the-container). Mount the checkout, model, tasks, and persistent output; adapt [configs/mai-gym-9b.yaml](../configs/mai-gym-9b.yaml) to paths and resources visible in the container. Keep resolved configuration outside the checkout and credentials in runtime environment variables.
4. For a training run, connect to the configured Ray cluster and run `mai-gym train --config /data/train.yaml` inside the container.
5. Return the image reference and identity, resolved configuration, launch and resume commands, output paths, verification logs, and any missing prerequisites. For a completed run, include the saved checkpoint path.
