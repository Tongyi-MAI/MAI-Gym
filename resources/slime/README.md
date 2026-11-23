# Bundled Slime training runtime

This directory contains Slime 0.2.4 from the MAI-Gym training integration,
including the runtime, model plugins, model configuration scripts, and upstream
Docker build recipes and patches. The public upstream project is
[THUDM/slime](https://github.com/THUDM/slime).

Run `mai-gym train --config /data/train.yaml` inside the MAI-Gym training image.
The launcher adds this source tree to the Python path. [Training quick start](../../README.md#4-train-a-gui-agent-with-online-rl-optional)

To install the trainer explicitly
inside an already provisioned GPU environment, run from this directory:

```bash
python -m pip install -e . --no-deps
```

The dependency list is in `requirements.txt`. CUDA extensions, Megatron-LM,
Megatron Bridge, and SGLang must be compatible with the model and hardware.
The Docker recipes provide dependency build instructions; a dependency-image
build is separate from validating the complete GUI-agent training setup.
The main NVIDIA recipe installs this bundled source. From this directory:

```bash
docker build -f docker/Dockerfile -t mai-gym-training:local .
```

Check the recipe's Megatron/SGLang versions and patches against your model and
context-parallel layout before using the image for training.

Global batch size counts flattened step/window samples. Rollout batch size
counts tasks, with multiple trajectories and training samples possible per task.

The Apache 2.0 license is preserved in `LICENSE`; additional attribution is in `NOTICE`
and source headers. Example datasets, deployment launchers, development-agent
files, and experiment artifacts are not bundled here.
