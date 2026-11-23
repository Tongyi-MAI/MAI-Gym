# Setup guides for coding agents

Open the MAI-Gym checkout in Claude Code, Codex, Cursor or another coding agent that can read files and run commands. Pick a prompt below and fill in the resources the agent may use. Give it credential environment-variable names and keep the values in your shell. For manual setup, start with [Quickstart](../site/docs/raw/index.md).

| Goal | Start here |
| --- | --- |
| Prepare a benchmark backend, its assets and native verifier | [Environment setup](envs.md), then the linked benchmark guide |
| Configure a model endpoint and run a benchmark evaluation | [Evaluation setup](eval.md) |
| Use the prebuilt Slime image and prepare task pools and managers | [Training setup](train.md) |
| Evaluate or explore Mocked Apps with the CLI | [Mocked Apps guide](../site/docs/raw/mocked-apps.md) |

## Prepare an evaluation

```text
Follow skills/eval.md for one MobileWorld task. My backend is <backend URL>.
Use <served model ID> at <API base URL> with the key in OPENAI_API_KEY.
Save my config and results in <local directory>. Run the task, check its
native evaluation score and reset, and show me the report and trajectory.
```

## Prepare training

```text
Follow skills/train.md on <GPU host> using the prebuilt Slime image.
My checkpoint is <path>, my task pools are <paths>, and their managers
are at <URLs>. Write the config in <local directory> with rollout.rounds: 1
and trainer.save_interval: 1. Run one training round and show me the checkpoint path.
```
