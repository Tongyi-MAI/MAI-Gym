# Evaluation setup for coding agents

[Setup index](README.md) · [Evaluation documentation](../site/docs/raw/evaluation.md) · [Environment guides](envs.md)

1. Read the [evaluation guide](../site/docs/raw/evaluation.md) and [envs.md](envs.md), then follow the selected benchmark's preparation guide.
2. Inspect the available host, model endpoint or checkpoint, assets, and credential variable names. Write resolved configuration outside the checkout and preserve source/asset pins.
3. Run [one benchmark task](../site/docs/raw/evaluation.md#2-run-one-task) with `mai-gym eval`. Require native full success and confirm reset and episode isolation with the [verification protocol](envs.md#verification-protocol). Retain failed attempts and verification logs.
4. Return the configuration paths, credential variable names, pinned runtime identities, verification and reset results, trajectory artifacts, and the exact evaluation command. Report missing prerequisites explicitly. Expand beyond single-task verification only when requested.

For the bundled browser-based Mocked Apps (42 Novel Apps), use `--benchmarks mocked-apps` with `configs/mocked-apps.yaml`. [Mocked Apps evaluation](../site/docs/raw/mocked-apps.md#evaluate-a-task)
