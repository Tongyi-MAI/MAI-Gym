# MacAgentBench preparation

[All environments](../envs.md)

Pinned official source and asset identities: [release descriptor](../../src/mai_gym/envs/macagentbench/release.json). This descriptor is separate from a paper profile; the local lock must also identify all required runtime assets.

## Sources and immutable assets

Official source: [MacAgentBench](https://github.com/JetAstra/MacAgentBench). Pin source and native checkpoint evaluators, macOS/BaseSystem assets, container image ID and all parameterized task files. Export the selected native task inventory and preserve every variant identity.

Use the [profile lock schema](../envs.md#immutable-profiles). Required source/asset pins must come from the selected evaluation profile. A missing pin or hash mismatch is a preparation failure.

## Host and dependencies

KVM-capable VM host, a pinned macOS guest/runtime image, SSH access to the guest and sufficient memory/disk for its private writable image.

Install the selected upstream revision in its own checkout/interpreter:

```bash
git clone https://github.com/JetAstra/MacAgentBench.git /opt/sources/macagentbench
cd /opt/sources/macagentbench
git checkout "$RECORDED_REVISION"
python3.12 -m venv /opt/venvs/macagentbench
/opt/venvs/macagentbench/bin/pip install paramiko astor beautifulsoup4 numpy pillow python-dateutil pytz requests fastapi uvicorn click pydantic pyyaml httpx openai
```

Set `PYTHONPATH=/opt/mai-gym/src:/opt/sources/macagentbench` for the worker. The framework manager/runner uses its own environment (`uv sync` in the framework checkout). Keep all API keys in runtime environment variables. `HF_TOKEN` is needed only for gated asset downloads; judge credentials are referenced by `judge.api_key_env`.

## Native preparation

Set MAC_AGENT_BENCH_MAC_HDD_IMG_PATH and MAC_AGENT_BENCH_BASESYSTEM_IMG_PATH to verified read-only source images. Configure a dedicated free guest SSH port in macos_config. The native launcher makes an owned container; MacOSEnv handles task setup and native checkpoints.

Worker settings (save with real nonsecret paths):

```json
{
  "profile_lock": "/opt/profiles/macagentbench-lock.json",
  "evidence_dir": "/opt/evidence/macagentbench",
  "macos_config": "/opt/profiles/macagentbench-native.yaml",
  "vm": {
    "image": "<official-image>@sha256:<digest>",
    "disk": "/opt/assets/mac_hdd_ng.img",
    "base_system": "/opt/assets/BaseSystem.img"
  }
}
```

Validate the profile before provisioning:

```bash
python -m mai_gym.envs.prepare --env macagentbench --config /opt/profiles/macagentbench-worker.json
```

## Canonical manager configuration

```yaml
devices:
- name: macagentbench-worker
  env_type: macagentbench
  device_type: computer
  infra_type: local
  connection: local
  max_sessions: 1
  sessions_per_backend: 1
  launch:
    command: /opt/venvs/macagentbench/bin/python -m mai_gym.envs.shared.worker --env
      macagentbench --config /opt/profiles/macagentbench-worker.json --port {port}
    port: 16800
  env_config:
    profile_lock: /opt/profiles/macagentbench-lock.json
```

Supported infrastructure: Local/SSH native VM worker, Docker/KVM runtime controlled by the worker, or attach.

The native YAML selects guest SSH credentials and an unused local SSH port. The worker creates an owned container through the generic Docker lifecycle and uses the pinned upstream entrypoint to create a fresh QCOW2 overlay. Record the Docker image and both read-only base files in the profile lock. Native SSH-readiness and screen-recording preparation hooks run before native task initialization; upstream agent loops are not imported.

## Readiness, reset and verification

Wait for guest SSH and a fresh screenshot. Verify native screen-recording/input permissions and task-specific application state before the first model action.

Use the [verification protocol](../envs.md#verification-protocol) with one actual manifest task, one trial, and a budget of **50 interaction steps**. Use CLI benchmark `macagentbench`. Preserve the exact task identity and parameters. Require full native success; fractional credit is retained separately.

Initialize the same task again in a fresh episode and confirm restoration of its initial application state, absence of prior episode mutations and lease isolation. Run initial/completed/reset verification where the native evaluator supports state checks. Retain original failed attempts, evaluator errors and reset verification logs. Judge-based tasks use the native trajectory protocol rather than an invented state-only score.

## Cleanup

Release the session lease, tear down the native task and stop the worker. Remove only its owned containers, private VM overlays, browser profiles and temporary volumes. Preserve the pinned read-only base assets and trajectory logs. Never prune shared Docker resources. Check disk headroom before starting the next heavyweight environment.
