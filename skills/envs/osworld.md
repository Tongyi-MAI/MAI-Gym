# OSWorld-Verified preparation

[All environments](../envs.md)

Pinned official source and asset identities: [release descriptor](../../src/mai_gym/envs/osworld/release.json). This descriptor is separate from a paper profile; the local lock must also identify all required runtime assets.

## Sources and immutable assets

Official source: [OSWorld](https://github.com/xlang-ai/OSWorld). Pin OSWorld source and native evaluator code, the selected Verified manifest, guest image and task assets. Keep current official releases separate from historical paper profiles.

Use the [profile lock schema](../envs.md#immutable-profiles). Required source/asset pins must come from the selected evaluation profile. A missing pin or hash mismatch is a preparation failure.

## Host and dependencies

Linux x86_64, Docker, KVM, RAM for an Ubuntu desktop guest and space for the pinned base QCOW2 plus disposable overlays.

Install the selected upstream revision in its own checkout/interpreter:

```bash
git clone https://github.com/xlang-ai/OSWorld.git /opt/sources/osworld
cd /opt/sources/osworld
git checkout "$RECORDED_REVISION"
python3.12 -m venv /opt/venvs/osworld
/opt/venvs/osworld/bin/pip install -r requirements.txt
```

Set `PYTHONPATH=/opt/mai-gym/src:/opt/sources/osworld` for the worker. The framework manager/runner uses its own environment (`uv sync` in the framework checkout). Keep all API keys in runtime environment variables. `HF_TOKEN` is needed only for gated asset downloads; judge credentials are referenced by `judge.api_key_env`.

## Native preparation

The native worker loads the pinned upstream DesktopEnv in its own interpreter. Set desktop.path_to_vm to the verified QCOW2 and use the selected release's task manifest. Native reset, setup and evaluation run inside this worker. Use a dedicated base/overlay directory for independent deployments.

Worker settings (save with real nonsecret paths):

```json
{
  "profile_lock": "/opt/profiles/osworld-lock.json",
  "evidence_dir": "/opt/evidence/osworld",
  "docker_image": "<runtime-repository>@sha256:<recorded-digest>",
  "desktop": {
    "provider_name": "docker",
    "path_to_vm": "/opt/assets/osworld.qcow2",
    "action_space": "pyautogui",
    "headless": true,
    "require_a11y_tree": false
  }
}
```

For the Docker provider, `docker_image` must match a verified Docker asset in the lock. The worker selects its immutable image ID while retaining the upstream provider's ports, reset, recovery and cleanup behavior.

Validate the profile before provisioning:

```bash
python -m mai_gym.envs.prepare --env osworld --config /opt/profiles/osworld-worker.json
```

## Canonical manager configuration

```yaml
manager:
  host: 127.0.0.1
  port: 8091
  traj_log_dir: /opt/evidence/osworld/trajectories
devices:
- name: osworld-worker
  env_type: osworld
  device_type: computer
  infra_type: local
  connection: local
  max_sessions: 1
  sessions_per_backend: 1
  launch:
    command: /opt/venvs/osworld/bin/python -m mai_gym.envs.shared.worker --env osworld
      --config /opt/profiles/osworld-worker.json --port {port}
    port: 16800
  env_config:
    runtime_protocol: native
    profile_lock: /opt/profiles/osworld-lock.json
```

For the two-benchmark evaluation example, save this YAML as `/opt/profiles/osworld-manager.yaml` and choose a writable trajectory directory. From the MAI-Gym checkout, start the manager in a separate terminal and leave it running.

```bash
uv run mai-gym serve --config /opt/profiles/osworld-manager.yaml
```

On the same host, `curl -fsS http://127.0.0.1:8091/health` should report `"ok": true` and the configured `traj_log_dir`. Set `benchmarks.osworld-verified.manager_url` to the manager address reachable from your runner. [Add OSWorld-Verified to an evaluation](../../site/docs/raw/evaluation.md#4-add-osworld-verified)

Supported infrastructure: local/SSH native worker controlling Docker/KVM, a packaged native worker with its own VM lifecycle, or attach. Native workers use one session per backend. Desktop-service configurations use runtime_protocol: legacy (the default), including the Docker and Kubernetes direct/pool/adopt profiles. osworld_cua selects the separately prepared CUA task bundle.

## Readiness, reset and verification

Check the native worker's /health and pinned /task/list, then initialize one leased episode and confirm a fresh guest screenshot. A healthy outer container alone does not establish guest readiness.

Use the [verification protocol](../envs.md#verification-protocol) with one actual manifest task, one trial, and a budget of **100 interaction steps**. Use CLI benchmark `osworld-verified`. Preserve the exact task identity and parameters. Require full native success; fractional credit is retained separately.

Initialize the same task again in a fresh episode and confirm restoration of its initial application state, absence of prior episode mutations and lease isolation. Run initial/completed/reset verification where the native evaluator supports state checks. Retain original failed attempts, evaluator errors and reset verification logs. Judge-based tasks use the native trajectory protocol rather than an invented state-only score.

## Cleanup

Release the session lease, tear down the native task and stop the worker. Remove only its owned containers, private VM overlays, browser profiles and temporary volumes. Preserve the pinned read-only base assets and trajectory logs. Never prune shared Docker resources. Check disk headroom before starting the next heavyweight environment.
