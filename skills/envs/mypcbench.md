# MyPCBench preparation

[All environments](../envs.md)

Pinned official source and asset identities: [release descriptor](../../src/mai_gym/envs/mypcbench/release.json). This descriptor is separate from a paper profile; the local lock must also identify all required runtime assets.

## Sources and immutable assets

Official source: [MyPCBench](https://github.com/ljang0/MyPCBench). Pin source/evaluator commits, the recorded 184-task manifest, Linux QCOW2, firmware and application state. The upstream current-image fetcher selects a daily image; restore a recorded image for a paper profile.

Use the [profile lock schema](../envs.md#immutable-profiles). Required source/asset pins must come from the selected evaluation profile. A missing pin or hash mismatch is a preparation failure.

## Host and dependencies

Linux x86_64, KVM, qemu-system-x86_64, qemu-img, OVMF firmware and an Ubuntu desktop base image. No GPU is required.

Install the selected upstream revision in its own checkout/interpreter:

```bash
git clone https://github.com/ljang0/MyPCBench.git /opt/sources/mypcbench
cd /opt/sources/mypcbench
git checkout "$RECORDED_REVISION"
python3.12 -m venv /opt/venvs/mypcbench
/opt/venvs/mypcbench/bin/pip install -r requirements.txt fastapi uvicorn click pydantic pyyaml httpx openai
```

Set `PYTHONPATH=/opt/mai-gym/src:/opt/sources/mypcbench` for the worker. The framework manager/runner uses its own environment (`uv sync` in the framework checkout). Keep all API keys in runtime environment variables. `HF_TOKEN` is needed only for gated asset downloads; judge credentials are referenced by `judge.api_key_env`.

## Native preparation

Set MYPCBENCH_OVMF_CODE and MYPCBENCH_OVMF_VARS to the verified firmware. Configure worker desktop.backend=qemu and desktop.qcow2_path. MyPCBenchEnv makes a unique private overlay and runs its native hard reset. The shared runner executes actions; native rubric bundle construction and offline full-trajectory grading supply the score.

Worker settings (save with real nonsecret paths):

```json
{
  "profile_lock": "/opt/profiles/mypcbench-lock.json",
  "evidence_dir": "/opt/evidence/mypcbench",
  "desktop": {
    "backend": "qemu",
    "qcow2_path": "/opt/assets/mypcbench.qcow2"
  },
  "judge": {
    "label": "<judge-profile>",
    "model": "<judge-model>",
    "base_url": "<API-base>",
    "api_key_env": "JUDGE_API_KEY"
  }
}
```

Validate the profile before provisioning:

```bash
python -m mai_gym.envs.prepare --env mypcbench --config /opt/profiles/mypcbench-worker.json
```

## Canonical manager configuration

```yaml
devices:
- name: mypcbench-worker
  env_type: mypcbench
  device_type: computer
  infra_type: local
  connection: local
  max_sessions: 1
  sessions_per_backend: 1
  launch:
    command: /opt/venvs/mypcbench/bin/python -m mai_gym.envs.shared.worker --env mypcbench
      --config /opt/profiles/mypcbench-worker.json --port {port}
    port: 16800
  env_config:
    profile_lock: /opt/profiles/mypcbench-lock.json
```

Supported infrastructure: Local/SSH QEMU worker, Docker/KVM runtime, or attach.

## Readiness, reset and verification

Check KVM and firmware access, guest Control API and screenshots, plus the task applications. A result.txt completion marker is not a native rubric grade.

Use the [verification protocol](../envs.md#verification-protocol) with one actual manifest task, one trial, and a budget of **100 interaction steps**. Use CLI benchmark `mypcbench`. Preserve the exact task identity and parameters. Require full native success; fractional credit is retained separately.

Initialize the same task again in a fresh episode and confirm restoration of its initial application state, absence of prior episode mutations and lease isolation. Run initial/completed/reset verification where the native evaluator supports state checks. Retain original failed attempts, evaluator errors and reset verification logs. Judge-based tasks use the native trajectory protocol rather than an invented state-only score.

## Cleanup

Release the session lease, tear down the native task and stop the worker. Remove only its owned containers, private VM overlays, browser profiles and temporary volumes. Preserve the pinned read-only base assets and trajectory logs. Never prune shared Docker resources. Check disk headroom before starting the next heavyweight environment.
