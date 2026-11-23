# AndroidWorld preparation

[All environments](../envs.md)

Pinned official source and asset identities: [release descriptor](../../src/mai_gym/envs/androidworld/release.json). This descriptor is separate from a paper profile; the local lock must also identify all required runtime assets.

## Sources and immutable assets

Official source: [android_world](https://github.com/google-research/android_world). Pin the AndroidWorld commit, APK/snapshot archive and seeded task manifest. Record instance seeds or complete native parameters; suite seeds and derived instance seeds are different values.

Use the [profile lock schema](../envs.md#immutable-profiles). Required source/asset pins must come from the selected evaluation profile. A missing pin or hash mismatch is a preparation failure.

## Host and dependencies

Linux x86_64 with accessible KVM, Android SDK/emulator, a Pixel 6 API-33 AVD, adb and a dedicated gRPC/console port pair.

Install the selected upstream revision in its own checkout/interpreter:

```bash
git clone https://github.com/google-research/android_world.git /opt/sources/androidworld
cd /opt/sources/androidworld
git checkout "$RECORDED_REVISION"
python3.12 -m venv /opt/venvs/androidworld
/opt/venvs/androidworld/bin/pip install -r requirements.txt fastapi uvicorn click pydantic pyyaml httpx openai
```

Set `PYTHONPATH=/opt/mai-gym/src:/opt/sources/androidworld` for the worker. The framework manager/runner uses its own environment (`uv sync` in the framework checkout). Keep all API keys in runtime environment variables. `HF_TOKEN` is needed only for gated asset downloads; judge credentials are referenced by `judge.api_key_env`.

## Native preparation

Launch a disposable copy of the recorded AndroidWorldAvd with emulator -avd AndroidWorldAvd -no-window -no-audio -no-snapshot-save -port 5554 -grpc 8554. Perform native emulator setup for a new AVD. Set worker adb_path, console_port and grpc_port. Native task initialize_task/is_successful/tear_down are used.

Download the accessibility-forwarder APK referenced by the pinned `android_env` package and record its SHA256 as a profile asset. Set `a11y_forwarder_apk` to that verified file to install those exact bytes through the native controller. When adb runs inside a separate container, mount the worker's source, dependency and temporary-data paths at the same paths there; set `TMPDIR` to an owned mounted directory.

Worker settings (save with real nonsecret paths):

```json
{
  "profile_lock": "/opt/profiles/androidworld-lock.json",
  "evidence_dir": "/opt/evidence/androidworld",
  "adb_path": "/opt/android-sdk/platform-tools/adb",
  "console_port": 5554,
  "grpc_port": 8554,
  "emulator_setup": false,
  "coordinate_space": "pixels"
}
```

Validate the profile before provisioning:

```bash
python -m mai_gym.envs.prepare --env androidworld --config /opt/profiles/androidworld-worker.json
```

## Canonical manager configuration

```yaml
devices:
- name: androidworld-worker
  env_type: androidworld
  device_type: mobile
  infra_type: local
  connection: local
  max_sessions: 1
  sessions_per_backend: 1
  launch:
    command: /opt/venvs/androidworld/bin/python -m mai_gym.envs.shared.worker --env
      androidworld --config /opt/profiles/androidworld-worker.json --port {port}
    port: 16800
  env_config:
    profile_lock: /opt/profiles/androidworld-lock.json
```

Supported infrastructure: Local/SSH native worker, a Docker worker with KVM/AVD mounts, or attach.

## Readiness, reset and verification

Use adb -s emulator-5554 shell getprop sys.boot_completed, then GET /health and acquire/init a real manifest task. Check its native goal and task-dependent max_steps in /task/list.

Use the [verification protocol](../envs.md#verification-protocol) with one actual manifest task, one trial, and the **task-specific step budget assigned by AndroidWorld's native complexity allocator**. Use CLI benchmark `androidworld`. Preserve the exact task identity and parameters. Require full native success; fractional credit is retained separately.

Initialize the same task again in a fresh episode and confirm restoration of its initial application state, absence of prior episode mutations and lease isolation. Run initial/completed/reset verification where the native evaluator supports state checks. Retain original failed attempts, evaluator errors and reset verification logs. Judge-based tasks use the native trajectory protocol rather than an invented state-only score.

## Cleanup

Release the session lease, tear down the native task and stop the worker. Remove only its owned containers, private VM overlays, browser profiles and temporary volumes. Preserve the pinned read-only base assets and trajectory logs. Never prune shared Docker resources. Check disk headroom before starting the next heavyweight environment.
