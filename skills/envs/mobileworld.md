# MobileWorld preparation

[All environments](../envs.md)

Pinned official source and asset identities: [release descriptor](../../src/mai_gym/envs/mobileworld/release.json). This descriptor is separate from a paper profile; the local lock must also identify all required runtime assets.

## Sources and immutable assets

Official source: [MobileWorld](https://github.com/Tongyi-MAI/MobileWorld). Pin the MobileWorld source/evaluator commit, task bundle, image ID, APKs, snapshots and optional verifier patch bundle. Restore the recorded manifest; a mutable latest image is not a paper pin.

Use the [profile lock schema](../envs.md#immutable-profiles). Required source/asset pins must come from the selected evaluation profile. A missing pin or hash mismatch is a preparation failure.

For an official image distribution, use an explicit official-profile label and record bundled sources as `{"kind":"docker","image":"<immutable reference>","id":"sha256:<image id>"}` for both runtime and evaluator. Each must match a verified Docker asset in the lock. This records the image's native code without inventing a Git revision. Paper-labeled locks require their recorded source commits.

## Host and dependencies

Linux x86_64, Docker, KVM for the emulator image, and space for Android snapshots.

Install the selected upstream revision in its own checkout/interpreter:

```bash
git clone https://github.com/Tongyi-MAI/MobileWorld.git /opt/sources/mobileworld
cd /opt/sources/mobileworld
git checkout "$RECORDED_REVISION"
uv sync
```

Set `PYTHONPATH=/opt/mai-gym/src:/opt/sources/mobileworld` for the worker. The framework manager/runner uses its own environment (`uv sync` in the framework checkout). Keep all API keys in runtime environment variables. `HF_TOKEN` is needed only for gated asset downloads; judge credentials are referenced by `judge.api_key_env`.

## Native preparation

Use the Android service in the pinned image. Snapshot restoration, native task initialization and teardown are owned by MobileWorldStrategy. Kubernetes profiles additionally stage task artifacts and apply only the configured verified patch bundle.

Worker settings (save with real nonsecret paths):

```json
{
  "profile_lock": "/opt/profiles/mobileworld-lock.json",
  "evidence_dir": "/opt/evidence/mobileworld",
  "snapshot": "<recorded-snapshot>"
}
```

From the MAI-Gym checkout, validate the profile before provisioning.

```bash
uv run python -m mai_gym.envs.prepare --env mobileworld --config /opt/profiles/mobileworld-worker.json
```

### Start one Android backend for the evaluation walkthrough

If you have a running backend, go straight to its readiness checks below. To start one on a Linux/KVM host, set `MOBILEWORLD_IMAGE` to the Docker image verified against your lock and `MOBILEWORLD_ENV_FILE` to the absolute path of your MobileWorld runtime environment file. Run the upstream CLI from a directory outside the source checkouts so the emulator uses its image-bundled code. [Upstream environment setup](https://github.com/Tongyi-MAI/MobileWorld#-quick-start)

```bash
uv run --project /opt/sources/mobileworld mw env run --count 1 \
  --name-prefix mai-gym-first-eval --backend-start-port 6800 \
  --image "$MOBILEWORLD_IMAGE" --env-file "$MOBILEWORLD_ENV_FILE"
```

Wait for the emulator to become ready. Put the backend URL with the port printed by the launcher in your copy of `mobileworld.yaml`, then return to the walkthrough. [Connect the model and backend](../../site/docs/raw/evaluation.md#1-connect-the-model-and-backend)

## Canonical manager configuration

```yaml
devices:
- name: mobileworld-worker
  env_type: mobileworld
  device_type: mobile
  infra_type: docker
  connection: local
  max_sessions: 1
  sessions_per_backend: 1
  launch:
    image: <pinned-image-reference>
    reuse_existing: false
    port_range_start: 16800
    name_prefix: mai-gym-mobileworld
  env_config:
    profile_lock: /opt/profiles/mobileworld-lock.json
```

Supported infrastructure: Docker locally, Docker over SSH, local/SSH service process, Kubernetes direct/pool/adopt, agent service, or attach.

For a multi-node deployment, follow the [Kubernetes fleet guide](../../site/docs/raw/kubernetes.md) to configure a pool, check session readiness and connect an evaluation.

## Readiness, reset and verification

Check `/health`, `/screenshot` and `/task/list`. Service versions exposing `/devices` can discover the device automatically; older images require an explicit device ID or one ready device in their health response. Initialize that device and confirm the task is registered before leasing it.

Use the [verification protocol](../envs.md#verification-protocol) with one actual manifest task, one trial, and a budget of **100 interaction steps**. Use CLI benchmark `mobileworld`. Preserve the exact task identity and parameters. Require full native success; fractional credit is retained separately.

Initialize the same task again in a fresh episode and confirm restoration of its initial application state, absence of prior episode mutations and lease isolation. Run initial/completed/reset verification where the native evaluator supports state checks. Retain original failed attempts, evaluator errors and reset verification logs. Judge-based tasks use the native trajectory protocol rather than an invented state-only score.

## Cleanup

Release the session lease, tear down the native task and stop the worker. Remove only its owned containers, private VM overlays, browser profiles and temporary volumes. Preserve the pinned read-only base assets and trajectory logs. Never prune shared Docker resources. Check disk headroom before starting the next heavyweight environment.
