# OSWorld 2.0 preparation

[All environments](../envs.md)

## Sources and immutable assets

Official source: [OSWorld-V2](https://github.com/xlang-ai/OSWorld-V2). The paper-labeled profile uses official OSWorld 2.1: task revision 0a1aadad95aa79b00b3783e717d865089ab06e26, task-manifest SHA256 c54d428329be5ca72742a6becd49ee83cf739df5dea429bba182e1f3f21badfb, native evaluator commit 11cea575561fb7800b5fb6b6cafa56f7a91de11f. Restore the matching release assets and service snapshots.

Use the [profile lock schema](../envs.md#immutable-profiles). Required source/asset pins must come from the selected evaluation profile. A missing pin or hash mismatch is a preparation failure.

For an official release profile, use the immutable identities in [release.json](../../src/mai_gym/envs/osworld_2/release.json), including the public `osworld-v2.1` source commit. Label the lock `official-v2.1`. The paper-labeled evaluator pin remains a separate profile and requires its exact archived source.

## Host and dependencies

Linux x86_64, Docker/KVM, an Ubuntu guest and the matching task, website and GitLab supporting services.

Install the selected upstream revision in its own checkout/interpreter:

```bash
git clone https://github.com/xlang-ai/OSWorld-V2.git /opt/sources/osworld_2
cd /opt/sources/osworld_2
git checkout "$RECORDED_REVISION"
python3.12 -m venv /opt/venvs/osworld_2
/opt/venvs/osworld_2/bin/pip install -r requirements.txt fastapi uvicorn click pydantic pyyaml httpx openai
```

Set `PYTHONPATH=/opt/mai-gym/src:/opt/sources/osworld_2` for the worker. The framework manager/runner uses its own environment (`uv sync` in the framework checkout). Keep all API keys in runtime environment variables. `HF_TOKEN` is needed only for gated asset downloads; judge credentials are referenced by `judge.api_key_env`.

## Native preparation

Use the release's native task hash manifest and set `tasks_root` to the matching downloaded Python task directory. Each task file is verified against that inventory. Configure worker desktop settings for its upstream DesktopEnv. The worker owns that VM lifecycle; the outer launcher owns only its process/container. Set `OSWORLD_FILE_BASE_URL` to the matching local asset directory before starting the worker. Start task-required supporting services on dedicated ports and retain their revisions.

Worker settings (save with real nonsecret paths):

```json
{
  "profile_lock": "/opt/profiles/osworld_2-lock.json",
  "evidence_dir": "/opt/evidence/osworld_2",
  "tasks_root": "/opt/assets/osworld_2_tasks",
  "desktop": {
    "provider_name": "docker",
    "path_to_vm": "/opt/assets/osworld_2.qcow2",
    "action_space": "pyautogui",
    "headless": true,
    "require_a11y_tree": false
  }
}
```

Validate the profile before provisioning:

```bash
python -m mai_gym.envs.prepare --env osworld_2 --config /opt/profiles/osworld_2-worker.json
```

## Canonical manager configuration

```yaml
devices:
- name: osworld_2-worker
  env_type: osworld_2
  device_type: computer
  infra_type: local
  connection: local
  max_sessions: 1
  sessions_per_backend: 1
  launch:
    command: /opt/venvs/osworld_2/bin/python -m mai_gym.envs.shared.worker --env osworld_2
      --config /opt/profiles/osworld_2-worker.json --port {port}
    port: 16800
  env_config:
    profile_lock: /opt/profiles/osworld_2-lock.json
```

Supported infrastructure: Local/SSH dependency-isolated worker, Docker/KVM worker, or attach.

## Readiness, reset and verification

Verify guest screenshots plus task-required services. Use native class setup and evaluate, including partial grades; only full native success satisfies single-task verification.

Use the [verification protocol](../envs.md#verification-protocol) with one actual manifest task, one trial, and a budget of **500 interaction steps**. Use CLI benchmark `osworld-2`. Preserve the exact task identity and parameters. Require full native success; fractional credit is retained separately.

Initialize the same task again in a fresh episode and confirm restoration of its initial application state, absence of prior episode mutations and lease isolation. Run initial/completed/reset verification where the native evaluator supports state checks. Retain original failed attempts, evaluator errors and reset verification logs. Judge-based tasks use the native trajectory protocol rather than an invented state-only score.

## Cleanup

Release the session lease, tear down the native task and stop the worker. Remove only its owned containers, private VM overlays, browser profiles and temporary volumes. Preserve the pinned read-only base assets and trajectory logs. Never prune shared Docker resources. Check disk headroom before starting the next heavyweight environment.
