# WebVoyager preparation

[All environments](../envs.md)

Pinned official source and asset identities: [release descriptor](../../src/mai_gym/envs/webvoyager/release.json). This descriptor is separate from a paper profile; the local lock must also identify all required runtime assets.

## Sources and immutable assets

Official source: [WebVoyager](https://github.com/MinorJerry/WebVoyager). Pin source/evaluator commits and the published WebVoyager_data.jsonl manifest. Preserve original task text, dates, IDs and prescribed start pages. Pin the browser binary and any reference assets.

Use the [profile lock schema](../envs.md#immutable-profiles). Required source/asset pins must come from the selected evaluation profile. A missing pin or hash mismatch is a preparation failure.

## Host and dependencies

Python, Chromium/Playwright dependencies, CPU/RAM for a fresh browser context and outbound access to the task websites.

Install the selected upstream revision in its own checkout/interpreter:

```bash
git clone https://github.com/MinorJerry/WebVoyager.git /opt/sources/webvoyager
cd /opt/sources/webvoyager
git checkout "$RECORDED_REVISION"
python3.12 -m venv /opt/venvs/webvoyager
/opt/venvs/webvoyager/bin/pip install fastapi uvicorn click pydantic pyyaml httpx openai playwright pillow
/opt/venvs/webvoyager/bin/playwright install --with-deps chromium
```

Set `PYTHONPATH=/opt/mai-gym/src:/opt/sources/webvoyager` for the worker. The framework manager/runner uses its own environment (`uv sync` in the framework checkout). Keep all API keys in runtime environment variables. `HF_TOKEN` is needed only for gated asset downloads; judge credentials are referenced by `judge.api_key_env`.

## Native preparation

The native JSONL fields id, ques and web supply task identity, instruction and start page. Configure judge references in the worker. The native evaluation/auto_eval.py evaluates the exported final response and last screenshots.

Worker settings (save with real nonsecret paths):

```json
{
  "profile_lock": "/opt/profiles/webvoyager-lock.json",
  "evidence_dir": "/opt/evidence/webvoyager",
  "judge": {
    "label": "<judge-profile>",
    "model": "<judge-model>",
    "base_url": "<API-base>",
    "api_key_env": "JUDGE_API_KEY",
    "image_count": 1
  }
}
```

Validate the profile before provisioning:

```bash
python -m mai_gym.envs.prepare --env webvoyager --config /opt/profiles/webvoyager-worker.json
```

## Canonical manager configuration

```yaml
devices:
- name: webvoyager-worker
  env_type: webvoyager
  device_type: web
  infra_type: local
  connection: local
  max_sessions: 1
  sessions_per_backend: 1
  launch:
    command: /opt/venvs/webvoyager/bin/python -m mai_gym.envs.shared.worker --env
      webvoyager --config /opt/profiles/webvoyager-worker.json --port {port}
    port: 16800
  env_config:
    profile_lock: /opt/profiles/webvoyager-lock.json
```

Supported infrastructure: Local/SSH browser worker, Docker, or attach.

## Readiness, reset and verification

Check the prescribed start page loads in a new context, screenshots are nonempty and the judge endpoint is reachable. Record website failures without replacing the task.

Use the [verification protocol](../envs.md#verification-protocol) with one actual manifest task, one trial, and a budget of **15 interaction steps**. Use CLI benchmark `webvoyager`. Preserve the exact task identity and parameters. Require full native success; fractional credit is retained separately.

Initialize the same task again in a fresh episode and confirm restoration of its initial application state, absence of prior episode mutations and lease isolation. Run initial/completed/reset verification where the native evaluator supports state checks. Retain original failed attempts, evaluator errors and reset verification logs. Judge-based tasks use the native trajectory protocol rather than an invented state-only score.

## Cleanup

Release the session lease, tear down the native task and stop the worker. Remove only its owned containers, private VM overlays, browser profiles and temporary volumes. Preserve the pinned read-only base assets and trajectory logs. Never prune shared Docker resources. Check disk headroom before starting the next heavyweight environment.
