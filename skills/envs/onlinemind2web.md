# Online-Mind2Web preparation

[All environments](../envs.md)

Pinned official source and asset identities: [release descriptor](../../src/mai_gym/envs/onlinemind2web/release.json). This descriptor is separate from a paper-reproduction profile; the local lock must also identify all required runtime assets.

## Sources and immutable assets

Official source: [Online-Mind2Web](https://github.com/OSU-NLP-Group/Online-Mind2Web). The separately labeled official-latest integration pins evaluator 1a8db93168d5d7870402b7f453160e1c557153c8 and dataset osunlp/Online-Mind2Web revision eacad896a84dc5b65e29b0b06e4699ab0544d701. Fetch and hash the approved dataset bytes. The paper 123-task subset requires its own recorded manifest; do not derive it by taking the first 123 current tasks.

Use the [profile lock schema](../envs.md#immutable-profiles). Required source/asset pins must come from the selected evaluation profile. A missing pin or hash mismatch is a preparation failure.

## Host and dependencies

Python 3.12+, Chromium/Playwright dependencies, website access, a judge endpoint and approved Hugging Face dataset access.

Install the selected upstream revision in its own checkout/interpreter:

```bash
git clone https://github.com/OSU-NLP-Group/Online-Mind2Web.git /opt/sources/onlinemind2web
cd /opt/sources/onlinemind2web
git checkout "$RECORDED_REVISION"
python3.12 -m venv /opt/venvs/onlinemind2web
/opt/venvs/onlinemind2web/bin/pip install fastapi uvicorn click pydantic pyyaml httpx openai playwright pillow backoff jsonschema
/opt/venvs/onlinemind2web/bin/playwright install --with-deps chromium
```

Set `PYTHONPATH=/opt/mai-gym/src:/opt/sources/onlinemind2web` for the worker. The framework manager/runner uses its own environment (`uv sync` in the framework checkout). Keep all API keys in runtime environment variables. `HF_TOKEN` is needed only for gated asset downloads; judge credentials are referenced by `judge.api_key_env`.

## Native preparation

Fetch the gated official dataset using HF_TOKEN after granting access. Use label official-latest for the pinned current dataset, or paper with the independently recorded subset. Native WebJudge selects key points/screenshots and grades factual executed actions. V2 result.json exports URLs, screenshots and a separate agent_final_answer.

Worker settings (save with real nonsecret paths):

```json
{
  "profile_lock": "/opt/profiles/onlinemind2web-lock.json",
  "evidence_dir": "/opt/evidence/onlinemind2web",
  "judge": {
    "label": "<judge-profile>",
    "model": "<judge-model>",
    "base_url": "<API-base>",
    "api_key_env": "JUDGE_API_KEY",
    "score_threshold": 3
  }
}
```

Validate the profile before provisioning:

```bash
python -m mai_gym.envs.prepare --env onlinemind2web --config /opt/profiles/onlinemind2web-worker.json
```

## Canonical manager configuration

```yaml
devices:
- name: onlinemind2web-worker
  env_type: onlinemind2web
  device_type: web
  infra_type: local
  connection: local
  max_sessions: 1
  sessions_per_backend: 1
  launch:
    command: /opt/venvs/onlinemind2web/bin/python -m mai_gym.envs.shared.worker --env
      onlinemind2web --config /opt/profiles/onlinemind2web-worker.json --port {port}
    port: 16800
  env_config:
    profile_lock: /opt/profiles/onlinemind2web-lock.json
```

Supported infrastructure: Local/SSH browser worker, Docker, or attach.

## Readiness, reset and verification

Check dataset authorization, prescribed task website, isolated context and screenshot capture. Starting from a search engine changes the task protocol unless the task explicitly prescribes it.

Use the [verification protocol](../envs.md#verification-protocol) with one actual manifest task, one trial, and a budget of **50 interaction steps**. Use CLI benchmark `onlinemind2web`. Preserve the exact task identity and parameters. Require full native success; fractional credit is retained separately.

Initialize the same task again in a fresh episode and confirm restoration of its initial application state, absence of prior episode mutations and lease isolation. Run initial/completed/reset verification where the native evaluator supports state checks. Retain original failed attempts, evaluator errors and reset verification logs. Judge-based tasks use the native trajectory protocol rather than an invented state-only score.

## Cleanup

Release the session lease, tear down the native task and stop the worker. Remove only its owned containers, private VM overlays, browser profiles and temporary volumes. Preserve the pinned read-only base assets and trajectory logs. Never prune shared Docker resources. Check disk headroom before starting the next heavyweight environment.
