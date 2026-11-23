# WebArena preparation

[All environments](../envs.md)

Pinned official source and asset identities: [release descriptor](../../src/mai_gym/envs/webarena/release.json). This descriptor is separate from a paper profile; the local lock must also identify all required runtime assets.

## Sources and immutable assets

Official source: [webarena](https://github.com/web-arena-x/webarena). Pin source/native evaluator commits, the 812-task manifest/config assets, site images, baseline databases, authentication preparation and browser identity. Restore the matching URLs and release; use immutable image IDs.

Use the [profile lock schema](../envs.md#immutable-profiles). Required source/asset pins must come from the selected evaluation profile. A missing pin or hash mismatch is a preparation failure.

## Host and dependencies

Linux Docker host, Chromium, resources for the self-hosted sites and dedicated ports/volumes for the site databases.

Install the selected upstream revision in its own checkout/interpreter:

```bash
git clone https://github.com/web-arena-x/webarena.git /opt/sources/webarena
cd /opt/sources/webarena
git checkout "$RECORDED_REVISION"
python3.11 -m venv /opt/venvs/webarena
/opt/venvs/webarena/bin/pip install -r requirements.txt fastapi uvicorn click pydantic pyyaml httpx openai
/opt/venvs/webarena/bin/playwright install --with-deps chromium
```

Set `PYTHONPATH=/opt/mai-gym/src:/opt/sources/webarena` for the worker. The framework manager/runner uses its own environment (`uv sync` in the framework checkout). Keep all API keys in runtime environment variables. `HF_TOKEN` is needed only for gated asset downloads; judge credentials are referenced by `judge.api_key_env`.

## Native preparation

Set sites.urls to the seven upstream site variables for the dedicated stack. Set sites.compose_file to a pinned compose file with project-owned containers, volumes, networks and free ports. Record every image identity and read-only seed file in the profile assets. Writable data uses disposable named volumes. Fixed container/resource names, external volumes and writable host binds are rejected. The published config_files/test.raw.json is supported directly; its native site placeholders are resolved consistently from sites.urls. sites.auth_command runs native authentication after reset and writes storage state to the episode's private auth directory.

Worker settings (save with real nonsecret paths):

```json
{
  "profile_lock": "/opt/profiles/webarena-lock.json",
  "evidence_dir": "/opt/evidence/webarena",
  "sites": {
    "compose_file": "/opt/profiles/webarena-sites.yaml",
    "auth_command": [
      "/opt/venvs/webarena/bin/python",
      "browser_env/auto_login.py",
      "--site_list", "gitlab",
      "--auth_folder", "{auth_dir}"
    ],
    "urls": {
      "REDDIT": "http://<site-host>:<forum-port>",
      "SHOPPING": "http://<site-host>:<shopping-port>",
      "SHOPPING_ADMIN": "http://<site-host>:<admin-port>",
      "GITLAB": "http://<site-host>:<gitlab-port>",
      "WIKIPEDIA": "http://<site-host>:<wiki-port>",
      "MAP": "http://<site-host>:<map-port>",
      "HOMEPAGE": "http://<site-host>:<homepage-port>"
    }
  }
}
```

Validate the profile before provisioning:

```bash
python -m mai_gym.envs.prepare --env webarena --config /opt/profiles/webarena-worker.json
```

## Canonical manager configuration

```yaml
devices:
- name: webarena-worker
  env_type: webarena
  device_type: web
  infra_type: local
  connection: local
  max_sessions: 1
  sessions_per_backend: 1
  launch:
    command: /opt/venvs/webarena/bin/python -m mai_gym.envs.shared.worker --env webarena
      --config /opt/profiles/webarena-worker.json --port {port}
    port: 16800
  env_config:
    profile_lock: /opt/profiles/webarena-lock.json
```

Supported infrastructure: Local/SSH browser/site worker, Docker with site lifecycle access, or attach to a dedicated native worker.

## Readiness, reset and verification

Verify every required site and authentication state, then initialize a real native task. Browser reset alone does not restore the databases. Native evaluation_harness supplies programmatic grading.

Use the [verification protocol](../envs.md#verification-protocol) with one actual manifest task, one trial, and a budget of **30 interaction steps**. Use CLI benchmark `webarena`. Preserve the exact task identity and parameters. Require full native success; fractional credit is retained separately.

Initialize the same task again in a fresh episode and confirm restoration of its initial application state, absence of prior episode mutations and lease isolation. Run initial/completed/reset verification where the native evaluator supports state checks. Retain original failed attempts, evaluator errors and reset verification logs. Judge-based tasks use the native trajectory protocol rather than an invented state-only score.

## Cleanup

Release the session lease, tear down the native task and stop the worker. Remove only its owned containers, private VM overlays, browser profiles and temporary volumes. Preserve the pinned read-only base assets and trajectory logs. Never prune shared Docker resources. Check disk headroom before starting the next heavyweight environment.
