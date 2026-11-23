# MobileGym preparation

[All environments](../envs.md)

## Sources and immutable assets

Official source: [mobilegym](https://github.com/Purewhiter/mobilegym). Pin the source/evaluator commit, Bench256 manifest with recorded instance parameters, built frontend and companion data archive. Do not replace Bench256 with the current complete template registry.

Use the [profile lock schema](../envs.md#immutable-profiles). Required source/asset pins must come from the selected evaluation profile. A missing pin or hash mismatch is a preparation failure. The [official snapshot descriptor](../../src/mai_gym/envs/mobilegym/release.json) records a source revision and the published companion archive SHA256. Download that archive, verify its SHA256, and extract it at the upstream checkout root before building the application.

Record the compiled `dist/` directory as a `tree` asset with a per-file SHA256 inventory. In profiled mode, `--static-dir` must select that verified frontend directory. Also inventory the extracted `mobilegym-data/` directory as a separate `tree` asset with `"role": "media"`. Keep the companion archive and browser binary as separate verified assets.

After verifying and extracting the archive, generate its media inventory and add the printed asset entry to the profile lock's `assets` list:

```bash
mkdir -p /opt/profiles
python3.12 - <<'PY'
import hashlib
import json
from pathlib import Path

root = Path('/opt/sources/mobilegym/mobilegym-data')
entries = []
for path in sorted(root.rglob('*')):
    if path.is_file():
        with path.open('rb') as stream:
            digest = hashlib.file_digest(stream, 'sha256').hexdigest()
        entries.append({'path': str(path.relative_to(root)), 'sha256': digest})
if not entries:
    raise SystemExit('Extract the verified companion archive before preparing media')
manifest = Path('/opt/profiles/mobilegym-media-files.json')
manifest.write_text(json.dumps(entries, indent=2) + '\n')
print(json.dumps({'kind': 'tree', 'role': 'media', 'path': str(root),
                  'manifest': str(manifest),
                  'sha256': hashlib.sha256(manifest.read_bytes()).hexdigest()}, indent=2))
PY
```

## Host and dependencies

Install CJK fonts in the browser worker image, for example `apt-get install fonts-noto-cjk`. Check a fresh screenshot for readable Chinese labels before evaluation. For a private font directory, pass its Fontconfig configuration through `FONTCONFIG_FILE` when starting the worker.

Node 22+, Python 3.12+, Chromium dependencies, CPU/RAM for isolated browser pages and space for the companion dataset. KVM is not required.

Install the selected upstream revision in its own checkout/interpreter:

```bash
git clone https://github.com/Purewhiter/mobilegym.git /opt/sources/mobilegym
cd /opt/sources/mobilegym
git checkout "$RECORDED_REVISION"
npm ci
npm run build
python3.12 -m venv /opt/venvs/mobilegym
/opt/venvs/mobilegym/bin/pip install -r bench_env/requirements.txt fastapi uvicorn click pydantic pyyaml httpx
/opt/venvs/mobilegym/bin/playwright install --with-deps chromium
```

Set `PYTHONPATH=/opt/mai-gym/src:/opt/sources/mobilegym` for the worker. The framework manager/runner uses its own environment (`uv sync` in the framework checkout). Keep all API keys in runtime environment variables. `HF_TOKEN` is needed only for gated asset downloads; judge credentials are referenced by `judge.api_key_env`.

## Native preparation

Restore and verify the recorded companion dataset, then build dist/. Start python -m mai_gym.envs.mobilegym.server_runtime --static-dir /opt/sources/mobilegym/dist --profile-config /opt/profiles/mobilegym-worker.json --num-pages 1 --port 16800. The profile filters listing and initialization to its manifest.

Both the browser's static service and the worker publish the verified media directory at `/cdn/`. The frontend remains separate from the media tree. The worker selects the profile's single `role: media` tree automatically, or accepts an explicit `media_dir` setting / `--media-dir` that matches a verified tree. Relative settings paths resolve from the profile lock's directory. Missing or unpinned media prevents profiled startup. Legacy workers can use `--media-dir` or an existing `dist/cdn/` directory.

Use the native **360×800 logical viewport**, with **3× device scale** producing **1080×2400 screenshots**. The physical screenshot dimensions must not be passed as viewport dimensions. `mai-gym eval` selects a native `mobile_use` policy for this environment, with MobileGym's app list and touch actions (tap, long press, swipe, drag, text, app launch and system buttons). Swipe scrolls with inertia; drag holds the touch while moving an item.

Worker settings (save with real nonsecret paths):

```json
{
  "profile_lock": "/opt/profiles/mobilegym-lock.json",
  "media_dir": "/opt/sources/mobilegym/mobilegym-data",
  "seed": 0,
  "evidence_dir": "/opt/evidence/mobilegym"
}
```

Select an explicit seed for an official-source profile; manifest-entry seeds override it. Paper profiles must restore their recorded instance parameters and seeds. Profiled workers use separate browser contexts and episode leases, retain native state/verifier artifacts, and report native progress separately from clean full completion.

Validate the profile before provisioning:

```bash
python -m mai_gym.envs.prepare --env mobilegym --config /opt/profiles/mobilegym-worker.json
```

## Canonical manager configuration

```yaml
devices:
- name: mobilegym-worker
  env_type: mobilegym
  device_type: mobile
  infra_type: local
  connection: local
  max_sessions: 1
  sessions_per_backend: 1
  launch:
    command: /opt/venvs/mobilegym/bin/python -m mai_gym.envs.mobilegym.server_runtime
      --static-dir /opt/sources/mobilegym/dist --profile-config /opt/profiles/mobilegym-worker.json
      --num-pages 1 --port {port}
    port: 16800
  env_config:
    profile_lock: /opt/profiles/mobilegym-lock.json
    media_dir: /opt/sources/mobilegym/mobilegym-data
```

Supported infrastructure: Local/SSH browser service, Docker, or attach. Its interaction device_type remains mobile.

## Readiness, reset and verification

GET /health, acquire a display, initialize a Bench256 task and fetch its native goal and screenshot. Every concurrent episode needs its own isolated page.

Fetch a known companion image at `/cdn/<app>/<asset-path>` from both the worker port and `--static-port`. Require HTTP 200, the expected content type and matching SHA256; missing files must return 404. Check browser image decoding and byte-range responses for media playback. `/state` reports `media_available`.

Use the [verification protocol](../envs.md#verification-protocol) with one actual manifest task, one trial, and a budget of **100 interaction steps**. Use CLI benchmark `mobilegym`. Preserve the exact task identity and parameters. Require full native success; fractional credit is retained separately.

Initialize the same task again in a fresh episode and confirm restoration of its initial application state, absence of prior episode mutations and lease isolation. Run initial/completed/reset verification where the native evaluator supports state checks. Retain original failed attempts, evaluator errors and reset verification logs. Judge-based tasks use the native trajectory protocol rather than an invented state-only score.

## Cleanup

Release the session lease, tear down the native task and stop the worker. Remove only its owned containers, private VM overlays, browser profiles and temporary volumes. Preserve the pinned read-only base assets and trajectory logs. Never prune shared Docker resources. Check disk headroom before starting the next heavyweight environment.
