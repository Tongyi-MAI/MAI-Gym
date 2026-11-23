# Benchmark environment preparation

Choose the benchmark you want to run. Each guide covers its host requirements, assets, setup and reset checks. Use the middle column both in the evaluation YAML under `benchmarks` and on the command line after `--benchmarks`.

| Interaction | Evaluation CLI ID | Preparation guide |
| --- | --- | --- |
| Mobile | `mobileworld` | [MobileWorld](envs/mobileworld.md) |
| Mobile | `androidworld` | [AndroidWorld](envs/androidworld.md) |
| Mobile | `mobilegym` | [MobileGym](envs/mobilegym.md) |
| Computer | `osworld-verified` | [OSWorld-Verified](envs/osworld.md) |
| Computer | `osworld-2` | [OSWorld 2.0](envs/osworld_2.md) |
| Computer | `macagentbench` | [MacAgentBench](envs/macagentbench.md) |
| Computer | `mypcbench` | [MyPCBench](envs/mypcbench.md) |
| Web | `webvoyager` | [WebVoyager](envs/webvoyager.md) |
| Web | `onlinemind2web` | [Online-Mind2Web](envs/onlinemind2web.md) |
| Web | `webarena` | [WebArena](envs/webarena.md) |
| Web | `mocked-apps` | [Mocked Apps (42 Novel Apps)](../site/docs/raw/mocked-apps.md#evaluate-a-task) |

Once your benchmark is ready, run one task with the shared MAI-Gym runner. [Evaluation walkthrough](../site/docs/raw/evaluation.md#preflight-and-first-task)

## Configuration

Each implementation in `src/mai_gym/envs/<environment>` owns task loading, initialization, verification and its launch profile. The shared runner makes the model decisions.

```yaml
devices:
  - name: benchmark-worker
    env_type: androidworld
    device_type: mobile
    infra_type: local
    connection: local
    max_sessions: 1
    sessions_per_backend: 1
    launch:
      command: /opt/venvs/androidworld/bin/python -m mai_gym.envs.shared.worker --env androidworld --config /opt/profiles/androidworld-worker.json --port {port}
      port: 16800
    env_config:
      profile_lock: /opt/profiles/androidworld-lock.json
```

The worker JSON contains environment-owned settings, including `profile_lock` and `evidence_dir`. Use a separate worker process/interpreter for each upstream dependency set. Set `PYTHONPATH` to the framework's `src` directory in that worker environment. A local/SSH command launches the worker; a Docker image can package the same command. `attach` connects to an already prepared endpoint. Docker over SSH uses `infra_type: docker` and `connection.ssh`; `infra_type: ssh` launches a remote process. Kubernetes pooling/adoption is available for MobileWorld and OSWorld profiles. Unsupported pairs fail before provisioning.

The manager and shared runner use Python 3.12+. A native worker may use Python 3.11 when required by its pinned upstream dependencies. Install that benchmark's dependencies in its own interpreter and expose the framework source through `PYTHONPATH`. Programmatic workers do not load an LLM judge client.

Legacy `type` and benchmark routing aliases remain valid YAML inputs. `mobileworld_k8s` and `osworld_k8s` constrain infrastructure; `osworld_cua` selects the CUA task bundle profile. Conflicting canonical and legacy fields are errors. Runtime details such as ReDroid, an API-33 AVD, a QCOW2 guest or Chromium belong to the environment profile rather than the interaction domain.

## Immutable profiles

Restore the recorded evaluation bundle before preparing a paper-reproduction profile. A task count is not a manifest, and the current upstream default branch is not an implicit paper revision. Keep source commits, native evaluator commits, task manifest bytes, application assets and VM/browser identities together.

Every native worker requires a JSON lock with this shape (paths are relative to the lock):

```json
{
  "benchmark": "androidworld",
  "label": "paper",
  "sources": {
    "runtime": {"path": "../sources/androidworld", "revision": "<40-character recorded commit>"},
    "evaluator": {"path": "../sources/androidworld", "revision": "<40-character recorded commit>"}
  },
  "task_manifest": {"path": "androidworld-tasks.json", "sha256": "<64-character recorded hash>"},
  "assets": [{"path": "../assets/avd-snapshot.tar", "sha256": "<64-character recorded hash>"}]
}
```

Docker assets instead use `{"kind":"docker","image":"<image reference>","id":"sha256:<64-character image id>"}`. Compare the cached image's identity before reuse. Tracked source modifications, missing assets and mismatching hashes stop preparation. Record any independently selected current upstream profile with a separate label and immutable revision. Keep secrets outside locks, configuration files and verification logs.

For extracted asset bundles or compiled browser frontends, use `{"kind":"tree","path":"../assets/bundle","manifest":"bundle-files.json","sha256":"<inventory SHA256>"}`. The inventory is a JSON list of `{"path":"relative/file","sha256":"<file SHA256>"}` entries. Preparation verifies the inventory and every declared file before starting the runtime. Keep its recorded source or asset-repository revision alongside the inventory.

Manifest entries may be task IDs or mappings with `task_id`, native asset `path`, category, recorded parameters and seed. Native JSONL task records are also supported. OSWorld category-to-ID manifests are accepted without sampling. Task variants need distinct IDs. AndroidWorld budgets are derived by its native allocator from the pinned task class and parameters.

MobileGym also accepts its published `bench_env/splits/test.txt` directly. MyPCBench accepts the canonical `all_tasks_with_grading.json` with its native instructions and rubrics inlined. OSWorld 2.0 uses its native task-file hash inventory. For current official releases, pin the source and asset revisions first and use an explicit `official-release` label; keep paper-reproduction locks separate.

<a name="canary-protocol"></a>

## Verification protocol

For a new manager, open a terminal in the MAI-Gym checkout, start the service with your device configuration and leave it running.

```bash
uv run mai-gym serve --config /opt/profiles/manager.yaml
```

In a second terminal, run one task on that manager. This example selects AndroidWorld and uses the served model ID configured in `eval.yaml`.

```bash
uv run mai-gym eval --config /opt/profiles/eval.yaml --benchmarks androidworld \
  --trials 1 --num-envs 1 --output /opt/evidence/androidworld
```

The evaluation config selects an actual manifest task through `benchmarks.<benchmark>.tasks`. Use the benchmark's native step budget. A passing validation requires a native full-success score; partial reward is reported separately. Retain failed attempts before retrying. Preserve screenshots, factual executed actions, final answers, verifier output and runtime/profile identities.

After each single-task verification, initialize the same task in a fresh episode and verify that the original state is restored and prior episode state is absent. For state verifiers, confirm failure initially, success after completion and failure after reset where the task supports it. Trajectory judges require their native evaluation protocol. Reset failures and missing grades remain unvalidated outcomes.

For an explicitly selected validation judge, put only references in worker settings:

```json
{"judge":{"label":"custom-judge","model":"<judge-model>","base_url":"<OpenAI-compatible API base>","api_key_env":"JUDGE_API_KEY","score_threshold":3}}
```

Inject the secret through the worker's runtime environment. A validation judge identity is recorded separately from the source/asset profile. Use fresh names, ports, volumes and VM overlays, keep disk headroom and remove only resources created by the validation run.
