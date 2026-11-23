# Evaluation Guide

[Home](../../../README.md) · [Quickstart](index.md) · **Evaluation** · [Training](training.md) · [Rollout](rollout.md) · [Kubernetes](kubernetes.md) · [Concepts](concepts.md)

Run a single MobileWorld task, inspect the native evaluation outcome, replay the episode in the trajectory dashboard, and extend the same configuration to multi-benchmark runs (such as MobileWorld + OSWorld-Verified or all supported benchmarks). For lightweight local browser evaluation without Docker or VMs, see [Mocked Apps](mocked-apps.md#evaluate-a-task) (`--benchmarks mocked-apps`).

> [!TIP]
> **Prefer agent-assisted setup?**
> Use the [coding-agent evaluation prompt](../../../skills/README.md#prepare-an-evaluation) in Claude Code, Codex, or Cursor to automate backend checks and trial runs.

---

## Supported Benchmarks & Native Protocols

`mai-gym eval` preserves each benchmark's native step limit, verifier, and primary evaluation metric (Paper Table 3). By default, evaluation runs with deterministic decoding (`temperature=0.0`) and `--trials 3`, reporting both `pass@n` (`pass_at_n`) and `pass^n` (`pass_all_n`).

| Domain | Benchmark | CLI ID (`--benchmarks`) | Tasks | Max Steps | Native Metric | Benchmark Guide |
| :--- | :--- | :--- | ---: | :--- | :--- | :--- |
| **Mobile** | MobileWorld (GUI-only) | `mobileworld` | 117 | 100 | `success_rate` | [Mobile Benchmarks](envs/mobile.md#mobileworld) |
| **Mobile** | AndroidWorld | `androidworld` | 116 | Task-dependent | `success_rate` | [Mobile Benchmarks](envs/mobile.md#androidworld) |
| **Mobile** | MobileGym | `mobilegym` | 256 | 100 | `partial_score` *(binary + partial)* | [Mobile Benchmarks](envs/mobile.md#mobilegym) |
| **Computer** | OSWorld-Verified | `osworld-verified` | 361 | 100 | `success_rate` | [Computer Benchmarks](envs/computer.md#osworld-verified) |
| **Computer** | OSWorld 2.0 | `osworld-2` | 108 | 500 | `binary_and_partial` | [Computer Benchmarks](envs/computer.md#osworld-20) |
| **Computer** | MacAgentBench | `macagentbench` | 676 | 50 | `success_rate` | [Computer Benchmarks](envs/computer.md#macagentbench) |
| **Computer** | MyPCBench | `mypcbench` | 184 | 100 | `perfect_task_rate` | [Computer Benchmarks](envs/computer.md#mypcbench) |
| **Web** | WebVoyager | `webvoyager` | 643 | 15 | `success_rate` | [Web Benchmarks](envs/web.md#webvoyager) |
| **Web** | Online-Mind2Web | `onlinemind2web` | 123 | 50 | `success_rate` | [Web Benchmarks](envs/web.md#online-mind2web) |
| **Web** | WebArena | `webarena` | 812 | 30 | `success_rate` | [Web Benchmarks](envs/web.md#webarena) |
| **Web** | Mocked Apps *(bundled)* | `mocked-apps` | 837 | Task-dependent | `binary_and_partial` | [Mocked Apps](mocked-apps.md#evaluate-a-task) |

---

## Prerequisites

- **MAI-Gym CLI** installed on the evaluation host ([Install the CLI](index.md#install-the-cli)).
- **MobileWorld Android backend** with a working snapshot and native verifier, healthy and exposing your target task ([Prepare and check the backend](index.md#prepare-a-mobileworld-environment)).
- **Vision model** available either as:
  - an **OpenAI-compatible API** (model ID, API base URL, and `OPENAI_API_KEY` if required), or
  - a **local Hugging Face checkpoint** on a GPU host with SGLang installed.

---

## 1. Connect the model and backend

From the repository root, copy the sample configs to a working directory outside the checkout. Keep both files in the same directory so the relative `manager_config: mobileworld.yaml` path resolves automatically.

```bash
EVAL_DIR="$HOME/mai-gym-eval"
mkdir -p "$EVAL_DIR"
cp -i configs/eval.yaml configs/mobileworld.yaml "$EVAL_DIR/"
```

Update the two files for your environment:

| File | Key fields to configure |
| :--- | :--- |
| `eval.yaml` | Set `model.name` to the served model ID and `model.base_url` to your API base (e.g., `http://127.0.0.1:8000/v1`). Credentials are read via `env/OPENAI_API_KEY`. Optionally set `model.agent_variant` if your model uses a specific action format, and `observations.screenshot_transport` (`inline` for small local runs, `oss` for large multi-node fleets). |
| `mobileworld.yaml` | Set `devices[0].connection.attach.backends[0].url` to your Android backend URL (e.g., `http://127.0.0.1:6800`). The sample manager listens on port `8090`. |

> [!NOTE]
> **Backend vs. Manager:** The **backend** runs the Android device/emulator. The **manager** leases devices to the runner, resets state per task, and records trajectories. When `manager_config: mobileworld.yaml` is set, `mai-gym eval` starts a local manager automatically for the run. If you already run a persistent manager (for example, a [Kubernetes fleet](kubernetes.md)), replace `manager_config` with `manager_url`.

---

<a name="preflight-and-first-task"></a>

## 2. Run one task

Start with `SetAlarmTask`, a MobileWorld Clock task with a native state verifier ([upstream task definition](https://github.com/Tongyi-MAI/MobileWorld/blob/main/src/mobile_world/tasks/definitions/native/set_alarm.py)). Confirm it is registered on your backend (or query `GET /tasks` on an existing MAI-Gym manager):

```bash
curl -fsS http://127.0.0.1:6800/task/list | uv run python -m json.tool
```

If your manifest differs, pick any GUI-only task ID from the response. Add `tasks` under `benchmarks.mobileworld` in `$EVAL_DIR/eval.yaml`:

```yaml
benchmarks:
  mobileworld:
    manager_config: mobileworld.yaml
    num_envs: 1
    tasks: [SetAlarmTask]
  osworld-verified:
    manager_url: http://127.0.0.1:8091
    num_envs: 1
```

> [!NOTE]
> `tasks` selects **which** tasks to run (omit `tasks` to run all discovered tasks for that benchmark). `--trials` controls how many independent trials to run for **each** selected task.

Launch a 1-trial evaluation and save outputs to `$EVAL_DIR/runs`:

```bash
uv run mai-gym eval --config "$EVAL_DIR/eval.yaml" \
  --benchmarks mobileworld --trials 1 --output "$EVAL_DIR/runs"
```

---

<a name="3-read-the-verdict-and-replay-the-episode"></a>

## 3. Read the evaluation outcome and replay the episode

When the run finishes, the CLI prints the path to `results.json` after `Evaluation results:`.

### Understanding evaluation metrics & automatic retries

| Field | Meaning |
| :--- | :--- |
| `metric` | Primary native metric for the benchmark (see table above). |
| `score` | Primary benchmark score (`partial_score` for MobileGym; `success_rate` otherwise). |
| `tasks` | Total number of tasks evaluated in the benchmark entry. |
| `success_rate` | Native full-success rate across scored tasks (`1.0` for a single passing task). |
| `partial_score` | Mean weighted partial reward across scored tasks. |
| `unscored_trials` | Attempts that still lack a score after retries (`0` indicates a clean infrastructure run). |
| `pass_at_n` | *(Multi-trial)* Fraction of scored tasks with at least one successful trial (`pass@n`). |
| `pass_all_n` | *(Multi-trial)* Fraction of fully scored tasks that succeeded on every trial (`pass^n`). |

- **Environment-failure isolation & retries:** When an attempt terminates due to an infrastructure error rather than an agent mistake, `mai-gym eval` records it as unscored and automatically reruns it up to `--max-retry-rounds` times (default: `5`; pass `0` to disable retries). Reruns remain visible in the dashboard, and the CLI exits with an error if any `unscored_trials` remain after retries.
- A **scored failure** still produces a valid native evaluation outcome and a full replayable trajectory.
- For a per-task breakdown, open `mobileworld/<model>/passn_report_*/summary.md` in the same run directory.

### Replay in the trajectory dashboard

To inspect screenshots, model responses, parsed actions, and evaluator outputs step by step:

1. Point `managers[].traj_root` in the dashboard config to the manager's trajectory directory:
   - **Auto-started manager:** use the absolute path to the `mobileworld/` directory beside the printed `results.json`.
   - **Persistent manager:** use its configured trajectory storage path (accessible from the dashboard host).
2. Start the Rust server and indexer, then open the run ID printed by the CLI. See [Dashboard setup](../../../src/dashboard-rs/README.md#build-and-run).

> [!IMPORTANT]
> **Validating a new backend?** Complete at least one task with a native full-success outcome, then reset the environment and verify that the initial state is cleanly restored (retaining failed attempts alongside success and reset logs). See the [Environment validation protocol](../../../skills/envs.md#verification-protocol).

---

## 4. Add OSWorld-Verified

To evaluate mobile and desktop tasks in a single command, prepare an OSWorld-Verified manager ([Computer Benchmarks](envs/computer.md#osworld-verified) · [setup skill](../../../skills/envs/osworld.md)) and point `benchmarks.osworld-verified.manager_url` in `$EVAL_DIR/eval.yaml` to its address (default `http://127.0.0.1:8091`; see [`configs/eval.yaml`](../../../configs/eval.yaml)). Add one registered OSWorld task to its `tasks` list to validate both domains together:

```bash
uv run mai-gym eval --config "$EVAL_DIR/eval.yaml" \
  --benchmarks mobileworld osworld-verified --trials 1 --output "$EVAL_DIR/runs"
```

The resulting `results.json` contains a separate entry for each benchmark. Remove each `tasks` filter when ready to run every task exposed by your managers.

> [!NOTE]
> For judge-based benchmarks, configure the native judge model and credentials separately from the evaluated agent model. See [Benchmarks Overview](environments.md).

---

## Model and Protocol Options

### Action format (`model.agent_variant`)

The default `unified` agent automatically selects the appropriate mobile (`mai_gym_mobile_use_windows_wise`) or desktop (`mai_gym_computer_use`) prompt and action parser for MobileWorld and OSWorld-Verified; MobileGym uses a native touch policy, and web benchmarks default to `general_e2e_agent`. List all registered agent variants:

```bash
uv run python -c 'from mai_gym.agents import list_agents; print(*sorted(list_agents()))'
```

### Screenshot transport (`observations.screenshot_transport`)

- `inline` *(default)*: sends screenshots directly inside HTTP responses—simplest for local or small-scale runs.
- `oss`: uploads screenshots to object storage and passes references through control messages, keeping manager bandwidth low at large fleet scales (requires an `observations.oss` block with credentials via `env/...`).

### OpenAI-compatible API override

Override the endpoint and served model ID directly from the command line:

```bash
uv run mai-gym eval --config "$EVAL_DIR/eval.yaml" \
  --model https://model-provider.example/v1 --model-name your-served-model \
  --benchmarks mobileworld --trials 1
```

### Local Hugging Face checkpoint (auto-served with SGLang)

On a GPU host with SGLang installed, pass the local checkpoint directory to `--model`—the CLI launches and tears down SGLang automatically for the duration of the run.

To reproduce the 10-benchmark evaluation from [Paper §3.3](https://arxiv.org/html/2610.20939v1#S3.SS3), point `MAI_GYM_CONFIG` to a YAML declaring all ten benchmark managers and their task selections:

```bash
export MAI_GYM_CONFIG=/absolute/path/to/prepared-eval.yaml
uv run --with sglang mai-gym eval --model /absolute/path/to/hf-checkpoint \
  --benchmarks mobileworld androidworld mobilegym \
    osworld-verified osworld-2 macagentbench mypcbench \
    webvoyager onlinemind2web webarena --trials 3
```

---

## Next Steps

- Debug individual tasks or control retries directly: [Rollout CLI](rollout.md)
- Inspect trajectories visually or query the diagnosis API: [Trajectory Dashboard](../../../src/dashboard-rs/README.md#http-diagnosis-api)
- Scale MobileWorld across multiple nodes: [Kubernetes Fleets](kubernetes.md)
- Evaluate Mocked Apps locally via `--benchmarks mocked-apps`: [Mocked Apps](mocked-apps.md#evaluate-a-task)
- Train a GUI agent with online RL: [RL Training Guide](training.md)
