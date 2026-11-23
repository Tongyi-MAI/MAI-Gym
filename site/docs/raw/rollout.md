# Rollout CLI

[Home](../../../README.md) · [Quickstart](index.md) · [Evaluation](evaluation.md) · [Training](training.md) · **Rollout** · [Kubernetes](kubernetes.md) · [Concepts](concepts.md)

`mai-gym rollout` runs GUI tasks directly for agent debugging, trajectory collection, and controlled multi-pass execution. It records screenshots, actions, and verifier scores on the exact same task runner used by `mai-gym eval`.

### When to use `rollout` vs. `eval` vs. `train`

| Command | Best for |
| :--- | :--- |
| `mai-gym rollout` | Direct single-environment trajectory collection, custom agent debugging, `--set` overrides, and fine-grained retry/resume control. |
| `mai-gym eval` | Standardized single- or multi-benchmark evaluations with native benchmark defaults, auto-served SGLang, and cross-benchmark summaries ([Evaluation Guide](evaluation.md)). |
| `mai-gym train` | Updating policy weights via online RL with Slime ([RL Training Guide](training.md)). |

---

## Configuration

Prepare the target backend and native verifier using the relevant benchmark guide ([Benchmarks Overview](environments.md) · [`skills/envs.md`](../../../skills/envs.md)), then adapt a local copy of [`configs/rollout.yaml`](../../../configs/rollout.yaml):

| Section | What it configures |
| :--- | :--- |
| `agent` | Policy type (`type`), served model ID (`model_name`), OpenAI-compatible endpoint (`base_url`), credential env reference (`api_key: env/...`), and step budget (`max_steps`). |
| `rollout` | Manager URL (`manager_url`), `env_type`, task filters, concurrency (`num_envs`), evaluation mode (`built_in`, `llm_judge`, `both`), and screenshot transport. |
| `manager` & `devices` | Local manager and device backend definitions used when auto-starting a manager. |

> [!IMPORTANT]
> Keep machine-specific configuration files outside the repository checkout and supply secrets only via runtime environment variables (for example, `env/OPENAI_API_KEY`).

---

## Running Rollouts

From the repository root, with your model endpoint and benchmark backend ready:

```bash
uv run mai-gym rollout --config /absolute/path/to/rollout.yaml
```

Filter specific registered task IDs and override concurrency:

```bash
uv run mai-gym rollout --config /absolute/path/to/rollout.yaml \
  --tasks your-task-id --num-envs 1
```

> [!NOTE]
> Use `--manager-url <URL>` (alias: `--server`) to override the manager URL at runtime. When the resolved manager URL is local and unreachable, `mai-gym rollout` automatically starts a manager for the lifetime of the run using the supplied YAML configuration.

---

## Passes, Scoring, Retries, and Resume

| Option | Behavior |
| :--- | :--- |
| `--pass-n N` | Run `N` independent passes (default: `1`). Multi-pass runs write aggregate `results.json` and `summary.md` reports. |
| `--evaluation-mode MODE` | Choose `built_in` (default native verifier), `llm_judge`, or `both` (judge modes require a judge-capable manager). |
| `--retry-until-scored` | Re-run only unscored attempts (`score is None`) caused by environment failures until all cells are scored or `--max-retry-rounds` is reached. |
| `--max-retry-rounds N` | Cap retry rounds (excluding the initial round); `0` allows unbounded retries when `--retry-until-scored` is enabled. |
| `--exact-train-id` | Use `rollout.train_id` verbatim without a random suffix. Set it to a prior run's printed ID to resume that run. |
| `--no-resume` | Re-run every requested task even when a saved `result.txt` already exists. |
| `--tasks-dir PATH --domains DOMAINS` | Discover generated OSWorld-style task files and optionally filter comma-separated domains. |
| `--set KEY=VALUE` | Override any YAML field by dotted path (repeatable, e.g., `--set agent.model_name=foo`). Explicit CLI flags take precedence. |

List all available flags and defaults:

```bash
uv run mai-gym rollout --help
```

---

## Output & Trajectory Inspection

At startup, `mai-gym rollout` prints the resolved `train_id` and `Run log root`.

- **Aggregate reports** (multi-pass or `--retry-until-scored`) distinguish native full success, partial scores, and unscored technical failures.
- **Visual replay:** Point the [Rust trajectory dashboard](../../../src/dashboard-rs/README.md) at the printed trajectory root to inspect screenshots, model reasoning traces, executed actions, and verifier scores.
