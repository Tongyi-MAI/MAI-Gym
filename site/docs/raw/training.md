# Online RL Training Guide

[Home](../../../README.md) · [Quickstart](index.md) · [Evaluation](evaluation.md) · **Training** · [Rollout](rollout.md) · [Kubernetes](kubernetes.md) · [Concepts](concepts.md)

Train GUI agents with online reinforcement learning using a single YAML configuration and the bundled [Slime](../../../resources/slime/README.md) runtime (`ghcr.io/tongyi-mai/mai_gym_slime:v1`, combining Megatron-LM for training with SGLang for rollouts). Rollout workers execute GUI tasks across mobile, computer, and web managers on the same harness used for evaluation, collect deterministic verifier rewards, update policy weights, and export Hugging Face checkpoints directly to your host.

> [!TIP]
> **Prefer agent-assisted setup?**
> Use the [coding-agent training prompt](../../../skills/README.md#prepare-training) in Claude Code, Codex, or Cursor. Looking for a minimal 4-GPU, single-task verification run first? Start with the [README training quickstart](../../../README.md#train) using [`configs/train-quickstart.yaml`](../../../configs/train-quickstart.yaml).

---

## How MAI-Gym Online RL Works

Long, screenshot-heavy GUI trajectories introduce three challenges beyond text-only RL. MAI-Gym addresses them directly in the rollout and training pipeline (Paper §3.5):

1. **Policy-Adaptive Task Sampling (`sampling.*`):** Group-based algorithms such as GRPO receive zero gradient when all `G` attempts on a task succeed or all fail. Before each round, the scheduler tracks per-task success rates: unseen and frontier tasks receive the full `algorithm.group_size` (`G = 8`) attempts, while never-solved tasks and mastered tasks (success rate `≥ η`, `success_threshold: 0.8`) receive small nonzero probe budgets (`fail_budget`, `mastered_budget`) so they are rechecked as the policy evolves. Advantages are computed only after all attempts in a group finish, avoiding completion-order or trajectory-length bias.
2. **Window-Wise History & Packing (`history.*`):** Instead of dropping the oldest screenshot every step (which invalidates the KV cache on every turn) or packing an entire 50–100-step multimodal trajectory into one sequence (which exceeds GPU memory), MAI-Gym groups turns into windows of `K` steps (`window_size: 10`) overlapping by `O` steps (`overlap: 2`). Within a window, observations are appended for KV-cache reuse; at window boundaries, older screenshots collapse to text placeholders while text history stays intact. During training, each window becomes one causal training sample with loss applied only to new turns (e.g., turns 1–10 in window 1, turns 11–18 in window 2 with turns 9–10 as context), replaying exact rollout tokens (`token_native_history: true`) and falling back to step-wise samples if prefixes diverge. Screenshots are decoded lazily per micro-batch with selective vision-tower activation recomputation.
3. **Train–Rollout Consistency:** Launch-time checks catch multimodal preprocessing mismatches, and during training MAI-Gym compares trainer and rollout log-probabilities on every generated token, monitoring the **`k₃` KL divergence estimator** to immediately surface kernel corruption, misaligned packed histories, or engine/trainer skew.

### Supported RL Algorithms (`algorithm.name`)

| Algorithm | Value | Notes |
| :--- | :--- | :--- |
| **GRPO** *(default)* | `grpo` | Group-normalized terminal reward baseline (`group_size >= 2`), per-token clipped surrogate loss, optional KL loss term (`kl_loss_coef`). Used for MAI-Gym-9B. |
| **DAPO** | `dapo` | Decoupled clip / dynamic sampling variant built on the GRPO estimator. |
| **GSPO** | `gspo` | Sequence-level importance-ratio policy optimization. |
| **REINFORCE++** | `reinforce++` | Critic-free baseline advantage estimator. |
| **PPO** | `ppo` | Clipped policy optimization with a learned value function. |
| **SAO** | `sao` | Asynchronous single-rollout updates with a learned value critic (requires `trainer.entrypoint`). |

---

## Prerequisites & Presets

- **Linux host with NVIDIA GPUs** and Docker GPU support (`--gpus all`).
- **Complete Hugging Face checkpoint** (weights + tokenizer files) for the Qwen3.5-9B architecture (e.g., [Qwen/Qwen3.5-9B](https://huggingface.co/Qwen/Qwen3.5-9B) or [Tongyi-MAI/MAI-Gym-9B](https://huggingface.co/Tongyi-MAI/MAI-Gym-9B)). For reproducing MAI-Gym-9B, the paper starts from an SFT warm-up checkpoint trained on mixed mobile and computer trajectories (Paper §4.1).
- **At least one verified benchmark manager** reachable from the training container. Verify one real task and its reset on every manager before launching training ([Evaluation preflight](evaluation.md#preflight-and-first-task)).

| Preset | Intended Use | Default Scale |
| :--- | :--- | :--- |
| [`configs/train-quickstart.yaml`](../../../configs/train-quickstart.yaml) | Single-task MobileWorld minimal verification run (GRPO, 1 round) | 1 node × 4 GPUs |
| [`configs/mai-gym-9b.yaml`](../../../configs/mai-gym-9b.yaml) | Joint mobile + computer RL starter preset (GRPO, window-wise history) | 4 nodes × 8 GPUs |

> [!NOTE]
> The starter preset uses inline screenshots. Set the trainer topology and the environment pool sizes to match your available resources. The CLI accepts `history.window` as an alias for `history.window_size`.

---

## 1. Prepare inputs on the host

Run the commands in this section on the **host**, from the MAI-Gym repository root. Create separate directories for read-only inputs (`$TRAIN_DATA`) and writable run outputs (`$TRAIN_RUNS`):

```bash
TRAIN_DATA="$HOME/mai-gym-data"
TRAIN_RUNS="$HOME/mai-gym-runs"
mkdir -p "$TRAIN_DATA" "$TRAIN_RUNS"
cp -i configs/mai-gym-9b.yaml "$TRAIN_DATA/train.yaml"
```

### Model checkpoint

Place the complete Hugging Face model directory (weights, config, and tokenizer) under `$TRAIN_DATA/hf-model/`.

### Task pools (JSONL or Parquet)

Place your task pools in `$TRAIN_DATA/` (the default preset expects `mobileworld.jsonl` and `osworld.jsonl`). Each row maps a natural-language task instruction to a registered task ID and environment type:

```json
{"prompt":"<task instruction>","label":"","metadata":{"task_id":"<registered task ID>","env_type":"mobileworld"}}
```

- Leave `label` empty when the benchmark's native verifier supplies the reward.
- Parquet task pools use the exact same schema.
- For combined multi-domain datasets, ensure each row's `metadata.env_type` matches its target manager pool.
- For custom MobileWorld tasks, install or register the task class on the emulator and verify its instruction first ([Register a custom training task](../../../README.md#train)).
- To train on the 42 bundled Novel Mocked Apps, point a pool at `env_type: mocked-apps` and a running `mai-gym serve --config configs/mocked-apps.yaml` manager ([Mocked Apps training](mocked-apps.md#use-the-manager-for-training)).

### Adapt `train.yaml`

Edit `$TRAIN_DATA/train.yaml` (see [`configs/mai-gym-9b.yaml`](../../../configs/mai-gym-9b.yaml)) to match your hardware, algorithm, and manager capacity:

| Section | Key settings |
| :--- | :--- |
| `model` | `checkpoint` (path or `env/MAI_GYM_CHECKPOINT`), `agent_variant` (default `unified`), `policy_variant` (`mai_ui` or `qwen35`), and `tool_parser`. |
| `tasks` | List of pools (`pool`, `env_type`, `manager_url`, `num_sessions`). Point each `manager_url` to a manager reachable from the training containers and cap `num_sessions` to the manager's capacity. |
| `algorithm` | `name` (`grpo`, `dapo`, `gspo`, `reinforce++`, `ppo`, `sao`), `group_size`, `learning_rate`, `kl_loss_coef` (applied to the loss, not the reward), `entropy_coef`, `per_token_loss`, and `reward_allocation`. |
| `sampling` | Policy-adaptive task sampling parameters: `success_threshold` (`η`), `fail_budget`, and `mastered_budget`. |
| `rollout` | `rounds` (set `1` for an initial validation run), `batch_size`, `concurrency`, `num_sessions`, `temperature`, `max_turns`, `max_total_length`, and `max_response_length`. |
| `history` | `window_size` (`K`, default `10`), `overlap` (`O`, default `2`, must satisfy `O < K`), `image_retention`, `token_native_history`, and `windowwise_train`. |
| `observations` | `screenshot_transport` (`inline` for small runs; `oss` + `observations.oss` credentials via `env/...` for large fleets). |
| `trainer` | `backend` (`slime`), `nodes`, `gpus_per_node`, `parallelism` (`tensor × pipeline × context × data == nodes × gpus_per_node`), `engine_gpus` (must divide `gpus_per_node`), `global_batch_size`, `save_interval`, `output` (fresh container path such as `/runs/first-run`), and optional `resume` / `ray_address`. |

---

<a name="start-the-container"></a>

## 2. Start the Slime container (Host)

From the repository root on the **host**, launch the prebuilt Slime container with the checkout mounted at `/workspace/MAI-Gym`, inputs read-only at `/data`, and outputs writable at `/runs`:

```bash
docker run --rm -it --gpus all --ipc=host --network=host \
  -v "$PWD:/workspace/MAI-Gym" \
  -v "$TRAIN_DATA:/data:ro" \
  -v "$TRAIN_RUNS:/runs" \
  -e MAI_GYM_ROOT=/workspace/MAI-Gym \
  -w /workspace/MAI-Gym \
  ghcr.io/tongyi-mai/mai_gym_slime:v1 bash
```

---

<a name="3-check-the-configuration-inside-the-container"></a>

## 3. Configure the environment (Container)

Run all remaining commands **inside the container**. Install the mounted MAI-Gym checkout in editable mode and export the paths referenced by `configs/mai-gym-9b.yaml`:

```bash
python -m pip install --no-deps -e .
export MAI_GYM_CHECKPOINT=/data/hf-model
export MAI_GYM_MOBILE_TASKS=/data/mobileworld.jsonl
export MAI_GYM_COMPUTER_TASKS=/data/osworld.jsonl
```

Validate your YAML configuration and inspect the resolved Slime command before touching Ray:

```bash
mai-gym train --config /data/train.yaml --dry-run
```

---

## 4. Launch training & locate checkpoints (Container)

### Start or connect to Ray

- **Single-node run:** start a local Ray head inside the container:
  ```bash
  ray start --head
  ```
- **Multi-node cluster:** connect to your existing Ray head, set `trainer.ray_address` in `train.yaml`, and ensure `/workspace/MAI-Gym`, `/data`, and `/runs` are mounted identically on every node.

Verify that Ray detects the GPU count requested in `train.yaml`, then launch training:

```bash
ray status
mai-gym train --config /data/train.yaml
```

### Output directory layout

With `trainer.output: /runs/first-run`, inspect `$TRAIN_RUNS/first-run/` **on the host**:

```text
$TRAIN_RUNS/first-run/
├── artifacts/
│   ├── adapter.yaml         # Resolved Slime adapter configuration
│   └── tasks.jsonl          # Combined task pool (when generated from tasks[])
├── checkpoints/             # Training state (includes latest_checkpointed_iteration.txt)
└── hf/
    └── iter_*/              # Exported Hugging Face checkpoints ready for eval
```

> [!NOTE]
> Use `trainer.save_interval: 1` during initial verification so a checkpoint and Hugging Face export are written after round 1.

---

## Resuming from a Checkpoint

To resume an interrupted or completed training run:

1. Keep compatible model and parallelism settings in `train.yaml`.
2. Set `trainer.resume` to the **container path** of the checkpoint directory containing `latest_checkpointed_iteration.txt` (for example, `/runs/first-run/checkpoints`).
3. Increase `rollout.rounds` to the desired total number of rounds and re-run:
   ```bash
   mai-gym train --config /data/train.yaml
   ```

---

## Next Steps

- Evaluate your exported checkpoint under `hf/iter_*/`: [Evaluation Guide](evaluation.md)
- Scale environment pools across nodes for high-concurrency RL: [Kubernetes Fleets](kubernetes.md)
- Inspect training trajectories and rewards visually: [Trajectory Dashboard](../../../src/dashboard-rs/README.md)
