# Extending MAI-Gym

[Home](../../../README.md) · [Quickstart](index.md) · [Development](development.md) · **Extending MAI-Gym** · [Platform & Core Concepts](concepts.md)

MAI-Gym isolates agent logic, task definitions, environment adapters, and deployment specifications behind explicit extension points (Paper §3.1 & Appendix A.2). Extending any one layer automatically works across `mai-gym eval`, `mai-gym rollout`, and `mai-gym train`.

---

## Four Extension Points

| Extension Point | What You Implement | Where It Lives |
| :--- | :--- | :--- |
| **1. Add Agent Harness** | Context construction, policy invocation, output parsing, and action dispatch. Once registered, all three workloads (`eval`, `rollout`, `train`) use it unchanged. | [`src/mai_gym/agents/`](../../../src/mai_gym/agents) |
| **2. Add Tasks** | Instruction, required initial assets, setup/reset logic, and programmatic verifier (or judge rubric) for an existing environment. | [`examples/mobileworld/open_settings.py`](../../../examples/mobileworld/open_settings.py) · `POST /register_task_set` |
| **3. Integrate Environments** | An environment adapter implementing the 5-operation contract below, preserving the benchmark's native initialization, action space, and verifier. | [`src/mai_gym/envs/`](../../../src/mai_gym/envs) |
| **4. Customize Environments** | An environment specification listing installed apps, supporting services, and launch/restore settings—reusing an existing adapter and deployment backend. | Device YAML (`devices[].env_config`, `launch`) |

---

## Environment Contract & Responsibilities

Every environment adapter implements five core operations behind the manager API (Paper Appendix A.2):

| Operation | Role |
| :--- | :--- |
| `reqEnvs(spec, n)` | Provisions or attaches `n` environment instances matching `spec`. |
| `observe(h)` | Returns the current observation (screenshot, optional CLI command output, metadata) for session `h`. |
| `reset(h, x)` | Restores session `h` to the reproducible initial state `s_0` of task `x`. |
| `step(h, a)` | Applies an action (or action batch) `a` to session `h` and returns updated state. |
| `eval(h, x)` | Invokes the native task verifier (code verifier and/or VLM judge) for task `x` on session `h`. |

Above this contract, MAI-Gym implements scheduling, leasing, trajectory persistence, and reward delivery **once** for all platforms:

| Concern | Environment Adapter | MAI-Gym Manager & Runner |
| :--- | :--- | :--- |
| **Capacity** | Reports health and available instances | Leases sessions, reclaims expired capacity, and assigns attempts |
| **Isolation** | Initializes task state and isolates concurrent instances | Keeps one attempt bound to one session for its entire lifetime |
| **Observation** | Returns screenshots, command output, and metadata | Selects screenshot transport (`inline` / `oss`) and records trajectories |
| **Action** | Executes the declared action vocabulary | Parses model outputs, dispatches idempotent actions, and logs results |
| **Task** | Installs task state and invokes its native verifier | Samples tasks, forms rollout groups, and delivers rewards to training |

### Reliability Guarantees
- **Session leases:** Prevent replaced or timed-out workers from mutating an instance owned by a newer attempt.
- **Idempotent actions:** Mutating requests carry caller-supplied action identifiers so network retries apply an action at most once.
- **Fault isolation:** Every operation has an explicit deadline, verifiers run in isolated process groups with resource limits, and infrastructure faults are recorded separately from zero rewards.

---

## Adding a Custom Task

### Option A: Live Task Registration (MobileWorld)
On the MAI-Gym manager, upload a task ZIP bundle via `POST /register_task_set` (or `POST /task/register` on the customized MobileWorld backend image) with a `set_id` and `package` archive. Registered tasks become immediately discoverable via `GET /tasks` without restarting the container.

### Option B: Custom Python Task Class
See [`examples/mobileworld/open_settings.py`](../../../examples/mobileworld/open_settings.py) for a minimal task class with a deterministic verifier checking the foreground Android package (`com.android.settings`). To validate any new task before training, verify the **initial → completed → reset** sequence (`0 → 1 → 0`): the verifier returns `0` at initial state `s₀`, `1` after task completion, and `0` again after environment reset.

---

## Adding an Agent Harness

1. Implement a policy class, prompt template (`.j2`), and action parser under [`src/mai_gym/agents/`](../../../src/mai_gym/agents).
2. Register the variant in `src/mai_gym/agents/variants.py` and verify it appears in:
   ```bash
   uv run python -c 'from mai_gym.agents import list_agents; print(*sorted(list_agents()))'
   ```
3. Reference the registered name via `model.agent_variant` in `eval.yaml` / `train.yaml` or `agent.type` in `rollout.yaml`.

For a minimal standalone mobile-use harness extraction, see [Minimal Mobile-Use Agent Subset](development.md#minimal-mobile-use-agent-subset).

---

## Adding a New Benchmark Environment

1. Implement a backend and strategy under `src/mai_gym/envs/<name>/` and register its action space (`MOBILE_USE`, `COMPUTER_USE`, or `WEB_USE`) in `src/mai_gym/agents/action_space.py`.
2. Declare its benchmark spec (CLI ID, `env_type`, default `max_steps`, native `metric`) in `src/mai_gym/benchmarks.py`.
3. Provide an immutable profile lock schema and single-task validation guide under `skills/envs/<name>.md` following [Benchmark Setup](../../../skills/envs.md#verification-protocol).

---

## Next Steps

- Review repository structure and test commands: [Development & Layout](development.md)
- Read the task & action space definitions: [Core Concepts](concepts.md)
- Inspect existing benchmark suites and setup guides: [Benchmarks Overview](environments.md) · [`skills/envs.md`](../../../skills/envs.md)
