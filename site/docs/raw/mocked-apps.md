# Mocked Apps

In MAI-Gym, **Mocked Apps** are self-contained, resettable application environments that let GUI agents practice and be evaluated thousands of times without external side effects. Mocked Apps comprise two categories:

1. **Real-World Clones** — Resettable clones of production applications studied in the [technical report](https://arxiv.org/abs/2610.20939) (spanning travel, banking, ride-hailing, food delivery, local services, and social media) for safe practice and clone-to-real transfer ([side-by-side comparison](https://tongyi-mai.github.io/MAI-Gym/#mock-vs-real)).
2. **Novel Apps** — **42 original browser-based HTML5 applications** (**837 verified tasks**, deterministic programmatic verifiers, and reference oracle action sequences) bundled in [`resources/novel-apps/`](../../../resources/novel-apps/) to evaluate adaptation to unfamiliar interfaces through reward alone ([interactive gallery](https://tongyi-mai.github.io/MAI-Gym/#apps)).

Open any bundled app's `index.html` directly in a browser—for example, [Squirrel Eats](../../../resources/novel-apps/squirrel_eats/index.html) to browse restaurants and place an order—or [browse all 42 apps](../../../resources/novel-apps/).

---

## Evaluate a task

Run these commands from the MAI-Gym repository root. Set `MODEL_URL` to an OpenAI-compatible vision endpoint and `MODEL_NAME` to its served model ID. If the endpoint requires authentication, export `OPENAI_API_KEY`.

```bash
uv sync --locked
uv run playwright install chromium
uv run mai-gym eval --config configs/mocked-apps.yaml --benchmarks mocked-apps \
  --model "$MODEL_URL" --model-name "$MODEL_NAME" --trials 1
```

The [preset](../../../configs/mocked-apps.yaml) starts the web backend and manager locally. It runs `kestrel_bank/T01_freeze` in Chromium, verifies the card state, and writes the evaluation outcome and weighted partial score to the printed `results.json`. Select other tasks by their `<app-id>/<task-id>` in the preset:

```yaml
benchmarks:
  mocked-apps:
    tasks: [kestrel_bank/T01_freeze, kestrel_bank/T02_query_payday]
    num_envs: 1
```

Task IDs, instructions, and step budgets are listed in each app's `tasks.json`. Omit `tasks` to evaluate all 837 tasks. Each episode resets to its declared seed and initial state.

## Use the manager for training

Start the same environment manager with `uv run mai-gym serve --config configs/mocked-apps.yaml`. In your training task pool, include rows with `env_type: "mocked-apps"`:

```json
{"prompt":"Freeze your card. Leave everything else on the card page alone.","label":"","metadata":{"env_type":"mocked-apps","task_id":"kestrel_bank/T01_freeze"}}
```

Set `model.agent_variant: mai_gym_computer_use` in the training configuration. Under `tasks`, point the pool at the running manager. The address below assumes the trainer can reach the manager on loopback.

```yaml
tasks:
  - pool: mocked-apps.jsonl
    env_type: mocked-apps
    manager_url: http://127.0.0.1:8092
    num_sessions: 1
```

Run `mai-gym train --config train.yaml` inside the training container. [Online RL training guide](training.md)

## App files

Each app directory includes three files, and the directory name is the app ID. Keep them together when copying an app.

| File | Contents |
| --- | --- |
| `index.html` | The interactive app, including its interface and state |
| `tasks.json` | Task instructions, initial conditions, and success criteria |
| `oracle.json` | Reference actions keyed by task ID |

The bundled Novel Apps collection contains 837 tasks across 42 apps. To try a task or watch a reference solution in the project demo, open the [interactive gallery](https://tongyi-mai.github.io/MAI-Gym/#apps).

Released under the [MIT License](../../../LICENSE).
