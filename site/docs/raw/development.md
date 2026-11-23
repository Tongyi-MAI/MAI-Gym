# Development & Repository Layout

[Home](../../../README.md) · [Quickstart](index.md) · [Evaluation](evaluation.md) · [Training](training.md) · [Rollout](rollout.md) · [Kubernetes](kubernetes.md) · [Concepts](concepts.md)

A reference to the **MAI-Gym** codebase layout, minimal mobile-agent extraction subset, and local development commands.

---

## Repository Layout

| Path | Contents |
| :--- | :--- |
| [`src/mai_gym/`](../../../src/mai_gym) | Core environment manager, `EnvClient`, benchmark adapters, agent harnesses, CLI, and evaluation runner |
| [`src/mai_gym_rl/`](../../../src/mai_gym_rl) | Online RL adapters, rollout buffer bridges, and Slime integration |
| [`src/dashboard-rs/`](../../../src/dashboard-rs) | Embedded Rust/SQLite trajectory dashboard server, indexer, frontend, and Rust tests |
| [`resources/slime/`](../../../resources/slime) | Bundled Slime RL training runtime, plugins, and third-party license notices |
| [`resources/novel-apps/`](../../../resources/novel-apps/) | 42 Novel Mocked Apps (837 tasks, native verifiers, and reference oracle trajectories) |
| [`configs/`](../../../configs) | Example manager, evaluation, rollout, Mocked Apps, and RL training YAML presets |
| [`examples/`](../../../examples) | Sample custom task definitions (such as `examples/mobileworld/open_settings.py`) |
| [`skills/`](../../../skills/README.md) | Agent-readable setup guides for benchmark preparation, evaluation, and training |
| [`site/docs/`](index.md) | Documentation website and Markdown sources |

---

## Minimal Mobile-Use Agent Subset

For lightweight deployments that only require the mobile-use agent loop, prompt templates, and action parser (without the environment manager or RL stack), extract this minimal subset:

```text
src/mai_gym/agents/
├── implementations/
│   ├── mai_gym_mobile_use.py
│   └── mai_gym_mobile_use_windows_wise.py
├── parsing/
│   └── mai_gym_mobile_use.py
└── prompts/
    ├── mai_gym_mobile_use_system.j2
    └── mai_gym_mobile_use_windows_wise_system.j2
```

---

## Local Verification

### Python

Requires the project environment synced via `uv sync --locked --python 3.12`:

```bash
uv run ruff check src/
uv run ruff format --check src/
```

### Trajectory Dashboard (Rust)

From [`src/dashboard-rs/`](../../../src/dashboard-rs/README.md):

```bash
cargo fmt --check
cargo test --locked
```

---

## Licensing & Attribution

When modifying or redistributing components, preserve upstream notices—especially third-party license and attribution files for the bundled Slime runtime under [`resources/slime/`](../../../resources/slime). See [Licensing](licensing.md) for the full component breakdown.
