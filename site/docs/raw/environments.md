# Benchmarks Overview

MAI-Gym integrates **10 standard GUI benchmarks** spanning **Mobile**, **Computer**, and **Web** environments alongside **Mocked Apps** (*Real-World Clones* and *42 bundled browser-based Novel Apps*). Every environment is exposed behind the same `EnvClient` interface and evaluated with the exact harness used for training, while preserving each benchmark's native step budget, action semantics, and verifier.

---

## Supported Suites at a Glance

Use the **CLI ID** in `eval.yaml` under `benchmarks:` and on the command line via `mai-gym eval --benchmarks <id>`.

| Domain | Benchmark / Suite | CLI ID | Tasks | Max Steps | Verifier Type | Native Metric | Guide | Setup Skill |
| :--- | :--- | :--- | ---: | :--- | :--- | :--- | :--- | :--- |
| **Mobile** | [MobileWorld](envs/mobile.md#mobileworld) | `mobileworld` | 117 | 100 | Programmatic state | `success_rate` | [Mobile](envs/mobile.md) | [`mobileworld.md`](../../../skills/envs/mobileworld.md) |
| **Mobile** | [AndroidWorld](envs/mobile.md#androidworld) | `androidworld` | 116 | Task-dependent | Programmatic state | `success_rate` | [Mobile](envs/mobile.md) | [`androidworld.md`](../../../skills/envs/androidworld.md) |
| **Mobile** | [MobileGym](envs/mobile.md#mobilegym) | `mobilegym` | 256 | 100 | Programmatic state | `partial_score` *(binary + partial)* | [Mobile](envs/mobile.md) | [`mobilegym.md`](../../../skills/envs/mobilegym.md) |
| **Computer** | [OSWorld-Verified](envs/computer.md#osworld-verified) | `osworld-verified` | 361 | 100 | Programmatic state | `success_rate` | [Computer](envs/computer.md) | [`osworld.md`](../../../skills/envs/osworld.md) |
| **Computer** | [OSWorld 2.0](envs/computer.md#osworld-20) | `osworld-2` | 108 | 500 | Programmatic state | `binary_and_partial` | [Computer](envs/computer.md) | [`osworld_2.md`](../../../skills/envs/osworld_2.md) |
| **Computer** | [MacAgentBench](envs/computer.md#macagentbench) | `macagentbench` | 676 | 50 | Programmatic checkpoints | `success_rate` | [Computer](envs/computer.md) | [`macagentbench.md`](../../../skills/envs/macagentbench.md) |
| **Computer** | [MyPCBench](envs/computer.md#mypcbench) | `mypcbench` | 184 | 100 | Rubric VLM judge | `perfect_task_rate` | [Computer](envs/computer.md) | [`mypcbench.md`](../../../skills/envs/mypcbench.md) |
| **Web** | [WebVoyager](envs/web.md#webvoyager) | `webvoyager` | 643 | 15 | VLM trajectory judge | `success_rate` | [Web](envs/web.md) | [`webvoyager.md`](../../../skills/envs/webvoyager.md) |
| **Web** | [Online-Mind2Web](envs/web.md#online-mind2web) | `onlinemind2web` | 123 | 50 | VLM judge (`WebJudge`) | `success_rate` | [Web](envs/web.md) | [`onlinemind2web.md`](../../../skills/envs/onlinemind2web.md) |
| **Web** | [WebArena](envs/web.md#webarena) | `webarena` | 812 | 30 | Programmatic state | `success_rate` | [Web](envs/web.md) | [`webarena.md`](../../../skills/envs/webarena.md) |
| **Web** | [Mocked Apps](mocked-apps.md) *(bundled)* | `mocked-apps` | 837 | Task-dependent | Programmatic state | `binary_and_partial` | [Mocked Apps](mocked-apps.md) | [`configs/mocked-apps.yaml`](../../../configs/mocked-apps.yaml) |

---

## Domain Guides

- **[Mocked Apps](mocked-apps.md)** — Resettable **Real-World Clones** and **42 bundled browser-based Novel Apps** (837 tasks with native verifiers and reference oracle trajectories). The fastest way to evaluate or train locally with zero Docker/KVM setup.
- **[Mobile Benchmarks](envs/mobile.md)** — **MobileWorld**, **AndroidWorld**, and **MobileGym**: Android container/emulator and mobile-viewport suites covering cross-app workflows, system settings, and Chinese/English touch UIs.
- **[Computer Benchmarks](envs/computer.md)** — **OSWorld-Verified**, **OSWorld 2.0**, **MacAgentBench**, and **MyPCBench**: Ubuntu and macOS desktop VM suites utilizing MAI-Gym's **Hybrid Action Space (GUI + CLI)**.
- **[Web Benchmarks](envs/web.md)** — **WebVoyager**, **Online-Mind2Web**, and **WebArena**: live-web and self-hosted multi-site browser benchmarks with native VLM judges and programmatic web evaluators.

---

## Reproducible Setup & Coding-Agent Skills

GUI benchmarks are sensitive to upstream source revisions, OS images, application snapshots, and task manifests. MAI-Gym separates **benchmark runtime isolation** (each benchmark runs in its own worker process/interpreter or container with pinned assets) from the shared `mai-gym` runner and manager:

- **Immutable profile locks:** Native workers verify source commits, task manifest SHA-256 hashes, Docker image digests, and VM/browser asset checksums before provisioning.
- **Single-task reset validation:** Before scaling to a full benchmark or RL fleet, validate one real manifest task end to end—confirming full native success, clean episode reset, and absence of residual state.
- **Judge isolation:** Benchmarks that require a VLM trajectory judge (`MyPCBench`, `WebVoyager`, `Online-Mind2Web`) configure the judge endpoint and API key environment variable independently from the evaluated agent model.

> [!TIP]
> **Full preparation instructions & coding-agent skills:**
> Detailed host checklists, JSON profile lock schemas, worker templates, and reset verification protocols live in [`skills/envs.md`](../../../skills/envs.md) and the per-benchmark guides under [`skills/envs/`](../../../skills/envs). You can read them directly or point Claude Code, Codex, or Cursor at [`skills/README.md`](../../../skills/README.md) to automate setup.

---

## Next Steps

- Run a single- or multi-benchmark evaluation: [Evaluation Guide](evaluation.md)
- Try the zero-Docker local suite first: [Mocked Apps](mocked-apps.md)
- Scale Android fleets across Kubernetes: [Kubernetes Fleets](kubernetes.md)
- Integrate a new benchmark adapter: [Extending MAI-Gym](extending.md)
