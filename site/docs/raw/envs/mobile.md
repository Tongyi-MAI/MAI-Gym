# Mobile Benchmarks

MAI-Gym evaluates mobile GUI agents across three standard Android and mobile-viewport benchmarks—**MobileWorld**, **AndroidWorld**, and **MobileGym**—spanning both in-distribution/task-OOD Android container suites and out-of-distribution mobile environments. All three run through MAI-Gym's shared mobile touch harness (`click`, `double_click`, `long_press`, `drag`/`swipe`, `type`, `open`, `system_button`, `wait`, `terminate`).

> [!TIP]
> Looking for detailed host checklists, immutable profile lock JSONs, or coding-agent bring-up prompts? Each benchmark below links directly to its full setup skill under [`skills/envs/`](../../../../skills/envs) and [`skills/envs.md`](../../../../skills/envs.md).

---

## MobileWorld

**MobileWorld** ([upstream repository](https://github.com/Tongyi-MAI/MobileWorld)) is a realistic Android benchmark featuring rich multi-app mobile tasks (alarms, messaging, notes, email, calendar, shopping, and system utilities) backed by deterministic state initialization and native evaluators. In MAI-Gym, **MobileWorld (GUI-only)** serves as the primary mobile evaluation suite and scales seamlessly from a single local Docker/KVM container to **20,000+ concurrent ReDroid pods** on Kubernetes.

| Property | Value |
| :--- | :--- |
| **CLI ID (`--benchmarks`)** | `mobileworld` |
| **Tasks & Scope** | 117 GUI-only tasks across everyday Android applications |
| **Step Budget** | 100 interaction steps |
| **Verification & Metric** | Native programmatic state verifier · `success_rate` |
| **Runtime Stack** | Containerized Android emulator (Docker/KVM locally) or ReDroid on Kubernetes |
| **MAI-Gym-9B Result** | **78.4%** *(+60.4 over Qwen3.5-9B base · Task OOD)* |

### Running in MAI-Gym

Start a MobileWorld backend (see [Quickstart](../index.md#prepare-a-mobileworld-environment) or [Kubernetes Fleets](../kubernetes.md)), point `mobileworld.yaml` at its URL, and evaluate:

```bash
uv run mai-gym eval --config "$HOME/mai-gym-eval/eval.yaml" \
  --benchmarks mobileworld --trials 1
```

- **Host requirements:** Linux `x86_64` with Docker, KVM, and disk space for Android snapshots (or a remote Kubernetes pool).
- **Detailed setup & agent skill:** [`skills/envs/mobileworld.md`](../../../../skills/envs/mobileworld.md) · [`release.json`](../../../../src/mai_gym/envs/mobileworld/release.json)

---

## AndroidWorld

**AndroidWorld** ([upstream repository](https://github.com/google-research/android_world)) is a widely adopted dynamic Android benchmark built on Google's Android Emulator (Pixel 6, API 33 AVD). Tasks are procedurally parameterized with seeds across 20 real Android applications (Contacts, Clock, Calendar, Files, Markor, OsmAnd, Simple Draw, Expense trackers, etc.) and verified programmatically by inspecting live Android SQLite databases, files, and device state. In MAI-Gym, AndroidWorld evaluates **Environment OOD** generalization from MobileWorld/OSWorld training to an unseen Android stack.

| Property | Value |
| :--- | :--- |
| **CLI ID (`--benchmarks`)** | `androidworld` |
| **Tasks & Scope** | 116 seeded Android tasks across 20 real apps |
| **Step Budget** | Task-dependent (derived from AndroidWorld's native complexity allocator) |
| **Verification & Metric** | Native programmatic state verifier (`is_successful`) · `success_rate` |
| **Runtime Stack** | Pixel 6 API-33 AVD via Android Emulator (`adb` + gRPC/console ports, KVM) |
| **MAI-Gym-9B Result** | **75.9%** *(+18.1 over Qwen3.5-9B base · Env OOD)* |

### Running in MAI-Gym

Configure an isolated AndroidWorld worker pointing at your pinned AVD and task manifest, then run:

```bash
uv run mai-gym eval --config /opt/profiles/eval.yaml \
  --benchmarks androidworld --trials 1
```

- **Host requirements:** Linux `x86_64` with KVM, Android SDK/emulator, Pixel 6 API-33 AVD, `adb`, and dedicated console/gRPC ports.
- **Detailed setup & agent skill:** [`skills/envs/androidworld.md`](../../../../skills/envs/androidworld.md) · [`release.json`](../../../../src/mai_gym/envs/androidworld/release.json)

---

## MobileGym

**MobileGym** ([upstream repository](https://github.com/Purewhiter/mobilegym)) is a browser-hosted suite of high-fidelity simulated Chinese mobile apps (**Bench256**) running inside Playwright Chromium at a native mobile viewport (**360×800 logical**, **3× DPR → 1080×2400 screenshots**). Because it requires no Android VM or KVM, MobileGym offers lightweight parallel execution while testing rich bilingual/Chinese UI grounding, multi-step workflows, and partial-credit state progression.

| Property | Value |
| :--- | :--- |
| **CLI ID (`--benchmarks`)** | `mobilegym` |
| **Tasks & Scope** | 256 tasks (`Bench256`) across simulated mobile apps with companion media assets |
| **Step Budget** | 100 interaction steps |
| **Verification & Metric** | Native programmatic state verifier · `partial_score` *(reports both binary & partial)* |
| **Runtime Stack** | Playwright Chromium + Node-built static frontend (no KVM required; CJK fonts needed) |
| **MAI-Gym-9B Result** | **46.5%** *(+37.5 over Qwen3.5-9B base · Env OOD)* |

### Running in MAI-Gym

`mai-gym eval` automatically configures the native mobile touch policy (supporting tap, long press, inertial swipe, drag, text input, app launch, and system buttons) for MobileGym:

```bash
uv run mai-gym eval --config /opt/profiles/eval.yaml \
  --benchmarks mobilegym --trials 1
```

- **Host requirements:** Node 22+, Python 3.12+, Playwright Chromium, CJK fonts (e.g., `fonts-noto-cjk`), and companion media assets.
- **Detailed setup & agent skill:** [`skills/envs/mobilegym.md`](../../../../skills/envs/mobilegym.md) · [`release.json`](../../../../src/mai_gym/envs/mobilegym/release.json)

---

## Next Steps

- Try browser-based mobile-sized Mocked Apps without any external setup: [Mocked Apps](../mocked-apps.md)
- Compare across all benchmarks: [Benchmarks Overview](../environments.md)
- Explore desktop suites: [Computer Benchmarks](computer.md)
