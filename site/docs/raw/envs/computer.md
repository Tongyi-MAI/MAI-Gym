# Computer Benchmarks

MAI-Gym integrates four desktop operating system benchmarks—**OSWorld-Verified**, **OSWorld 2.0**, **MacAgentBench**, and **MyPCBench**—covering Ubuntu Linux and macOS environments across office suites, browsers, IDEs, file systems, media tools, and multi-service developer workflows. On all computer benchmarks, MAI-Gym enables its **Hybrid Action Space (GUI + CLI)**, allowing agents to seamlessly combine `pyautogui` visual interactions with direct `<bash>` shell execution.

> [!TIP]
> Looking for detailed VM/image pinning, immutable profile lock JSONs, or coding-agent bring-up prompts? Each benchmark below links directly to its full setup skill under [`skills/envs/`](../../../../skills/envs) and [`skills/envs.md`](../../../../skills/envs.md).

---

## OSWorld-Verified

**OSWorld-Verified** ([upstream repository](https://github.com/xlang-ai/OSWorld)) is the standard verified benchmark for open-ended Ubuntu desktop computer use, spanning LibreOffice (Calc, Writer, Impress), Chrome, GIMP, VS Code, Thunderbird, VLC, OS file management, and multi-app workflows. Each task boots from a reproducible Ubuntu VM snapshot and is graded deterministically by native Python evaluators inspecting application files, settings, and outputs.

| Property | Value |
| :--- | :--- |
| **CLI ID (`--benchmarks`)** | `osworld-verified` |
| **Tasks & Scope** | 361 verified Ubuntu desktop tasks across productivity, creative, web, and OS apps |
| **Step Budget** | 100 interaction steps |
| **Action Space** | Hybrid GUI (`pyautogui`) + CLI (`<bash>`) |
| **Verification & Metric** | Native programmatic state evaluator · `success_rate` |
| **Runtime Stack** | Ubuntu desktop VM (Docker/KVM with pinned QCOW2 base + disposable overlays) |
| **MAI-Gym-9B Result** | **76.2%** *(+34.4 over Qwen3.5-9B base · Task OOD)* |

### Running in MAI-Gym

Start an OSWorld-Verified manager (`mai-gym serve`) and evaluate directly—or pair it with MobileWorld in a single command (see [Add OSWorld-Verified](../evaluation.md#4-add-osworld-verified)):

```bash
uv run mai-gym eval --config "$HOME/mai-gym-eval/eval.yaml" \
  --benchmarks osworld-verified --trials 1
```

- **Host requirements:** Linux `x86_64`, Docker, KVM, RAM for an Ubuntu desktop VM, and disk space for the base QCOW2 + private overlays.
- **Detailed setup & agent skill:** [`skills/envs/osworld.md`](../../../../skills/envs/osworld.md) · [`release.json`](../../../../src/mai_gym/envs/osworld/release.json)

---

## OSWorld 2.0

**OSWorld 2.0** ([upstream repository](https://github.com/xlang-ai/OSWorld-V2), v2.1 profile) raises the difficulty bar with long-horizon, multi-step professional desktop workflows backed by supporting local websites and GitLab services. Tasks allow up to **500 steps** and grade multi-criterion progress via both strict full-completion binary indicators and weighted partial scores.

| Property | Value |
| :--- | :--- |
| **CLI ID (`--benchmarks`)** | `osworld-2` |
| **Tasks & Scope** | 108 long-horizon desktop + multi-service tasks (Ubuntu VM + local web/GitLab services) |
| **Step Budget** | 500 interaction steps |
| **Action Space** | Hybrid GUI (`pyautogui`) + CLI (`<bash>`) |
| **Verification & Metric** | Native programmatic class evaluators · `binary_and_partial` |
| **Runtime Stack** | Ubuntu desktop VM (Docker/KVM) + pinned supporting web & GitLab services |
| **MAI-Gym-9B Result** | **3.7% binary / 11.3% partial** *(+3.7 / +8.8 over Qwen3.5-9B base · Env OOD)* |

### Running in MAI-Gym

```bash
uv run mai-gym eval --config /opt/profiles/eval.yaml \
  --benchmarks osworld-2 --trials 1
```

- **Host requirements:** Linux `x86_64`, Docker/KVM, Ubuntu guest QCOW2, and dedicated ports for matching task/web/GitLab services.
- **Detailed setup & agent skill:** [`skills/envs/osworld_2.md`](../../../../skills/envs/osworld_2.md) · [`release.json`](../../../../src/mai_gym/envs/osworld_2/release.json)

---

## MacAgentBench

**MacAgentBench** ([upstream repository](https://github.com/JetAstra/MacAgentBench)) evaluates desktop GUI agents in a **macOS** environment, testing cross-OS generalization beyond Linux desktops (Finder, macOS system settings, native macOS apps, shortcuts, and workflows). Tasks are verified via native macOS checkpoint evaluators executed over isolated guest sessions.

| Property | Value |
| :--- | :--- |
| **CLI ID (`--benchmarks`)** | `macagentbench` |
| **Tasks & Scope** | 676 macOS desktop tasks across native apps and system workflows |
| **Step Budget** | 50 interaction steps |
| **Action Space** | Hybrid GUI + CLI |
| **Verification & Metric** | Native macOS checkpoint evaluators · `success_rate` |
| **Runtime Stack** | KVM-hosted macOS guest VM with per-episode QCOW2 overlays and SSH control |
| **MAI-Gym-9B Result** | **41.3%** *(+20.6 over Qwen3.5-9B base · Env OOD)* |

### Running in MAI-Gym

```bash
uv run mai-gym eval --config /opt/profiles/eval.yaml \
  --benchmarks macagentbench --trials 1
```

- **Host requirements:** KVM-capable VM host, pinned macOS disk & BaseSystem images, SSH access to the guest, and memory/disk headroom for writable overlays.
- **Detailed setup & agent skill:** [`skills/envs/macagentbench.md`](../../../../skills/envs/macagentbench.md) · [`release.json`](../../../../src/mai_gym/envs/macagentbench/release.json)

---

## MyPCBench

**MyPCBench** ([upstream repository](https://github.com/ljang0/MyPCBench)) evaluates practical everyday desktop computer tasks on a clean **Ubuntu desktop VM** managed directly through QEMU and OVMF UEFI firmware. Unlike purely programmatic suites, MyPCBench constructs native rubric bundles and grades completed episodes via offline full-trajectory VLM rubric evaluation, reporting the `perfect_task_rate` metric.

| Property | Value |
| :--- | :--- |
| **CLI ID (`--benchmarks`)** | `mypcbench` |
| **Tasks & Scope** | 184 personal-computer desktop tasks with inlined grading rubrics |
| **Step Budget** | 100 interaction steps |
| **Action Space** | Hybrid GUI + CLI |
| **Verification & Metric** | Native rubric-based VLM trajectory grading · `perfect_task_rate` |
| **Runtime Stack** | Ubuntu desktop VM via QEMU (`qemu-system-x86_64`) + OVMF firmware (Linux KVM, no GPU required) |
| **MAI-Gym-9B Result** | **15.2%** *(+8.7 over Qwen3.5-9B base · Env OOD)* |

### Running in MAI-Gym

Configure both the QEMU worker and your rubric judge endpoint (`JUDGE_API_KEY`) in the worker settings, then run:

```bash
uv run mai-gym eval --config /opt/profiles/eval.yaml \
  --benchmarks mypcbench --trials 1
```

- **Host requirements:** Linux `x86_64`, KVM, `qemu-system-x86_64`, `qemu-img`, OVMF firmware, Ubuntu base QCOW2, and a judge API endpoint.
- **Detailed setup & agent skill:** [`skills/envs/mypcbench.md`](../../../../skills/envs/mypcbench.md) · [`release.json`](../../../../src/mai_gym/envs/mypcbench/release.json)

---

## Next Steps

- Learn how the Hybrid Action Space combines GUI and `<bash>` CLI: [Platform & Core Concepts](../concepts.md#hybrid-action-space)
- Compare across all benchmarks: [Benchmarks Overview](../environments.md)
- Explore browser benchmarks: [Web Benchmarks](web.md)
