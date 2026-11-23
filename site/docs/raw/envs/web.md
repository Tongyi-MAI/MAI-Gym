# Web Benchmarks

MAI-Gym integrates three standard browser benchmarks—**WebVoyager**, **Online-Mind2Web**, and **WebArena**—covering both open live-web navigation and deterministic self-hosted web applications. For lightweight, resettable offline browser apps bundled directly inside this repository, see [Mocked Apps](../mocked-apps.md) (`--benchmarks mocked-apps`).

> [!TIP]
> Looking for detailed browser/site compose files, immutable profile lock JSONs, or coding-agent bring-up prompts? Each benchmark below links directly to its full setup skill under [`skills/envs/`](../../../../skills/envs) and [`skills/envs.md`](../../../../skills/envs.md).

---

## WebVoyager

**WebVoyager** ([upstream repository](https://github.com/MinorJerry/WebVoyager)) tests open-ended information seeking and interaction across 15 live real-world websites (search, flights, hotels, shopping, recipes, academic databases, maps, and media). Each task starts on a prescribed live URL and runs within a tight **15-step budget**, evaluated by WebVoyager's native VLM trajectory judge (`auto_eval.py`) over the final answer and trailing screenshots.

| Property | Value |
| :--- | :--- |
| **CLI ID (`--benchmarks`)** | `webvoyager` |
| **Tasks & Scope** | 643 tasks across 15 live real-world websites |
| **Step Budget** | 15 interaction steps |
| **Verification & Metric** | Native VLM trajectory judge (`evaluation/auto_eval.py`) · `success_rate` |
| **Runtime Stack** | Isolated Playwright Chromium contexts + outbound internet access |

### Running in MAI-Gym

```bash
uv run mai-gym eval --config /opt/profiles/eval.yaml \
  --benchmarks webvoyager --trials 1
```

- **Host requirements:** Python 3.12+, Playwright Chromium, outbound internet access to target websites, and an OpenAI-compatible VLM judge endpoint (`JUDGE_API_KEY`).
- **Detailed setup & agent skill:** [`skills/envs/webvoyager.md`](../../../../skills/envs/webvoyager.md) · [`release.json`](../../../../src/mai_gym/envs/webvoyager/release.json)

---

## Online-Mind2Web

**Online-Mind2Web** ([upstream repository](https://github.com/OSU-NLP-Group/Online-Mind2Web)) evaluates real-world web task completion across diverse live websites with a **50-step budget**. Episodes are evaluated using the benchmark's native **WebJudge** pipeline, which identifies key task requirements, selects critical screenshots along the trajectory, and grades factual executed actions and the final response.

| Property | Value |
| :--- | :--- |
| **CLI ID (`--benchmarks`)** | `onlinemind2web` |
| **Tasks & Scope** | 123 tasks (paper subset; full gated dataset via `osunlp/Online-Mind2Web` on Hugging Face) |
| **Step Budget** | 50 interaction steps |
| **Verification & Metric** | Native `WebJudge` VLM trajectory evaluator · `success_rate` |
| **Runtime Stack** | Isolated Playwright Chromium contexts + outbound internet access |

### Running in MAI-Gym

```bash
uv run mai-gym eval --config /opt/profiles/eval.yaml \
  --benchmarks onlinemind2web --trials 1
```

- **Host requirements:** Python 3.12+, Playwright Chromium, approved Hugging Face access (`HF_TOKEN` for dataset download), outbound web access, and a judge API endpoint (`JUDGE_API_KEY`).
- **Detailed setup & agent skill:** [`skills/envs/onlinemind2web.md`](../../../../skills/envs/onlinemind2web.md) · [`release.json`](../../../../src/mai_gym/envs/onlinemind2web/release.json)

---

## WebArena

**WebArena** ([upstream repository](https://github.com/web-arena-x/webarena)) provides a completely **self-hosted, reproducible web benchmark** consisting of **812 tasks** across seven functional websites: an e-commerce storefront (`Shopping`), store back-office (`Shopping Admin`), forum (`Reddit`), code forge (`GitLab`), encyclopedia (`Wikipedia`), maps (`Map`), and a start portal (`Homepage`). Because the entire stack runs locally via Docker Compose, WebArena avoids live-internet flakiness and evaluates every task using deterministic programmatic verifiers (`evaluation_harness`).

| Property | Value |
| :--- | :--- |
| **CLI ID (`--benchmarks`)** | `webarena` |
| **Tasks & Scope** | 812 tasks across 7 self-hosted functional web applications |
| **Step Budget** | 30 interaction steps |
| **Verification & Metric** | Native programmatic web & database evaluators (`evaluation_harness`) · `success_rate` |
| **Runtime Stack** | Docker Compose 7-site stack + Playwright Chromium with per-episode authentication state |

### Running in MAI-Gym

```bash
uv run mai-gym eval --config /opt/profiles/eval.yaml \
  --benchmarks webarena --trials 1
```

- **Host requirements:** Linux Docker host with sufficient RAM/disk for the 7 self-hosted site containers and volumes, plus Playwright Chromium (native worker uses Python 3.11 when required by upstream pins).
- **Detailed setup & agent skill:** [`skills/envs/webarena.md`](../../../../skills/envs/webarena.md) · [`release.json`](../../../../src/mai_gym/envs/webarena/release.json)

---

## Looking for Zero-Setup Browser Environments?

If you want deterministic, locally resettable web apps without hosting Docker Compose stacks or relying on live websites, try the **42 bundled Novel Apps** in [Mocked Apps](../mocked-apps.md) (`--benchmarks mocked-apps` with [`configs/mocked-apps.yaml`](../../../../configs/mocked-apps.yaml)).

---

## Next Steps

- Try the bundled local browser apps: [Mocked Apps](../mocked-apps.md)
- Compare across all 10 benchmarks: [Benchmarks Overview](../environments.md)
- Run multi-benchmark evaluations: [Evaluation Guide](../evaluation.md)
