<div align="center">
  <h1><img src="site/assets/mai-gym-logo.png" width="56" align="middle" alt="MAI-Gym logo"> MAI-Gym</h1>
  <p><strong>An Open All-in-One Platform for GUI Agent Evaluation and RL Training across Mobile and Computer Environments</strong></p>
  <p>
    <a href="https://tongyi-mai.github.io/MAI-Gym/"><img src="https://img.shields.io/badge/Website-Project_Page-7457D6?style=flat-square" alt="Project website"></a>
    <a href="https://arxiv.org/abs/2610.20939"><img src="https://img.shields.io/badge/arXiv-2610.20939-B31B1B?style=flat-square" alt="Paper on arXiv"></a>
    <a href="https://huggingface.co/Tongyi-MAI/MAI-Gym-9B"><img src="https://img.shields.io/badge/Model-MAI--Gym--9B-FFD21E?style=flat-square&amp;logo=huggingface&amp;logoColor=000" alt="MAI-Gym-9B on Hugging Face"></a>
  </p>

  [Framework](#framework) · [Main Results](#main-results) · [News](#news) · [Citation](#citation)
</div>

MAI-Gym provides a shared infrastructure for evaluating GUI agents, collecting trajectories, and running online reinforcement learning across mobile, computer, and web environments. Framework source code will be released soon.

## Framework

<p align="center">
  <img src="site/assets/framework-architecture.png" width="100%" alt="MAI-Gym framework architecture">
</p>

- **One harness and one rollout path.** Evaluation, data collection, and RL training use the same agent harness, prompt construction, action parser, and interaction loop, preventing implementation drift between training and evaluation.
- **Unified environment management.** `EnvClient` exposes a common interface for instance allocation, observation, task reset, action execution, and evaluation. Environment adapters preserve each benchmark's native initialization, actions, and verifiers.
- **Cross-platform deployment.** The same manager serves mobile, computer, and web instances hosted on Kubernetes, Docker, physical devices, or existing endpoints. MAI-Gym has been tested with up to 20,000 concurrent environments.
- **Reliable, elastic rollouts.** Asynchronous provisioning, warm pools, exact task resets, session leases, idempotent actions, operation deadlines, and isolated verifiers contain infrastructure failures and keep them out of the training signal.
- **Efficient multimodal history.** Window-wise history preserves reusable prompt prefixes for KV-cache reuse while bounding recent screenshots. Sticky routing keeps each trajectory on the inference engine that holds its cache.
- **Verifiable tasks and diagnostics.** Task-defined code verifiers and VLM judges produce rewards; persisted trajectories and the dashboard connect aggregate scores to screenshots, model outputs, actions, verifier results, and environment status.
- **Extensible task and environment synthesis.** MAI-Gym supports resettable mocked apps, coherent cross-app worlds, execution-validated tasks, and four extension points: agent harnesses, tasks, environment adapters, and environment specifications.
- **Online RL integration.** The trajectory buffer feeds synchronous or asynchronous training backends, with support for GRPO, DAPO, GSPO, REINFORCE++, SAO, and PPO through Slime.

## Main results

Mobile and computer benchmark results (%). **Task OOD** denotes unseen tasks in training environments; **Env OOD** denotes unseen environments.

| Model | Size | Access | MobileWorld GUI-only<br>Task OOD | AndroidWorld<br>Env OOD | MobileGym<br>Env OOD | OSWorld-Verified<br>Task OOD | OSWorld 2.0<br>Env OOD | MacAgentBench<br>Env OOD | MyPCBench<br>Env OOD |
| --- | ---: | :---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| *Frontier models* | | | | | | | | | |
| Claude Opus 4.8 | — | API | 69.2 | 75.0 | 50.7 | 83.4 | 20.3 / 54.8 | 58.4 | 62.0 |
| GPT-5.5 | — | API | 62.4 | 74.1 | 48.1 | 78.7 | 13.9 / 47.5 | 66.7 | 47.3 |
| Qwen3.8-Max | 2.4T-A95B | API | 77.8 | 85.3 | 61.3 | 86.1 | 19.4 / 46.7 | 74.3 | 48.6 |
| Qwen3.8-27B | 27B | Open | 70.1 | 81.9 | 57.5 | 84.3 | 19.4 / 48.0 | 65.4 | 28.5 |
| *GUI agent models* | | | | | | | | | |
| GUI-Owl-1.5-8B | 8B | Open | 38.2 | 71.6 | 15.1 | 52.9 | 0.0 / 1.6 | 10.2 | 1.8 |
| GUI-Owl-1.5-32B | 32B | Open | 43.9 | 69.8 | 20.3 | 56.5 | 0.9 / 2.5 | 17.2 | 6.3 |
| UI-Venus-2-9B | 9B | Open | 75.2 | 80.2 | 52.7 | 70.8 | 0.0 / 7.5 | 36.2 | 6.3 |
| SCALECUA-Qwen3.5-9B | 9B | Open | 12.8 | 47.4 | 13.5 | 68.7 | 0.0 / 2.2 | 3.9 | 1.1 |
| Qwen-CUA | 397B-A17B | Open | — | — | — | 86.2 | 18.5 / 48.4 | 69.2 | 58.7 |
| EvoCUA-1.5-32B | 32B | Unreleased | — | — | — | 63.2 | — | — | — |
| CUA-GYM-35B-A3B | 35B-A3B | Unreleased | — | — | — | 62.1 | — | — | — |
| CUA-GYM-397B-A17B | 397B-A17B | Unreleased | — | — | — | 72.6 | — | — | — |
| *Ours* | | | | | | | | | |
| Qwen3.5-9B (base model) | 9B | Open | 18.0 | 57.8 | 9.0 | 41.8 | 0.0 / 2.5 | 20.7 | 6.5 |
| **MAI-Gym-9B** | **9B** | **Open** | **78.4**<br><sub>(+60.4)</sub> | **75.9**<br><sub>(+18.1)</sub> | **46.5**<br><sub>(+37.5)</sub> | **76.2**<br><sub>(+34.4)</sub> | **3.7 / 11.3**<br><sub>(+3.7 / +8.8)</sub> | **41.3**<br><sub>(+20.6)</sub> | **15.2**<br><sub>(+8.7)</sub> |

Results use each benchmark's native evaluation metric. MobileGym reports partial score; OSWorld 2.0 reports binary accuracy / partial score. Parenthesized values are absolute percentage-point gains over Qwen3.5-9B. See the [paper](https://arxiv.org/abs/2610.20939) for evaluation protocols and baseline sources.

<a name="news"></a>

## 🔥 News

<!-- Add new entries at the top, newest first. -->
- **[2026-10]** 🚀 MAI-Gym is released with the [paper](https://arxiv.org/abs/2610.20939), [project website](https://tongyi-mai.github.io/MAI-Gym/), and [MAI-Gym-9B model weights](https://huggingface.co/Tongyi-MAI/MAI-Gym-9B). Framework source code is coming soon.

<a name="citation"></a>

## 📄 Citation

```bibtex
@article{maigym2026,
  title   = {{MAI-Gym}: An Open All-in-One Platform for GUI Agent Evaluation
             and RL Training across Mobile and Computer Environments},
  author  = {Panrong Tong and Quyu Kong and Pengxiang Li and Tianyu Xia and
             Hanzhang Zhou and Lei Wang and Chenglin Cai and Long Li and Long Chen and
             Jianan Zhang and Zhengwen Zeng and Xu Zhang and Gongjie Zhang and
             Yue Wang and Steven Hoi},
  journal = {arXiv preprint arXiv:2610.20939},
  year    = {2026},
  url     = {https://arxiv.org/abs/2610.20939}
}
```
