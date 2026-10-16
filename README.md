<h1 align="center">
  <img src="docs/assets/mai-gym-logo.png" width="48" height="48" align="middle" alt="MAI-Gym logo">
  MAI-Gym
</h1>

<p align="center">
  <strong>An Open All-in-One Platform for GUI Agent Evaluation and RL Training<br>across Mobile and Computer Environments</strong>
</p>

<p align="center">
  <a href="https://tongyi-mai.github.io/MAI-Gym/">🌐 Website &amp; Interactive Demos</a> ·
  <a href="https://arxiv.org/abs/2610.20939">📄 Paper</a> ·
  <a href="https://huggingface.co/Tongyi-MAI/MAI-Gym-9B">🤗 MAI-Gym-9B</a>
</p>

<p align="center"><strong>MAI-UI Team · Alibaba Group</strong></p>

**MAI-Gym** unifies GUI agent evaluation, data collection, and online reinforcement learning across **mobile, computer, and web environments**. A shared rollout pipeline connects agents, interactive environments, task verifiers, and RL training, keeping agent behavior consistent across all three workloads.

With MAI-Gym, we train **MAI-Gym-9B**, a unified 9B agent for mobile and computer use. It achieves **78.4% on MobileWorld** and **76.2% on OSWorld-Verified**, leading open GUI agents under 10B parameters on these benchmarks in our [paper](https://arxiv.org/abs/2610.20939).

**Source code: coming soon.** This initial release shares the project overview, research results, and links to the paper and interactive demos.

![MAI-Gym overview: GUI environments, evaluation, parallel rollouts, RL training, adaptive synthesis, and model results](docs/assets/overview.webp)

## Highlights

- **One rollout pipeline, three workloads.** Reuse the same agent harness and environment interfaces for evaluation, trajectory collection, and online RL.
- **10 GUI benchmarks.** Evaluate across mobile, computer, and web environments through a common interface while preserving each benchmark's task setup and verification.
- **20,000 parallel environments.** Scale interaction with environment provisioning, warm pools, session management, and screenshot storage separate from manager traffic.
- **Adaptive app and task synthesis.** Diagnose failed trajectories, build resettable applications and coherent cross-app worlds, and generate tasks with execution-validated verifiers.
- **A unified 9B policy.** Train across mobile and computer tasks, with transfer from a mocked train-ticketing app to its real counterpart on a physical phone.
- **Inspectable trajectories.** Trace benchmark scores to individual runs, screenshots, model outputs, actions, and verifier results through the dashboard.

## MAI-Gym-9B

MAI-Gym-9B is trained from **Qwen3.5-9B** using mixed-domain GUI reinforcement learning. The table below reports success rates (%) from the [project website](https://tongyi-mai.github.io/MAI-Gym/#results); gains are absolute percentage points over the base model.

| Domain | Benchmark | Qwen3.5-9B (base) | MAI-Gym-9B | Gain |
| --- | --- | ---: | ---: | ---: |
| Mobile | MobileWorld (GUI-only) | 18.0 | **78.4** | +60.4 |
| Mobile | AndroidWorld | 57.8 | **75.9** | +18.1 |
| Mobile | MobileGym | 9.0 | **46.5** | +37.5 |
| Computer | OSWorld-Verified | 41.8 | **76.2** | +34.4 |
| Computer | MacAgentBench | 20.7 | **41.3** | +20.6 |
| Computer | MyPCBench | 6.5 | **15.2** | +8.7 |

**Simulation-to-real transfer.** RL in a mocked Railway12306 app improves task success in the real app on a physical phone from **28.8% to 48.1%**. See the [paper](https://arxiv.org/abs/2610.20939) for the evaluation protocol and full results, including OSWorld 2.0.

The **MAI-Gym-9B checkpoint will be available upon release** at [Tongyi-MAI/MAI-Gym-9B](https://huggingface.co/Tongyi-MAI/MAI-Gym-9B).

## Platform

### Shared environment interface

Agents interact with environments through one lifecycle:

```text
acquire → reset → (observe → act)* → evaluate → release
```

Environment adapters retain native action spaces, reset procedures, and task verifiers. The manager handles provisioning, session ownership, task registration, reward evaluation, and trajectory recording. Deployment can use Kubernetes, Docker, or existing devices and services.

| Domain | Example environments |
| --- | --- |
| Mobile | MobileWorld, AndroidWorld, MobileGym, mocked mobile apps |
| Computer | OSWorld / OSWorld-Verified, OSWorld 2.0, MacAgentBench, MyPCBench |
| Web | WebArena, WebVoyager, locally served applications |

### Evaluate → diagnose → synthesize → retrain

The synthesis workflow starts with a realistic world of application data, files, and cross-app relationships. It then creates task instructions, initialization logic, and code-based verifiers. Execution tests validate the setup and scoring before tasks enter the training pool. Diagnosis of successful and failed attempts guides the next round of synthesis.

Our research covers **49 resettable mocked apps**, including **42 novel apps**, and **2,000 verified tasks**, alongside app and task synthesis skills. The [interactive website](https://tongyi-mai.github.io/MAI-Gym/#apps) lets you explore the novel apps, try tasks, check completion, and watch reference solutions. It also includes [real-versus-mocked app comparisons](https://tongyi-mai.github.io/MAI-Gym/#mock-vs-real).

### Online reinforcement learning

Rollout workers collect trajectories and verifier rewards through the same interfaces used for evaluation. The RL integration connects these trajectories to Slime, with support for synchronous and asynchronous rollout collection. The platform supports GRPO, DAPO, GSPO, REINFORCE++, SAO, and PPO.

## Citation

If you use MAI-Gym in your research, please cite our [paper](https://arxiv.org/abs/2610.20939):

```bibtex
@article{maigym2026,
  title   = {MAI-Gym: An Open All-in-One Platform for GUI Agent Evaluation
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
