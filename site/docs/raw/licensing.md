# Licensing & Third-Party Notices

[Home](../../../README.md) · [Quickstart](index.md) · [Development](development.md) · [Concepts](concepts.md)

## Project License

MAI-Gym's original framework code, RL adapters, Rust dashboard code, documentation, and Mocked Apps (42 Novel Apps: code, task definitions, and reference oracle solutions) are released under the [**MIT License**](../../../LICENSE).

The Trajectory Dashboard is a standalone subproject and includes a local copy of the MIT License so it can be distributed independently:

- Trajectory Dashboard: [`src/dashboard-rs/LICENSE`](../../../src/dashboard-rs/LICENSE)

---

## Third-Party Components

Bundled third-party components retain their original copyright notices and licenses:

| Component | License | Notice / License File |
| :--- | :--- | :--- |
| **Slime RL runtime** (bundled) | Apache-2.0 | [`resources/slime/LICENSE`](../../../resources/slime/LICENSE) · [`NOTICE`](../../../resources/slime/NOTICE) |
| **TileLang-derived kernels** (in Slime) | MIT | [`resources/slime/third_party/tilelang/LICENSE`](../../../resources/slime/third_party/tilelang/LICENSE) *(files & revisions listed in Slime `NOTICE`)* |
| **uPlot** (dashboard charting library) | MIT | [`src/dashboard-rs/static/uplot-LICENSE`](../../../src/dashboard-rs/static/uplot-LICENSE) |

---

## External Datasets, Assets, and Models

> [!NOTE]
> The MAI-Gym MIT License does **not** replace third-party licenses or grant rights to separately downloaded benchmark datasets, application assets, VM/emulator images, or pretrained model weights. Always follow the license terms published by their respective upstream maintainers.
