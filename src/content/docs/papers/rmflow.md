---
title: "RMFlow: Refined Mean Flow by a Noise-Injection Step for Multimodal Generation"
description: RMFlow 在 coarse MeanFlow transport 後加入 tailored noise-injection refinement。
sidebar:
  order: 5
---

## Metadata

- **Authors**：Yuhao Huang, Shih-Hsin Wang, Andrea L. Bertozzi, Bao Wang
- **Venue**：ICLR 2026
- **Year**：2026
- **Primary tasks**：text-to-image、context-to-molecule、time-series / dynamical-system generation
- **Core topics**：MeanFlow refinement、noise injection、likelihood / Wasserstein control、1-NFE multimodal generation
- **Data / benchmarks**：QM9；Lorenz / FitzHugh–Nagumo trajectory benchmark；synthetic 1D mixture / 2D checkerboard；COCO 2017 5k split（additional text-image evaluation）
- **Sources**：[ICLR Proceedings](https://proceedings.iclr.cc/paper_files/paper/2026/hash/7b8c48c00dd5e3090ba0976e297fae5c-Abstract-Conference.html) · [arXiv](https://arxiv.org/abs/2602.00849)

## Core question

MeanFlow 的 1-NFE transport 很快，但在更複雜 multimodal generation 中，coarse transport 可能留下明顯 sample / distribution error。RMFlow 問的是：

> **能不能保留一次 MeanFlow evaluation 的速度，同時用一個很便宜的 refinement 把結果拉回更好的 target distribution？**

## Method

RMFlow 先做 1-NFE MeanFlow coarse transport，再加入 tailored noise-injection refinement。概念上是 two-stage，但 neural flow 仍只 evaluation 一次；額外步驟是 noise refinement，不是再跑一個昂貴 ODE solver。

## Why noise is not arbitrary

Paper 將 training objective 拆成 Wasserstein-style MeanFlow control 與 likelihood-related term。NLL term 為

$$
\mathcal L_{NLL}=\mathbb E\left[\left\|(x_{data}+\sigma_{min}\epsilon)-(x_0+\hat u_{0,1}(x_0;\theta))\right\|^2\right].
$$

Joint objective：

$$
\mathcal L_{RMFlow}=\mathcal L_{CMFM}+\lambda_1\mathcal L_{NLL}+\lambda_2\mathbb E\|\phi_\omega(c)\|^2.
$$

因此 refinement 與 likelihood / KL control 有理論關係，不只是「人工加一點 noise」。

## Relation to MeanFlow / iMF

- MeanFlow：改 modeled quantity，直接學 average transport；
- iMF：改 JVP tangent / regression formulation；
- RMFlow：接受 coarse 1-NFE transport 可能還有 gap，從 output refinement 與 distribution objective 下手。

這三者代表不同層次的改進，不應混成同一種 trick。

## Scope

Paper 展示 text-to-image、context-to-molecule、time-series generation。QM9 是 context-to-molecule 的主要資料；dynamical-system experiments 延續 Lorenz / FitzHugh–Nagumo trajectory setup；論文 appendix 另外用 COCO 2017 的 5k split 比較 text-image CLIP score。這些資料的角色並不完全相同，因此重現時要分清 training data、benchmark 與 additional evaluation split。

## Interpretation

RMFlow 最大的啟發是：MeanFlow 後續 research 不一定只能繼續改 JVP target。若 one-step endpoint quality 才是 bottleneck，可以把問題拆成「coarse deterministic transport + low-cost stochastic refinement」。

## Related data

- [Datasets｜訓練資料集與任務地圖](/research-handbook/datasets/)
