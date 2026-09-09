---
title: Improving Flow Matching by Aligning Flow Divergence
description: 以 divergence alignment 補強 conditional flow matching 對 probability path accuracy 的控制。
sidebar:
  order: 6
---

## Metadata

- **Authors**：Yuhao Huang, Taos Transue, Shih-Hsin Wang, William M. Feldman, Hong Zhang, Bao Wang
- **Venue**：ICML 2025 · PMLR 267:25813–25834
- **Year**：2025
- **Primary tasks**：2D density estimation、DNA sequence generation、dynamical-system trajectory sampling、video prediction
- **Core topics**：Conditional Flow Matching、probability-path error、flow divergence、PDE characterization、Hutchinson trace estimator
- **Data / benchmarks**：2D checkerboard synthetic data；DNA benchmark following the discrete / Dirichlet-flow setup；Lorenz and FitzHugh–Nagumo systems；KTH Actions video
- **Sources**：[PMLR](https://proceedings.mlr.press/v267/huang25ag.html) · [ICML](https://icml.cc/virtual/2025/poster/45878) · [Code](https://github.com/Utah-Math-Data-Science/Flow_Div_Matching)

## Core question

Conditional Flow Matching 可以讓 vector-field regression loss 變小，但這是否足以保證 learned probability path 本身也正確？這篇 paper 的核心問題就是：

> **「velocity 預測得像」與「整個 distribution evolution 正確」是不是同一件事？**

答案是否定的，因此作者從 probability-path error 的 PDE 出發，找出還需要控制 divergence gap。

## Theoretical bridge

Paper 建立 learned path $\hat p_t$ 與 exact path $p_t$ 的 error dynamics，並得到 total-variation control。這把 distribution-path error 和 divergence-related training quantity 直接連起來。

## Objective

實務上同時保留 CFM loss 與 conditional divergence matching：

$$
\mathcal L_{FDM}=\lambda_1\mathcal L_{CFM}+\lambda_2\mathcal L_{CDM}.
$$

所以方法不是放棄 velocity regression，而是增加一個 distribution evolution 的 constraint。

## Intuition: divergence 是什麼？

Velocity field 告訴每個 point 往哪裡走；divergence 則描述附近 flow 是在「擴散出去」還是「擠壓進來」。兩個 vector fields 在每個點方向接近，並不保證它們對 density 的 compression / expansion 完全相同。

這就是為什麼只 match vector value 可能還不足以 match probability path。

## Computation

高維 divergence 需要 Jacobian trace，paper 使用 Hutchinson trace estimator 降低成本，避免顯式建立完整 Jacobian。

## Benchmarks across modalities

Code / paper 將同一 objective 放到多種資料型態：2D checkerboard density estimation、DNA discrete sequence generation、Lorenz / FitzHugh–Nagumo trajectory sampling，以及 KTH video prediction。這個設計很適合拿來判斷一個 flow objective 是否只是「對某個 image benchmark 有效」，還是真的具有跨 modality 的 dynamics 意義。

## Interpretation

這條研究線與 MeanFlow 非常互補：MeanFlow 問「模型該預測 local 還是 finite-time quantity」；Flow Divergence Matching 問「不管你怎麼 parameterize，learned field 對 distribution path 的控制是否足夠」。對 future fast generators，這提供一個比單純 sample FID 更接近 dynamics correctness 的 training lens。

## Related data

- [Datasets｜訓練資料集與任務地圖](/research-handbook/datasets/)
