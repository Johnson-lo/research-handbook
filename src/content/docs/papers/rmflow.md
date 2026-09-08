---
title: "RMFlow: Refined Mean Flow by a Noise-Injection Step for Multimodal Generation"
description: RMFlow 在 coarse MeanFlow transport 後加入 tailored noise-injection refinement。
sidebar:
  order: 3
---

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

Paper 展示 text-to-image、context-to-molecule、time-series generation；也報告在有限 GPU 資源下的訓練設定，說明這條 branch 對實驗資源受限的研究環境也具有實作吸引力。

## Interpretation

RMFlow 最大的啟發是：MeanFlow 後續 research 不一定只能繼續改 JVP target。若 one-step endpoint quality 才是 bottleneck，可以把問題拆成「coarse deterministic transport + low-cost stochastic refinement」。
