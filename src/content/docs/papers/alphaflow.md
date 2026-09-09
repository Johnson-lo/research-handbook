---
title: "AlphaFlow: Understanding and Improving MeanFlow Models"
description: AlphaFlow 解析 MeanFlow objective 的 gradient conflict，並用 curriculum 從 trajectory flow matching 漸進到 MeanFlow。
sidebar:
  order: 3
---

## Metadata

- **Authors**：Huijie Zhang, Aliaksandr Siarohin, Willi Menapace, Michael Vasilkovsky, Sergey Tulyakov, Qing Qu, Ivan Skorokhodov
- **Venue**：ICLR 2026
- **Year**：2026
- **Primary task**：ImageNet-1K 256×256 one-step / two-step generative modeling with MeanFlow-style objectives
- **Core topics**：MeanFlow、optimization conflict、trajectory flow matching、trajectory consistency、curriculum learning
- **Data / benchmark**：ImageNet-1K 256×256
- **Main metrics**：FID、NFE、training convergence / gradient interaction
- **Sources**：[ICLR Proceedings](https://proceedings.iclr.cc/paper_files/paper/2026/hash/e8c20cafe841cba3e31a17488dc9c3f1-Abstract-Conference.html) · [Code](https://github.com/snap-research/alphaflow)

## Core question

MeanFlow 想學 large-stride average velocity，但實際 training 卻大量抽樣 boundary case $r=t$。在 boundary 上，MeanFlow 退化成更接近普通 Flow Matching 的 supervision。問題是：**如果目標是學 long-interval transport，為什麼訓練中反而需要大量 local / boundary supervision？**

這個現象和一個很自然的研究假設連在一起：是否應先把 $r=t$ 或接近 $r=t$ 的能力學穩，再逐步學更大的 temporal gap？

:::caution[證據邊界]
AlphaFlow 支持「先偏向 Flow Matching / trajectory-flow-matching，再平滑轉向完整 MeanFlow」的 curriculum；它不是直接證明「嚴格只用 $r=t$ pretrain 一段固定時間」一定是最佳策略。
:::

## Objective decomposition

AlphaFlow 將 MeanFlow objective 分解成兩個概念上不同的部分：

1. **Trajectory Flow Matching**：讓 trajectory 上的 flow prediction 接近正確 transport。
2. **Trajectory Consistency**：讓不同 interval / trajectory representation 彼此一致。

作者的 gradient analysis 顯示，這兩部分在 training 中有強烈負相關的 gradient，代表它們會互相拉扯，造成 optimization conflict 與 slow convergence。

## Why does $r=t$ matter?

當

$$
r=t,
$$

interval length

$$
\Delta t=t-r=0.
$$

由 boundary relation

$$
u(z_t,t,t)=v(z_t,t),
$$

此時 average velocity collapses to the instantaneous velocity。也就是說，$r=t$ supervision 把問題拉回一個比較 local、比較接近 Flow Matching 的 learning task。

AlphaFlow 的分析指出，這種 boundary supervision 可以降低 trajectory-flow-matching 部分的 error，同時避免直接和 consistency term 在同一個大 interval 上強烈衝突；代價是原始 MeanFlow 需要花大量 training samples 在 boundary case。

## AlphaFlow 的做法

作者提出一族以 $\alpha$ 控制的 objectives，將 trajectory flow matching、Shortcut Model 與 MeanFlow 放進共同框架。Training 使用 curriculum：

$$
\alpha: 1 \longrightarrow 0,
$$

讓 optimization 從較偏 trajectory flow matching 的 regime，逐漸轉到完整 MeanFlow。

直覺上就是：**先把較窄、較 local 的 flow-fitting task 穩定下來，再逐步承擔 high-variance 的 long-interval consistency / MeanFlow task。**

## Evidence

論文在 ImageNet-1K 256×256、vanilla DiT backbones、from-scratch setting 下報告 $\alpha$-Flow-XL/2+：

- **1-NFE FID 2.58**
- **2-NFE FID 2.15**

## Research interpretation

這篇 paper 對「$r=t$ 是否應先學」最大的價值，不只是結果數字，而是提供一個 optimization explanation：boundary / Flow-Matching-like supervision 可能扮演 **先建立容易且低 variance 的能力，再逐步切換到完整 MeanFlow** 的角色。

這和另一篇 [Understanding, Accelerating, and Improving MeanFlow Training](/research-handbook/papers/meanflow-training/) 的 controlled temporal-gap experiments 相互補強：一篇從 objective decomposition / gradient conflict 解釋 curriculum，另一篇從 instantaneous velocity 與不同 $\Delta t$ 的 interaction 直接量測 training dynamics。

## Related pages

- [Research Problem Map](/research-handbook/tracks/problem-map/)
- [MeanFlow paper](/research-handbook/papers/meanflow/)
- [MeanFlow Training Dynamics](/research-handbook/papers/meanflow-training/)
- [Datasets｜訓練資料集與任務地圖](/research-handbook/datasets/)
