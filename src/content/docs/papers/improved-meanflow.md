---
title: "Improved Mean Flows: On the Challenges of Fastforward Generative Models"
description: iMF 重構 MeanFlow objective，並延伸 guidance、conditioning 與 system design。
sidebar:
  order: 2
---

## Metadata

- **Authors**：Zhengyang Geng, Yiyang Lu, Zongze Wu, Eli Shechtman, J. Zico Kolter, Kaiming He
- **Venue**：CVPR 2026 · pp. 30467–30476
- **Year**：2026
- **Primary task**：ImageNet-1K 256×256 class-conditional one-step / few-step image generation
- **Core topics**：MeanFlow objective reformulation、state-conditioned v-loss、classifier-free guidance、in-context conditioning
- **Data / benchmark**：ImageNet-1K 256×256
- **Sources**：[CVF Open Access](https://openaccess.thecvf.com/content/CVPR2026/html/Geng_Improved_Mean_Flows_On_the_Challenges_of_Fastforward_Generative_Models_CVPR_2026_paper.html) · [arXiv](https://arxiv.org/abs/2512.02012)

## Core question

Original MeanFlow 已經能 one-step generate，但 training target 同時依賴 ground-truth sample quantity 與 network 本身。iMF 問的是：

> **能不能把 MeanFlow 改寫成更接近 standard regression 的 prediction function，同時保留 one-step inference？**

## Step 1 — 先看 Original MF 的 v-loss view

將 original MF 重寫後，compound predictor 可以表示為

$$
V_\theta(z_t,e-x)
=u_\theta(z_t,r,t)
+(t-r)\operatorname{JVP}_{sg}(u_\theta;e-x).
$$

Loss 是

$$
\mathcal L=\mathbb E\|V_\theta(z_t,e-x)-(e-x)\|^2.
$$

這暴露出 predictor 額外依賴 sample-specific conditional tangent $e-x$。

## Step 2 — iMF 的核心 replacement

iMF parameterize 一個 marginal-like velocity estimate $v_\theta(z_t,t)$，並改成

$$
V_\theta(z_t)
=u_\theta(z_t,r,t)
+(t-r)\operatorname{JVP}_{sg}(u_\theta;v_\theta).
$$

Supervision 仍是

$$
\mathcal L_{iMF}=\mathbb E\|V_\theta(z_t)-(e-x)\|^2.
$$

因此 **$e-x$ 沒有從 loss 消失；它只是從 JVP tangent 消失。**

## The notation trap: $v_\theta$ vs JVP vs $V_\theta$

這是最容易混淆的地方：

- $u_\theta$：average velocity model；
- 小寫 $v_\theta$：instantaneous / marginal-like velocity estimate，用來當 tangent；
- $\operatorname{JVP}(u_\theta;v_\theta)$：對 $u_\theta$ 做 directional derivative，$v_\theta$ 只是方向；
- 大寫 $V_\theta$：$u_\theta$ 加 JVP correction 後的 compound predictor；
- $e-x$：sample-level conditional supervision。

互動式符號拆解與完整公式請看 [Improved MeanFlow deep dive](/research-handbook/meanflow/improved-meanflow/)。

## Where does $v_\theta$ come from?

Boundary variant：

$$
v_\theta(z_t,t)\equiv u_\theta(z_t,t,t).
$$

也可以用 training-only auxiliary $v$-head，額外以 $\|v_\theta-(e-x)\|^2$ 訓練。

## Why this matters

同一個 $z_t$ 可能由不同 sampled pairs 產生，所以 conditional $e-x$ 有 sample variance。Original MF 把它當 JVP tangent，variance 可能經 Jacobian-vector product 放大；iMF 改用 state-conditioned $v_\theta(z_t)$，讓 $V_\theta$ 成為只由 current state / conditioning 決定的 prediction function。

## Objective evidence

| Setting | 1-NFE FID ↓ |
|---|---:|
| Original MF-B/2 w/ CFG | 6.17 |
| iMF boundary | 5.97 |
| iMF aux v-head | 5.68 |
| Original MF-XL/2 w/ CFG | 3.43 |
| MF-XL/2 + iMF boundary objective | 2.99 |

## System-level changes

Paper 另外加入 flexible CFG、$\Omega$-conditioning、in-context conditioning、Transformer block changes 與 longer training。完整 iMF-XL/2 system 報告 FID **1.72**，因此不能把 1.72 全部歸因於 objective replacement。

## Interpretation

iMF 的價值是把「MeanFlow 的 target construction 問題」重新表述成 regression-function design：prediction path 應該由 state 決定，而不是額外依賴 sampled conditional tangent。這也把 future work 從單純 target patch 推向 interval difficulty、endpoint error、gradient geometry 與 refinement 等更廣的問題。

## Related data

- [Datasets｜訓練資料集與任務地圖](/research-handbook/datasets/)
