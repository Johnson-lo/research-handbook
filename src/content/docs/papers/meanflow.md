---
title: Mean Flows for One-step Generative Modeling
description: MeanFlow 以 interval-average velocity 建立 one-step generative modeling framework。
sidebar:
  order: 1
---

## Metadata

- **Authors**：Zhengyang Geng, Mingyang Deng, Xingjian Bai, J. Zico Kolter, Kaiming He
- **Venue**：NeurIPS 2025 · Oral
- **Year**：2025
- **Primary task**：ImageNet-1K 256×256 class-conditional one-step / few-step image generation
- **Core topics**：Flow Matching、interval-average velocity、MeanFlow Identity、JVP、one-step generation
- **Data / benchmark**：ImageNet-1K 256×256
- **Main metrics**：FID、Inception Score、NFE
- **Sources**：[NeurIPS](https://papers.neurips.cc/paper_files/paper/2025/hash/6d13e085b79d454da5910e4ca82a3d9d-Abstract-Conference.html) · [arXiv](https://arxiv.org/abs/2505.13447)

## Core question

Flow Matching 已經能穩定學到 instantaneous velocity，但 inference 仍需要 ODE solver 反覆 query local field。這篇 paper 問的是：

> **能不能直接學 finite-time average transport，而不是每一步只問 local direction？**

## Baseline recap

Linear path：

$$
z_t=(1-t)x+te,
$$

sample-level conditional velocity：

$$
e-x.
$$

Flow Matching 訓練 $v_\theta(z_t,t)$ 去 regression $e-x$，但 inference 仍要解 $dz_t/dt=v_\theta(z_t,t)$。

## Core idea

MeanFlow 把 modeled object 改成 interval-average velocity：

$$
u(z_t,r,t)=\frac{1}{t-r}\int_r^t v(z_\tau,\tau)d\tau.
$$

因此

$$
z_r=z_t-(t-r)u(z_t,r,t).
$$

One-step 直接得到

$$
z_0=z_1-u_\theta(z_1,0,1).
$$

## Why the derivation is necessary

難點在於 $u$ 沒有像 $e-x$ 一樣直接的 per-sample GT。從

$$
(t-r)u=\int_r^t v(z_\tau,\tau)d\tau
$$

對 $t$ 微分：

$$
v=u+(t-r)\frac{du}{dt}.
$$

而

$$
\frac{du}{dt}=\partial_z u\,v+\partial_tu
=\operatorname{JVP}(u;v).
$$

若偏微分、total derivative 或 JVP 不熟，先看 [Calculus & JVP](/research-handbook/concepts/calculus-jvp/)；完整逐步推導在 [MeanFlow deep dive](/research-handbook/meanflow/meanflow/)。

## Training objective

Original MF 用 sample conditional velocity $e-x$ 近似 JVP tangent：

$$
u_{tgt}=(e-x)-(t-r)\operatorname{JVP}(u_\theta;e-x),
$$

$$
\mathcal L_{MF}=\mathbb E\|u_\theta-\operatorname{sg}(u_{tgt})\|^2.
$$

## Training vs inference

**Training**：抽 $x,e,t,r$，直接構造 $z_t$；不需要從 noise rollout 整條 trajectory。JVP 用來建立 average-velocity training relation。

**Inference**：只需要 $u_\theta$；one-step 使用 $z_0=z_1-u_\theta(z_1,0,1)$。

## Evidence

ImageNet 256×256、1-NFE、from scratch，MF-XL/2 報告 FID **3.43**。

## Interpretation

這篇 paper 的真正創新不是「把 local velocity regression 做得更準」，而是改變 network output 的時間尺度：從 local derivative 變成 finite-interval average transport。代價是 training target 不再直接，JVP 與 network-dependent target 因而變成後續研究核心。

## Limitation / next paper

Original MF 把 sample-specific $e-x$ 放入 JVP tangent。這個設計在 [Improved MeanFlow](/research-handbook/papers/improved-meanflow/) 中被重新檢視，並改寫成 state-conditioned v-loss predictor。

## Related data

- [Datasets｜訓練資料集與任務地圖](/research-handbook/datasets/)
