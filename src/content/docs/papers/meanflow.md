---
title: Mean Flows for One-step Generative Modeling
description: MeanFlow 以 interval-average velocity 建立 one-step generative modeling framework。
sidebar:
  order: 1
---

## Metadata

- **Authors**：Zhengyang Geng, Mingyang Deng, Xingjian Bai, J. Zico Kolter, Kaiming He
- **Venue**：NeurIPS 2025 · Oral
- **Topic**：one-step generation, flow matching, average velocity
- **Sources**：[NeurIPS](https://papers.neurips.cc/paper_files/paper/2025/hash/6d13e085b79d454da5910e4ca82a3d9d-Abstract-Conference.html) · [arXiv](https://arxiv.org/abs/2505.13447)

## Summary

MeanFlow 的核心 representation shift 是從 instantaneous velocity $v$ 轉向 interval-average velocity $u$。這使 one-step inference 可以直接使用一個跨完整時間區間的 transport quantity，而不必在 inference 時反覆 query local velocity。

## Problem

一般 Flow Matching 在 inference 時數值積分 ODE：state 每改變一次，就可能需要重新估計 local velocity，因此通常需要 multiple NFEs。MeanFlow 的目標是建立一個可以直接支援 one-step transport 的 field representation。

## Core method

MeanFlow 定義

$$
u(z_t,r,t)=\frac{1}{t-r}\int_r^t v(z_\tau,\tau)\,d\tau.
$$

並利用 MeanFlow Identity

$$
v=u+(t-r)\frac{du}{dt}
$$

把 average velocity 與 instantaneous velocity 連接起來。$du/dt$ 透過 total derivative / JVP 計算，因此 training 不需要顯式積分完整 ground-truth trajectory。

## Training vs inference

**Training**：以 sampled pair 的 conditional velocity $e-x$ 建構 supervision，並透過 JVP-based target construction 訓練 $u_\theta$。

**Inference**：在 one-step case，直接使用

$$
z_0=z_1-u_\theta(z_1,0,1).
$$

## Evidence

論文在 ImageNet 256×256、1-NFE、from-scratch setting 下報告 MF-XL/2 的 FID 為 **3.43**。

## Research interpretation

這篇 paper 最重要的變化不是單純讓 local velocity 更準，而是改變模型所學 quantity 的時間尺度：$u$ 本身承載 finite-time transport 的語意。這也直接引出後續對 target construction、JVP tangent 與 training stability 的研究。

## Related pages

- [Research Track：MeanFlow Evolution](/research-handbook/meanflow/story/)
- [MeanFlow mathematical deep dive](/research-handbook/meanflow/meanflow/)
- [Flow Matching concept](/research-handbook/foundations/flow-matching/)
