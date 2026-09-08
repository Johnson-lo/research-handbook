---
title: "Understanding, Accelerating, and Improving MeanFlow Training"
description: Controlled experiments on the learning order of instantaneous velocity, small-gap average velocity, and large-gap MeanFlow transport.
sidebar:
  order: 4
---

## Metadata

- **Authors**：Jin-Young Kim, Hyojun Go, Lea Bogensperger, Julius Erbach, Nikolai Kalischek, Federico Tombari, Konrad Schindler, Dominik Narnhofer
- **Venue**：CVPR 2026
- **Topic**：MeanFlow training dynamics, instantaneous velocity, average velocity, temporal gap, curriculum
- **Sources**：[CVF Open Access](https://openaccess.thecvf.com/content/CVPR2026/html/Kim_Understanding_Accelerating_and_Improving_MeanFlow_Training_CVPR_2026_paper.html) · [arXiv](https://arxiv.org/abs/2511.19065) · [Code](https://github.com/seahl0119/ImprovedMeanFlow)

## Core question

MeanFlow 同時牽涉 instantaneous velocity $v$ 與 interval-average velocity $u$。但兩者真的可以從 training 一開始就平等地一起學嗎？或者存在一個依賴順序：

$$
\text{instantaneous }v
\rightarrow
\text{small-gap }u
\rightarrow
\text{large-gap }u?
$$

這篇 paper 直接用 controlled experiments 研究這個問題。

## Key findings

作者得到三個非常重要的 observation：

1. **well-established instantaneous velocity 是學 average velocity 的 prerequisite。** 如果 $v$ 沒先形成，average velocity learning 會失敗或不穩。
2. **small temporal gap 的 average-velocity supervision 會幫助 $v$；large gap 則會傷害 $v$。**
3. **large-gap average velocity 的穩定 learning，依賴先前已形成的 accurate instantaneous velocity 與 small-gap average velocity。**

這讓 temporal gap $\Delta t=t-r$ 不再只是 sampling hyperparameter，而是 training curriculum 的難度軸。

## Connection to the $r=t$ hypothesis

當

$$
r=t,
$$

有

$$
\Delta t=0,
$$

而 MeanFlow boundary relation 給出

$$
u(z_t,t,t)=v(z_t,t).
$$

因此「先學 $r=t$」可以被理解成 training curriculum 最 local 的極限：先建立 instantaneous velocity。這篇 paper **強烈支持「先把 $v$ 建立起來，再從 small $\Delta t$ 往 large $\Delta t$ 擴張」**。

但需要精確區分：paper 實驗支持的是 **instantaneous-first + short-to-long interval curriculum**；這並不自動等價於「所有模型都應先做一段純 $r=t$ pretraining」。後者仍是一個可以獨立驗證的具體實驗設計。

## Training scheme

作者根據上述 observation 設計 enhanced training：先加速 instantaneous velocity 的形成，之後把 training emphasis 從 short interval 逐步移向 long interval。這正好符合 one-step generation 的需求：large-gap $u$ 是最後必須學會的能力，但不是一開始最適合強迫模型學的能力。

## Evidence

同樣使用 DiT-XL backbone、ImageNet 256×256：

- conventional MeanFlow：**1-NFE FID 3.43**
- enhanced training：**1-NFE FID 2.87**
- 或者可在約 **2.5× shorter training time** 下達到 conventional MeanFlow 相近表現

## Why this matters

這篇 paper 把一個常見但容易被忽略的問題變成可測量的 research axis：**MeanFlow 不同 temporal gaps 不是同等難度，也不是互相獨立的 tasks。**

對後續研究，這直接產生幾個可做的 hypothesis：

- strict $r=t$ warm-up vs small-gap warm-up，哪一個更有效？
- curriculum transition 應由 epoch、loss plateau、task-affinity score，還是 gradient alignment 控制？
- robotics/action generation 的 trajectory horizon 是否也存在相似的 short-horizon → long-horizon curriculum？

## Related pages

- [Research Problem Map](/research-handbook/tracks/problem-map/)
- [AlphaFlow](/research-handbook/papers/alphaflow/)
- [MeanFlow paper](/research-handbook/papers/meanflow/)
