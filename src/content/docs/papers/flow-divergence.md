---
title: Improving Flow Matching by Aligning Flow Divergence
description: 以 divergence alignment 補強 conditional flow matching 對 probability path accuracy 的控制。
sidebar:
  order: 4
---

## Core question

Conditional Flow Matching 可以讓 vector-field regression loss 變小，但這是否足以保證 learned probability path 本身也正確？這篇 paper 的核心問題就是：

> **「velocity 預測得像」與「整個 distribution evolution 正確」是不是同一件事？**

答案是否定的，因此作者從 probability-path error 的 PDE 出發，找出還需要控制 divergence gap。

## Theoretical bridge

Paper 建立 learned path $\hat p_t$ 與 exact path $p_t$ 的 error dynamics，並得到 total-variation bound。核心結果之一是

$$
\operatorname{TV}(p_t,\hat p_t)\le \frac12\mathcal L_{CDM}.
$$

這把 distribution-path error 和 divergence-related training quantity 直接連起來。

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

## Interpretation

這條研究線與 MeanFlow 非常互補：MeanFlow 問「模型該預測 local 還是 finite-time quantity」；Flow Divergence Matching 問「不管你怎麼 parameterize，learned field 對 distribution path 的控制是否足夠」。對 future fast generators，這提供一個比單純 sample FID 更接近 dynamics correctness 的 training lens。
