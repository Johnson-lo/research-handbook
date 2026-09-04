---
title: Mean Flows for One-step Generative Modeling
sidebar:
  order: 1
---

## Paper 狀態

- Venue：NeurIPS 2025 Oral
- 在本站的角色：original MeanFlow baseline

## 一句話總結

MeanFlow 不再只預測 instantaneous velocity，而是學習 interval-average velocity field，並推導出一個 differential identity，使這個 average field 在 training 時不需要顯式積分整條 ground-truth trajectory 就能被訓練。

## 閱讀重點

- Motivation：降低 generative ODE sampling 所需的 NFE
- Core object：average velocity $u(z_t,r,t)$
- Key derivation：MeanFlow Identity
- Training mechanism：JVP-based target construction
- Inference：$z_0=z_1-u_\theta(z_1,0,1)$
- Reported ImageNet-256 one-step FID：MF-XL/2 為 3.43

## 我的理解

核心轉變不是去學「更好的 local velocity」，而是直接學一個語意上已經包含 finite-time transport 的 quantity。
