---
title: Improved Mean Flows — On the Challenges of Fastforward Generative Models
sidebar:
  order: 2
---

## Paper 狀態

- Venue：CVPR 2026 Spotlight
- 在本站的角色：對 original MeanFlow 的直接修正與擴充

## 一句話總結

iMF 將 original network-dependent MeanFlow target 改寫成由 average-velocity network parameterize 的 velocity regression view，並把 JVP 中 sample-specific 的 conditional tangent 換成 estimated marginal velocity，同時利用 conditioning 讓 CFG 可以在 inference 時彈性調整。

## 主要解決問題

1. Original MF 的 target 依賴目前的 network。
2. Original MF 在 training 時固定 CFG scale。

## 主要實驗結果

ImageNet 256×256、1 NFE、from scratch、no distillation：FID 1.72。

## Research interpretation

iMF 之後，「只修正 network-dependent target」本身不應再被視為足夠的新 novelty。更值得探索的方向包括 temporal difficulty、endpoint-quality mismatch、gradient geometry 與 flow-map parameterization。
