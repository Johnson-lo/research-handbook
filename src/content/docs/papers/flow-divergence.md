---
title: Improving Flow Matching by Aligning Flow Divergence
description: 以 divergence alignment 補強 conditional flow matching 對 probability path accuracy 的控制。
sidebar:
  order: 4
---

## Metadata

- **Authors**：Yuhao Huang, Taos Transue, Shih-Hsin Wang, William M. Feldman, Hong Zhang, Bao Wang
- **Venue**：ICML 2025
- **Topic**：flow matching, divergence, probability-path error
- **Sources**：[PMLR](https://proceedings.mlr.press/v267/huang25ag.html) · [arXiv](https://arxiv.org/abs/2602.00869) · [Code](https://github.com/Utah-Math-Data-Science/Flow_Div_Matching)

## Summary

這篇 paper 的核心問題不是 sampling NFE，而是：**conditional flow matching loss 足以讓 velocity regression 變好，但是否足以保證 learned probability path 本身準確？** 作者從 probability-path error 出發，導出新的 PDE characterization 與 solution，並用它建立 training objective。

## Theoretical contribution

Paper 證明 learned 與 exact probability paths 的 total-variation gap，可以由 CFM loss 與 associated divergence loss 的組合上界控制。這使 divergence 不再只是額外 regularizer，而是直接連到 distribution-path error 的理論量。

## Objective

方法同時匹配 flow 與 flow divergence。研究焦點因此從單純的 pointwise vector regression，擴展到 learned vector field 是否產生正確 probability evolution。

## Evidence

Paper 在 dynamical systems、DNA sequences 與 videos 等 benchmark 展示改善，並強調不需要犧牲 generation efficiency。

## Research interpretation

這條研究線和 MeanFlow 的問題不同但互補：MeanFlow 改變「模型要預測什麼」以降低 NFE；flow-divergence alignment 則追問「即使 velocity regression 看起來合理，probability path 是否真的對」。這種 distribution-level constraint 對 fast generative models 也可能是重要的評估與 training lens。

## Related pages

- [Research Track：Flow Matching Objectives](/research-handbook/tracks/flow-matching-objectives/)
- [Flow Matching concept](/research-handbook/foundations/flow-matching/)
