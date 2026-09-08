---
title: RMFlow: Refined Mean Flow by a Noise-Injection Step for Multimodal Generation
description: RMFlow 在 coarse MeanFlow transport 後加入 tailored noise-injection refinement。
sidebar:
  order: 3
---

## Metadata

- **Authors**：Yuhao Huang, Shih-Hsin Wang, Andrea L. Bertozzi, Bao Wang
- **Publication**：arXiv 2026
- **Topic**：MeanFlow, refinement, multimodal generation
- **Source**：[arXiv](https://arxiv.org/abs/2602.00849)

## Summary

RMFlow 將 MeanFlow 的 coarse 1-NFE transport 與後續的 tailored noise-injection refinement step 結合。它不是單純增加一個一般的 multi-step ODE solver，而是以額外 refinement mechanism 改善 single-evaluation transport 後的 sample quality。

## Method positioning

Paper 將 flow path 的 average velocity 交由 neural network 近似，並提出新的 loss，在 probability paths 之間的 Wasserstein-distance minimization 與 sample likelihood 之間取得平衡。這個設計讓 refinement 不只是任意加入 noise，而是和 distribution-level objective 綁在一起。

## Scope

論文把方法展示在多模態生成任務，包括 text-to-image、context-to-molecule 與 time-series generation，顯示 MeanFlow-style fast transport 不只侷限於單一 image benchmark。

## Research interpretation

RMFlow 對 MeanFlow 主線的意義在於：one-step / few-step 方法的後續研究不一定只能繼續修改 JVP target，也可以把問題重新拆成 coarse transport 與 distribution-aware refinement。這形成和 iMF objective reformulation 不同的研究分支。

## Related pages

- [Research Track：MeanFlow Evolution](/research-handbook/meanflow/story/)
- [MeanFlow paper](/research-handbook/papers/meanflow/)
- [Improved MeanFlow paper](/research-handbook/papers/improved-meanflow/)
