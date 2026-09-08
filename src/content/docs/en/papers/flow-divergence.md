---
title: Improving Flow Matching by Aligning Flow Divergence
description: Divergence alignment strengthens conditional flow matching with control over probability-path accuracy.
sidebar:
  order: 4
---

## Metadata

- **Authors**: Yuhao Huang, Taos Transue, Shih-Hsin Wang, William M. Feldman, Hong Zhang, Bao Wang
- **Venue**: ICML 2025
- **Topic**: flow matching, divergence, probability-path error
- **Sources**: [PMLR](https://proceedings.mlr.press/v267/huang25ag.html) · [arXiv](https://arxiv.org/abs/2602.00869) · [Code](https://github.com/Utah-Math-Data-Science/Flow_Div_Matching)

## Summary

The central question is not sampling NFE but whether a good conditional flow-matching regression loss is sufficient to guarantee an accurate learned probability path. The paper derives a PDE characterization of probability-path error and uses it to motivate a stronger training objective.

## Theoretical contribution

The paper shows that the total-variation gap between learned and exact probability paths can be upper-bounded by a combination of the CFM loss and an associated divergence loss. Divergence is therefore tied directly to distribution-path error rather than serving only as an auxiliary regularizer.

## Objective

The method simultaneously matches the flow and its divergence. The focus moves from pointwise vector regression toward whether the learned field induces the correct probability evolution.

## Evidence

The paper reports improvements on benchmarks spanning dynamical systems, DNA sequences, and videos while maintaining generation efficiency.

## Research interpretation

This track is complementary to MeanFlow. MeanFlow changes what the model predicts in order to reduce NFE; divergence alignment asks whether an apparently good velocity regression actually yields the right probability path. Distribution-level constraints of this kind may also matter for fast generative models.

## Related pages

- [Research Track: Flow Matching Objectives](/research-handbook/en/tracks/flow-matching-objectives/)
- [Flow Matching concept](/research-handbook/en/foundations/flow-matching/)
