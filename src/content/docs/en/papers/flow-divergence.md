---
title: Improving Flow Matching by Aligning Flow Divergence
description: Divergence alignment strengthens conditional flow matching with control over probability-path accuracy.
sidebar:
  order: 6
---

## Metadata

- **Authors**: Yuhao Huang, Taos Transue, Shih-Hsin Wang, William M. Feldman, Hong Zhang, Bao Wang
- **Venue**: ICML 2025 · PMLR 267:25813–25834
- **Year**: 2025
- **Primary tasks**: 2D density estimation, DNA sequence generation, dynamical-system trajectory sampling, video prediction
- **Core topics**: Conditional Flow Matching, probability-path error, flow divergence, PDE characterization, Hutchinson trace estimation
- **Data / benchmarks**: 2D checkerboard synthetic data; DNA benchmark following the discrete / Dirichlet-flow setup; Lorenz and FitzHugh–Nagumo systems; KTH Actions video
- **Sources**: [PMLR](https://proceedings.mlr.press/v267/huang25ag.html) · [ICML](https://icml.cc/virtual/2025/poster/45878) · [Code](https://github.com/Utah-Math-Data-Science/Flow_Div_Matching)

## Core question

A low Conditional Flow Matching regression loss does not automatically guarantee that the learned probability path is accurate. The paper asks whether matching velocity values is enough to match the entire distribution evolution.

## Theoretical bridge

The paper derives PDE error dynamics between the learned path $\hat p_t$ and exact path $p_t$, giving total-variation control in terms of the flow/divergence errors.

## Objective

The practical objective keeps the CFM term and adds conditional divergence matching:

$$
\mathcal L_{FDM}=\lambda_1\mathcal L_{CFM}+\lambda_2\mathcal L_{CDM}.
$$

The method therefore augments pointwise vector regression with a constraint on distribution evolution.

## Intuition for divergence

A velocity field says where nearby points move; divergence describes whether the local flow expands or compresses volume. Two fields can be close pointwise yet induce different density compression / expansion, which is why vector matching alone may not fully control the probability path.

## Computation

High-dimensional divergence requires a Jacobian trace, so the paper uses a Hutchinson trace estimator rather than explicitly forming the full Jacobian.

## Benchmarks across modalities

The paper/code applies the same idea to 2D checkerboard density estimation, discrete DNA generation, Lorenz / FitzHugh–Nagumo trajectory sampling, and KTH video prediction. This breadth is useful when asking whether a flow objective captures dynamics beyond a single image benchmark.

## Interpretation

This direction is complementary to MeanFlow: MeanFlow changes the temporal object being predicted, while divergence alignment asks whether the learned dynamics produce the correct probability evolution regardless of parameterization.

## Related data

- [Datasets | Training Data & Task Map](/research-handbook/en/datasets/)
