---
title: Improving Flow Matching by Aligning Flow Divergence
description: Divergence alignment strengthens conditional flow matching with control over probability-path accuracy.
sidebar:
  order: 4
---

## Core question

A low Conditional Flow Matching regression loss does not automatically guarantee that the learned probability path is accurate. The paper asks whether matching velocity values is enough to match the entire distribution evolution.

## Theoretical bridge

The paper derives PDE error dynamics between the learned path $\hat p_t$ and exact path $p_t$, leading to a total-variation control such as

$$
\operatorname{TV}(p_t,\hat p_t)\le \frac12\mathcal L_{CDM}.
$$

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

## Interpretation

This direction is complementary to MeanFlow: MeanFlow changes the temporal object being predicted, while divergence alignment asks whether the learned dynamics produce the correct probability evolution regardless of parameterization.
