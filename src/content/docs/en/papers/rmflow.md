---
title: "RMFlow: Refined Mean Flow by a Noise-Injection Step for Multimodal Generation"
description: RMFlow adds a tailored noise-injection refinement after coarse MeanFlow transport.
sidebar:
  order: 3
---

## Metadata

- **Authors**: Yuhao Huang, Shih-Hsin Wang, Andrea L. Bertozzi, Bao Wang
- **Publication**: arXiv 2026
- **Topic**: MeanFlow, refinement, multimodal generation
- **Source**: [arXiv](https://arxiv.org/abs/2602.00849)

## Summary

RMFlow combines a coarse 1-NFE MeanFlow transport with a tailored noise-injection refinement step. Rather than simply adding a generic multi-step ODE solver, it introduces a separate refinement mechanism after fast transport.

## Method positioning

The paper approximates the average velocity of the flow path with a neural network and proposes a new loss balancing Wasserstein-distance minimization between probability paths with sample likelihood. The refinement is therefore tied to a distribution-level objective rather than being an arbitrary noise perturbation.

## Scope

The method is evaluated across multimodal generation settings including text-to-image, context-to-molecule, and time-series generation, broadening the scope of MeanFlow-style fast transport beyond a single image benchmark.

## Research interpretation

RMFlow shows a different branch from iMF: future one-step / few-step work does not have to modify only the JVP target. The problem can also be decomposed into coarse transport followed by distribution-aware refinement.

## Related pages

- [Research Track: MeanFlow Evolution](/research-handbook/en/meanflow/story/)
- [MeanFlow paper](/research-handbook/en/papers/meanflow/)
- [Improved MeanFlow paper](/research-handbook/en/papers/improved-meanflow/)
