---
title: "RMFlow: Refined Mean Flow by a Noise-Injection Step for Multimodal Generation"
description: RMFlow adds a tailored noise-injection refinement after coarse MeanFlow transport.
sidebar:
  order: 5
---

## Metadata

- **Authors**: Yuhao Huang, Shih-Hsin Wang, Andrea L. Bertozzi, Bao Wang
- **Venue**: ICLR 2026
- **Year**: 2026
- **Primary tasks**: text-to-image, context-to-molecule, time-series / dynamical-system generation
- **Core topics**: MeanFlow refinement, noise injection, likelihood / Wasserstein control, 1-NFE multimodal generation
- **Data / benchmarks**: QM9; Lorenz / FitzHugh–Nagumo trajectory benchmark; synthetic 1D mixture / 2D checkerboard; COCO 2017 5k split for additional text-image evaluation
- **Sources**: [ICLR Proceedings](https://proceedings.iclr.cc/paper_files/paper/2026/hash/7b8c48c00dd5e3090ba0976e297fae5c-Abstract-Conference.html) · [arXiv](https://arxiv.org/abs/2602.00849)

## Core question

A 1-NFE MeanFlow transport is fast, but coarse transport can leave visible sample and distribution errors in multimodal generation. RMFlow asks whether one can retain a single flow evaluation while adding a low-cost refinement that improves the target distribution.

## Method

RMFlow performs coarse 1-NFE MeanFlow transport followed by a tailored noise-injection refinement. The method is conceptually two-stage, but the learned flow is evaluated only once; the refinement is not another expensive ODE solve.

## Training objective

The likelihood-related term is

$$
\mathcal L_{NLL}=\mathbb E\left[\left\|(x_{data}+\sigma_{min}\epsilon)-(x_0+\hat u_{0,1}(x_0;\theta))\right\|^2\right],
$$

and the joint objective is

$$
\mathcal L_{RMFlow}=\mathcal L_{CMFM}+\lambda_1\mathcal L_{NLL}+\lambda_2\mathbb E\|\phi_\omega(c)\|^2.
$$

The refinement is therefore tied to likelihood / KL control rather than being an arbitrary noise perturbation.

## Relation to MeanFlow and iMF

MeanFlow changes the modeled quantity, iMF changes the JVP tangent / regression formulation, and RMFlow accepts that coarse 1-NFE transport may still leave an endpoint gap and attacks it through distribution-aware refinement.

## Scope

The paper covers text-to-image, context-to-molecule, and time-series generation. QM9 is the context-to-molecule benchmark; the dynamical-system setup follows Lorenz / FitzHugh–Nagumo trajectories; and the appendix additionally evaluates image-text alignment on a 5k COCO 2017 split. These roles should not be conflated with one another when documenting training versus evaluation data.

## Interpretation

A key research lesson is that post-MeanFlow work need not only patch the JVP target. If endpoint quality is the bottleneck, the system can be decomposed into coarse deterministic transport plus low-cost stochastic refinement.

## Related data

- [Datasets | Training Data & Task Map](/research-handbook/en/datasets/)
