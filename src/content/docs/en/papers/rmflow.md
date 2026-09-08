---
title: "RMFlow: Refined Mean Flow by a Noise-Injection Step for Multimodal Generation"
description: RMFlow adds a tailored noise-injection refinement after coarse MeanFlow transport.
sidebar:
  order: 3
---

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

## Interpretation

A key research lesson is that post-MeanFlow work need not only patch the JVP target. If endpoint quality is the bottleneck, the system can be decomposed into coarse deterministic transport plus low-cost stochastic refinement.
