---
title: Mean Flows for One-step Generative Modeling
description: MeanFlow models interval-average velocity for one-step generative modeling.
sidebar:
  order: 1
---

## Metadata

- **Authors**: Zhengyang Geng, Mingyang Deng, Xingjian Bai, J. Zico Kolter, Kaiming He
- **Venue**: NeurIPS 2025 · Oral
- **Topic**: one-step generation, flow matching, average velocity
- **Sources**: [NeurIPS](https://papers.neurips.cc/paper_files/paper/2025/hash/6d13e085b79d454da5910e4ca82a3d9d-Abstract-Conference.html) · [arXiv](https://arxiv.org/abs/2505.13447)

## Summary

MeanFlow shifts the modeled quantity from instantaneous velocity $v$ to interval-average velocity $u$. This gives one-step inference direct access to a finite-time transport quantity instead of repeatedly querying a local velocity field.

## Problem

Standard Flow Matching solves an ODE numerically at inference time. As the state moves, the local velocity may change, so multiple NFEs are usually required. MeanFlow seeks a field representation that can support direct one-step transport.

## Core method

MeanFlow defines

$$
u(z_t,r,t)=\frac{1}{t-r}\int_r^t v(z_\tau,\tau)\,d\tau.
$$

The MeanFlow Identity

$$
v=u+(t-r)\frac{du}{dt}
$$

connects the average and instantaneous fields. The total derivative $du/dt$ is computed through a JVP, avoiding explicit integration of a full ground-truth trajectory during training.

## Training vs inference

**Training** uses the sampled conditional velocity $e-x$ as supervision and constructs the MeanFlow target through a JVP-based relation.

**Inference** uses, in the one-step case,

$$
z_0=z_1-u_\theta(z_1,0,1).
$$

## Evidence

On ImageNet 256×256 with 1 NFE and training from scratch, the paper reports **FID 3.43** for MF-XL/2.

## Research interpretation

The main shift is not simply a more accurate local velocity estimator. MeanFlow changes the temporal scale of the modeled object: $u$ directly represents finite-time transport. This creates the target-construction and JVP questions addressed by later work.

## Related pages

- [Research Track: MeanFlow Evolution](/research-handbook/en/meanflow/story/)
- [MeanFlow mathematical deep dive](/research-handbook/en/meanflow/meanflow/)
- [Flow Matching concept](/research-handbook/en/foundations/flow-matching/)
