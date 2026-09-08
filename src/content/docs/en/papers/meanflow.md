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

## Core question

Flow Matching learns an instantaneous velocity well, but inference still repeatedly queries a local field through an ODE solver. MeanFlow asks whether finite-time average transport can be modeled directly instead.

## Core idea

MeanFlow defines

$$
u(z_t,r,t)=\frac{1}{t-r}\int_r^t v(z_\tau,\tau)d\tau,
$$

so $z_r=z_t-(t-r)u(z_t,r,t)$ and one-step inference becomes $z_0=z_1-u_\theta(z_1,0,1)$.

## Why the derivation is necessary

The interval average $u$ has no simple per-sample label. Differentiating $(t-r)u=\int_r^t v(z_\tau,\tau)d\tau$ yields

$$
v=u+(t-r)\frac{du}{dt},
$$

with

$$
\frac{du}{dt}=\partial_z u\,v+\partial_tu=\operatorname{JVP}(u;v).
$$

See [Calculus & JVP](/research-handbook/en/concepts/calculus-jvp/) for the beginner explanation and [MeanFlow deep dive](/research-handbook/en/meanflow/meanflow/) for the full derivation.

## Training objective

Original MF uses the sample conditional velocity $e-x$ as the JVP tangent approximation:

$$
u_{tgt}=(e-x)-(t-r)\operatorname{JVP}(u_\theta;e-x),
$$

$$
\mathcal L_{MF}=\mathbb E\|u_\theta-\operatorname{sg}(u_{tgt})\|^2.
$$

## Training vs inference

**Training** directly constructs $z_t$ from sampled $x,e,t,r$ and uses the JVP relation; no trajectory rollout is needed. **Inference** uses only $u_\theta$.

## Evidence

On ImageNet 256×256 with 1 NFE and training from scratch, MF-XL/2 reports **FID 3.43**.

## Interpretation

The main innovation is a change in temporal scale of the modeled object: from a local derivative to finite-interval average transport. The cost is a less direct training relation, which creates the JVP and target-construction issues addressed by later work.
