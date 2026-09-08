---
title: "Improved Mean Flows: On the Challenges of Fastforward Generative Models"
description: iMF reformulates the MeanFlow objective and extends guidance, conditioning, and system design.
sidebar:
  order: 2
---

## Metadata

- **Authors**: Zhengyang Geng, Yiyang Lu, Zongze Wu, Eli Shechtman, J. Zico Kolter, Kaiming He
- **Publication**: arXiv 2512.02012, revised 2026
- **Topic**: MeanFlow, objective reformulation, classifier-free guidance
- **Source**: [arXiv](https://arxiv.org/abs/2512.02012)

## Core question

Original MeanFlow already enables one-step generation, but its training target depends on both sample-ground-truth quantities and the network itself. iMF asks whether the formulation can be recast as a more standard prediction function while preserving one-step inference.

## Original MF as a v-loss

$$
V_\theta(z_t,e-x)=u_\theta(z_t,r,t)+(t-r)\operatorname{JVP}_{sg}(u_\theta;e-x),
$$

$$
\mathcal L=\mathbb E\|V_\theta(z_t,e-x)-(e-x)\|^2.
$$

The predictor visibly depends on the sample-specific conditional tangent $e-x$.

## iMF replacement

iMF introduces a marginal-like velocity estimate $v_\theta(z_t,t)$ and uses

$$
V_\theta(z_t)=u_\theta(z_t,r,t)+(t-r)\operatorname{JVP}_{sg}(u_\theta;v_\theta),
$$

with unchanged supervision

$$
\mathcal L_{iMF}=\mathbb E\|V_\theta(z_t)-(e-x)\|^2.
$$

## The notation trap

- $u_\theta$: average-velocity model;
- lowercase $v_\theta$: instantaneous / marginal-like velocity estimate used as tangent;
- $\operatorname{JVP}(u_\theta;v_\theta)$: directional derivative of $u_\theta$; $v_\theta$ is the direction, not the differentiated function;
- capital $V_\theta$: compound predictor after adding the JVP correction;
- $e-x$: sample-level conditional supervision.

See the [Improved MeanFlow deep dive](/research-handbook/en/meanflow/improved-meanflow/) for the interactive notation explorer.

## Where $v_\theta$ comes from

The boundary variant uses $v_\theta(z_t,t)\equiv u_\theta(z_t,t,t)$. An alternative training-only auxiliary head predicts $v_\theta$ with an additional Flow Matching loss $\|v_\theta-(e-x)\|^2$.

## Why this matters

Conditional $e-x$ can vary across sampled pairs consistent with the same state. Passing that sample-specific tangent through a JVP can amplify variance. iMF instead uses state-conditioned $v_\theta(z_t)$ so the compound predictor becomes a legitimate state-conditioned regression function.

## Objective evidence

| Setting | 1-NFE FID ↓ |
|---|---:|
| Original MF-B/2 w/ CFG | 6.17 |
| iMF boundary | 5.97 |
| iMF aux v-head | 5.68 |
| Original MF-XL/2 w/ CFG | 3.43 |
| MF-XL/2 + iMF boundary objective | 2.99 |

## System-level changes

The complete method also adds flexible CFG, $\Omega$-conditioning, in-context conditioning, Transformer block changes, and longer training. The final iMF-XL/2 system reports **FID 1.72**, which is therefore a system-level result rather than an objective-only result.
