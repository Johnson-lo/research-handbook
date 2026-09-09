---
title: "Understanding, Accelerating, and Improving MeanFlow Training"
description: Controlled experiments on the learning order of instantaneous velocity, small-gap average velocity, and large-gap MeanFlow transport.
sidebar:
  order: 4
---

## Metadata

- **Authors**: Jin-Young Kim, Hyojun Go, Lea Bogensperger, Julius Erbach, Nikolai Kalischek, Federico Tombari, Konrad Schindler, Dominik Narnhofer
- **Venue**: CVPR 2026
- **Year**: 2026
- **Primary task**: ImageNet-1K 256×256 MeanFlow training dynamics / one-step image generation
- **Core topics**: instantaneous velocity, average velocity, temporal gap, short-to-long curriculum, training acceleration
- **Data / benchmark**: ImageNet-1K 256×256
- **Main metrics**: FID, training time / convergence, instantaneous / average velocity interaction
- **Sources**: [CVF Open Access](https://openaccess.thecvf.com/content/CVPR2026/html/Kim_Understanding_Accelerating_and_Improving_MeanFlow_Training_CVPR_2026_paper.html) · [arXiv](https://arxiv.org/abs/2511.19065) · [Code](https://github.com/seahl0119/ImprovedMeanFlow)

## Core question

MeanFlow couples instantaneous velocity $v$ and interval-average velocity $u$. Should these be learned equally from the beginning, or is there a dependency order?

$$
\text{instantaneous }v
\rightarrow
\text{small-gap }u
\rightarrow
\text{large-gap }u
$$

## Key findings

The controlled experiments show that: (1) a well-established instantaneous velocity is a prerequisite for learning average velocity; (2) small-gap average-velocity supervision helps instantaneous velocity, while large gaps can degrade it; and (3) smooth learning of large-gap average velocities depends on prior formation of accurate instantaneous and small-gap velocities.

## Connection to the $r=t$ hypothesis

At $r=t$, $\Delta t=0$ and

$$
u(z_t,t,t)=v(z_t,t).
$$

Thus an $r=t$-first idea is the most local limiting case of an instantaneous-first curriculum. The paper strongly supports **instantaneous-first followed by short-to-long interval learning**, but does not by itself prove that a strict $r=t$-only pretraining stage is universally optimal.

## Training scheme and evidence

The enhanced scheme accelerates formation of instantaneous velocity, then shifts emphasis from short to long intervals. With the same DiT-XL backbone on ImageNet 256×256, it improves 1-NFE FID from **3.43 to 2.87**, or matches the baseline with roughly **2.5× shorter training time**.

## Research consequences

This makes temporal gap a measurable curriculum axis rather than a passive sampling variable. Natural follow-up questions include strict $r=t$ warm-up versus small-gap warm-up, adaptive transition criteria, and whether analogous short-horizon → long-horizon curricula appear in robotics/action generation.

## Related pages

- [Research Problem Map](/research-handbook/en/tracks/problem-map/)
- [AlphaFlow](/research-handbook/en/papers/alphaflow/)
- [MeanFlow paper](/research-handbook/en/papers/meanflow/)
- [Datasets | Training Data & Task Map](/research-handbook/en/datasets/)
