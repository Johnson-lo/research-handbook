---
title: "AlphaFlow: Understanding and Improving MeanFlow Models"
description: AlphaFlow analyzes gradient conflict inside MeanFlow and uses a curriculum from trajectory flow matching toward MeanFlow.
sidebar:
  order: 3
---

## Metadata

- **Authors**: Huijie Zhang, Aliaksandr Siarohin, Willi Menapace, Michael Vasilkovsky, Sergey Tulyakov, Qing Qu, Ivan Skorokhodov
- **Venue**: ICLR 2026
- **Topic**: MeanFlow, optimization conflict, trajectory flow matching, trajectory consistency, curriculum
- **Sources**: [ICLR Proceedings](https://proceedings.iclr.cc/paper_files/paper/2026/hash/e8c20cafe841cba3e31a17488dc9c3f1-Abstract-Conference.html) · [Code](https://github.com/snap-research/alphaflow)

## Core question

MeanFlow aims to learn large-stride average velocity, yet training spends substantial supervision on the boundary case $r=t$, where the problem becomes much more Flow-Matching-like. Why is local/boundary supervision useful when the ultimate goal is long-interval transport?

:::caution[Evidence boundary]
AlphaFlow supports a curriculum that begins closer to trajectory flow matching and smoothly transitions to MeanFlow. It does not prove that a strict, fixed-duration $r=t$-only pretraining phase is universally optimal.
:::

## Objective decomposition

AlphaFlow decomposes the MeanFlow objective into trajectory flow matching and trajectory consistency. Gradient analysis shows strong negative correlation between the two, creating optimization conflict and slow convergence.

## Why $r=t$ is special

At

$$
r=t,
$$

we have $\Delta t=t-r=0$ and the boundary relation

$$
u(z_t,t,t)=v(z_t,t).
$$

The average-velocity task collapses to instantaneous velocity, making the supervision more local and Flow-Matching-like.

## Curriculum

The paper introduces an $\alpha$-Flow family and anneals

$$
\alpha:1\longrightarrow0,
$$

moving from a trajectory-flow-matching-dominant regime toward full MeanFlow.

## Evidence

On ImageNet-1K 256×256 with vanilla DiT backbones, $\alpha$-Flow-XL/2+ reports **FID 2.58 at 1 NFE** and **2.15 at 2 NFE**.

## Research interpretation

AlphaFlow gives an optimization explanation for why boundary / Flow-Matching-like supervision can be useful: establish a lower-variance flow-fitting capability before progressively taking on the higher-variance long-interval MeanFlow task. This complements the temporal-gap experiments in [Understanding, Accelerating, and Improving MeanFlow Training](/research-handbook/en/papers/meanflow-training/).

## Related pages

- [Research Problem Map](/research-handbook/en/tracks/problem-map/)
- [MeanFlow paper](/research-handbook/en/papers/meanflow/)
- [MeanFlow Training Dynamics](/research-handbook/en/papers/meanflow-training/)
