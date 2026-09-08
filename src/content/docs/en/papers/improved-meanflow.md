---
title: Improved Mean Flows: On the Challenges of Fastforward Generative Models
description: iMF reformulates the MeanFlow objective and extends guidance, conditioning, and system design.
sidebar:
  order: 2
---

## Metadata

- **Authors**: Zhengyang Geng, Yiyang Lu, Zongze Wu, Eli Shechtman, J. Zico Kolter, Kaiming He
- **Venue**: CVPR 2026
- **Topic**: MeanFlow, objective reformulation, classifier-free guidance
- **Sources**: [CVPR Open Access](https://openaccess.thecvf.com/content/CVPR2026/html/Geng_Improved_Mean_Flows_On_the_Challenges_of_Fastforward_Generative_Models_CVPR_2026_paper.html) · [arXiv](https://arxiv.org/abs/2512.02012)

## Summary

Improved MeanFlow addresses both the original MeanFlow training objective and guidance flexibility. The central objective change replaces the sample-specific $e-x$ JVP tangent with an estimated marginal velocity $v_\theta(z_t)$ while retaining $e-x$ as supervision.

## Objective reformulation

Original MF can be written as the equivalent v-loss

$$
V_\theta=u_\theta+(t-r)\operatorname{JVP}_{sg}(u_\theta;e-x),
$$

$$
\mathcal L=\mathbb E\|V_\theta-(e-x)\|^2.
$$

iMF instead uses

$$
V_\theta(z_t)=u_\theta(z_t)+(t-r)\operatorname{JVP}_{sg}(u_\theta;v_\theta),
$$

while supervision remains

$$
\mathcal L=\mathbb E\|V_\theta(z_t)-(e-x)\|^2.
$$

## Additional system changes

Beyond the objective reformulation, the paper adds flexible guidance conditioning, $\Omega$-conditioning, in-context conditioning, Transformer block changes, and longer training. These should be separated from objective-only improvements when interpreting the final result.

## Evidence

Controlled objective evidence includes MF-B/2 improving from 6.17 to 5.97 / 5.68 and an MF-XL/2 boundary-objective setting improving from 3.43 to 2.99. The complete iMF-XL/2 system reports **FID 1.72**.

## Research interpretation

The paper reframes the network-dependent target as a compound-predictor regression problem and clarifies the distinction between a sample-specific conditional tangent and a marginal field. The final 1.72 result is a system-level outcome combining objective, guidance, conditioning, architecture, and training changes.

## Related pages

- [Research Track: MeanFlow Evolution](/research-handbook/en/meanflow/story/)
- [Improved MeanFlow deep dive](/research-handbook/en/meanflow/improved-meanflow/)
- [Original MeanFlow paper](/research-handbook/en/papers/meanflow/)
