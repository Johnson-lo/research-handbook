---
title: Improved Mean Flows: On the Challenges of Fastforward Generative Models
description: iMF 重構 MeanFlow objective，並延伸 guidance、conditioning 與 system design。
sidebar:
  order: 2
---

## Metadata

- **Authors**：Zhengyang Geng, Yiyang Lu, Zongze Wu, Eli Shechtman, J. Zico Kolter, Kaiming He
- **Venue**：CVPR 2026
- **Topic**：MeanFlow, objective reformulation, classifier-free guidance
- **Sources**：[CVPR Open Access](https://openaccess.thecvf.com/content/CVPR2026/html/Geng_Improved_Mean_Flows_On_the_Challenges_of_Fastforward_Generative_Models_CVPR_2026_paper.html) · [arXiv](https://arxiv.org/abs/2512.02012)

## Summary

Improved MeanFlow 直接處理 original MeanFlow 的 training objective 與 guidance flexibility。核心 objective change 是把 JVP tangent 從 sample-specific $e-x$ 改成 estimated marginal velocity $v_\theta(z_t)$，但 $e-x$ 仍然保留為 supervision。

## Objective reformulation

Original MF 可用 equivalent v-loss 表示為

$$
V_\theta=u_\theta+(t-r)\operatorname{JVP}_{sg}(u_\theta;e-x),
$$

$$
\mathcal L=\mathbb E\|V_\theta-(e-x)\|^2.
$$

iMF 則改為

$$
V_\theta(z_t)=u_\theta(z_t)+(t-r)\operatorname{JVP}_{sg}(u_\theta;v_\theta),
$$

而 supervision 仍是

$$
\mathcal L=\mathbb E\|V_\theta(z_t)-(e-x)\|^2.
$$

## Additional system changes

除了 objective reformulation，paper 也加入 flexible guidance conditioning、$\Omega$-conditioning、in-context conditioning、Transformer block changes 與 longer training。這些變化需要和 objective-only improvement 分開解讀。

## Evidence

較 controlled 的 objective evidence 包括 MF-B/2 由 6.17 改善到 5.97 / 5.68，以及 MF-XL/2 的 boundary-objective setting 由 3.43 改善到 2.99。完整 iMF-XL/2 system 最終報告 **FID 1.72**。

## Research interpretation

這篇 paper 把「network-dependent target」重新表述成一個更接近 standard regression 的 compound predictor 問題，也凸顯 sample-specific conditional tangent 與 marginal field 之間的差異。最終 1.72 則是 objective、guidance、conditioning、architecture 與 training changes 的 system-level 結果。

## Related pages

- [Research Track：MeanFlow Evolution](/research-handbook/meanflow/story/)
- [Improved MeanFlow deep dive](/research-handbook/meanflow/improved-meanflow/)
- [Original MeanFlow paper](/research-handbook/papers/meanflow/)
