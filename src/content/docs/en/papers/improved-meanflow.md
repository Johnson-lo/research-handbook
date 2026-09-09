---
title: "Improved Mean Flows: On the Challenges of Fastforward Generative Models"
description: iMF reformulates the MeanFlow objective and extends guidance, conditioning, and system design.
sidebar:
  order: 2
---

## Metadata

- **Authors**: Zhengyang Geng, Yiyang Lu, Zongze Wu, Eli Shechtman, J. Zico Kolter, Kaiming He
- **Venue**: CVPR 2026 · pp. 30467–30476
- **Year**: 2026
- **Primary task**: ImageNet-1K 256×256 class-conditional one-step / few-step image generation
- **Core topics**: MeanFlow objective reformulation, state-conditioned v-loss, classifier-free guidance, in-context conditioning
- **Data / benchmark**: ImageNet-1K 256×256
- **Sources**: [CVF Open Access](https://openaccess.thecvf.com/content/CVPR2026/html/Geng_Improved_Mean_Flows_On_the_Challenges_of_Fastforward_Generative_Models_CVPR_2026_paper.html) · [arXiv](https://arxiv.org/abs/2512.02012)

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

## Reading note | Could shared adaLN become a multi-condition bottleneck?

### Separate paper evidence from hypothesis

The paper directly establishes that iMF adds **flexible guidance conditioning, $\Omega$-conditioning, in-context conditioning, and Transformer block changes**. It does **not** directly prove that adaLN fails under many conditions. The following should therefore be treated as a **research hypothesis / potential gap**, not a paper conclusion.

### Hypothesis

If multiple conditions are first compressed into embeddings and then merged before producing one shared set of adaLN modulation parameters, heterogeneous signals may be forced through the same low-dimensional modulation bottleneck.

For MeanFlow / iMF, the condition set can be richer than ordinary class-conditional DiT:

- semantic / class condition;
- current time $t$;
- interval start $r$ or gap $t-r$;
- guidance variables such as $\Omega=\{\omega,t_{min},t_{max}\}$;
- future multimodal or robotics conditions such as image, language, state, and action history.

These signals have different physical roles. Potential failure modes worth testing include condition interference, scale mismatch, gradient conflict, modulation-capacity bottlenecks, and poor compositional generalization.

### Transfer ideas from other fields

| Transfer idea | Intuition | Candidate MeanFlow adaptation |
|---|---|---|
| **Condition-specific modulators** | FiLM / conditional normalization generate feature-wise modulation from conditioning information | Let each condition produce its own $\Delta\gamma_k,\Delta\beta_k,\Delta\alpha_k$ before fusion |
| **Global/local separation** | SPADE-style designs preserve spatial conditions through a dedicated modulation path | Keep $t,r,\Omega$ in global adaLN; send image/token/spatial conditions through cross-attention or spatial modulation |
| **In-context / token conditioning** | Multimodal Transformers preserve condition tokens rather than collapsing everything into one vector | Keep heterogeneous conditions as tokens and let each block query them |
| **Learnable gating / MoE** | Multi-task models use gates to balance shared experts under different task relationships | Learn block-wise gates over semantic/time/guidance modulation branches |
| **Condition dropout** | CFG and compositional training expose the model to missing condition subsets | Randomly drop individual conditions or subsets to test compositional robustness |
| **Gradient-conflict handling** | PCGrad and GradNorm explicitly address multi-task gradient interference / imbalance | Measure gradient cosine similarities across auxiliary losses or condition-specific branches, then project or reweight when needed |
| **Low-rank / orthogonal modulation** | Adapter-style methods restrict parameter updates to structured subspaces | Give condition branches separate low-rank or approximately orthogonal modulation bases |
| **Time-dependent fusion** | Diffusion models often need different conditioning strengths at different noise/time regimes | Make fusion weights depend on $t$ or $t-r$ |

### Minimal ablation ladder

1. baseline: merge all conditions → one adaLN MLP;
2. separate-then-sum: one small MLP per condition, then sum modulation;
3. separate + learned gate;
4. hybrid: temporal / interval / guidance variables in adaLN, semantic/context conditions in cross-attention;
5. orthogonal / low-rank modulation;
6. condition-drop curriculum.

Useful diagnostics include 1/2-NFE FID, single-condition vs combined-condition quality, block-wise modulation norms, cosine similarity between condition modulation vectors, gradient cosine similarity, sensitivity to $t/r/\Omega$, and generalization to unseen condition combinations.

A concise research question is:

> **Does shared adaLN become a conditioning bottleneck when MeanFlow must simultaneously encode semantic, temporal, interval, and guidance conditions?**

A method direction could be **Factorized Conditional Modulation for MeanFlow**: independent condition-specific modulation branches fused by learned gates, low-rank bases, or time-dependent weighting.

## Related data

- [Datasets | Training Data & Task Map](/research-handbook/en/datasets/)
