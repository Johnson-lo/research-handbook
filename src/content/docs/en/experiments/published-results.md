---
title: MeanFlow vs iMF — Published Experimental Results
sidebar:
  order: 1
---

## Objective-focused ablation

The most useful results for isolating the objective change are not necessarily the final 1.72 FID system result.

| Setting | 1-NFE FID ↓ | Interpretation |
|---|---:|---|
| MF-B/2 original MF, fixed CFG | 6.17 | baseline |
| iMF, boundary $v_\theta=u_\theta(z_t,t,t)$ | 5.97 | change JVP tangent / predictor parameterization |
| iMF, auxiliary $v$-head | 5.68 | stronger marginal-velocity estimate |
| MF-XL/2 original MF | 3.43 | large-model baseline |
| MF-XL/2 + iMF boundary objective | 2.99 | objective improvement at XL scale |

## iMF system ablation

| Stage | Params | 1-NFE FID ↓ |
|---|---:|---:|
| best objective result | 133M | 5.68 |
| $\omega$-conditioning | 133M | 5.52 |
| $\Omega$-conditioning incl. CFG interval | 133M | 4.57 |
| in-context conditioning | 89M | 4.09 |
| + SwiGLU / RMSNorm / RoPE | 89M | 3.82 |
| + longer training (640 epochs) | 89M | 3.39 |

## System-level comparison

| Size | MF FID ↓ | iMF FID ↓ |
|---|---:|---:|
| B/2 | 6.17 | 3.39 |
| M/2 | 5.01 | 2.27 |
| L/2 | 3.84 | 1.86 |
| XL/2 | 3.43 | 1.72 |

:::caution[Do not over-attribute the final 1.72]
The paper changes more than the objective: CFG conditioning and the conditioning architecture also change. Removing adaLN-zero changes the parameter/compute profile, so the B/M/L/XL system-level rows are not a perfectly controlled objective-only comparison.
:::
