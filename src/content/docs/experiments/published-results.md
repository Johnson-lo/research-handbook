---
title: MeanFlow vs iMF — 公開實驗結果
sidebar:
  order: 1
---

## Objective-focused ablation

如果要隔離 objective change 的效果，最有資訊量的結果不一定是最終的 FID 1.72，而是較受控制的 objective ablation。

| Setting | 1-NFE FID ↓ | Interpretation |
|---|---:|---|
| MF-B/2 original MF, fixed CFG | 6.17 | baseline |
| iMF, boundary $v_\theta=u_\theta(z_t,t,t)$ | 5.97 | 改變 JVP tangent / predictor parameterization |
| iMF, auxiliary $v$-head | 5.68 | 較強的 marginal-velocity estimate |
| MF-XL/2 original MF | 3.43 | large-model baseline |
| MF-XL/2 + iMF boundary objective | 2.99 | XL scale 的 objective improvement |

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

:::caution[不要把最終 1.72 全部歸因於 objective 改動]
Paper 不只改 objective，也改了 CFG conditioning 與 conditioning architecture。移除 adaLN-zero 後 parameter / compute profile 也不同，因此 B/M/L/XL 的 system-level rows 並不是完全受控制的 objective-only comparison。
:::
