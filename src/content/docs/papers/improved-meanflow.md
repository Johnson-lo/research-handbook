---
title: "Improved Mean Flows: On the Challenges of Fastforward Generative Models"
description: iMF 重構 MeanFlow objective，並延伸 guidance、conditioning 與 system design。
sidebar:
  order: 2
---

## Metadata

- **Authors**：Zhengyang Geng, Yiyang Lu, Zongze Wu, Eli Shechtman, J. Zico Kolter, Kaiming He
- **Venue**：CVPR 2026 · pp. 30467–30476
- **Year**：2026
- **Primary task**：ImageNet-1K 256×256 class-conditional one-step / few-step image generation
- **Core topics**：MeanFlow objective reformulation、state-conditioned v-loss、classifier-free guidance、in-context conditioning
- **Data / benchmark**：ImageNet-1K 256×256
- **Sources**：[CVF Open Access](https://openaccess.thecvf.com/content/CVPR2026/html/Geng_Improved_Mean_Flows_On_the_Challenges_of_Fastforward_Generative_Models_CVPR_2026_paper.html) · [arXiv](https://arxiv.org/abs/2512.02012)

## Core question

Original MeanFlow 已經能 one-step generate，但 training target 同時依賴 ground-truth sample quantity 與 network 本身。iMF 問的是：

> **能不能把 MeanFlow 改寫成更接近 standard regression 的 prediction function，同時保留 one-step inference？**

## Step 1 — 先看 Original MF 的 v-loss view

將 original MF 重寫後，compound predictor 可以表示為

$$
V_\theta(z_t,e-x)
=u_\theta(z_t,r,t)
+(t-r)\operatorname{JVP}_{sg}(u_\theta;e-x).
$$

Loss 是

$$
\mathcal L=\mathbb E\|V_\theta(z_t,e-x)-(e-x)\|^2.
$$

這暴露出 predictor 額外依賴 sample-specific conditional tangent $e-x$。

## Step 2 — iMF 的核心 replacement

iMF parameterize 一個 marginal-like velocity estimate $v_\theta(z_t,t)$，並改成

$$
V_\theta(z_t)
=u_\theta(z_t,r,t)
+(t-r)\operatorname{JVP}_{sg}(u_\theta;v_\theta).
$$

Supervision 仍是

$$
\mathcal L_{iMF}=\mathbb E\|V_\theta(z_t)-(e-x)\|^2.
$$

因此 **$e-x$ 沒有從 loss 消失；它只是從 JVP tangent 消失。**

## The notation trap: $v_\theta$ vs JVP vs $V_\theta$

這是最容易混淆的地方：

- $u_\theta$：average velocity model；
- 小寫 $v_\theta$：instantaneous / marginal-like velocity estimate，用來當 tangent；
- $\operatorname{JVP}(u_\theta;v_\theta)$：對 $u_\theta$ 做 directional derivative，$v_\theta$ 只是方向；
- 大寫 $V_\theta$：$u_\theta$ 加 JVP correction 後的 compound predictor；
- $e-x$：sample-level conditional supervision。

互動式符號拆解與完整公式請看 [Improved MeanFlow deep dive](/research-handbook/meanflow/improved-meanflow/)。

## Where does $v_\theta$ come from?

Boundary variant：

$$
v_\theta(z_t,t)\equiv u_\theta(z_t,t,t).
$$

也可以用 training-only auxiliary $v$-head，額外以 $\|v_\theta-(e-x)\|^2$ 訓練。

## Why this matters

同一個 $z_t$ 可能由不同 sampled pairs 產生，所以 conditional $e-x$ 有 sample variance。Original MF 把它當 JVP tangent，variance 可能經 Jacobian-vector product 放大；iMF 改用 state-conditioned $v_\theta(z_t)$，讓 $V_\theta$ 成為只由 current state / conditioning 決定的 prediction function。

## Objective evidence

| Setting | 1-NFE FID ↓ |
|---|---:|
| Original MF-B/2 w/ CFG | 6.17 |
| iMF boundary | 5.97 |
| iMF aux v-head | 5.68 |
| Original MF-XL/2 w/ CFG | 3.43 |
| MF-XL/2 + iMF boundary objective | 2.99 |

## System-level changes

Paper 另外加入 flexible CFG、$\Omega$-conditioning、in-context conditioning、Transformer block changes 與 longer training。完整 iMF-XL/2 system 報告 FID **1.72**，因此不能把 1.72 全部歸因於 objective replacement。

## 閱讀心得｜adaLN 多條件 conditioning 可能是一個研究破口

### 先分清楚 paper 已證明什麼、我們推測什麼

Paper 的直接內容是：iMF 除了 objective reformulation，也加入 **flexible guidance conditioning、$\Omega$-conditioning、in-context conditioning 與 Transformer block changes**。Paper 並沒有直接證明「adaLN 一定會在多 condition 下失效」，因此下面這一段要視為 **research hypothesis / potential gap**，不是 paper conclusion。

### 我目前最在意的問題

如果多個條件都先被壓成 embedding，最後再一起送進同一組 adaLN / modulation 參數，可能出現一個 architecture-level 問題：

> **不同 condition 的語意、尺度與作用位置被迫共用同一個低維 modulation bottleneck。**

對一般 class-conditional DiT，condition 很單純，這不一定是問題；但 MeanFlow / iMF 可能同時需要處理：

- class / semantic condition；
- current time $t$；
- interval start $r$ 或 gap $t-r$；
- guidance-related condition，例如 $\Omega=\{\omega,t_{min},t_{max}\}$；
- future multimodal / robot setting 中的 image、language、state、action history 等 heterogeneous conditions。

這些 condition 的「物理角色」其實不同：有的是 semantic identity，有的是時間座標，有的是 integration interval，有的是 guidance control。若只把它們相加、concat 後經同一個 MLP 再生成一組 adaLN scale/shift/gate，可能會有：

1. **condition interference**：一個 condition 的 modulation 方向覆蓋另一個；
2. **scale mismatch**：不同 embedding 的數值尺度讓某些 condition 長期主導；
3. **loss-gradient conflict**：不同 condition 對 block modulation 的最佳更新方向互相衝突；
4. **information bottleneck**：多個 heterogeneous controls 被壓成同一組 $\gamma,\beta,\alpha$，容量未必夠；
5. **poor compositional generalization**：訓練看過單一 condition 組合，但新組合時 modulation 不一定可線性組合。

### 其他領域有哪些可轉移的解法？

下面不是 MeanFlow 已驗證方案，而是我認為值得 transfer 的候選：

| Transfer idea | 其他領域的直覺 | 可怎麼套到 MeanFlow / iMF |
|---|---|---|
| **Condition-specific modulators** | FiLM / conditional normalization 常讓不同 condition 產生自己的 feature-wise modulation | 每個 condition 各自產生 $\Delta\gamma_k,\Delta\beta_k,\Delta\alpha_k$，最後再 fusion，而不是先把 condition embedding 混成一個 |
| **Global vs local condition separation** | SPADE 類方法把 spatial condition 與 global semantic condition 分開注入 | $t,r,\Omega$ 走 global adaLN；image / token / spatial condition 走 cross-attention 或 spatial modulation |
| **Cross-attention / in-context tokens** | 多模態 Transformer 常保留 condition tokens，不先壓成單一向量 | heterogeneous conditions 保留成 tokens，讓 network 自己在不同 layer query，而不是所有 condition 都塞進 adaLN |
| **Learnable gating / mixture-of-experts** | 多任務模型用 gating 決定不同 task/condition 要走多少 expert | 對每個 block 學 condition gate，決定 semantic/time/guidance 哪一支 modulation 應該占多少權重 |
| **Condition dropout / compositional training** | CFG 與 compositional diffusion 會隨機 drop condition，強迫模型分辨不同控制來源 | 訓練時隨機 drop 單一 condition 或 condition subset，測試是否能改善組合泛化 |
| **Gradient conflict handling** | multi-task learning 的 PCGrad / GradNorm 直接處理多目標 gradient conflict | 如果 semantic / interval / guidance losses 或 auxiliary heads 互相打架，可量測 cosine similarity，再做 gradient surgery 或 reweighting |
| **Low-rank / orthogonal modulation subspaces** | parameter-efficient tuning 會限制不同 adapter 的更新子空間 | 讓不同 condition 的 adaLN modulation 使用低秩或近似正交 basis，降低互相覆蓋 |
| **Time-dependent fusion** | diffusion 中不同時間段需要不同 conditioning strength | fusion weight 顯式依賴 $t$ 或 $t-r$，例如 early/late interval 使用不同 condition priority |

### 我覺得最值得先做的 ablation

這個破口的優點是：**不需要一開始就發明很大的新模型**，先做 architecture ablation 就能判斷問題是否真的存在。

可以從下列順序做：

1. **Baseline**：所有 condition embedding 合併 → 單一 adaLN MLP；
2. **Separate-then-sum**：每個 condition 各自一個 small MLP → modulation 相加；
3. **Separate + learned gate**：每個 condition 各自 modulation，再由 block-wise gate 決定權重；
4. **Hybrid**：$t,r,\Omega$ 留在 adaLN；semantic / context condition 改走 cross-attention；
5. **Orthogonal / low-rank modulation**：限制不同 condition 的 modulation basis；
6. **Condition-drop curriculum**：隨機 drop condition subset，測 compositional robustness。

至少要量：

- 1-NFE / 2-NFE FID；
- 各 condition 單獨與組合時的 performance；
- block-wise modulation norm；
- 不同 condition modulation vector 的 cosine similarity；
- gradient cosine similarity；
- 對 $t$、$r$、$\Omega$ 的 sensitivity；
- unseen condition combinations 的 generalization。

### 一個更具體、我認為有論文味道的假設

可以把問題寫成：

> **Does shared adaLN become a conditioning bottleneck when MeanFlow must simultaneously encode semantic, temporal, interval, and guidance conditions?**

如果實驗真的看到不同 condition 的 modulation directions 高度衝突，那方法可以很自然地往：

**Factorized Conditional Modulation for MeanFlow**

發展：每個 condition 先獨立產生 modulation，再以 learned gate / low-rank basis / time-dependent fusion 組合。

這個方向和單純「換 backbone」不同，因為它直接對準 MeanFlow 比一般 diffusion 多出來的 **interval / guidance condition structure**。

## Interpretation

iMF 的價值是把「MeanFlow 的 target construction 問題」重新表述成 regression-function design：prediction path 應該由 state 決定，而不是額外依賴 sampled conditional tangent。這也把 future work 從單純 target patch 推向 interval difficulty、endpoint error、gradient geometry、conditioning architecture 與 refinement 等更廣的問題。

## Related data

- [Datasets｜訓練資料集與任務地圖](/research-handbook/datasets/)
