---
title: "One-step Latent-free Image Generation with Pixel Mean Flows"
description: pMF 的 x-prediction / velocity-loss 分離、與 RMFlow 的嚴格數學對接、NLL/KL 證明，以及在單張 RTX 3080 10GB 上的實驗設計。
sidebar:
  order: 6
---

## Metadata

- **Paper**：One-step Latent-free Image Generation with Pixel Mean Flows
- **Authors**：Yiyang Lu, Susie Lu, Qiao Sun, Hanhong Zhao, Zhicheng Jiang, Xianbang Wang, Tianhong Li, Zhengyang Geng, Kaiming He
- **Year**：2026
- **Primary task**：ImageNet-1K 256×256 / 512×512 one-step latent-free image generation
- **Core topics**：Pixel MeanFlow、x-prediction、velocity-space loss、JVP、one-step generation、latent-free generation
- **Paper**：[arXiv:2601.22158](https://arxiv.org/abs/2601.22158)
- **Official code**：[Lyy-iiis/pMF](https://github.com/Lyy-iiis/pMF)
- **Related notes**：[Improved MeanFlow](/research-handbook/papers/improved-meanflow/) · [RMFlow](/research-handbook/papers/rmflow/)

:::note[這一頁目前的研究目的]
這頁除了整理 pMF paper 本身，也記錄目前要實作的研究主線：

> **Can RMFlow-style likelihood refinement improve Pixel MeanFlow without destroying the manifold-friendly x-prediction that pMF was designed to preserve?**

重點不是把兩個 loss 直接拼起來，而是先嚴格確認：

1. pMF 的 endpoint x-prediction 是否能精確對應 RMFlow 的 conditional mean？
2. latent-free 是否仍允許 RMFlow 的 Gaussian noise injection / likelihood construction？
3. $\sigma=0$ 與 $\sigma>0$ 各自代表什麼？
4. 哪些 RMFlow theorem 可以直接搬到 pMF，哪些目前還不能宣稱？
5. 在 RTX 3080 10GB 的限制下，如何做最乾淨且可解釋的實驗？
:::

---

# 1. pMF 解決的是什麼問題？

一般 diffusion / flow image generator 常同時有兩個特徵：

1. inference 需要很多 sampling steps；
2. 為了降低 computation，通常在 VAE latent space 裡生成。

pMF 想同時移除這兩件事：**直接在 pixel space 做 one-step generation**。

它的核心不是「把 MeanFlow 原封不動搬到 pixel space」，而是把兩個空間刻意拆開：

$$
\boxed{
\text{network output space}=x\text{-prediction},
\qquad
\text{loss space}=\text{velocity / MeanFlow}
}
$$

直覺是：velocity target 可能非常 noisy、維度高且離 natural image manifold 很遠；clean image 本身則被假設落在較低維的 image manifold。於是 network 先預測 image-like quantity，再透過解析 transformation 轉成 average velocity 來算 MeanFlow loss。

---

# 2. pMF 的時間 convention

令 clean image 為 $x$，prior noise 為

$$
\epsilon_0\sim\mathcal N(0,I).
$$

pMF 使用線性 path

$$
\boxed{
z_t=(1-t)x+t\epsilon_0
}
\tag{1}
$$

因此：

$$
t=0\Rightarrow z_0=x,
\qquad
t=1\Rightarrow z_1=\epsilon_0.
$$

也就是：

$$
\boxed{
\text{pMF time: data at }0\;\longrightarrow\;\text{noise at }1
}
$$

它的 instantaneous velocity 是

$$
\frac{dz_t}{dt}=\epsilon_0-x.
\tag{2}
$$

所以 sample-level conditional velocity target 可以寫成

$$
v_t=\epsilon_0-x.
$$

官方 code 也用等價形式

$$
v_t=\frac{z_t-x}{t},
$$

因為由 (1)

$$
z_t-x=t(\epsilon_0-x).
$$

---

# 3. pMF 的 x-prediction 與 average velocity

MeanFlow 用 $u_{r,t}(z_t)$ 表示區間 $[r,t]$ 上的 average velocity。因為生成方向是從較大的 $t$ 往較小的 $r$，step 寫成

$$
z_r=z_t-(t-r)u_{r,t}(z_t).
\tag{3}
$$

pMF 定義 image-like prediction

$$
\boxed{
x_\theta(z_t,r,t)=z_t-tu_\theta(z_t,r,t)
}
\tag{4}
$$

反過來就是

$$
\boxed{
u_\theta(z_t,r,t)=\frac{z_t-x_\theta(z_t,r,t)}{t}
}
\tag{5}
$$

這正是官方 `pmfDiT.py` 最後做的 transformation：network head 先產生 image-like output，再用 $(x-x_{pred})/t$ 轉成 $u$。

## 3.1 One-step endpoint

令

$$
t=1,\qquad r=0.
$$

由 (4)：

$$
\boxed{
x_\theta(\epsilon_0,0,1)
=\epsilon_0-u_\theta(\epsilon_0,0,1)
}
\tag{6}
$$

這就是 pMF 的 1-NFE deterministic output。

因此 pMF 雖然用 velocity-space objective training，真正 one-step generator 的 endpoint quantity 本身是 image-space 的 $x_\theta$。

---

# 4. pMF 的 training loss：為什麼仍然是 velocity loss？

pMF / iMF 類 objective 不直接對 $x_\theta$ 做主要 supervision，而是建立 compound velocity predictor。

概念上可寫成

$$
V_\theta
=
u_\theta+(t-r)\frac{d u_\theta}{dt},
\tag{7}
$$

實作上 derivative 由 JVP 計算。pMF 官方 code 同時有 auxiliary $v$ head，並使用 predicted instantaneous velocity 作為 JVP tangent。

主要 regression target 仍然是 guided instantaneous velocity $v_g$：

$$
\mathcal L_u
\propto
\|V_\theta-v_g\|^2,
\tag{8}
$$

以及 auxiliary velocity loss

$$
\mathcal L_v
\propto
\|v_\theta-v_g\|^2.
\tag{9}
$$

官方實作另外可以從

$$
\hat x=z_t-tu_\theta
\tag{10}
$$

取得 `pred_x`，再加 LPIPS / ConvNeXt perceptual loss。

這個 code structure 對我們非常重要，因為 RMFlow likelihood branch 需要的 conditional mean 正好可以放在這個 image-like endpoint 上，而不需要把 backbone 改回直接 velocity prediction。

---

# 5. RMFlow 的時間 convention 與 pMF 相反

RMFlow / original MeanFlow 常用另一個方向：

$$
s=0:\text{prior},
\qquad
s=1:\text{target}.
$$

而 pMF 是

$$
t=1:\text{prior},
\qquad
t=0:\text{target}.
$$

因此令

$$
\boxed{s=1-t.}
\tag{11}
$$

如果定義同一條 trajectory

$$
y_s:=z_{1-s},
$$

則 chain rule 給出

$$
\frac{dy_s}{ds}
=\frac{dz_t}{dt}\frac{dt}{ds}
=-\frac{dz_t}{dt}.
\tag{12}
$$

因此時間反轉之後 velocity 符號反轉：

$$
\boxed{
v^{\rm RMF}=-v^{\rm pMF}.}
\tag{13}
$$

對 average velocity 也是同樣的 sign relation：

$$
\boxed{
u_{0,1}^{\rm RMF}=-u_{0,1}^{\rm pMF}.}
\tag{14}
$$

---

# 6. Proposition 1：pMF endpoint 精確等於 RMFlow conditional mean

RMFlow 的 deterministic MeanFlow intermediate output 定義為

$$
m_\theta(x_0)
:=x_0+u_{0,1}^{\rm RMF}(x_0;\theta).
\tag{15}
$$

把 prior sample 對齊：

$$
x_0\equiv\epsilon_0.
$$

再由 (14)：

$$
\begin{aligned}
m_\theta(\epsilon_0)
&=\epsilon_0+u_{0,1}^{\rm RMF}(\epsilon_0)\\
&=\epsilon_0-u_{0,1}^{\rm pMF}(\epsilon_0).
\end{aligned}
$$

而 pMF endpoint (6) 正是

$$
x_\theta(\epsilon_0,0,1)
=\epsilon_0-u_{0,1}^{\rm pMF}(\epsilon_0).
$$

因此：

$$
\boxed{
m_\theta(\epsilon_0)
=x_\theta(\epsilon_0,0,1).}
\tag{16}
$$

:::tip[這一個 correspondence 是精確的]
這不是 heuristic，也不是「兩個量看起來很像」。只要兩邊描述同一條 trajectory，並以 $s=1-t$ 對齊時間方向，RMFlow 的 one-step conditional mean 和 pMF 的 endpoint x-prediction 是同一個量。
:::

這是目前 pMF × RMFlow 最重要的數學橋樑。

---

# 7. Latent-free 並不妨礙 RMFlow 的 Gaussian likelihood

RMFlow 的 likelihood construction 並不要求「一定要有 VAE latent」。

它需要的是：

1. 一個 deterministic conditional mean $m_\theta(x_0)$；
2. 一個 non-degenerate Gaussian conditional variance；
3. 能寫出 $p_\theta(x_{tgt}\mid x_0)$。

pMF 的 (16) 已經提供 deterministic conditional mean。空間從 latent 變成 pixel，只是維度與 geometry 改變，**Gaussian density 在數學上仍然完全可以定義**。

真正的研究問題不是「能不能 inject noise」，而是：

> 在高維 pixel space 使用 isotropic Gaussian likelihood，是否仍是一個對 image quality 有效的 inductive bias？

---

# 8. Strict RMFlow × pMF：$\sigma>0$ 時不能只改 NLL

這是實作時最重要的嚴格界線。

RMFlow 先定義一個 smoothed intermediate target

$$
\boxed{
x_\sigma=x+\sigma\epsilon_1,
\qquad
\epsilon_1\sim\mathcal N(0,I),
\qquad
0\le\sigma<\sigma_{\min}.
}
\tag{17}
$$

接著再加入第二段 noise：

$$
\boxed{
x_{\rm tgt}
=x_\sigma+
\sqrt{\sigma_{\min}^2-\sigma^2}\,\epsilon_2,
\qquad
\epsilon_2\sim\mathcal N(0,I).
}
\tag{18}
$$

且 $\epsilon_1\perp\epsilon_2$。

因此如果要做 **strict RMFlow + pMF**，pMF 的 flow endpoint 也必須從 clean $x$ 改為 $x_\sigma$：

$$
\boxed{
z_t=(1-t)x_\sigma+t\epsilon_0.}
\tag{19}
$$

此時 instantaneous target 也變成

$$
v_t=\epsilon_0-x_\sigma.
\tag{20}
$$

如果 $\sigma>0$ 卻仍使用原本 clean path

$$
z_t=(1-t)x+t\epsilon_0,
$$

只是額外在 loss target 加 noise，那就不能嚴格稱為 RMFlow transport；比較準確的名稱是 **RMFlow-inspired likelihood regularization**。

---

# 9. Proposition 2：兩段 noise 的 marginal target variance

把 (17) 代入 (18)：

$$
x_{\rm tgt}
=x
+\sigma\epsilon_1
+\sqrt{\sigma_{\min}^2-\sigma^2}\epsilon_2.
$$

因為兩個 Gaussian 獨立：

$$
\operatorname{Cov}
\left[
\sigma\epsilon_1
+\sqrt{\sigma_{\min}^2-\sigma^2}\epsilon_2
\right]
$$

$$
=\sigma^2I+(\sigma_{\min}^2-\sigma^2)I
=\sigma_{\min}^2I.
$$

所以存在新的 $\epsilon\sim\mathcal N(0,I)$，使

$$
\boxed{
x_{\rm tgt}\overset d=x+\sigma_{\min}\epsilon.}
\tag{21}
$$

也就是最終 target distribution 永遠是 data distribution 與 $\mathcal N(0,\sigma_{\min}^2I)$ 的 convolution；$\sigma$ 決定的是「多少 smoothing 在 MeanFlow transport 前完成、多少留給 final injection」。

---

# 10. Proposition 3：pMF endpoint 對應的 conditional Gaussian

定義

$$
\tau^2:=\sigma_{\min}^2-\sigma^2>0.
\tag{22}
$$

由 Proposition 1，令

$$
m_\theta(\epsilon_0)
:=x_\theta(\epsilon_0,0,1).
$$

RMFlow-style generator 是

$$
\boxed{
\hat x_{\rm tgt}
=m_\theta(\epsilon_0)+\tau\epsilon_2.
}
\tag{23}
$$

condition on $\epsilon_0$ 後，$m_\theta(\epsilon_0)$ 是 deterministic vector，而唯一 randomness 是

$$
\epsilon_2\sim\mathcal N(0,I).
$$

由 Gaussian affine transformation：

$$
\boxed{
p_\theta(x_{\rm tgt}\mid\epsilon_0)
=\mathcal N
\left(
m_\theta(\epsilon_0),
\tau^2I
\right).
}
\tag{24}
$$

這個推導與資料在 latent space 或 pixel space 無關。

---

# 11. Proposition 4：Gaussian NLL 的完整形式

若 image dimension 為 $d$，由 (24)：

$$
\log p_\theta(x_{\rm tgt}\mid\epsilon_0)
=
-\frac d2\log(2\pi\tau^2)
-\frac{1}{2\tau^2}
\|x_{\rm tgt}-m_\theta(\epsilon_0)\|^2.
\tag{25}
$$

因此 full negative log-likelihood 是

$$
\boxed{
\mathcal L_{\rm full\text{-}NLL}
=
\mathbb E\left[
\frac{1}{2\tau^2}
\|x_{\rm tgt}-m_\theta(\epsilon_0)\|^2
+\frac d2\log(2\pi\tau^2)
\right].
}
\tag{26}
$$

若 $\tau$ 固定，第二項與 $\theta$ 無關，因此 optimizing mean 等價於 minimizing

$$
\boxed{
\mathcal L_{\rm NLL\text{-}reg}
=
\mathbb E
\|x_{\rm tgt}-m_\theta(\epsilon_0)\|^2.
}
\tag{27}
$$

但要注意：full NLL 與 unnormalized squared loss 的 gradient **差一個固定倍率**：

$$
\nabla_\theta\mathcal L_{\rm full\text{-}NLL}
=
\frac{1}{2\tau^2}
\nabla_\theta\mathcal L_{\rm NLL\text{-}reg}.
\tag{28}
$$

所以可以說它們有相同 minimizer / gradient direction（固定 $\tau$ 時），但不能字面寫成兩個 gradient 完全相等。

---

# 12. Proposition 5：final noise 對 deterministic mean 的期望梯度做了什麼？

由

$$
x_{\rm tgt}=x_\sigma+\tau\epsilon_2,
$$

且 $\epsilon_2$ 與其他變數獨立，固定 $(x_\sigma,\epsilon_0)$ 後：

$$
\begin{aligned}
&\mathbb E_{\epsilon_2}
\|x_\sigma+\tau\epsilon_2-m_\theta\|^2\\
&=
\|x_\sigma-m_\theta\|^2
+2\tau(x_\sigma-m_\theta)^T\mathbb E[\epsilon_2]
+\tau^2\mathbb E\|\epsilon_2\|^2\\
&=
\boxed{
\|x_\sigma-m_\theta\|^2+d\tau^2
}.
\end{aligned}
\tag{29}
$$

因此對 deterministic mean parameters：

$$
\boxed{
\nabla_\theta
\mathbb E_{\epsilon_2}
\|x_{\rm tgt}-m_\theta\|^2
=
\nabla_\theta
\|x_\sigma-m_\theta\|^2.
}
\tag{30}
$$

如果用 full NLL，則再乘固定係數 $1/(2\tau^2)$。

## 這個結果非常重要

final injection 的 $\epsilon_2$：

- 對 **expected conditional-mean optimum** 不會產生新的位置；
- sampled training 時會增加 stochasticity；
- 但它讓 $p_\theta(x_{\rm tgt}|\epsilon_0)$ 變成 non-degenerate density，因此 likelihood / KL derivation 才有意義；
- inference 時也真正改變生成 distribution。

所以對 pMF manifold geometry 真正有直接影響的是第一段

$$
\boxed{x\rightarrow x_\sigma=x+\sigma\epsilon_1,}
$$

而不是第二段 $\epsilon_2$ 本身。

---

# 13. $\sigma=0$：最乾淨的 bridge case

令

$$
\sigma=0.
$$

則

$$
x_\sigma=x,
\qquad
\tau=\sigma_{\min}.
$$

因此 pMF 原本 clean-image path 完全不需要改：

$$
z_t=(1-t)x+t\epsilon_0.
$$

final target 是

$$
x_{\rm tgt}=x+\sigma_{\min}\epsilon_2.
\tag{31}
$$

由 (29)：

$$
\mathbb E_{\epsilon_2}
\|x+\sigma_{\min}\epsilon_2-m_\theta\|^2
=
\|x-m_\theta\|^2+d\sigma_{\min}^2.
\tag{32}
$$

所以如果使用 squared regression form：

$$
\nabla_\theta
\mathbb E\|x_{\rm tgt}-m_\theta\|^2
=
\nabla_\theta\mathbb E\|x-m_\theta\|^2.
\tag{33}
$$

而 full NLL 則是

$$
\boxed{
\nabla_\theta\mathcal L_{\rm full\text{-}NLL}
=
\frac{1}{2\sigma_{\min}^2}
\nabla_\theta
\mathbb E\|x-m_\theta\|^2.
}
\tag{34}
$$

因此：

> **固定 isotropic variance、$\sigma=0$ 時，RMFlow 的 mean-learning component 在 expectation 下就是 endpoint x-MSE 的 constant-rescaled version。**

這表示實驗上一定要把「endpoint reconstruction supervision 的效果」和「真正 RMFlow stochastic generation 的效果」拆開。

---

# 14. Proposition 6：Jensen lower bound 與 KL control 可以搬到 pMF endpoint

模型 marginal density 為

$$
p_\theta(y)
=
\mathbb E_{\epsilon_0\sim p_0}
[p_\theta(y\mid\epsilon_0)].
\tag{35}
$$

對任意固定 $y$，由 $\log$ concave 與 Jensen inequality：

$$
\begin{aligned}
\log p_\theta(y)
&=\log
\mathbb E_{\epsilon_0}
[p_\theta(y\mid\epsilon_0)]\\
&\ge
\mathbb E_{\epsilon_0}
[\log p_\theta(y\mid\epsilon_0)].
\end{aligned}
\tag{36}
$$

再對 target marginal $p_{\rm tgt}$ 取 expectation：

$$
\boxed{
\mathbb E_{y\sim p_{\rm tgt}}
[\log p_\theta(y)]
\ge
\mathbb E_{\epsilon_0,y}
[\log p_\theta(y\mid\epsilon_0)].
}
\tag{37}
$$

代入 Gaussian log-likelihood (25)：

$$
\mathbb E_{\epsilon_0,y}
[\log p_\theta(y|\epsilon_0)]
=
-\frac{1}{2\tau^2}
\mathbb E\|y-m_\theta(\epsilon_0)\|^2
+C_\tau,
\tag{38}
$$

其中

$$
C_\tau=-\frac d2\log(2\pi\tau^2)
$$

與 $\theta$ 無關。

因此存在常數

$$
A=\frac{1}{2\tau^2}>0
$$

使

$$
\boxed{
-A\mathcal L_{\rm NLL\text{-}reg}+C_\tau
\le
\mathbb E_{p_{\rm tgt}}
[\log p_\theta(y)].
}
\tag{39}
$$

另一方面，KL definition：

$$
D_{KL}(p_{\rm tgt}\|p_\theta)
=
\mathbb E_{p_{\rm tgt}}
\left[
\log\frac{p_{\rm tgt}(y)}{p_\theta(y)}
\right].
$$

展開：

$$
D_{KL}
=
\mathbb E_{p_{\rm tgt}}[\log p_{\rm tgt}(y)]
-
\mathbb E_{p_{\rm tgt}}[\log p_\theta(y)].
$$

而 entropy

$$
H(p_{\rm tgt})
=-\mathbb E_{p_{\rm tgt}}
[\log p_{\rm tgt}(y)].
$$

所以

$$
\boxed{
\mathbb E_{p_{\rm tgt}}[\log p_\theta(y)]
=-H(p_{\rm tgt})
-D_{KL}(p_{\rm tgt}\|p_\theta).
}
\tag{40}
$$

把 (39) 與 (40) 合起來：

$$
\boxed{
-A\mathcal L_{\rm NLL\text{-}reg}+C_\tau
\le
-H(p_{\rm tgt})
-D_{KL}(p_{\rm tgt}\|p_\theta).
}
\tag{41}
$$

因為 $p_{\rm tgt}$ 固定時 $H(p_{\rm tgt})$ 與 $\theta$ 無關，所以降低 conditional Gaussian regression loss 會提高一個 marginal expected log-likelihood lower bound；這就是 RMFlow likelihood / KL argument 可以搬到 pMF endpoint 的核心。

:::note[這部分為什麼可以直接 transfer？]
上述 proof 只使用：

- prior $\epsilon_0$；
- deterministic conditional mean $m_\theta(\epsilon_0)$；
- Gaussian conditional density；
- Jensen inequality；
- KL / entropy identity。

沒有任何一步要求 VAE latent，也沒有要求 mean 一定由 original MeanFlow parameterization 表示。因此由 Proposition 1 將 $m_\theta$ 換成 pMF endpoint $x_\theta(\epsilon_0,0,1)$ 是合法的。
:::

---

# 15. 一個需要註記的 RMFlow 公式一致性問題

RMFlow 的 noise construction 使用

$$
\tau^2=\sigma_{\min}^2-\sigma^2
$$

以及 injection standard deviation

$$
\tau=\sqrt{\sigma_{\min}^2-\sigma^2}.
$$

這會導出 Gaussian quadratic coefficient

$$
\frac{1}{2(\sigma_{\min}^2-\sigma^2)}.
$$

目前 arXiv HTML 的 Appendix A 某個 rendered equation 顯示成 $2(\sigma_{\min}-\sigma)^2$ 的 denominator；這和前面的 noise construction 並不代數一致。研究與實作時應以 generative construction / conditional variance 本身為基準，並在正式寫 paper 前再次對照作者 source / official code，避免把可能的排版或 notation typo 傳進我們的方法。

---

# 16. 哪些結論目前已經嚴格成立？

| Claim | Status | 理由 |
|---|---|---|
| pMF endpoint $x_\theta$ 可作 RMFlow conditional mean | **成立** | 時間反轉 $s=1-t$ 後精確相等 |
| latent-free pixel space 可以定義 RMFlow Gaussian conditional density | **成立** | Gaussian likelihood 不依賴 VAE |
| RMFlow NLL / Jensen / KL proof 可作用於 pMF endpoint | **成立** | proof 只依賴 conditional density 與 marginalization |
| $\sigma=0$ 時 expected NLL mean-gradient 等價於 endpoint MSE（差固定 scale） | **成立** | Gaussian cross term expectation 為 0 |
| $\sigma>0$ 時 strict RMFlow 必須把 flow endpoint 改成 $x_\sigma$ | **成立於 RMFlow construction** | 第一階段 transport target 就是 smoothed intermediate distribution |
| 原 RMFlow 的 Wasserstein bound 可原封不動套到完整 pMF loss | **尚未證明** | pMF 改了 parameterization、JVP tangent 與 auxiliary objective，需要逐項檢查 theorem assumptions |
| isotropic Gaussian likelihood 在 pixel space 會改善 FID | **未知，需實驗** | 這是研究 hypothesis，不是理論保證 |
| RMFlow smoothing 一定會破壞 pMF image-manifold advantage | **未知，需實驗** | 這正是 C→D ablation 要回答的問題 |

因此目前最安全的論文表述是：

> **The likelihood/KL construction transfers exactly to the pMF endpoint parameterization; whether pMF’s full velocity objective inherits RMFlow’s Wasserstein-side guarantee is treated separately and is not assumed without proof.**

---

# 17. 為什麼這個組合比 RMFlow + iMF 更適合先實作？

RMFlow + iMF 的主要困難是 likelihood mean 要如何和 iMF 的 $v_\theta$ / JVP tangent / compound $V_\theta$ 重新對齊。那條線不是錯，但需要先處理 target semantics 與 theorem assumptions。

pMF 則已經顯式提供 image endpoint：

$$
x_\theta=z_t-tu_\theta.
$$

因此 RMFlow conditional mean 不需要重新發明：

$$
\boxed{m_\theta=x_\theta(\epsilon_0,0,1).}
$$

所以目前研究優先級定為：

1. **pMF × RMFlow**：主線；
2. **RMFlow × iMF**：暫停，但保留作 Phase 2 / comparison branch。

---

# 18. 論文級 ablation：A / B / C / D 必須分開

| Exp | Flow endpoint | Extra endpoint objective | Final noise | Interpretation |
|---|---|---|---|---|
| **A** | $x$ | 無 | 無 | Official pMF baseline |
| **B** | $x$ | clean endpoint MSE | 無 | Reconstruction supervision ablation |
| **C** | $x$ ($\sigma=0$) | RMFlow Gaussian NLL | $\sigma_{\min}\epsilon_2$ | Strict bridge case；flow path 不變 |
| **D** | $x_\sigma=x+\sigma\epsilon_1$ | RMFlow Gaussian NLL | $\sqrt{\sigma_{\min}^2-\sigma^2}\epsilon_2$ | Full RMFlow smoothing |

## A → B

回答：

> 單純加 endpoint image supervision，是否已經能改善 one-step pMF？

## B → C

因為固定 variance 下 expected NLL mean-gradient 與 endpoint MSE 只差 scale，B / C 的主要差異應放在 **probabilistic interpretation 與 stochastic inference**，而不是假裝 NLL magically 產生另一個 deterministic optimum。

## C → D

這是目前最重要的研究問題：

$$
\boxed{
\text{clean manifold target}
\quad\text{vs}\quad
\text{RMFlow likelihood smoothing}
}
$$

也就是：

> **Likelihood smoothing 是否會改善 global distribution coverage，但同時犧牲 pMF 想保留的 image-manifold / high-frequency structure？**

---

# 19. Strict objective 的建議寫法

令原 pMF objective 為

$$
\mathcal L_{pMF}
=\mathcal L_u+\mathcal L_v+\mathcal L_{perc}.
$$

RMFlow branch 定義

$$
\mathcal L_{RMF}
=
\mathbb E
\left[
\frac{1}{2\tau^2}
\|x_{\rm tgt}-x_\theta(\epsilon_0,0,1)\|^2
\right],
\tag{42}
$$

忽略與 $\theta$ 無關的 log-normalizer。

joint objective：

$$
\boxed{
\mathcal L
=\mathcal L_{pMF}
+\lambda_{RMF}\mathcal L_{RMF}.
}
\tag{43}
$$

但要特別注意：$\mathcal L_{RMF}$ 要 evaluate **endpoint $(t,r)=(1,0)$** 的 prediction。對 arbitrary sampled $(t,r)$ 直接把 $x_\theta(z_t,r,t)$ 拉向 final data，理論上比較像 auxiliary reconstruction loss，不再等價於 RMFlow terminal likelihood proof。

---

# 20. 10GB VRAM 下的 unbiased endpoint estimator

最忠實的做法是：

1. 一次 random $(t,r)$ forward 算 $\mathcal L_{pMF}$；
2. 再一次 endpoint $(1,0)$ forward 算 $\mathcal L_{RMF}$。

但對單張 RTX 3080 10GB，這可能太貴。

一個較乾淨的 single-forward stochastic estimator 是引入

$$
M\sim\operatorname{Bernoulli}(p_{end}).
$$

- 若 $M=0$：照原 pMF distribution sample $(t,r)$，使用
  $$
  \frac{1}{1-p_{end}}\mathcal L_{pMF}.
  $$
- 若 $M=1$：強制 $(t,r)=(1,0)$，使用
  $$
  \frac{\lambda_{RMF}}{p_{end}}\mathcal L_{RMF}.
  $$

則

$$
\begin{aligned}
\mathbb E_M[\hat{\mathcal L}]
&=(1-p_{end})\frac{1}{1-p_{end}}\mathcal L_{pMF}
+p_{end}\frac{\lambda_{RMF}}{p_{end}}\mathcal L_{RMF}\\
&=\boxed{
\mathcal L_{pMF}+\lambda_{RMF}\mathcal L_{RMF}
}.
\end{aligned}
\tag{44}
$$

所以這是 **unbiased objective estimator**，不是隨便把一部分 batch 改成 endpoint。

缺點是 gradient variance 會增加；實驗時必須記錄 $p_{end}$，並和「額外 endpoint forward」版本在小模型上做 sanity check。

也可以對 batch 內不同 example 使用 Bernoulli mask，只要 weighting 保持一致。

---

# 21. Official code 對應位置

官方 `Lyy-iiis/pMF` main branch 是 JAX training implementation。

核心位置：

- `pmf.py`
  - `z_t = (1-t) * x + t * e`
  - `v_t = (z_t - x) / t`
  - JVP construction
  - `V = u + (t-r) * stop_gradient(du_dt)`
  - `pred_x = z_t - t * u`
  - `loss_u`, `loss_v`, LPIPS / ConvNeXt losses
- `models/pmfDiT.py`
  - network heads 先產生 image-like output
  - 再用 `(x - pred) / t` 轉成 $u,v$

因此第一版研究**不需要改 backbone**。

要新增的是：

1. RMFlow config；
2. $\sigma,\sigma_{min},\lambda_{RMF},p_{end}$；
3. strict $x_\sigma$ construction；
4. endpoint NLL branch；
5. inference final noise；
6. A/B/C/D experiment configs；
7. logging：`loss_rmf`, endpoint MSE, $\tau$, $\sigma$, FID 等。

---

# 22. Jin 容器目前的實際限制

目前實驗機：

- **GPU**：NVIDIA GeForce RTX 3080
- **VRAM**：10 GB
- **System RAM**：125 GiB
- **Driver**：580.173.02
- **reported CUDA capability**：CUDA 13.0
- **Disk**：充足
- base Python：3.13.5
- base env 尚無 JAX / Flax / Optax

官方 pMF main training code 是 **JAX + TPU-oriented setup**，官方 `install.sh` 直接安裝 TPU JAX；不能原封不動在 Jin 上執行。

官方另有 PyTorch branch，但 README 說明主要提供 GPU inference / checkpoint usage，training 仍指回 JAX implementation。因此第一階段應優先保留 JAX training semantics，不要同時做「RMFlow modification + full JAX→PyTorch port」，否則變因太多。

目前 JAX 官方文件對 Linux NVIDIA CUDA 13 推薦 `jax[cuda13]`，而 RTX 3080 (Ampere, SM 8.6) 符合 CUDA 13 wheel 的最低 SM 7.5 要求。實驗環境仍應建立 isolated env，並先做 device / JVP / bf16 sanity check，再跑 pMF。

---

# 23. 第一階段實驗順序

## Phase 0 — Environment / baseline

先完全不改模型：

1. clone official pMF；
2. 建立 GPU JAX isolated environment；
3. 確認 `jax.devices()` 只有 RTX 3080 GPU；
4. 跑最小 forward / backward / JVP test；
5. 跑官方 model initialization；
6. 用極小 batch 做 1–10 training steps，確認 loss finite、VRAM 可接受；
7. 若 256×256 pMF-B/16 直接 OOM，再做 reduced smoke-test config，不要先改方法。

## Phase 1 — A vs B

先測 clean endpoint supervision 是否有 signal。

## Phase 2 — C: $\sigma=0$

加入 strict Gaussian conditional / final injection，但保持 clean pMF path。

## Phase 3 — D: $\sigma>0$

真正將 flow endpoint 改為 $x_\sigma$，測 manifold-vs-smoothing tradeoff。

## Phase 4 — frequency-aware covariance

如果 isotropic pixel Gaussian 破壞 texture / high-frequency detail，再考慮

$$
\Sigma\neq\tau^2I
$$

例如 frequency-aware covariance / weighted likelihood。這可以自然接到 Frequency-Aware Flow Matching 的研究線，但不應在第一版同時加入。

---

# 24. 最低限度要記錄的 metrics

- training `loss_u`
- training `loss_v`
- perceptual losses
- endpoint clean MSE
- RMFlow NLL / normalized NLL
- gradient norm
- NaN / overflow incidence
- peak VRAM
- one-step FID
- IS（若 evaluator 可行）
- LPIPS / perceptual quality
- sample grid
- 若做 $\sigma>0$：不同 $\sigma/\sigma_{min}$ 下的結果

對 C / D 最重要的是畫：

$$
\sigma/\sigma_{min}
\quad\text{vs}\quad
\text{FID / LPIPS / endpoint MSE}.
$$

---

# 25. Open-source code 與論文發表

pMF 官方 repository 使用 **MIT License**。MIT 明確允許 use、copy、modify、merge、publish、distribute 等行為，因此可以：

$$
\text{fork official pMF}
\rightarrow
\text{implement research modification}
\rightarrow
\text{run experiments}
\rightarrow
\text{publish paper}.
$$

需要分清楚兩層：

### Software license

公開修改版 code 時保留原始 copyright notice 與 MIT license。

### Academic attribution

論文必須引用 pMF、RMFlow，以及實際使用到的其他方法 / checkpoints / datasets；Implementation Details 應明確說明是基於官方 pMF codebase 修改。

此外 ImageNet、pretrained checkpoints 與第三方 dependencies 有各自的 terms，不能把 pMF 的 MIT license 當成它們全部的 license。

---

# 26. 目前研究主線的最精確表述

目前不把題目簡化成「RMFlow + pMF」。更準確的 scientific question 是：

> **Pixel MeanFlow obtains its advantage by predicting an image-manifold quantity while optimizing in velocity space. RMFlow obtains likelihood control by smoothing the endpoint distribution and adding a Gaussian refinement. Can the latter improve distributional fidelity without erasing the former's manifold advantage?**

對應的核心 tradeoff：

$$
\boxed{
\text{manifold-friendly x-prediction}
\quad\longleftrightarrow\quad
\text{likelihood-friendly Gaussian smoothing}
}
$$

這個 tradeoff 由 $\sigma$ 直接控制，因此 $\sigma$ 不只是一個 tuning hyperparameter，而是目前方法分析最重要的 axis。

---

# 27. Research status

### 已確認

- pMF endpoint mean 對應 RMFlow mean：數學成立。
- Gaussian likelihood / Jensen / KL proof 不依賴 VAE：可以 transfer。
- $\sigma=0$ 是最乾淨 bridge case。
- $\sigma>0$ strict case 必須修改 pMF flow endpoint。
- fixed isotropic variance 下，NLL 對 mean 的 expected gradient 和 endpoint regression 只差常數倍率。
- pMF official code 是可修改的 MIT codebase。

### 待驗證

- pMF full objective 是否保有 RMFlow 所引用的 Wasserstein-side bound。
- 10GB VRAM 能否直接跑 pMF-B/16 training；很可能需要 tiny smoke-test configuration / microbatching。
- clean endpoint supervision 是否真的改善 pMF。
- RMFlow final stochasticity 是否改善 FID。
- $\sigma>0$ 是否造成 perceptual / high-frequency degradation。
- frequency-aware covariance 是否比 isotropic covariance 更適合 pixel space。

---

# 28. 下一步

目前先暫停 RMFlow × iMF implementation，保留為 Phase 2 comparison branch。

第一個實作 checkpoint 不是 RMFlow modification，而是：

$$
\boxed{
\text{official pMF code}
\rightarrow
\text{RTX 3080 GPU-JAX environment}
\rightarrow
\text{baseline smoke test}
}
$$

只有 baseline forward / backward / JVP 都穩定後，才開始 A/B/C/D modification。這樣任何 performance 或 numerical 問題才有辦法定位來源。
