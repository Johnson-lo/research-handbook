---
title: "RMFlow: Refined Mean Flow by a Noise-Injection Step for Multimodal Generation"
description: 從 noise injection、Gaussian likelihood、期望值、NLL、ELBO / Jensen lower bound 到 KL，再討論與 Improved MeanFlow 結合的可行性。
sidebar:
  order: 5
---

## Metadata

- **Authors**：Yuhao Huang, Shih-Hsin Wang, Andrea L. Bertozzi, Bao Wang
- **Venue**：ICLR 2026
- **Year**：2026
- **Primary tasks**：text-to-image、context-to-molecule、time-series / dynamical-system generation
- **Core topics**：MeanFlow refinement、noise injection、likelihood / Wasserstein control、1-NFE multimodal generation
- **Sources**：[ICLR Proceedings](https://proceedings.iclr.cc/paper_files/paper/2026/hash/7b8c48c00dd5e3090ba0976e297fae5c-Abstract-Conference.html) · [arXiv](https://arxiv.org/abs/2602.00849) · [Paper PDF](https://publications.sci.utah.edu/publications/Hua2026a/2602.00849v1.pdf)

:::note[這一頁的閱讀目標]
這頁不是只整理 RMFlow「做了什麼」，而是把我們討論時最容易卡住的地方完整拆開：

1. 為什麼 noise 的係數是 $\sqrt{\sigma_{\min}^2-\sigma^2}$，但最後的 $x_{\mathrm{tgt}}$ 又是 $x_{\mathrm{data}}+\sigma_{\min}\epsilon$？
2. 為什麼 $\hat x_{\mathrm{tgt}}\mid x_0$ 會是 Gaussian，而且 mean / variance 分別是那些東西？
3. Gaussian likelihood 如何一步一步變成 Equation (9) 的 log-likelihood？
4. $\mathbb E_{x_0}$ 與 $\mathbb E_{x_0,x_{\mathrm{tgt}}}$ 到底差在哪裡？
5. NLL 如何透過 Jensen inequality 變成 expected log-likelihood 的 lower bound？
6. 為什麼 expected log-likelihood 又等於 $-H-D_{KL}$？
7. 「把 NLL 加進 MeanFlow loss」究竟多控制了什麼？
8. 如果 Improved MeanFlow 把 JVP tangent 改成 $v_\theta$，RMFlow 的 likelihood 想法還能不能用？
:::

---

# 1. RMFlow 真正想解的問題

MeanFlow 的核心優勢是：不用沿 ODE 路徑做很多次 neural network evaluation，而是直接學 interval-average velocity，讓 1-NFE generation 成為可能。

但是 1-NFE 有一個很直接的風險：**一次 deterministic coarse transport 必須把整個 multimodal distribution 一次送到正確位置。** 一旦 learned transport 有誤差，沒有後續 integration step 幫你逐步修正。

RMFlow 的想法是把問題拆成兩件事：

$$
\text{prior}
\xrightarrow{\text{1-NFE MeanFlow}}
\text{coarse intermediate sample}
\xrightarrow{\text{cheap Gaussian noise}}
\text{final sample}.
$$

所以 RMFlow 不是把 MeanFlow 推翻，而是接受：

> **1-NFE transport 先負責把大方向送對；最後再利用一個有明確 probability model 的 noise injection，讓 endpoint 可以做 likelihood training。**

這個「有明確 probability model」非常重要。因為只要最後一步是 Gaussian，我們就能把 conditional density 寫出來，接著才有 NLL、ELBO / Jensen lower bound、KL 的整套推導。

---

# 2. Equation (5)：原本 MeanFlow 控制的是 Wasserstein gap

Paper 先引用既有 MeanFlow / flow-map 理論：存在某個常數 $M>0$，使

$$
M\,\mathcal L_{\mathrm{CMFM}}(\theta)
\ge
W_2^2(p_{\mathrm{tgt}},p_\theta).
\tag{5}
$$

其中

$$
W_2^2(p_{\mathrm{tgt}},p_\theta)
=
\inf_{\gamma\in\Pi(p_{\mathrm{tgt}},p_\theta)}
\mathbb E_{(x,y)\sim\gamma}\|x-y\|^2.
$$

直覺是：$\mathcal L_{\mathrm{CMFM}}$ 小，代表 exact probability flow 與 learned flow 的 discrepancy 被壓小，因此最終 distribution 的 Wasserstein distance 也受到控制。

:::caution[這不是 RMFlow 在本文重新證明的 theorem]
Equation (5) 是 paper 引用既有工作得到的控制式。RMFlow 後面的新理論重點不是重新證明 (5)，而是再加入一條 **likelihood / KL 方向的控制**。
:::

這也回答「本來就有 loss，為什麼還要多一個？」：原本的 CMFM loss 主要是 **flow / transport geometry** 的控制；RMFlow 想再加入 **endpoint density / likelihood** 的控制。

---

# 3. Equation (6)–(8)：noise injection 到底在做什麼？

## 3.1 Intermediate target：Equation (6)

RMFlow 不先要求 MeanFlow 直接輸出最終 noisy target，而是先運送到

$$
x_1=x_{\mathrm{data}}+\sigma\epsilon_1,
\qquad
\epsilon_1\sim\mathcal N(0,I),
\qquad
\sigma<\sigma_{\min}.
\tag{6}
$$

也就是說：第一階段的 target 還是 data 附近的一個 Gaussian-smoothed sample，但噪聲標準差只有 $\sigma$。

## 3.2 再補一段 noise：Equation (7)

第二階段加入

$$
x_{\mathrm{tgt}}
=
x_1+
\sqrt{\sigma_{\min}^2-\sigma^2}\,\epsilon_2,
\qquad
\epsilon_2\sim\mathcal N(0,I).
\tag{7}
$$

### 你之前卡住的地方：$x_{\mathrm{tgt}}$ 不是「兩個 variance 相減開根號」

$\sqrt{\sigma_{\min}^2-\sigma^2}$ 是 **第二段新增 noise 的標準差**，不是 $x_{\mathrm{tgt}}$ 本身的標準差。

把 (6) 代進 (7)：

$$
\begin{aligned}
x_{\mathrm{tgt}}
&=x_{\mathrm{data}}
+\sigma\epsilon_1
+\sqrt{\sigma_{\min}^2-\sigma^2}\,\epsilon_2.
\end{aligned}
$$

假設 $\epsilon_1,\epsilon_2$ 彼此獨立，兩段 Gaussian noise 的 covariance 會相加：

$$
\operatorname{Cov}
\left[
\sigma\epsilon_1
+\sqrt{\sigma_{\min}^2-\sigma^2}\epsilon_2
\right]
=
\sigma^2I+(\sigma_{\min}^2-\sigma^2)I
=
\sigma_{\min}^2I.
$$

因此可以定義一個新的 standard Gaussian

$$
\epsilon
=
\frac{
\sigma\epsilon_1+
\sqrt{\sigma_{\min}^2-\sigma^2}\epsilon_2
}{\sigma_{\min}},
$$

則

$$
\epsilon\sim\mathcal N(0,I),
$$

所以在 **distribution 的意義** 下，

$$
x_{\mathrm{tgt}}
\overset{d}{=}
x_{\mathrm{data}}+\sigma_{\min}\epsilon.
$$

:::tip[$\overset{d}{=}$ 非常重要]
這不是說原本的 $\epsilon_1$ 或 $\epsilon_2$ 其中一個突然變成 $\epsilon$，而是說「這個線性組合的 distribution 等價於一個新的 standard Gaussian」。
:::

所以整件事可以記成：

$$
\boxed{
\underbrace{\sigma^2}_{\text{第一段 variance}}
+
\underbrace{(\sigma_{\min}^2-\sigma^2)}_{\text{第二段 variance}}
=
\underbrace{\sigma_{\min}^2}_{\text{最後總 variance}}
}
$$

---

## 3.3 Generation equation：Equation (8)

MeanFlow 的 1-NFE intermediate output 是

$$
x_1=x_0+\hat u_{0,1}(x_0;\theta).
$$

因此把 (7) 接在後面：

$$
\boxed{
\hat x_{\mathrm{tgt}}
=
x_0+\hat u_{0,1}(x_0;\theta)
+
\sqrt{\sigma_{\min}^2-\sigma^2}\,\epsilon_2
}
\tag{8}
$$

這就是 RMFlow 的 final generator。

雖然概念上是 two-stage，但昂貴的 neural flow 只 evaluate 一次；第二段只是 sampled Gaussian noise 的加法，因此 NFE 仍算 1。

---

# 4. 為什麼 $\hat x_{\mathrm{tgt}}\mid x_0$ 是那個 Gaussian？

這是後面 NLL 推導最重要的一步。

先定義

$$
\mu_\theta(x_0)
:=
x_0+\hat u_{0,1}(x_0;\theta),
$$

以及

$$
\tau^2
:=
\sigma_{\min}^2-\sigma^2.
$$

Equation (8) 就變成

$$
\hat x_{\mathrm{tgt}}
=
\mu_\theta(x_0)+\tau\epsilon_2,
\qquad
\epsilon_2\sim\mathcal N(0,I).
$$

現在 **condition on $x_0$**。一旦 $x_0$ 被固定，$\mu_\theta(x_0)$ 就是一個 deterministic vector；唯一剩下的 randomness 是 $\epsilon_2$。

Gaussian 的 affine transformation 性質告訴我們：若

$$
\epsilon_2\sim\mathcal N(0,I),
$$

則

$$
\mu+\tau\epsilon_2
\sim
\mathcal N(\mu,\tau^2I).
$$

因此

$$
\boxed{
\hat x_{\mathrm{tgt}}\mid x_0
\sim
\mathcal N\left(
 x_0+\hat u_{0,1}(x_0;\theta),
 (\sigma_{\min}^2-\sigma^2)I
\right)
}
$$

### 為什麼 mean 是 $x_0+\hat u$？

因為

$$
\mathbb E[\epsilon_2]=0,
$$

所以

$$
\mathbb E[\hat x_{\mathrm{tgt}}\mid x_0]
=
\mu_\theta(x_0)+\tau\mathbb E[\epsilon_2]
=
\mu_\theta(x_0).
$$

### 為什麼 variance 是 $(\sigma_{\min}^2-\sigma^2)I$？

因為

$$
\operatorname{Cov}(\tau\epsilon_2)
=
\tau^2\operatorname{Cov}(\epsilon_2)
=
\tau^2I.
$$

注意：這裡算的是 **given $x_0$ 的 conditional variance**，所以只剩最後一段 injection noise 的 variance。這和前一節「從 $x_{\mathrm{data}}$ 看最後 target 的總 variance 是 $\sigma_{\min}^2$」並不矛盾；兩者 conditioning 的對象不同。

---

# 5. Equation (9)：從 Gaussian density 推到 conditional log-likelihood

對一個 $d$ 維 Gaussian

$$
y\sim\mathcal N(\mu,\Sigma),
$$

其 density 是

$$
p(y)
=
\frac{1}{(2\pi)^{d/2}|\Sigma|^{1/2}}
\exp\left(
-\frac12(y-\mu)^T\Sigma^{-1}(y-\mu)
\right).
$$

在 RMFlow 中

$$
\Sigma=\tau^2I,
\qquad
\tau^2=\sigma_{\min}^2-\sigma^2.
$$

因此

$$
\Sigma^{-1}=\frac1{\tau^2}I,
$$

而

$$
|\Sigma|
=|\tau^2I|
=(\tau^2)^d.
$$

所以

$$
|\Sigma|^{1/2}=\tau^d.
$$

代回 Gaussian density：

$$
p_\theta(x_{\mathrm{tgt}}\mid x_0)
=
\frac{1}{(2\pi\tau^2)^{d/2}}
\exp\left(
-\frac{1}{2\tau^2}
\|x_{\mathrm{tgt}}-\mu_\theta(x_0)\|^2
\right).
$$

取 log：

$$
\begin{aligned}
\log p_\theta(x_{\mathrm{tgt}}\mid x_0)
&=
-\frac d2\log(2\pi\tau^2)
-
\frac{1}{2\tau^2}
\|x_{\mathrm{tgt}}-\mu_\theta(x_0)\|^2.
\end{aligned}
$$

把 $\tau^2=\sigma_{\min}^2-\sigma^2$ 與 $\mu_\theta=x_0+\hat u_{0,1}$ 換回去：

$$
\boxed{
\log p_\theta(x_{\mathrm{tgt}}\mid x_0)
=
-
\frac{1}{2(\sigma_{\min}^2-\sigma^2)}
\left\|
 x_{\mathrm{tgt}}-
 \bigl(x_0+\hat u_{0,1}(x_0;\theta)\bigr)
\right\|^2
+C
}
\tag{9}
$$

其中

$$
C=-\frac d2\log\left(2\pi(\sigma_{\min}^2-\sigma^2)\right).
$$

如果 $\sigma$ 與 $\sigma_{\min}$ 是固定 hyperparameters，$C$ 對 $\theta$ 是常數，所以 training 時可以忽略它。

同理，前面的 scaling

$$
A:=\frac{1}{2(\sigma_{\min}^2-\sigma^2)}>0
$$

也是固定常數。因此 maximizing conditional log-likelihood，等價於 minimizing 那個 squared error。

---

# 6. 先把「期望值」講清楚：$\mathbb E_{x_0}$ 與 $\mathbb E_{x_0,x_{\mathrm{tgt}}}$ 差在哪？

## 6.1 Expectation 就是「按照機率做平均」

離散情況：

$$
\mathbb E[X]=\sum_x x\,P(X=x).
$$

連續情況：

$$
\mathbb E[f(X)]
=
\int f(x)p(x)\,dx.
$$

所以 $\mathbb E$ 的下標告訴你：**到底要對哪些 random variable 平均。**

## 6.2 只對 $x_0$ 取 expectation

例如固定某個 $x_{\mathrm{tgt}}$：

$$
\mathbb E_{x_0}
\left[
\log p_\theta(x_{\mathrm{tgt}}\mid x_0)
\right]
=
\int q(x_0)
\log p_\theta(x_{\mathrm{tgt}}\mid x_0)
\,dx_0.
$$

這裡 $x_{\mathrm{tgt}}$ 暫時不平均；你可以把它想成「先拿著一個固定 target，看所有不同 $x_0$ 的平均 log-likelihood」。

## 6.3 對 $x_0$ 與 $x_{\mathrm{tgt}}$ 都取 expectation

再把不同 target 也平均：

$$
\mathbb E_{x_{\mathrm{tgt}}}
\left[
\mathbb E_{x_0}
\left[f(x_0,x_{\mathrm{tgt}})\right]
\right].
$$

如果 unguided setting 中 $x_0\sim q$ 與 $x_{\mathrm{tgt}}\sim p_{\mathrm{tgt}}$ 是用 product measure 取樣，則

$$
\mathbb E_{x_{\mathrm{tgt}}}
\mathbb E_{x_0}[f]
=
\mathbb E_{x_0,x_{\mathrm{tgt}}}[f].
$$

展開就是

$$
\int p_{\mathrm{tgt}}(x_{\mathrm{tgt}})
\int q(x_0)
 f(x_0,x_{\mathrm{tgt}})
\,dx_0\,dx_{\mathrm{tgt}}.
$$

### 你可以這樣記

- $\mathbb E_{x_0}$：只平均「不同起點」。
- $\mathbb E_{x_{\mathrm{tgt}}}$：只平均「不同 target」。
- $\mathbb E_{x_0,x_{\mathrm{tgt}}}$：兩個來源都平均。

:::caution[guided generation 要更小心]
若有 condition $c$，RMFlow 使用 $x_0\sim q_\omega(x_0\mid c)$，而 data 也來自 $p_{\mathrm{data}}(x\mid c)$。此時最嚴格的寫法應該把 $c$ conditioning 保留，而不是直接假設所有變數無條件 independent。對固定 $c$，同一套推導改寫成 $p_\theta(x_{\mathrm{tgt}}\mid c)$ 即可。
:::

---

# 7. Equation (10)：NLL 為什麼最後看起來就是 MSE？

Paper 用

$$
x_{\mathrm{tgt}}
\overset{d}{=}
x_{\mathrm{data}}+\sigma_{\min}\epsilon,
\qquad
\epsilon\sim\mathcal N(0,I).
$$

代到 Equation (9)。如果我們要 **maximize** log-likelihood

$$
\log p_\theta(x_{\mathrm{tgt}}\mid x_0),
$$

等價於 minimize negative log-likelihood：

$$
-\log p_\theta(x_{\mathrm{tgt}}\mid x_0).
$$

因為 $A>0$、$C$ 都是固定常數，optimization 上等價於 minimize

$$
\left\|
 x_{\mathrm{tgt}}-
 \bigl(x_0+\hat u_{0,1}(x_0;\theta)\bigr)
\right\|^2.
$$

再用 $x_{\mathrm{tgt}}=x_{\mathrm{data}}+\sigma_{\min}\epsilon$ 表示 target sample：

$$
\boxed{
\mathcal L_{\mathrm{NLL}}
:=
\mathbb E_{x_0,x_{\mathrm{data}},\epsilon}
\left[
\left\|
(x_{\mathrm{data}}+\sigma_{\min}\epsilon)
-
(x_0+\hat u_{0,1}(x_0;\theta))
\right\|^2
\right]
}
\tag{10}
$$

### 一個精確的命名提醒

Paper 把它叫 $\mathcal L_{\mathrm{NLL}}$，但 Equation (10) 已經把真正 Gaussian NLL 裡的 positive scaling $A$ 與 additive constant 拿掉了。

所以從嚴格數學角度，它是「**與 Gaussian NLL optimization 等價的 scaled / shifted surrogate**」。在 $\tau^2$ 固定時這完全沒問題。

但如果未來你把 $\tau^2$ 也設成 learnable，那麼

$$
\frac{1}{2\tau^2}
\|\cdot\|^2
+\frac d2\log \tau^2
$$

裡面的 variance-dependent term 就不能隨便丟掉。

---

# 8. Equation (11)：NLL 如何連到 expected log-likelihood 與 KL？

Paper 的核心 theorem 是

$$
\boxed{
-A\,\mathcal L_{\mathrm{NLL}}+C
\le
\mathbb E_{x_{\mathrm{tgt}}}
[\log p_\theta(x_{\mathrm{tgt}})]
=
-H(p_{\mathrm{tgt}})
-D_{\mathrm{KL}}(p_{\mathrm{tgt}}\|p_\theta)
}
\tag{11}
$$

其中

$$
A=\frac{1}{2(\sigma_{\min}^2-\sigma^2)}.
$$

你之前說的「ANLL」若是指這裡，其實 paper **沒有另外定義一個叫 ANLL 的 loss**；這裡只是

$$
A\times\mathcal L_{\mathrm{NLL}}
$$

這個 positive scaling。

下面把 Appendix 的 Equations (13)–(17) 完整重推一次。

---

# 9. Equation (13)：為什麼 marginal likelihood 是對 $x_0$ 的 expectation？

因為 $x_0$ 在 generator 中可以視為 latent variable。

對 final sample $x_{\mathrm{tgt}}$，marginal density 是把所有可能的 $x_0$ 積分掉：

$$
p_\theta(x_{\mathrm{tgt}})
=
\int q(x_0)
 p_\theta(x_{\mathrm{tgt}}\mid x_0)
\,dx_0.
$$

這正是 expectation：

$$
p_\theta(x_{\mathrm{tgt}})
=
\mathbb E_{x_0}
\left[
 p_\theta(x_{\mathrm{tgt}}\mid x_0)
\right].
$$

所以

$$
\log p_\theta(x_{\mathrm{tgt}})
=
\log
\mathbb E_{x_0}
\left[p_\theta(x_{\mathrm{tgt}}\mid x_0)\right].
$$

接著使用 Jensen inequality。

因為 $\log$ 是 concave function：

$$
\log\mathbb E[Z]
\ge
\mathbb E[\log Z].
$$

令

$$
Z=p_\theta(x_{\mathrm{tgt}}\mid x_0)>0,
$$

得到

$$
\boxed{
\log p_\theta(x_{\mathrm{tgt}})
\ge
\mathbb E_{x_0}
\left[
\log p_\theta(x_{\mathrm{tgt}}\mid x_0)
\right]
}
\tag{13}
$$

這就是 ELBO-like lower bound 的來源。

### 為什麼叫 lower bound？

右邊永遠不大於左邊，所以右邊是 $\log p_\theta(x_{\mathrm{tgt}})$ 的「下界」。

而且 Jensen gap 可以寫得更具體。若 $q(x_0)$ 是 model prior，則

$$
\log p_\theta(x_{\mathrm{tgt}})
-
\mathbb E_{q(x_0)}
[\log p_\theta(x_{\mathrm{tgt}}\mid x_0)]
=
D_{\mathrm{KL}}
\bigl(
q(x_0)
\|p_\theta(x_0\mid x_{\mathrm{tgt}})
\bigr)
\ge0.
$$

所以 bound 可能是鬆的；它不是說 conditional objective 就「等於」marginal likelihood。

---

# 10. Equation (14)：從固定 target 變成所有 target 的平均

Equation (13) 對每一個 $x_{\mathrm{tgt}}$ 都成立：

$$
\log p_\theta(x_{\mathrm{tgt}})
\ge
\mathbb E_{x_0}
[\log p_\theta(x_{\mathrm{tgt}}\mid x_0)].
$$

因此兩邊再對真實 target distribution $p_{\mathrm{tgt}}$ 取 expectation：

$$
\mathbb E_{x_{\mathrm{tgt}}}
[\log p_\theta(x_{\mathrm{tgt}})]
\ge
\mathbb E_{x_{\mathrm{tgt}}}
\mathbb E_{x_0}
[\log p_\theta(x_{\mathrm{tgt}}\mid x_0)].
$$

在 product-measure 的 unguided 情況下可寫成 joint expectation：

$$
\boxed{
\mathbb E_{x_{\mathrm{tgt}}}
[\log p_\theta(x_{\mathrm{tgt}})]
\ge
\mathbb E_{x_0,x_{\mathrm{tgt}}}
[\log p_\theta(x_{\mathrm{tgt}}\mid x_0)]
}
\tag{14}
$$

這就是你之前問的：

> 為什麼前面是對 $x_0$ 取 expectation，後面突然是對 $x_0,x_{\mathrm{tgt}}$？

答案是：**Equation (13) 先對每個固定 target 做 latent $x_0$ averaging；Equation (14) 再把所有 target 也平均一次。**

---

# 11. Equation (15)：把 Gaussian log-likelihood 代回去

由 Equation (9)：

$$
\log p_\theta(x_{\mathrm{tgt}}\mid x_0)
=
-A
\|x_{\mathrm{tgt}}-\mu_\theta(x_0)\|^2
+C,
$$

其中

$$
A=\frac1{2(\sigma_{\min}^2-\sigma^2)}.
$$

所以

$$
\begin{aligned}
&\mathbb E_{x_0,x_{\mathrm{tgt}}}
[\log p_\theta(x_{\mathrm{tgt}}\mid x_0)]\\
&=
-A
\mathbb E_{x_0,x_{\mathrm{tgt}}}
\left[
\|x_{\mathrm{tgt}}-\mu_\theta(x_0)\|^2
\right]
+C.
\end{aligned}
$$

而 Equation (10) 正是在 Monte Carlo 取樣 $x_{\mathrm{tgt}}=x_{\mathrm{data}}+\sigma_{\min}\epsilon$ 下，估計這個 expected squared error，因此

$$
\boxed{
\mathbb E_{x_0,x_{\mathrm{tgt}}}
[\log p_\theta(x_{\mathrm{tgt}}\mid x_0)]
=
-A\mathcal L_{\mathrm{NLL}}+C
}
\tag{15}
$$

於是和 Equation (14) 合起來：

$$
\boxed{
-A\mathcal L_{\mathrm{NLL}}+C
\le
\mathbb E_{x_{\mathrm{tgt}}}
[\log p_\theta(x_{\mathrm{tgt}})]
}
\tag{16}
$$

這就是「NLL 提高 likelihood lower bound」真正的數學意思。

---

# 12. 你之前對 lower bound 的理解：要修正哪一點？

你可以把它理解成：

$$
\mathcal L_{\mathrm{NLL}}\downarrow
\quad\Longrightarrow\quad
-A\mathcal L_{\mathrm{NLL}}+C\uparrow.
$$

也就是 **minimize NLL 會把 expected log-likelihood 的已知下界往上推。**

但有兩個很重要的限定：

1. 它是 lower bound，不是 equality；Jensen gap 可能存在。
2. 所以「lower bound 變高」不代表每一次 SGD step 都能數學保證真實 expected log-likelihood 一定單調上升。

比較精確的說法是：

> **RMFlow optimize 一個可計算的 surrogate，使 likelihood 的 lower bound 變好；這提供了往較高 marginal likelihood / 較小 forward KL 移動的理論方向。**

---

# 13. Equation (17)：為什麼 expected log-likelihood = $-H-D_{KL}$？

KL 定義：

$$
D_{\mathrm{KL}}(p_{\mathrm{tgt}}\|p_\theta)
=
\mathbb E_{x\sim p_{\mathrm{tgt}}}
\left[
\log\frac{p_{\mathrm{tgt}}(x)}{p_\theta(x)}
\right].
$$

拆開 log：

$$
D_{\mathrm{KL}}
=
\mathbb E_{p_{\mathrm{tgt}}}[\log p_{\mathrm{tgt}}(x)]
-
\mathbb E_{p_{\mathrm{tgt}}}[\log p_\theta(x)].
$$

entropy 定義：

$$
H(p_{\mathrm{tgt}})
=
-\mathbb E_{p_{\mathrm{tgt}}}
[\log p_{\mathrm{tgt}}(x)].
$$

所以

$$
\mathbb E_{p_{\mathrm{tgt}}}
[\log p_{\mathrm{tgt}}(x)]
=
-H(p_{\mathrm{tgt}}).
$$

代回 KL：

$$
D_{\mathrm{KL}}
=
-H(p_{\mathrm{tgt}})
-
\mathbb E_{p_{\mathrm{tgt}}}[\log p_\theta(x)].
$$

移項：

$$
\boxed{
\mathbb E_{x_{\mathrm{tgt}}}
[\log p_\theta(x_{\mathrm{tgt}})]
=
-H(p_{\mathrm{tgt}})
-D_{\mathrm{KL}}(p_{\mathrm{tgt}}\|p_\theta)
}
\tag{17}
$$

由於 $p_{\mathrm{tgt}}$ 是 fixed target distribution，所以 $H(p_{\mathrm{tgt}})$ 對 model parameter $\theta$ 是常數。

因此：

$$
\max_\theta
\mathbb E_{p_{\mathrm{tgt}}}[\log p_\theta]
\quad\Longleftrightarrow\quad
\min_\theta
D_{\mathrm{KL}}(p_{\mathrm{tgt}}\|p_\theta).
$$

這就是 likelihood 與 forward KL 的關係。

---

# 14. 「原本就有 loss，加入 KL / NLL 到底有什麼好處？」

這裡最容易混淆的是：RMFlow **不是直接把 $D_{KL}$ 數值算出來加進 loss**。

實際加入的是

$$
\mathcal L_{\mathrm{NLL}},
$$

而理論告訴我們它會抬高 expected log-likelihood 的 lower bound；expected log-likelihood 又和 forward KL 相差一個固定 entropy。

所以三個 quantity 的角色不同：

| Quantity | 主要控制對象 | 直覺 |
|---|---|---|
| $\mathcal L_{\mathrm{CMFM}}$ | velocity / probability path consistency | transport 路徑有沒有學對 |
| $W_2$ | global transport geometry | 把 probability mass 搬多遠 |
| likelihood / forward KL | endpoint density matching | target sample 在 model 底下有沒有足夠高 density |

對 multimodal distribution，單靠 transport regression 小，不代表每個 target mode 都一定得到好的 endpoint density。加入 likelihood-related term，相當於再從另一個 distributional criterion 施加約束。

這也是 RMFlow 的核心研究價值：

$$
\boxed{
\text{path-level control}
+
\text{endpoint likelihood control}
}
$$

而不是單純「多加一個 MSE」。

---

# 15. Equation (12)：Joint objective 每一項到底在做什麼？

RMFlow 最終 objective：

$$
\boxed{
\mathcal L_{\mathrm{RMFlow}}(\theta,\omega)
=
\underbrace{\mathcal L_{\mathrm{CMFM}}}_{\mathrm{I}}
+
\lambda_1
\underbrace{\mathcal L_{\mathrm{NLL}}}_{\mathrm{II}}
+
\lambda_2
\underbrace{
\mathbb E_{(x_{\mathrm{data}},c)}
\|\phi_\omega(c)\|^2
}_{\mathrm{III}}
}
\tag{12}
$$

## Term I — CMFM

維持 MeanFlow 本來的平均速度 / probability-path learning，並承接 Equation (5) 的 Wasserstein-style control。

## Term II — NLL surrogate

直接看 1-NFE endpoint

$$
\mu_\theta(x_0)=x_0+\hat u_{0,1}(x_0;\theta)
$$

是否能對 noisy target 給出高 likelihood。

它的 gradient 很直觀。忽略常數後：

$$
\mathcal L_{\mathrm{NLL}}
=
\mathbb E\|y-g_\theta(x_0)\|^2,
$$

其中

$$
g_\theta(x_0)=x_0+\hat u_{0,1}(x_0;\theta).
$$

因此

$$
\nabla_\theta\mathcal L_{\mathrm{NLL}}
=
2\,\mathbb E
\left[
J_{g_\theta}(x_0)^T
(g_\theta(x_0)-y)
\right].
$$

也就是 endpoint error 會直接回傳到 final one-step map。

## Term III — guidance prior regularization

Guided generation 使用

$$
x_0=\phi_\omega(c)+\sigma_c\epsilon.
$$

所以 $\|\phi_\omega(c)\|^2$ 可以視為對 conditional prior mean 的 regularization。

Paper 也提醒：這一項如果過大，反而可能明顯傷害 performance，所以 $\lambda_2$ 不是越大越好。

---

# 16. RMFlow 與 Improved MeanFlow 能不能結合？

這是我們討論到最值得進一步做研究的地方。

先講結論：

> **likelihood / NLL 這一段理論，本身並不要求你一定使用 original CMFM target。只要 inference generator 最後仍能寫成一個 differentiable mean map，再加已知 Gaussian noise，Equation (9)–(17) 的 likelihood proof 就仍成立。**

但「Wasserstein theorem + likelihood theorem 同時成立」要分開檢查，不能直接把 original MeanFlow 的所有保證原封不動搬到 iMF。

## 16.1 一個更一般的 proposition

假設任意一個 one-step model 有 deterministic mean map

$$
g_\theta(x_0),
$$

並定義 stochastic final generator

$$
y=g_\theta(x_0)+\tau\epsilon,
\qquad
\epsilon\sim\mathcal N(0,I),
\qquad
\tau>0.
$$

則

$$
y\mid x_0
\sim
\mathcal N(g_\theta(x_0),\tau^2I).
$$

完全不管 $g_\theta$ 是用 original MeanFlow、Improved MeanFlow、distillation、或其他 objective 學出來的，都有

$$
\log p_\theta(y\mid x_0)
=
-\frac{1}{2\tau^2}
\|y-g_\theta(x_0)\|^2+C.
$$

因此同樣由 Jensen：

$$
-\frac{1}{2\tau^2}
\mathbb E\|y-g_\theta(x_0)\|^2+C
\le
\mathbb E_y[\log p_\theta(y)].
$$

所以 **RMFlow 的 likelihood regularizer 對 one-step mean map 是 model-agnostic 的。**

## 16.2 套到 Improved MeanFlow

在目前 iMF 的 formulation 中，改動重點是 training objective 裡 JVP 的 tangent：

$$
V_\theta
=
u_\theta
+(t-r)\operatorname{JVP}_{sg}(u_\theta;v_\theta),
$$

而 supervision 仍可寫成對 sample-level conditional velocity 的 regression。

最終 inference 仍然使用 learned mean-flow map $u_\theta$ 做 transport。因此可以考慮

$$
\boxed{
\mathcal L_{\mathrm{iMF+RM}}
=
\mathcal L_{\mathrm{iMF}}
+
\lambda_{\mathrm{NLL}}
\mathbb E
\left\|
 x_{\mathrm{tgt}}
-
\bigl(x_0+u_\theta(x_0,0,1)\bigr)
\right\|^2
}
$$

再在 sampling 時使用

$$
\hat x_{\mathrm{tgt}}
=
x_0+u_\theta(x_0,0,1)
+\tau\epsilon.
$$

只要這個 final conditional model 是 Gaussian，上面的 likelihood / KL lower-bound proof 不需要改。

### 所以「iMF target 改成 $v_\theta$ 後就不能用 RMFlow」嗎？

不是。

RMFlow NLL 關心的是 **final generated mean $g_\theta(x_0)$**，而不是你在 JVP 內部到底用 $e-x$ 還是 $v_\theta$ 當 tangent。

真正要問的是：

$$
\boxed{
\text{$v_\theta$ 的改寫是否仍能學出可用的 final }u_\theta\text{ map？}
}
$$

只要答案是 yes，NLL term 就可以直接作用在 endpoint。

---

# 17. 但「可加」不等於「所有 theorem 自動保留」

這裡要把兩件事拆開。

## 17.1 Likelihood proof：可以保留

Equation (9)–(17) 只依賴：

1. 有一個 one-step deterministic mean map $g_\theta(x_0)$；
2. final noise 是 Gaussian；
3. variance $\tau^2>0$；
4. 相關 expectation 有限，可以合法交換 / 計算。

它**不依賴 CMFM target 的具體 JVP 寫法**。

所以 iMF + Gaussian refinement 的 likelihood lower bound 是可以嚴格重建的。

## 17.2 Wasserstein guarantee：不能直接宣稱自動保留

Equation (5) 的 bound 是針對特定 MeanFlow / flow-map training setup 的既有理論結果。

如果 iMF 把 objective reformulate 成新的 $V_\theta$ regression，你不能只因為「看起來很像」就直接寫

$$
M\mathcal L_{\mathrm{iMF}}\ge W_2^2.
$$

要嚴格成立，需要另外證明例如：

- iMF objective 與原本 CMFM 在 population optimum 上等價；或
- iMF loss 能 upper-bound 原本 CMFM discrepancy；或
- 直接建立新的 stability / transport error theorem。

所以最嚴格的研究表述應該是：

> **iMF + RMFlow likelihood term 在機率模型上是相容的；但若要聲稱同時繼承 original RMFlow 的 Wasserstein-control theorem，還需要額外 proof。**

這也是一個很合理的研究破口。

---

# 18. Boundary $v_\theta$ 與 auxiliary $v$-head 時，NLL gradient 會去哪裡？

這點實作上也很重要。

## Boundary form

若

$$
v_\theta(z_t,t)
\equiv
u_\theta(z_t,t,t),
$$

則 $v_\theta$ 與 $u_\theta$ 完全共享 parameters。NLL 對 final $u_\theta$ 回傳的 gradient，自然也會更新同一組 backbone parameters。

## Auxiliary $v$-head

如果 $v_\theta$ 是額外 head：

$$
\mathcal L_v
=
\mathbb E\|v_\theta-(e-x)\|^2,
$$

那麼 RMFlow NLL 直接依賴的是 final $u_\theta$ map，不一定直接依賴 auxiliary head。

因此 auxiliary head 是否收到 NLL gradient，取決於它與 $u_\theta$ 是否共享 backbone / computational graph。

一個合理 hybrid objective 是

$$
\mathcal L
=
\mathcal L_{\mathrm{iMF}}
+\beta\mathcal L_v
+\lambda_{\mathrm{NLL}}\mathcal L_{\mathrm{NLL}}
+\lambda_2\mathcal L_{\mathrm{guide}}.
$$

這裡每一項扮演的角色要分清楚，否則很容易變成「加很多 loss 但不知道哪一項在改善什麼」。

---

# 19. Paper 裡兩個值得特別注意的 notation consistency 問題

## 19.1 Appendix 的 Gaussian denominator 與正文不一致

正文 Equation (8) / (9) 對應的 conditional variance 明確是

$$
\tau^2
=\sigma_{\min}^2-\sigma^2.
$$

因此正確 scaling 應該是

$$
A
=
\frac{1}{2(\sigma_{\min}^2-\sigma^2)}.
$$

但 Appendix 的 Theorem 4.1 proof 排版中出現了

$$
\frac{1}{2(\sigma_{\min}-\sigma)^2},
$$

這和 Equation (8) 所定義的 noise variance 不一致。

本頁所有推導以正文 Equation (8) / (9) 的 stochastic model 為準，也就是使用

$$
\sigma_{\min}^2-\sigma^2.
$$

## 19.2 Theorem 4.1 的 constants 文字也有小不一致

Equation (11) 寫的是

$$
-A\mathcal L_{\mathrm{NLL}}+C,
$$

但 theorem 敘述附近有一處寫成「$A,B>0$ are constants」。依上下文應該是在談 $A$ 與 additive constant $C$。

這些不影響主推導，但閱讀時不要被符號誤導。

---

# 20. 為什麼新增 NLL 會增加 training memory？

Paper 後半段特別提到：$\mathcal L_{\mathrm{RMFlow}}$ 比單純 $\mathcal L_{\mathrm{CMFM}}$ 多出 gradient pathways。

原因是原本 MeanFlow objective 已經包含 JVP / stop-gradient 設計；現在又要讓 final one-step endpoint

$$
x_0+\hat u_{0,1}(x_0;\theta)
$$

參與一條額外 likelihood gradient path。

對大模型，這會增加 activation / backward graph 的 memory footprint。

所以 paper 的大規模策略是：

1. 先用 $\mathcal L_{\mathrm{CMFM}}$ train MeanFlow；
2. 再用 PEFT 對 $\mathcal L_{\mathrm{RMFlow}}$ fine-tune；
3. molecule task 還另外加入 physical-feedback policy gradient。

這也代表 paper 的最終實驗結果不應簡化成「只多一個 NLL 就得到全部 improvement」。training recipe、PEFT、guidance encoder、task-specific feedback 都可能有貢獻。

---

# 21. 實驗結果要怎麼讀才不會過度解讀？

## Synthetic distributions

1D Gaussian mixture：

| Method | NFE | TV ↓ | KL ↓ |
|---|---:|---:|---:|
| MeanFlow | 1 | 1.4422 | 0.8074 |
| MeanFlow | 8 | 0.7977 | 0.4074 |
| MeanFlow | 32 | 0.6737 | 0.1017 |
| RMFlow | 1 | 0.7567 | 0.2332 |

1-NFE RMFlow 明顯改善 1-NFE MeanFlow，甚至在這個 experiment 的 KL / TV 上優於 8-NFE MeanFlow，但仍不是所有指標都超過 32-NFE MeanFlow。

2D checkerboard：

| Method | NFE | TV ↓ | KL ↓ |
|---|---:|---:|---:|
| MeanFlow | 1 | 0.238 | 0.311 |
| MeanFlow | 8 | 0.167 | 0.139 |
| MeanFlow | 32 | 0.155 | 0.118 |
| RMFlow | 1 | 0.173 | 0.163 |

這裡 1-NFE RMFlow 把 gap 大幅縮小，但 8 / 32-NFE MeanFlow 仍稍好。

## QM9 molecule generation

Paper 的 1-NFE results：

| Method | Atomic stability ↑ | Molecule stability ↑ | NFE |
|---|---:|---:|---:|
| MeanFlow w/ contexts | 98.4% | 84.3% | 1 |
| RMFlow w/ contexts | 98.9% | 93.2% | 1 |
| RMFlow w/ contexts + RLPF | 98.9% | 93.5% | 1 |

這支持「coarse one-step transport + refinement」對結構 validity 有實質幫助；但 +RLPF 的版本又加入 task-specific physical reward，因此不能全部歸因於 NLL。

## Text-to-image

COCO FID-30K：

| Method | NFE | FID ↓ |
|---|---:|---:|
| MeanFlow | 1 | 27.31 |
| RMFlow | 1 | 18.91 |

這是明顯 improvement，但跨 paper 比較時仍要注意 architecture、training data、teacher / discriminator、resolution 與 compute budget 都不同。

---

# 22. RMFlow 的 limitation 反而給出下一步研究方向

Paper 自己指出兩個方向：

### Multi-step RMFlow

現在 proof 是針對「一次 MeanFlow transport + 一次 Gaussian refinement」。若每個 transport step 後都加 noise，就要重新設計 joint probability model 與 likelihood objective。

### Noise variance 不應永遠固定

目前

$$
\tau^2=\sigma_{\min}^2-\sigma^2
$$

是 fixed design。未來可能考慮 learnable / adaptive schedule。

但如果 $\tau$ 變成 learnable，就必須把完整 Gaussian NLL 的

$$
\frac{1}{2\tau^2}\|y-\mu_\theta\|^2
+\frac d2\log\tau^2
$$

都保留，否則 model 可以透過任意改 variance 破壞 likelihood objective 的意義。

---

# 23. 最後用一條完整邏輯鏈把 RMFlow 記起來

$$
\boxed{
\begin{aligned}
&\text{1-NFE MeanFlow gives }\mu_\theta(x_0)
=x_0+\hat u_{0,1}(x_0;\theta)\\[2mm]
&\Downarrow\\[-1mm]
&\hat x_{\mathrm{tgt}}
=\mu_\theta(x_0)+\tau\epsilon,
\quad
\tau^2=\sigma_{\min}^2-\sigma^2\\[2mm]
&\Downarrow\\[-1mm]
&\hat x_{\mathrm{tgt}}\mid x_0
\sim\mathcal N(\mu_\theta(x_0),\tau^2I)\\[2mm]
&\Downarrow\\[-1mm]
&\log p_\theta(x_{\mathrm{tgt}}\mid x_0)
=-\frac{1}{2\tau^2}\|x_{\mathrm{tgt}}-\mu_\theta(x_0)\|^2+C\\[2mm]
&\Downarrow\\[-1mm]
&\mathcal L_{\mathrm{NLL}}
\propto
\mathbb E\|x_{\mathrm{tgt}}-\mu_\theta(x_0)\|^2\\[2mm]
&\Downarrow\;\text{(Jensen)}\\[-1mm]
&-A\mathcal L_{\mathrm{NLL}}+C
\le
\mathbb E_{p_{\mathrm{tgt}}}[\log p_\theta(x)]\\[2mm]
&\Downarrow\\[-1mm]
&\mathbb E_{p_{\mathrm{tgt}}}[\log p_\theta(x)]
=-H(p_{\mathrm{tgt}})-D_{KL}(p_{\mathrm{tgt}}\|p_\theta).
\end{aligned}
}
$$

所以 RMFlow 的核心不是「加 noise 讓 sample 看起來更隨機」，而是：

> **加一個可寫出 conditional density 的 stochastic refinement，讓 one-step MeanFlow endpoint 可以被 likelihood / KL-aware objective 直接約束。**

而對我們接下來最值得驗證的 hybrid idea，數學上可以先拆成兩個獨立問題：

$$
\boxed{
\text{iMF objective 是否維持好的 transport guarantee？}
}
$$

以及

$$
\boxed{
\text{在同一個 final }u_\theta\text{ map 上加入 RMFlow NLL，是否改善 endpoint distribution？}
}
$$

第二個問題的 likelihood lower-bound proof 已經可以沿用；第一個問題才是需要另外嚴格證明的新理論部分。

---

## Related pages

- [Mean Flows for One-step Generative Modeling](/research-handbook/papers/meanflow/)
- [Improved Mean Flows](/research-handbook/papers/improved-meanflow/)
- [MeanFlow → Improved MeanFlow → RMFlow research track](/research-handbook/meanflow/story/)
- [Improved MeanFlow 深入：$v_\theta$、JVP、$V_\theta$](/research-handbook/meanflow/improved-meanflow/)
