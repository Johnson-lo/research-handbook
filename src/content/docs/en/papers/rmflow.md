---
title: "RMFlow: Refined Mean Flow by a Noise-Injection Step for Multimodal Generation"
description: Full derivation of RMFlow noise injection, Gaussian likelihood, expectations, Jensen lower bound, KL relation, and rigorous compatibility with Improved MeanFlow.
sidebar:
  order: 5
---

import RMFlowNoiseDemo from '../../../components/RMFlowNoiseDemo.astro';

## Metadata

- **Authors**: Yuhao Huang, Shih-Hsin Wang, Andrea L. Bertozzi, Bao Wang
- **Venue**: ICLR 2026
- **Year**: 2026
- **Primary tasks**: text-to-image, context-to-molecule, time-series / dynamical-system generation
- **Core topics**: MeanFlow refinement, Gaussian noise injection, likelihood / KL control, 1-NFE multimodal generation
- **Sources**: [ICLR Proceedings](https://proceedings.iclr.cc/paper_files/paper/2026/hash/7b8c48c00dd5e3090ba0976e297fae5c-Abstract-Conference.html) · [arXiv](https://arxiv.org/abs/2602.00849)

# 1. Start from the physical picture

For a prior sample $x_0$, a one-step MeanFlow model first produces a deterministic coarse endpoint

$$
\mu_\theta(x_0)=x_0+u_\theta(x_0,0,1).
$$

A deterministic point has no non-degenerate local density by itself. RMFlow therefore adds a cheap Gaussian refinement

$$
Y=\mu_\theta(x_0)+\tau\varepsilon,
\qquad \varepsilon\sim\mathcal N(0,I).
$$

Conditioned on $x_0$, the mean is now fixed and only the Gaussian noise remains random, hence

$$
Y\mid x_0\sim\mathcal N(\mu_\theta(x_0),\tau^2I).
$$

This is the key move: the coarse one-step output is turned into a conditional distribution whose density can be written analytically.

<RMFlowNoiseDemo lang="en" />

---

# 2. Why the two variances add up to $\sigma_{\min}^2$

RMFlow uses an intermediate noisy target

$$
X_1=X_{\mathrm{data}}+\sigma\varepsilon_1,
\qquad \varepsilon_1\sim\mathcal N(0,I),
$$

then injects

$$
X_{\mathrm{tgt}}
=X_1+\sqrt{\sigma_{\min}^2-\sigma^2}\,\varepsilon_2,
\qquad \varepsilon_2\sim\mathcal N(0,I),
$$

with independent $\varepsilon_1,\varepsilon_2$.

Substitution gives

$$
X_{\mathrm{tgt}}
=X_{\mathrm{data}}+\sigma\varepsilon_1
+\sqrt{\sigma_{\min}^2-\sigma^2}\,\varepsilon_2.
$$

For independent Gaussian vectors,

$$
\operatorname{Cov}(a\varepsilon_1+b\varepsilon_2)
=(a^2+b^2)I.
$$

Therefore

$$
\operatorname{Cov}(X_{\mathrm{tgt}}\mid X_{\mathrm{data}})
=\sigma^2I+(\sigma_{\min}^2-\sigma^2)I
=\sigma_{\min}^2I.
$$

Equivalently, for a fresh $\varepsilon\sim\mathcal N(0,I)$,

$$
X_{\mathrm{tgt}}\overset d=
X_{\mathrm{data}}+\sigma_{\min}\varepsilon.
$$

The quantity $\sqrt{\sigma_{\min}^2-\sigma^2}$ is the **standard deviation of the second noise increment**, not the final standard deviation.

---

# 3. From the conditional Gaussian to Equation (9)

Define

$$
\tau^2:=\sigma_{\min}^2-\sigma^2,
\qquad
\mu_\theta(x_0):=x_0+\hat u_{0,1}(x_0;\theta).
$$

Then

$$
\hat X_{\mathrm{tgt}}
=\mu_\theta(x_0)+\tau\varepsilon_2,
$$

so

$$
\hat X_{\mathrm{tgt}}\mid x_0
\sim\mathcal N(\mu_\theta(x_0),\tau^2I).
$$

For a $d$-dimensional isotropic Gaussian,

$$
p_\theta(y\mid x_0)
=\frac{1}{(2\pi\tau^2)^{d/2}}
\exp\left(
-\frac{\|y-\mu_\theta(x_0)\|^2}{2\tau^2}
\right).
$$

Taking logs,

$$
\log p_\theta(y\mid x_0)
=-\frac{1}{2\tau^2}\|y-\mu_\theta(x_0)\|^2
-\frac d2\log(2\pi\tau^2).
$$

Hence

$$
\boxed{
\log p_\theta(x_{\mathrm{tgt}}\mid x_0)
=-\frac{1}{2(\sigma_{\min}^2-\sigma^2)}
\left\|x_{\mathrm{tgt}}-
\bigl(x_0+\hat u_{0,1}(x_0;\theta)\bigr)
\right\|^2+C
}
$$

with

$$
C=-\frac d2\log\left(2\pi(\sigma_{\min}^2-\sigma^2)\right).
$$

When the variance is fixed, maximizing this conditional log-likelihood is equivalent to minimizing the squared distance.

---

# 4. Why the NLL becomes an MSE-looking objective

The exact Gaussian negative log-likelihood is

$$
-\log p_\theta(y\mid x_0)
=\frac{1}{2\tau^2}\|y-\mu_\theta(x_0)\|^2
+\frac d2\log(2\pi\tau^2).
$$

If $\tau$ is fixed, both the scale $1/(2\tau^2)$ and the additive constant are independent of $\theta$. Therefore the same optimizer is obtained from

$$
\boxed{
\mathcal L_{\mathrm{NLL}}
=
\mathbb E
\left\|
(X_{\mathrm{data}}+\sigma_{\min}\varepsilon)
-
(X_0+\hat u_{0,1}(X_0;\theta))
\right\|^2
}
$$

up to a fixed positive scale and constant shift.

If $\tau$ were learnable, this simplification would no longer be valid; the $\log \tau^2$ term must then be retained.

---

# 5. $\mathbb E_{x_0}$ versus $\mathbb E_{x_0,x_{\mathrm{tgt}}}$

For a fixed target $y$, the model marginalizes the latent prior variable $X_0$:

$$
p_\theta(y)
=\int q(x_0)p_\theta(y\mid x_0)dx_0
=\mathbb E_{X_0\sim q}[p_\theta(y\mid X_0)].
$$

So the expectation

$$
\mathbb E_{X_0}[\log p_\theta(y\mid X_0)]
$$

averages over different prior samples while keeping $y$ fixed.

If we then average over targets $Y\sim p_{\mathrm{tgt}}$ as well,

$$
\mathbb E_Y\mathbb E_{X_0}
[\log p_\theta(Y\mid X_0)]
$$

is, under the product sampling measure, the joint expectation

$$
\mathbb E_{X_0,Y}[\log p_\theta(Y\mid X_0)].
$$

Thus the second expectation does not replace the first; it adds averaging over the target population.

---

# 6. Jensen lower bound, step by step

Because $\log$ is concave,

$$
\log\mathbb E[Z]\ge\mathbb E[\log Z]
$$

for every positive random variable $Z$.

Set

$$
Z=p_\theta(y\mid X_0)>0.
$$

Then

$$
\begin{aligned}
\log p_\theta(y)
&=\log\int q(x_0)p_\theta(y\mid x_0)dx_0\\
&=\log\mathbb E_{X_0}[p_\theta(y\mid X_0)]\\
&\ge\mathbb E_{X_0}[\log p_\theta(y\mid X_0)].
\end{aligned}
$$

Averaging both sides over $Y\sim p_{\mathrm{tgt}}$ gives

$$
\mathbb E_Y[\log p_\theta(Y)]
\ge
\mathbb E_{Y,X_0}[\log p_\theta(Y\mid X_0)].
$$

Using the Gaussian log-likelihood,

$$
\mathbb E_{Y,X_0}[\log p_\theta(Y\mid X_0)]
=-A\mathcal L_{\mathrm{NLL}}+C,
$$

where

$$
A=\frac{1}{2(\sigma_{\min}^2-\sigma^2)}>0.
$$

Therefore

$$
\boxed{
-A\mathcal L_{\mathrm{NLL}}+C
\le
\mathbb E_Y[\log p_\theta(Y)]
}
$$

and minimizing $\mathcal L_{\mathrm{NLL}}$ raises this computable lower bound.

This does **not** imply that the true marginal likelihood increases monotonically at every SGD step, because the Jensen gap can change.

---

# 7. Why expected log-likelihood equals $-H-D_{KL}$

By definition,

$$
D_{\mathrm{KL}}(p_{\mathrm{tgt}}\|p_\theta)
=
\mathbb E_{p_{\mathrm{tgt}}}
\left[
\log\frac{p_{\mathrm{tgt}}(Y)}{p_\theta(Y)}
\right].
$$

Expanding the logarithm,

$$
D_{\mathrm{KL}}
=
\mathbb E_{p_{\mathrm{tgt}}}[\log p_{\mathrm{tgt}}(Y)]
-
\mathbb E_{p_{\mathrm{tgt}}}[\log p_\theta(Y)].
$$

Since

$$
H(p_{\mathrm{tgt}})
=-\mathbb E_{p_{\mathrm{tgt}}}[\log p_{\mathrm{tgt}}(Y)],
$$

we obtain

$$
\boxed{
\mathbb E_{p_{\mathrm{tgt}}}[\log p_\theta(Y)]
=-H(p_{\mathrm{tgt}})
-D_{\mathrm{KL}}(p_{\mathrm{tgt}}\|p_\theta)
}
$$

and because $p_{\mathrm{tgt}}$ is fixed, maximizing expected log-likelihood is equivalent to minimizing forward KL.

---

# 8. Why add this when MeanFlow already has a loss?

The objectives constrain different objects:

| Quantity | What it constrains | Interpretation |
|---|---|---|
| $\mathcal L_{\mathrm{CMFM}}$ | probability-flow / average-velocity consistency | whether the transport field is learned correctly |
| $W_2$ | global transport geometry | how far probability mass must move |
| likelihood / forward KL | endpoint density matching | whether target samples have high density under the final model |

So RMFlow is not merely “adding another MSE.” It introduces a second distributional criterion that acts directly on the one-step endpoint.

---

# 9. Rigorous feasibility proof for Improved MeanFlow + RMFlow

The crucial question is whether iMF replacing the JVP tangent with $v_\theta$ breaks the RMFlow likelihood construction. It does not, under explicit assumptions.

## Proposition 1 — Gaussian refinement is objective-agnostic

Assume:

1. $X_0\sim q$;
2. a one-step model defines a measurable deterministic map
   $$g_\theta:\mathbb R^d\to\mathbb R^d;$$
3. $\tau>0$ is fixed;
4. $\varepsilon\sim\mathcal N(0,I)$ is independent of $X_0$;
5. the final generator is
   $$Y=g_\theta(X_0)+\tau\varepsilon.$$

Then for every $x_0$,

$$
Y\mid X_0=x_0
\sim\mathcal N(g_\theta(x_0),\tau^2I),
$$

and

$$
\log p_\theta(y\mid x_0)
=-\frac{1}{2\tau^2}\|y-g_\theta(x_0)\|^2+C.
$$

If the required expectations are finite,

$$
-\frac{1}{2\tau^2}
\mathbb E\|Y-g_\theta(X_0)\|^2+C
\le
\mathbb E_Y[\log p_\theta(Y)].
$$

### Proof

The first claim is the affine-transformation property of a Gaussian. The second follows by writing the Gaussian density and taking logs. The third follows from

$$
p_\theta(y)=\int q(x_0)p_\theta(y\mid x_0)dx_0
$$

and Jensen's inequality. No step uses the training objective that produced $g_\theta$. QED.

The result is deliberately model-agnostic: $g_\theta$ may come from original MeanFlow, Improved MeanFlow, distillation, or another one-step training method.

---

# 10. Apply Proposition 1 to iMF

Improved MeanFlow changes the training-time compound predictor. A representative form is

$$
V_\theta(z_t)
=u_\theta(z_t,r,t)
+(t-r)\operatorname{JVP}_{sg}(u_\theta;v_\theta).
$$

The key change is that the JVP tangent is state-conditioned through $v_\theta$, rather than using the sample-specific conditional velocity directly.

However, the one-step inference endpoint is still defined by the learned mean-flow map. Using this site's convention,

$$
g_\theta(x_0)
:=x_0+u_\theta(x_0,0,1).
$$

Substituting this $g_\theta$ into Proposition 1 gives

$$
Y\mid X_0=x_0
\sim
\mathcal N\left(
 x_0+u_\theta(x_0,0,1),
 \tau^2I
\right),
$$

so a valid endpoint objective is

$$
\boxed{
\mathcal L_{\mathrm{endpoint}}
=
\mathbb E
\left\|
Y-igl(X_0+u_\theta(X_0,0,1)\bigr)
\right\|^2
}
$$

with the same likelihood lower bound

$$
-\frac{1}{2\tau^2}\mathcal L_{\mathrm{endpoint}}+C
\le
\mathbb E_Y[\log p_\theta(Y)].
$$

Therefore the RMFlow likelihood argument does **not** depend on whether the JVP tangent is $e-x$ or $v_\theta$. It depends only on the final one-step map and the Gaussian refinement.

---

# 11. Proposition 2 — the endpoint NLL produces a valid iMF gradient

Let

$$
g_\theta(x_0)=x_0+u_\theta(x_0,0,1)
$$

and assume $g_\theta$ is differentiable in $\theta$ and that differentiation may be interchanged with expectation. For

$$
\mathcal L_R(\theta)
=\mathbb E\|Y-g_\theta(X_0)\|^2,
$$

we have

$$
\boxed{
\nabla_\theta\mathcal L_R
=
2\mathbb E\left[
J_{g_\theta}(X_0)^T
(g_\theta(X_0)-Y)
\right]
}
$$

so the endpoint likelihood term directly updates the same map used for one-step inference.

For the boundary variant

$$
v_\theta(z_t,t)=u_\theta(z_t,t,t),
$$

$v_\theta$ and $u_\theta$ share parameters, so the endpoint loss updates the same backbone.

For an auxiliary $v$-head, the likelihood proof still holds, but whether that head receives the endpoint gradient depends on parameter sharing and graph connectivity.

---

# 12. A mathematically valid hybrid objective

A reasonable combined objective is

$$
\boxed{
\mathcal L_{\mathrm{hybrid}}
=
\mathcal L_{\mathrm{iMF}}
+\beta\mathcal L_v
+\lambda_{\mathrm{NLL}}\mathcal L_{\mathrm{endpoint}}
+\lambda_2\mathcal L_{\mathrm{guide}}
}
$$

where the terms have distinct roles:

- $\mathcal L_{\mathrm{iMF}}$: state-conditioned MeanFlow regression;
- $\mathcal L_v$: optional auxiliary instantaneous-velocity supervision;
- $\mathcal L_{\mathrm{endpoint}}$: one-step endpoint likelihood control;
- $\mathcal L_{\mathrm{guide}}$: conditional-prior regularization if used.

This proves that the hybrid objective is probabilistically and differentially well-defined. It does **not** prove that it must outperform iMF or RMFlow individually.

---

# 13. What is still unproven

Three claims must not be inferred from the previous propositions.

First, one cannot automatically claim

$$
M\mathcal L_{\mathrm{iMF}}\ge W_2^2
$$

because the original Wasserstein-style result is tied to a specific CMFM / flow-map error. A new theorem would need to compare the iMF objective to that discrepancy or derive a fresh stability bound.

Second, a higher likelihood lower bound does not imply that FID must improve:

$$
\mathcal L_{\mathrm{NLL}}\downarrow
\not\Rightarrow
\mathrm{FID}\downarrow
$$

as a theorem.

Third, the lower bound itself need not track the true marginal likelihood monotonically because the Jensen gap may change.

These are not weaknesses of the compatibility proof; they identify the actual open research questions.

---

# 14. The whole logic in one chain

$$
\boxed{
\begin{aligned}
X_0
&\xrightarrow{\text{iMF / MeanFlow one-step}}
\mu_\theta(X_0)\\
&\xrightarrow{+\tau\varepsilon}
Y,\\[1mm]
Y\mid X_0
&\sim\mathcal N(\mu_\theta(X_0),\tau^2I)\\
&\Downarrow\\
-\log p_\theta(Y\mid X_0)
&=\frac{\|Y-\mu_\theta(X_0)\|^2}{2\tau^2}+C\\
&\Downarrow\;\text{Jensen}\\
-A\mathcal L_{\mathrm{endpoint}}+C
&\le \mathbb E[\log p_\theta(Y)]\\
&= -H(p_{\mathrm{tgt}})-D_{KL}(p_{\mathrm{tgt}}\|p_\theta).
\end{aligned}
}
$$

The strict conclusion is therefore:

> **RMFlow's Gaussian endpoint-likelihood mechanism is compatible with Improved MeanFlow because it only requires a deterministic final one-step map; it does not require the original sample-specific JVP tangent.**

What remains to be proved separately is whether iMF preserves the original CMFM Wasserstein-control theorem and whether the hybrid improves empirical quality.

## Related pages

- [Improved MeanFlow deep dive](/research-handbook/en/meanflow/improved-meanflow/)
- [MeanFlow research track](/research-handbook/en/meanflow/story/)
