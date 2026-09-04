# Research Handbook — Development Progress

Last updated: 2026-09-05

## Current product direction

The site has been redesigned from a category-first wiki into a **causal research story**:

`Flow Matching → why multi-NFE → MeanFlow average velocity → MeanFlow Identity / JVP → Original MF objective issue → iMF core change → evidence → open research questions`.

The previous structure (Foundations / MeanFlow / Paper Notes / Interactive Lab / Experiments as peer navigation categories) was too fragmented because readers had to infer why one concept led to the next.

## Current information architecture

### Start Here
- Home
- Reading Guide

### Main Story: Flow Matching → iMF
- The Full Story (`meanflow/story`)
- MeanFlow Deep Dive
- Improved MeanFlow Deep Dive

### Concept Reference
- Distribution & Sampling
- Flow Matching / conditional vs marginal velocity
- Interactive Lab

### Evidence & Sources
- Paper Notes
- Published Experiments

### About

Foundations, Paper Notes, Interactive Lab, and Experiments are supporting references rather than competing reading paths.

## Bilingual architecture

- Traditional Chinese is the default/root locale (`zh-TW`).
- English mirrors all main V1 pages under `/en/` using matching slugs.
- Chinese: `https://johnson-lo.github.io/research-handbook/`
- English: `https://johnson-lo.github.io/research-handbook/en/`
- Starlight language switcher is enabled and story pages map one-to-one between locales.

## New interaction design

### `DerivationStepper.astro`
Purpose: make MeanFlow Identity derivation sequential rather than presenting a wall of equations.

Steps:
1. `(t-r)u = ∫v` as accumulated displacement.
2. Differentiate both sides with respect to `t`.
3. Apply product rule to obtain `u + (t-r)du/dt = v`.
4. Expand `du/dt = ∂_z u · v + ∂_t u = JVP(u;v)`.

Features:
- Prev / Next controls.
- Play / Pause automatic progression.
- Native MathML inside the interactive stepper.
- Traditional-Chinese and English explanatory copy.

### `MFvsIMFBridge.astro`
Purpose: make the difference between Original MeanFlow and Improved MeanFlow visually obvious.

Default view isolates the core objective-level change:
- Original MF JVP tangent: `e-x` (sample-specific conditional velocity).
- iMF JVP tangent: `v_theta(z_t)` (model-estimated marginal-like velocity).

Shared pieces are kept conceptually fixed:
- same sampled `x,e,t,r` and constructed `z_t`,
- same `e-x` supervision,
- same average-velocity inference object `u_theta`,
- same one-step inference equation.

A Full Flow toggle reveals the common parts. A separate block distinguishes the core objective change from flexible CFG, Ω-conditioning, in-context conditioning, architecture changes, and longer training.

## Main story page

Bilingual pages:
- `src/content/docs/meanflow/story.mdx`
- `src/content/docs/en/meanflow/story.mdx`

Narrative sections:
1. Why Flow Matching needs multiple NFEs.
2. MeanFlow's shift from instantaneous velocity to average velocity.
3. Why average velocity lacks a convenient sampled GT.
4. Dynamic derivation of MeanFlow Identity and why JVP appears.
5. Original MF objective and why `e-x` inside the JVP matters.
6. iMF's `e-x → v_theta(z_t)` JVP-tangent change.
7. Controlled objective evidence vs the final system-level FID 1.72.
8. Remaining research questions.

## Technical content that remains invariant

- `z_t=(1-t)x+te`, `t=0` data, `t=1` prior/noise.
- `e-x` is sample-specific conditional velocity supervision.
- `v(z,t)=E[e-x|z,t]` is the marginal instantaneous velocity field.
- MeanFlow learns average velocity `u(z,r,t)`.
- MeanFlow Identity: `v=u+(t-r)du/dt`.
- JVP is the total derivative of `u` along the flow/time direction.
- Original MF can be viewed as a v-loss reparameterized by u-pred.
- iMF changes the JVP tangent from `e-x` to `v_theta(z_t)` while keeping `e-x` as supervision.
- 1-NFE inference remains `z_0=z_1-u_theta(z_1,0,1)`.

## Published evidence preserved

Objective-focused:
- MF-B/2 original: FID 6.17
- iMF boundary: 5.97
- iMF auxiliary v-head: 5.68
- MF-XL/2 original: 3.43
- MF-XL/2 + iMF boundary objective: 2.99

System-level:
- iMF-XL/2: FID 1.72

The site explicitly warns that the final 1.72 includes objective, CFG, conditioning architecture, and training changes.

## Deployment and QA

GitHub Pages is enabled with GitHub Actions.

Narrative redesign build/deploy:
- workflow run `33900551542`
- build: **success**
- deploy: **success**

Artifact-level QA on the deployed Pages build:
- Chinese and English home/story/deep-dive routes are present.
- 342 internal `/research-handbook/...` links checked: **0 missing**.
- Chinese and English story pages both contain rendered KaTeX equations.
- `DerivationStepper` is present on both story locales with localized Prev/Next/Play controls.
- `MFvsIMFBridge` is present on both story locales and includes the `e-x → v_theta(z_t)` comparison.
- Chinese ↔ English story links are present.
- Pagefind produced both `zh-tw` and `en` indexes.
- A missing default favicon discovered during QA was fixed by adding `public/favicon.svg`.

The execution environment could not reliably launch Chromium headless for pixel-level browser QA, so final visual/mobile judgment should be done in a normal browser. The generated static artifact itself is internally consistent and deploys successfully.

## Next tasks

1. Human visual review of the Chinese main story on desktop/mobile.
2. Adjust density, spacing, or explanatory order based on actual reading experience.
3. Reduce any remaining duplicate prose between story, deep dives, and paper notes.
4. Add a compact research-map / related-work graph only after the story flow is stable.
5. Expand open research questions and later MeanFlow papers without breaking the causal narrative.

## Development continuity rule

Whenever the public site changes materially, update this file with pages/components added or removed, major content decisions, deployment state, known bugs/blockers, and next concrete tasks. Keep it synchronized with the Notion development report.
