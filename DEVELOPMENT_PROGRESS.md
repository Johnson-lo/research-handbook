# Research Handbook — Development Progress

Last updated: 2026-09-04

## Project goal

Build a public research handbook / knowledge base for Johnson-lo, focused on Flow Matching, MeanFlow, fast generative modeling, and robotics. The site is not a Notion mirror and not a chronological blog. It is curated public research knowledge and a long-term technical portfolio.

## Product decisions

- Framework: **Astro + Starlight**.
- Hosting: **GitHub Pages** with the official Astro GitHub Action.
- Language architecture: **Traditional Chinese is the default/root locale (`zh-TW`) and English lives under `/en/`**.
- Technical terminology and paper names remain in English where that improves precision.
- Public-site structure: Home / Handbook / Paper Notes / Interactive Lab / Experiments / About.
- Notion remains the private scratchpad and working research space.
- Public site separates paper-supported claims, personal interpretation / mental models, and open research questions.

## Bilingual URL architecture

- Traditional Chinese: `https://johnson-lo.github.io/research-handbook/`
- English: `https://johnson-lo.github.io/research-handbook/en/`
- Starlight locale switcher is configured for `繁體中文` and `English`.
- Root files and `/en/` files use matching slugs so translated pages map one-to-one.
- Sidebar labels have Traditional-Chinese defaults and English translations.
- Interactive Astro components accept a locale prop and localize explanatory UI copy while keeping mathematical notation shared.

## Information architecture (V1)

### Home
- Research focus
- Entry points: Foundations / Papers / Interactive
- MeanFlow research map

### Handbook
- Foundations: Distribution & Sampling, Probability Path, Flow Matching, Conditional vs Marginal Velocity
- MeanFlow: Average Velocity, MeanFlow Identity, JVP, Original Objective, 1-NFE Inference
- Improved MeanFlow: network-dependent target, equivalent v-loss, iMF objective, boundary velocity, auxiliary v-head, flexible CFG, in-context conditioning, training vs inference

### Paper Notes
- Mean Flows for One-step Generative Modeling
- Improved Mean Flows: On the Challenges of Fastforward Generative Models
- Later: RMFlow, TEMF, AlphaFlow, variance reduction, Decoupled MeanFlow, etc.

### Interactive Lab
- Probability Path Explorer
- Conditional vs Marginal Velocity Explorer
- JVP / interval Explorer
- MF vs iMF Objective Explorer
- Later: 1-NFE vs multi-NFE explorer

### Experiments
- Published MF vs iMF results
- Later: reproduction logs, GPU/VRAM, ablations, implementation notes

## Implemented in GitHub V1

Repository: `Johnson-lo/research-handbook`

- Astro + Starlight project scaffold
- GitHub Pages workflow
- custom theme CSS
- Home and guide
- Foundations pages
- MeanFlow and Improved MeanFlow handbook chapters
- MeanFlow and iMF paper notes
- Interactive Lab index
- Published results page
- About page
- Traditional-Chinese versions of all current public pages
- English mirror under `src/content/docs/en/`
- Starlight bilingual locale configuration and language switcher
- localized interactive components:
  - `ProbabilityPath.astro`
  - `VelocityGTExplorer.astro`
  - `JVPExplorer.astro`
  - `MFIMFLossExplorer.astro`

## Core technical content encoded

- `z_t=(1-t)x+te`, with `t=0` data and `t=1` prior/noise.
- Training constructs `z_t` directly; it is not an inference rollout.
- `e-x` is sample-specific conditional velocity supervision.
- `v(z,t)=E[e-x|z,t]` is the marginal instantaneous velocity field.
- MeanFlow learns average velocity `u(z,r,t)`.
- MeanFlow identity: `v=u+(t-r)du/dt`.
- JVP is the total derivative of `u` along the flow and explicit time conditioning.
- Original MF apparent target is network-dependent.
- Original MF can be equivalently viewed as a v-loss re-parameterized by u-pred.
- iMF replaces the JVP tangent `e-x` with predicted marginal `v_theta(z_t)` while keeping `e-x` as supervision.
- iMF 1-NFE inference still uses `z_0=z_1-u_theta(z_1,0,1)`.

## Published experimental results added

Objective-focused results:
- MF-B/2 fixed-CFG baseline: FID 6.17
- iMF boundary velocity: FID 5.97
- iMF auxiliary v-head: FID 5.68
- MF-XL/2 original MF: FID 3.43
- MF-XL/2 + iMF boundary objective: FID 2.99

System-level MF → iMF FID:
- B/2: 6.17 → 3.39
- M/2: 5.01 → 2.27
- L/2: 3.84 → 1.86
- XL/2: 3.43 → 1.72

Important caveat: the final 1.72 result includes objective, flexible CFG, conditioning architecture, and training/system changes; it must not be attributed purely to the loss reformulation.

## Deployment state

- Repository is public and GitHub Pages is enabled with **Source = GitHub Actions**.
- Deployment target: `https://johnson-lo.github.io/research-handbook/`.
- GitHub Actions build and deploy both succeeded after Pages was enabled (workflow run `33859386608`, attempt 2).
- Initial CI issues already fixed:
  1. package manager detection (`package-manager: npm@latest`),
  2. Starlight v0.42 sidebar schema migration,
  3. Astro 7 MDX math parsing (`starlight-katex`, `@astrojs/markdown-remark`, `unified()`).
- Current bilingual change triggers a fresh build/deployment that must be verified before considering V1 bilingual release complete.

## Next development steps

1. Verify bilingual CI build and Pages deployment.
2. Test Chinese root and English `/en/` routes.
3. Test the Starlight language switcher and one-to-one translated page mapping.
4. Test all internal links under `/research-handbook` and `/research-handbook/en`.
5. Validate equations and all four interactive components on desktop and mobile.
6. Expand the iMF chapter with flexible CFG / in-context conditioning details and stronger experimental interpretation.
7. Add a dedicated research map and related-work graph.
8. Add later MeanFlow papers from Notion in curated form rather than bulk-copying.
9. Add reproduction / compute-resource pages when experiments begin.

## Development continuity rule

Whenever the public site changes materially, update this file with pages/components added or removed, major content decisions, deployment state, known bugs/blockers, and next concrete tasks.

This file is the project handoff checkpoint for future ChatGPT sessions and should remain synchronized with the Notion development report.
