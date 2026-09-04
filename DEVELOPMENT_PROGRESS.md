# Research Handbook — Development Progress

Last updated: 2026-09-04

## Project goal

Build a public research handbook / knowledge base for Johnson-lo, focused on Flow Matching, MeanFlow, fast generative modeling, and robotics. The site is not a Notion mirror and not a chronological blog. It is curated public research knowledge.

## Product decisions

- Framework: **Astro + Starlight**.
- Hosting: **GitHub Pages** with the official Astro GitHub Action.
- Language: **Traditional Chinese as the main explanatory language**, while preserving English technical terms and paper terminology.
- Public-site structure: Home / Handbook / Paper Notes / Interactive Lab / Experiments / About.
- Notion remains the private scratchpad and working research space.
- Public site separates paper-supported claims, personal interpretation / mental models, and open research questions.

## Information architecture (V1)

### Home
- Research focus
- Entry points: Learn / Papers / Interactive
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
- MF-XL/2 original objective: FID 3.43
- MF-XL/2 + iMF boundary objective: FID 2.99

System-level MF → iMF FID:
- B/2: 6.17 → 3.39
- M/2: 5.01 → 2.27
- L/2: 3.84 → 1.86
- XL/2: 3.43 → 1.72

Important caveat: the final 1.72 result includes objective, flexible CFG, conditioning architecture, and training/system changes; it must not be attributed purely to the loss reformulation.

## Deployment state

- Repository created and V1 source uploaded to `main`.
- Deployment target: `https://johnson-lo.github.io/research-handbook/`
- GitHub Actions **build now succeeds**.
- Fixed three CI issues during initial deployment:
  1. package manager detection by setting `package-manager: npm@latest` in the Astro Pages action,
  2. Starlight v0.42 sidebar schema migration,
  3. MDX math parsing by adding `starlight-katex`, `@astrojs/markdown-remark`, and switching Astro to the `unified()` Markdown processor.
- Current blocker is no longer the site build. The deploy job reaches `actions/deploy-pages@v5` but GitHub returns `404 Not Found` with the explicit message: **Ensure GitHub Pages has been enabled**.
- Required user-side setting: GitHub repository → Settings → Pages → Build and deployment → Source = **GitHub Actions**. After this setting is enabled, rerun the failed workflow or push a new commit.

## Next development steps

1. Enable GitHub Pages for the repository with Source = GitHub Actions.
2. Re-run deployment and verify the public URL.
3. Test all internal links under the `/research-handbook` base path.
4. Validate interactive components on desktop and mobile.
5. Translate/expand the V1 public prose into the intended Traditional-Chinese-first style.
6. Expand iMF chapter with flexible CFG / in-context conditioning details and stronger experimental interpretation.
7. Add a dedicated research map and related-work graph.
8. Add later MeanFlow papers from Notion in curated form rather than bulk-copying.
9. Add reproduction / compute-resource pages when experiments begin.

## Development continuity rule

Whenever the public site changes materially, update this file with pages/components added or removed, major content decisions, deployment state, known bugs/blockers, and next concrete tasks.

This file is the project handoff checkpoint for future ChatGPT sessions and should remain synchronized with the Notion development report.
