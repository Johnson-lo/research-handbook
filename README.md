# Johnson's Research Handbook

A bilingual public research handbook for Flow Matching, MeanFlow, fast generative modeling, and robotics.

- 繁體中文（default）: `https://johnson-lo.github.io/research-handbook/`
- English: `https://johnson-lo.github.io/research-handbook/en/`

## Stack

- Astro 7
- Starlight
- Markdown / MDX
- KaTeX for equations
- Vanilla Astro components for interactive demos
- Starlight i18n (`zh-TW` root + `en`)
- GitHub Pages deployment through GitHub Actions

## Development

```bash
npm install
npm run dev
```

## Content convention

The public site separates:

- paper-supported claims,
- personal interpretation / mental models,
- open research questions.

Traditional Chinese is the primary authoring language. English mirrors use matching slugs so each page can be switched between languages.

## Deployment target

Repository: `Johnson-lo/research-handbook`

If the repository name changes, update `base` in `astro.config.mjs`.
