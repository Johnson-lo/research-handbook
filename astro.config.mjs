import { defineConfig } from 'astro/config';
import { unified } from '@astrojs/markdown-remark';
import starlight from '@astrojs/starlight';
import { starlightKatex } from 'starlight-katex';

export default defineConfig({
  site: 'https://johnson-lo.github.io',
  base: '/research-handbook',
  markdown: {
    processor: unified(),
  },
  integrations: [
    starlight({
      title: "Johnson's Research Handbook",
      description: 'Flow Matching · MeanFlow · Fast Generative Modeling · Robotics',
      lastUpdated: true,
      customCss: ['./src/styles/custom.css'],
      plugins: [starlightKatex()],
      defaultLocale: 'root',
      locales: {
        root: {
          label: '繁體中文',
          lang: 'zh-TW',
        },
        en: {
          label: 'English',
          lang: 'en',
        },
      },
      social: [
        { icon: 'github', label: 'GitHub', href: 'https://github.com/Johnson-lo' }
      ],
      sidebar: [
        {
          label: '開始',
          translations: { en: 'Start Here' },
          items: [
            { label: '首頁', translations: { en: 'Home' }, slug: '' },
            { label: '閱讀指南', translations: { en: 'How to use this handbook' }, slug: 'guide' }
          ]
        },
        {
          label: '基礎',
          translations: { en: 'Foundations' },
          items: [{ autogenerate: { directory: 'foundations' } }]
        },
        {
          label: 'MeanFlow',
          items: [{ autogenerate: { directory: 'meanflow' } }]
        },
        {
          label: '論文筆記',
          translations: { en: 'Paper Notes' },
          items: [{ autogenerate: { directory: 'papers' } }]
        },
        {
          label: '互動實驗室',
          translations: { en: 'Interactive Lab' },
          items: [{ autogenerate: { directory: 'interactive' } }]
        },
        {
          label: '實驗',
          translations: { en: 'Experiments' },
          items: [{ autogenerate: { directory: 'experiments' } }]
        },
        { label: '關於', translations: { en: 'About' }, slug: 'about' }
      ]
    })
  ]
});
