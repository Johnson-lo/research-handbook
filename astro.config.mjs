import { defineConfig } from 'astro/config';
import { unified } from '@astrojs/markdown-remark';
import starlight from '@astrojs/starlight';
import remarkMath from 'remark-math';
import rehypeKatex from 'rehype-katex';

export default defineConfig({
  site: 'https://johnson-lo.github.io',
  base: '/research-handbook',
  markdown: {
    processor: unified({
      remarkPlugins: [remarkMath],
      rehypePlugins: [[rehypeKatex, { strict: false }]],
    }),
  },
  integrations: [
    starlight({
      title: "Johnson's Research Handbook",
      description: 'Flow Matching · MeanFlow · Fast Generative Modeling · Robotics',
      lastUpdated: true,
      customCss: ['./src/styles/custom.css'],
      defaultLocale: 'root',
      locales: {
        root: { label: '繁體中文', lang: 'zh-TW' },
        en: { label: 'English', lang: 'en' },
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
            { label: '閱讀指南', translations: { en: 'Reading Guide' }, slug: 'guide' }
          ]
        },
        {
          label: '主線：Flow Matching → iMF',
          translations: { en: 'Main Story: Flow Matching → iMF' },
          items: [
            { label: '完整故事線', translations: { en: 'The Full Story' }, slug: 'meanflow/story' },
            { label: 'MeanFlow 深入', translations: { en: 'MeanFlow Deep Dive' }, slug: 'meanflow/meanflow' },
            { label: 'Improved MeanFlow 深入', translations: { en: 'Improved MeanFlow Deep Dive' }, slug: 'meanflow/improved-meanflow' }
          ]
        },
        {
          label: '概念參考',
          translations: { en: 'Concept Reference' },
          items: [
            { autogenerate: { directory: 'foundations' } },
            { label: '互動實驗室', translations: { en: 'Interactive Lab' }, slug: 'interactive' }
          ]
        },
        {
          label: '證據與來源',
          translations: { en: 'Evidence & Sources' },
          items: [
            { autogenerate: { directory: 'papers' } },
            { autogenerate: { directory: 'experiments' } }
          ]
        },
        { label: '關於', translations: { en: 'About' }, slug: 'about' }
      ]
    })
  ]
});
