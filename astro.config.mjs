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
      description: 'Generative Modeling · Flow-based Methods · Robotics',
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
        { label: '首頁', translations: { en: 'Home' }, slug: '' },
        {
          label: 'Paper Library｜論文庫',
          translations: { en: 'Paper Library' },
          items: [
            { label: '論文總覽', translations: { en: 'Overview' }, slug: 'papers' },
            { label: 'Mean Flows for One-step Generative Modeling', slug: 'papers/meanflow' },
            { label: 'Improved Mean Flows', slug: 'papers/improved-meanflow' },
            { label: 'RMFlow', slug: 'papers/rmflow' },
            { label: 'Improving Flow Matching by Aligning Flow Divergence', slug: 'papers/flow-divergence' }
          ]
        },
        {
          label: 'Research Tracks｜研究脈絡',
          translations: { en: 'Research Tracks' },
          items: [
            { label: '研究脈絡總覽', translations: { en: 'Overview' }, slug: 'tracks' },
            { label: 'MeanFlow Evolution', slug: 'meanflow/story' },
            { label: 'Flow Matching Objectives', slug: 'tracks/flow-matching-objectives' }
          ]
        },
        {
          label: 'Concepts｜核心概念',
          translations: { en: 'Concepts' },
          items: [
            { label: '概念總覽', translations: { en: 'Overview' }, slug: 'concepts' },
            { label: 'Calculus & JVP', slug: 'concepts/calculus-jvp' },
            { label: 'Distribution & Sampling', slug: 'foundations/distribution-and-sampling' },
            { label: 'Flow Matching', slug: 'foundations/flow-matching' },
            { label: 'Interactive Lab', slug: 'interactive' }
          ]
        },
        {
          label: 'Experiments｜實驗與證據',
          translations: { en: 'Experiments & Evidence' },
          items: [
            { autogenerate: { directory: 'experiments' } }
          ]
        },
        { label: '關於本站', translations: { en: 'About' }, slug: 'about' }
      ]
    })
  ]
});
