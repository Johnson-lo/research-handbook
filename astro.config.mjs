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
      social: [
        { icon: 'github', label: 'GitHub', href: 'https://github.com/Johnson-lo' }
      ],
      sidebar: [
        {
          label: 'Start Here',
          items: [
            { label: 'Home', slug: '' },
            { label: 'How to use this handbook', slug: 'guide' }
          ]
        },
        {
          label: 'Foundations',
          items: [{ autogenerate: { directory: 'foundations' } }]
        },
        {
          label: 'MeanFlow',
          items: [{ autogenerate: { directory: 'meanflow' } }]
        },
        {
          label: 'Paper Notes',
          items: [{ autogenerate: { directory: 'papers' } }]
        },
        {
          label: 'Interactive Lab',
          items: [{ autogenerate: { directory: 'interactive' } }]
        },
        {
          label: 'Experiments',
          items: [{ autogenerate: { directory: 'experiments' } }]
        },
        { label: 'About', slug: 'about' }
      ]
    })
  ]
});
