import { defineConfig } from 'astro/config';
import node from '@astrojs/node';
import react from '@astrojs/react';
import tailwindcss from '@tailwindcss/vite';

// A standalone Node server: `npm run build && npm start`. Swap the adapter for your host (Vercel, Netlify, Cloudflare)
// and set SITE_URL so share links and link previews carry your origin.
export default defineConfig({
  site: process.env.SITE_URL ?? 'http://localhost:4321',
  output: 'server',
  adapter: node({ mode: 'standalone' }),
  integrations: [react()],
  image: { service: { entrypoint: 'astro/assets/services/noop' } },
  vite: { plugins: [tailwindcss()] },
});
