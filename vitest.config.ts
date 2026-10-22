import { defineConfig } from 'vitest/config';

// Unit tests for the LLM Sizer engine and the guide structure helpers (pure TypeScript, no environment).
// Kept on the plain Vite config on purpose: these must run without the Astro plugins.
export default defineConfig({
  test: {
    include: ['src/lib/llm-sizer/**/*.test.ts', 'src/lib/guide-structure.test.ts', 'src/lib/remark-callouts.test.ts', 'src/lib/remark-strip-comments.test.ts', 'src/lib/visits.test.ts', 'src/lib/launch-price.test.ts', 'src/lib/refund.test.ts', 'src/lib/video-views.test.ts', 'src/lib/free-chapter.test.ts'],
    environment: 'node',
  },
});
