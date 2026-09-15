import { defineConfig } from 'vitest/config';

// Unit tests for the LLM Sizer engine (pure TypeScript, no site imports, no environment).
// Kept on the plain Vite config on purpose: the engine must run without the Astro plugins.
export default defineConfig({
  test: {
    include: ['src/lib/llm-sizer/**/*.test.ts'],
    environment: 'node',
  },
});
