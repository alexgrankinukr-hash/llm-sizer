import { describe, expect, it } from 'vitest';
import { readFileSync } from 'node:fs';

describe('test tooling', () => {
  it('reads the machine catalog from the repo', () => {
    const doc = JSON.parse(readFileSync(new URL('../../../../public/data/llm-sizer/machines.json', import.meta.url), 'utf8')) as { machines: unknown[] };
    expect(doc.machines.length).toBeGreaterThan(40);
  });
});
