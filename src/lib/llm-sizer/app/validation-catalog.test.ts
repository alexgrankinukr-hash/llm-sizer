/**
 * The validation rows (public/data/llm-sizer/validation.json) name catalog models. Their active-parameter counts and
 * cache bytes per token must be the catalog's own, so that a fit judged on those rows says something about the live
 * catalog-to-engine path and the two cannot drift apart.
 */
import { existsSync, readFileSync } from 'node:fs';
import { describe, expect, it } from 'vitest';
import { loadModel } from '../engine/__fixtures__/load';

interface Row {
  id: string;
  model_id?: string;
  active_params_b: number;
  arch: { kv_bytes_per_token_8bit: number };
}
const rows = (JSON.parse(readFileSync(new URL('../../../../public/data/llm-sizer/validation.json', import.meta.url), 'utf8')) as { rows: Row[] }).rows;
/** every validation model that is in the catalog has a fixture copied from the export */
const CATALOG_IDS = ['deepseek-v4-flash-0731', 'gemma-4-26b-a4b', 'gemma-4-31b', 'gemma-4-e4b', 'glm-5.2', 'glm-5.3-flash', 'gpt-oss-120b', 'gpt-oss-20b', 'kimi-k2.6', 'qwen3-coder-next', 'qwen3.5-397b-a17b', 'qwen3.6-35b-a3b', 'qwen3.8-27b'];

describe('validation rows agree with the catalog records they name', () => {
  it('every catalog model named by a row has a fixture', () => {
    for (const id of CATALOG_IDS) expect(existsSync(new URL(`../engine/__fixtures__/${id}.json`, import.meta.url)), id).toBe(true);
    const named = new Set(rows.map((r) => r.model_id).filter((x): x is string => !!x));
    for (const id of CATALOG_IDS) expect(named.has(id), `${id} is listed but no row names it`).toBe(true);
  });

  for (const id of CATALOG_IDS) {
    it(`${id}: active parameters and cache bytes per token match the record`, () => {
      const m = loadModel(id);
      const active = (m.params_active ?? m.params_total ?? 0) / 1e9;
      for (const r of rows.filter((x) => x.model_id === id)) {
        expect(Math.abs(r.active_params_b - active), `${r.id}: row ${r.active_params_b}B, catalog ${active.toFixed(2)}B`).toBeLessThanOrEqual(0.05 + 0.005 * active);
        expect(r.arch.kv_bytes_per_token_8bit, `${r.id}: cache bytes`).toBe(m.architecture?.full_bytes_per_token_8bit);
      }
    });
  }
});
