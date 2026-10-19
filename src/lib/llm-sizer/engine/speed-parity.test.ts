/**
 * The fixed-cost speed model computed by the engine must reproduce the milliseconds fit_speed.py emitted for the
 * same inputs (scripts/llm-sizer/tests/fixtures/speed-cases.json). Python and TypeScript mirror each other function
 * for function; this is the pin that keeps them from drifting.
 */
import { readFileSync } from 'node:fs';
import { describe, expect, it } from 'vitest';
import { attendedTokens, floorMs, quantFamilyOf } from './speed';
import { loadFactors } from './__fixtures__/load';
import type { ArchClass, MlxGroup, Runtime } from './types';

interface Case {
  id: string;
  chip: string;
  platform?: 'apple' | 'cuda';
  bandwidth_gbs: number;
  profile: { t0_ms: number; b_eff_ratio: number };
  active_params_b: number;
  bits: number;
  kv_bytes_per_token_8bit: number;
  kv_bits: number;
  context_tokens: number;
  quant_label: string;
  runtime: Runtime;
  arch_class: ArchClass;
  mlx_group: MlxGroup;
  mlx_ctx?: 'pre_m5' | 'm5plus';
  attention: string;
  /** the expert-layer rule's input (0.17) */
  moe_layers?: number | null;
  /** a sparse-attention model's budget (speed_model.py:sparse_budget), null for the rest */
  sparse_budget?: number | null;
  gb_per_token: number;
  expected_ms: number;
  expected_tok_s: number;
}

const file = JSON.parse(readFileSync(new URL('../../../../scripts/llm-sizer/tests/fixtures/speed-cases.json', import.meta.url), 'utf8')) as { cases: Case[] };
const factors = loadFactors();

describe('speed-model parity with fit_speed.py', () => {
  it('the fixture carries cases for both runtimes and more than one architecture class', () => {
    expect(file.cases.length).toBeGreaterThanOrEqual(6);
    expect(new Set(file.cases.map((c) => c.runtime))).toEqual(new Set(['gguf', 'mlx']));
    expect(new Set(file.cases.map((c) => c.arch_class)).size).toBeGreaterThan(1);
    // the expert-layer rule is pinned on at least one small mixture on MLX
    expect(file.cases.some((c) => c.runtime === 'mlx' && c.mlx_group.startsWith('small_active') && c.moe_layers)).toBe(true);
    // a sparse-attention row past its budget, so the index term and the capped read are pinned on both sides
    expect(file.cases.some((c) => c.sparse_budget && c.runtime === 'mlx' && c.context_tokens > c.sparse_budget)).toBe(true);
  });

  for (const c of file.cases) {
    it(`${c.id}: ${c.expected_tok_s} tok/s`, () => {
      const sm = factors.speed_model!;
      const budget = c.sparse_budget ?? null;
      const attended = attendedTokens(c.context_tokens, c.runtime, budget);
      const gb = (c.active_params_b * 1e9 * c.bits) / 8 / 1e9 + (c.kv_bytes_per_token_8bit * (c.kv_bits / 8) * attended) / 1e9;
      expect(gb).toBeCloseTo(c.gb_per_token, 5);
      const family = quantFamilyOf({ label: c.quant_label, format: c.runtime === 'mlx' ? 'mlx' : 'gguf' } as never);
      const parts = floorMs({ t0Ms: c.profile.t0_ms, bEffRatio: c.profile.b_eff_ratio }, c.bandwidth_gbs, gb, family, c.arch_class, c.runtime, c.mlx_group, c.attention, c.context_tokens, sm, c.platform ?? 'apple', c.mlx_ctx ?? 'pre_m5', { chip: c.chip, moeLayers: c.moe_layers }, budget);
      expect(parts.ms).toBeCloseTo(c.expected_ms, 2);
      expect(1000 / parts.ms).toBeCloseTo(c.expected_tok_s, 1);
    });
  }
});
