import { readFileSync } from 'node:fs';
import { describe, expect, it } from 'vitest';
import { loadFactors, loadMachines, loadModel, machineById } from './__fixtures__/load';
import { evaluateCell } from './index';
import { customMachine } from './custom';
import { estimatePrefill, feelsLikeWait, firstWordWaitS, prefillConstant, prefillCoresOf, prefillD0, prefillRateAt, prefillTokS } from './prefill';
import type { Settings } from './types';

const factors = loadFactors();
const pf = factors.prefill!;
const base: Omit<Settings, 'memoryGb'> = { workApps: 0, override: false, contextTokens: 32768, kvBits: 16, runtime: 'gguf', qualityFloorBits: 2 };
const s = (memoryGb: number, over: Partial<Settings> = {}): Settings => ({ ...base, memoryGb, ...over });

interface Case {
  id: string; chip: string; gpu_cores: number | null; platform: 'apple' | 'cuda'; quant_label: string; format: 'gguf' | 'mlx'; arch_class: string; runtime: 'gguf' | 'mlx';
  active_params_b: number; kv_layers: number; layers: number; context_tokens: number; mlx_group: 'pre_m5' | 'm5plus';
  expected_k: number; expected_source: string; expected_tok_s: number; expected_d0: number; expected_wait_8k_s: number; expected_wait_ctx_s: number; expected_rate_at_ctx: number;
}
const cases = (JSON.parse(readFileSync(new URL('../../../../scripts/llm-sizer/tests/fixtures/prefill-cases.json', import.meta.url), 'utf8')) as { cases: Case[] }).cases;

describe('prefill parity with speed_model.py', () => {
  it('every fixture case reproduces: the constant, its source, the rate, D0 and the waits', () => {
    for (const c of cases) {
      const k = prefillConstant(pf, { chip: c.chip, family: c.chip, gpu_cores: c.gpu_cores ? [c.gpu_cores] : [], platform: c.platform });
      expect(k.source, c.id).toBe(c.expected_source);
      expect(k.k, c.id).toBeCloseTo(c.expected_k, 6);
      const tokS = prefillTokS(pf, k.k!, c.quant_label, c.format, c.arch_class, c.runtime, c.active_params_b, c.mlx_group);
      expect(tokS, c.id).toBeCloseTo(c.expected_tok_s, 2);
      const d0 = prefillD0(pf, { architecture: { kv_layers: c.kv_layers, layers: c.layers } as never }).d0;
      expect(d0, c.id).toBeCloseTo(c.expected_d0, 2);
      expect(firstWordWaitS(tokS, d0, 8192), c.id).toBeCloseTo(c.expected_wait_8k_s, 3);
      expect(firstWordWaitS(tokS, d0, c.context_tokens), c.id).toBeCloseTo(c.expected_wait_ctx_s, 3);
      expect(prefillRateAt(tokS, d0, c.context_tokens), c.id).toBeCloseTo(c.expected_rate_at_ctx, 3);
    }
  });
});

describe('the prefill estimate on real cells', () => {
  const mbp = machineById('macbook-pro-m5-max-40c');
  it('Qwen 3.8 27B at Q4 on an M5 Max 40-core: about 670 tok/s at a short prompt, a 32K prompt waits about a minute and a half', () => {
    const cell = evaluateCell(loadModel('qwen3.8-27b'), mbp, s(128), factors);
    const p = cell.prefill!;
    expect(p.source).toBe('measured');
    expect(p.parts).toMatchObject({ cores: 40, generation: 'M5', kquant: 0.88, cls: 'dense', classFactor: 1, mlx: 1 });
    expect(p.tokS!).toBeGreaterThan(640);
    expect(p.tokS!).toBeLessThan(700);
    expect(p.waits.map((w) => w.tokens)).toEqual([8192, 32768, 131072]);
    const w32 = p.waits[1].seconds;
    expect(w32).toBeGreaterThan(75);
    expect(w32).toBeLessThan(105);
    expect(p.waits[0].seconds * 4).toBeLessThan(w32); // worse than linear: the context already read slows the rest
    expect(p.atContext).toMatchObject({ tokens: 32768 });
    expect(p.atContext!.rateTokS).toBeLessThan(p.tokS!);
    expect(p.notes).toEqual([]);
  });
  it('the file\'s bits do not matter, the cores do, and a 32-core M5 Max reads at 0.8 of the 40-core', () => {
    const q8 = evaluateCell(loadModel('qwen3.8-27b'), mbp, s(128, { quant: loadModel('qwen3.8-27b').quants.find((q) => q.label === 'Q8_0') }), factors).prefill!;
    const q4 = evaluateCell(loadModel('qwen3.8-27b'), mbp, s(128), factors).prefill!;
    expect(q8.tokS!).toBeCloseTo(q4.tokS! / 0.88, 3); // Q8_0 is a plain file, Q4_K_M a K-quant one: the only difference
    const c32 = evaluateCell(loadModel('qwen3.8-27b'), machineById('macbook-pro-m5-max-32c'), s(36), factors).prefill!;
    expect(c32.tokS! / q4.tokS!).toBeCloseTo(0.8, 6);
  });
  it('an M5 Ultra has no rows: its generation\'s rate on 80 cores, flagged; an M6 mini takes the newest generation, flagged as assumed', () => {
    const ultra = evaluateCell(loadModel('glm-5.2'), machineById('mac-studio-m5-ultra'), s(512, { override: 1 }), factors).prefill!;
    expect(ultra.source).toBe('generation');
    expect(ultra.parts).toMatchObject({ cores: 80, cls: 'moe_latent' });
    expect(ultra.notes.some((n) => n.includes('no measured prefill row'))).toBe(true);
    expect(ultra.notes.some((n) => n.includes('assumed'))).toBe(true); // the latent-attention class factor has no rows
    const m6 = evaluateCell(loadModel('qwen3.8-27b'), machineById('mac-mini-m6'), s(32), factors).prefill!;
    expect(m6.source).toBe('assumed');
    expect(m6.parts).toMatchObject({ cores: 12, generation: 'M5' });
  });
  it('Flash Next is its own class, a special build has no estimate, and a compromise carries the fix\'s configuration', () => {
    const fn = loadModel('qwen3.8-flash-next');
    const studio = evaluateCell(fn, machineById('mac-studio-m5-ultra'), s(256), factors).prefill!;
    expect(studio.parts).toMatchObject({ cls: 'flash_next', classFactor: 0.31 });
    const ssd = evaluateCell(fn, mbp, s(64, { quant: fn.special_builds![0] }), factors).prefill!;
    expect(ssd.tokS).toBeNull();
    expect(ssd.notes[0]).toMatch(/streams from the SSD/);
    const fix = evaluateCell(fn, mbp, s(64), factors); // Q4 ask → the SSD build is the fix
    expect(fix.verdict).toBe('compromise');
    expect(fix.prefill?.tokS).toBeNull();
    const noFit = evaluateCell(loadModel('kimi-k3'), mbp, s(128), factors);
    expect(noFit.prefill).toBeNull();
  });
  it('MLX pays its factor by generation, the Spark uses its own constant, a ROCm box has none', () => {
    const mlxQ = loadModel('qwen3.8-27b').quants.find((q) => q.format === 'mlx')!;
    const m5 = evaluateCell(loadModel('qwen3.8-27b'), mbp, s(128, { runtime: 'mlx', quant: mlxQ }), factors).prefill!;
    expect(m5.parts?.mlx).toBe(pf.mlx.m5plus.factor);
    const m1 = evaluateCell(loadModel('qwen3.8-27b'), machineById('macbook-pro-m1-max'), s(64, { runtime: 'mlx', quant: mlxQ }), factors).prefill!;
    expect(m1.parts?.mlx).toBe(pf.mlx.pre_m5.factor);
    const spark = evaluateCell(loadModel('qwen3.8-27b'), machineById('nvidia-dgx-spark'), s(128), factors).prefill!;
    expect(spark.source).toBe('measured');
    expect(spark.parts).toMatchObject({ cores: null, k: pf.by_chip['DGX Spark'].value });
    const box = customMachine({ name: 'Halo', memoryGb: 128, bandwidthGbs: 256, platform: 'rocm' });
    const rocm = evaluateCell(loadModel('qwen3.8-27b'), box, s(128), factors).prefill!;
    expect(rocm.tokS).toBeNull();
    expect(rocm.source).toBe('none');
  });
  it('a row with two GPU bins reads on the one its price buys at that size, and names the larger with its price', () => {
    const q = loadModel('qwen3.8-27b');
    // the 48 GB Mac mini M5 Pro is $2,299 with the 16-core GPU; the 20-core is +$200 (the reader's report, 2026-09-24)
    const mini = evaluateCell(q, machineById('mac-mini-m5-pro'), s(48), factors).prefill!;
    expect(mini.parts?.cores).toBe(16);
    expect(mini.upgrade).toMatchObject({ cores: 20, ratio: 1.25, usd: 200 });
    expect(mini.upgrade!.tokS).toBeCloseTo(mini.tokS! * 1.25, 6);
    expect(mini.upgrade!.waits.map((w) => w.tokens)).toEqual(mini.waits.map((w) => w.tokens));
    expect(mini.upgrade!.waits[1].seconds).toBeCloseTo(mini.waits[1].seconds / 1.25, 6);
    // 64 GB on the MacBook Pro M5 Pro comes only with the 20-core GPU: that is what it reads on, nothing to upgrade
    const mbp64 = evaluateCell(q, machineById('macbook-pro-m5-pro'), s(64), factors).prefill!;
    expect(mbp64.parts?.cores).toBe(20);
    expect(mbp64.upgrade).toBeNull();
    expect(evaluateCell(q, machineById('macbook-pro-m5-pro'), s(48), factors).prefill!.upgrade).toMatchObject({ cores: 20, usd: 200 });
    // the M5 Ultra: 64 cores at 96 and 256 GB (+$1,300 for 80), 80 at 512 GB
    const ultra96 = evaluateCell(q, machineById('mac-studio-m5-ultra'), s(96), factors).prefill!;
    expect(ultra96.parts?.cores).toBe(64);
    expect(ultra96.upgrade).toMatchObject({ cores: 80, usd: 1300 });
    // an older machine follows the same rule, without a price for the larger bin
    const m1max = evaluateCell(q, machineById('macbook-pro-m1-max'), s(64), factors).prefill!;
    expect(m1max.parts?.cores).toBe(24);
    expect(m1max.upgrade).toMatchObject({ cores: 32, usd: null });
    // one bin per row: nothing to name
    expect(evaluateCell(q, mbp, s(128), factors).prefill!.upgrade).toBeNull();
  });
  it('without a memory size a two-bin row takes its smaller bin', () => {
    expect(prefillCoresOf(machineById('mac-mini-m5-pro'))).toBe(16);
    expect(prefillCoresOf(machineById('mac-studio-m5-ultra'), 512)).toBe(80);
    expect(prefillCoresOf(machineById('nvidia-dgx-spark'), 128)).toBeNull();
  });
  it('feels-like bands for the wait', () => {
    expect(feelsLikeWait(pf, 2)).toBe('instant');
    expect(feelsLikeWait(pf, 10)).toBe('short wait');
    expect(feelsLikeWait(pf, 60)).toBe('get a coffee');
    expect(feelsLikeWait(pf, 600)).toBe('painful');
  });
});
