import { describe, expect, it } from 'vitest';
import { accelerations, chipKey, estimateSpeed, feelsLike, mlxContextGroupOf, resolveEfficiency, runtimeFactor, tierOf } from './speed';
import { loadFactors, loadModel, loadMultiplierFactors, machineById, quantByLabel } from './__fixtures__/load';
import type { Machine } from './types';

const factors = loadFactors();
// the multiplier model, retired in 0.9 and kept until its branch is deleted: its suites run on the last multiplier factor file
const legacy = loadMultiplierFactors();

describe('efficiency resolution', () => {
  it('strips the GPU bin from the chip name and finds the measured row', () => {
    expect(chipKey('M5 Max (40-core GPU)')).toBe('M5 Max');
    const r = resolveEfficiency(machineById('macbook-pro-m5-max-40c'), factors);
    expect(r.source).toBe('measured');
    expect(r.value).toBeCloseTo(factors.efficiency.by_chip['M5 Max'].value!, 3);
  });

  it('M6 mini uses the assumed base value; M5 Ultra the explicit assumption', () => {
    const m6 = resolveEfficiency(machineById('mac-mini-m6'), factors);
    expect(m6.source).toBe('assumed');
    expect(m6.value).toBeCloseTo(factors.efficiency.assumed['M6'].value!, 3);
    const ultra = resolveEfficiency(machineById('mac-studio-m5-ultra'), factors);
    expect(ultra.source).toBe('assumed');
    expect(ultra.value).toBeCloseTo(factors.efficiency.assumed['M5 Ultra'].value!, 3);
  });

  it('the Spark resolves through its platform to the measured Spark row', () => {
    const r = resolveEfficiency(machineById('nvidia-dgx-spark'), factors);
    expect(r.value).toBeCloseTo(0.748, 3); // five llama.cpp maintainer rows, build b7941
    expect(r.source).toBe('measured');
    expect(tierOf(machineById('nvidia-dgx-spark'))).toBe('spark');
  });

  it('an unknown Apple chip falls to its tier; ROCm has no factor yet', () => {
    const custom: Machine = { id: 'x', kind: 'mac', family: 'Custom', chip: 'M9 Pro', memory_options_gb: [64], bandwidth_gbs: 400, platform: 'apple', price_usd: null, custom: true };
    const r = resolveEfficiency(custom, factors);
    expect(r.source).toBe('tier');
    expect(r.value).toBeCloseTo(factors.efficiency.by_tier.pro.value!, 3);
    const halo: Machine = { ...custom, platform: 'rocm', chip: 'Ryzen AI Max+ 395' };
    expect(resolveEfficiency(halo, factors)).toEqual({ value: null, source: 'none', key: null });
  });
});

describe('decode speed, multiplier model (retired in METHODOLOGY 0.9)', () => {
  const ultra = machineById('mac-studio-m5-ultra');
  const spark = machineById('nvidia-dgx-spark');
  const glm = loadModel('glm-5.3-flash');
  const q4 = quantByLabel(glm, 'UD-Q4_K_XL');

  it('GLM-5.3 Flash Q4 on the M5 Ultra: bandwidth × efficiency ÷ GB per token', () => {
    const s = estimateSpeed({ model: glm, quant: q4, machine: ultra, contextTokens: 8192, kvBits: 8, factors: legacy });
    const expected = (1200 * legacy.efficiency.assumed['M5 Ultra'].value!) / s.gbPerToken;
    expect(s.tokS).toBeCloseTo(expected, 6);
    expect(s.gbPerToken).toBeCloseTo((18e9 * q4.bits!) / 8 / 1e9 + (5632 * 8192) / 1e9, 3);
    expect(s.tokS!).toBeGreaterThan(50);
    expect(s.tokS!).toBeLessThan(70);
    expect(s.efficiencySource).toBe('assumed');
    expect(s.notes.join(' ')).toMatch(/no measured rows/);
    expect(s.feelsLike?.label).toBe('fast');
  });

  it('the same model on the Spark is roughly three times slower', () => {
    const s = estimateSpeed({ model: glm, quant: q4, machine: spark, contextTokens: 8192, kvBits: 8, factors: legacy });
    expect(s.tokS!).toBeGreaterThan(16);
    expect(s.tokS!).toBeLessThan(20);
    expect(s.feelsLike?.label).toBe('comfortable');
  });

  it('speed falls with context in proportion to the cache re-read', () => {
    const drop = (id: string, label: string) => {
      const m = loadModel(id);
      const q = quantByLabel(m, label);
      const a = estimateSpeed({ model: m, quant: q, machine: ultra, contextTokens: 8192, kvBits: 8, factors: legacy }).tokS!;
      const b = estimateSpeed({ model: m, quant: q, machine: ultra, contextTokens: 131072, kvBits: 8, factors: legacy }).tokS!;
      return 1 - b / a;
    };
    expect(drop('glm-5.2', 'UD-Q4_K_XL')).toBeGreaterThan(0.1);
    expect(drop('glm-5.2', 'UD-Q4_K_XL')).toBeLessThan(0.18);
    expect(drop('glm-5.3-flash', 'UD-Q4_K_XL')).toBeLessThan(0.08);
    expect(drop('qwen3.8-flash-next', 'UD-Q4_K_XL')).toBeGreaterThan(0.2);
    expect(drop('qwen3.8-27b', 'Q4_K_M')).toBeGreaterThan(0.15);
    expect(drop('qwen3.8-27b', 'Q4_K_M')).toBeLessThan(0.25);
  });

  it('a special build has no speed figure anywhere; the publisher\'s measurement is quoted as a note', () => {
    const fn = loadModel('qwen3.8-flash-next');
    for (const id of ['macbook-pro-m5-max-40c', 'mac-mini-m4-pro', 'mac-studio-m5-ultra']) {
      const s = estimateSpeed({ model: fn, quant: fn.special_builds![0], machine: machineById(id), contextTokens: 32768, kvBits: 8, factors: legacy });
      expect(s.tokS).toBeNull();
      expect(s.efficiencySource).toBe('none');
      expect(s.notes[0]).toBe('the publisher measured 36 tok/s on an M5 Max (MacBook Pro M5 Max 64 GB); no speed is estimated for a build that streams from the SSD');
    }
  });

  it('no bandwidth or no factor → no number', () => {
    const halo: Machine = { id: 'h', kind: 'prebuilt', family: 'Halo', chip: 'Ryzen AI Max+ 395', memory_options_gb: [128], bandwidth_gbs: 256, platform: 'rocm', price_usd: null };
    const s = estimateSpeed({ model: glm, quant: q4, machine: halo, contextTokens: 8192, kvBits: 8, factors: legacy });
    expect(s.tokS).toBeNull();
    expect(s.efficiencySource).toBe('none');
  });
});

describe('runtime factor, multiplier model (retired in METHODOLOGY 0.9)', () => {
  const max = machineById('macbook-pro-m5-max-40c');
  const qwen = loadModel('qwen3.8-27b');
  const mlx4 = quantByLabel(qwen, 'MLX-4bit', 'lmstudio-community/Qwen3.8-27B-MLX-4bit');

  it('dense 27B on MLX gets 1.15× up to 24K, fades to 1.0 at 36K', () => {
    expect(runtimeFactor(mlx4, qwen, max, 8192, legacy)).toEqual({ factor: 1.15, applied: 'mlx' });
    expect(runtimeFactor(mlx4, qwen, max, 24000, legacy).factor).toBeCloseTo(1.15, 6);
    expect(runtimeFactor(mlx4, qwen, max, 30000, legacy).factor).toBeCloseTo(1.075, 6);
    expect(runtimeFactor(mlx4, qwen, max, 36000, legacy).factor).toBe(1);
    expect(runtimeFactor(mlx4, qwen, max, 131072, legacy).factor).toBe(1);
    const s = estimateSpeed({ model: qwen, quant: mlx4, machine: max, contextTokens: 8192, kvBits: 8, factors: legacy });
    expect(s.runtimeApplied).toBe('mlx');
    expect(s.tokS!).toBeGreaterThan(33);
    expect(s.tokS!).toBeLessThan(38);
  });

  it('a mixture-of-experts model on MLX gets the MoE factor; GGUF stays 1.0; MLX off Apple stays 1.0', () => {
    const gemma = loadModel('gemma-4-26b-a4b');
    const gmlx = gemma.quants.find((q) => q.format === 'mlx')!;
    expect(runtimeFactor(gmlx, gemma, max, 8192, legacy).factor).toBe(legacy.runtime.mlx.moe.factor);
    expect(runtimeFactor(quantByLabel(qwen, 'Q4_K_M', 'lmstudio-community/Qwen3.8-27B-GGUF'), qwen, max, 8192, legacy)).toEqual({ factor: 1, applied: 'gguf' });
    expect(runtimeFactor(mlx4, qwen, machineById('nvidia-dgx-spark'), 8192, legacy).applied).toBe('gguf');
  });
});

describe('accelerations and feels-like', () => {
  it('reports MTP and draft ranges only where the model has them', () => {
    const qwen = accelerations(loadModel('qwen3.8-27b'), factors);
    expect(qwen.map((a) => a.kind)).toEqual(['mtp', 'draft']);
    expect(qwen[0].min).toBeCloseTo(1.6, 2);
    expect(qwen[1].max).toBeGreaterThan(3);
    const gemma = accelerations(loadModel('gemma-4-26b-a4b'), factors);
    expect(gemma.map((a) => a.kind)).toEqual(['draft']);
    expect(gemma[0].median).toBeCloseTo(1.318, 2); // 4B active → the small-active-MoE band
  });

  it('feels-like bands', () => {
    expect(feelsLike(3, factors)?.label).toBe('painful');
    expect(feelsLike(20, factors)?.label).toBe('comfortable');
    expect(feelsLike(200, factors)?.label).toBe('cloud-like');
    expect(feelsLike(null, factors)).toBeNull();
  });
});

describe('fixed-cost model (METHODOLOGY §6.1; dormant until factors.speed_model.kind is "floor")', async () => {
  const { archClassOf, floorMs, mlxGroupOf, quantFamilyOf, resolveChipProfile, speedModelKind } = await import('./speed');
  const sm = factors.speed_model!;
  const floor: typeof factors = { ...factors, speed_model: { ...sm, kind: 'floor' } };
  const m4max40 = machineById('macbook-pro-m4-max-40c');
  const m4max32 = machineById('macbook-pro-m4-max-32c');
  const ultra = machineById('mac-studio-m5-ultra');
  const spark = machineById('nvidia-dgx-spark');
  const glm = loadModel('glm-5.3-flash');
  const qwen = loadModel('qwen3.8-27b');
  const oss = loadModel('gpt-oss-20b');

  it('the factor file carries the model block; the kind decides which branch runs', () => {
    expect(sm.kind === 'multiplier' || sm.kind === 'floor').toBe(true);
    expect(speedModelKind(floor)).toBe('floor');
    expect(speedModelKind({ ...factors, speed_model: undefined })).toBe('multiplier');
    for (const cls of ['dense', 'moe', 'moe_hybrid', 'moe_latent', 'deepseek_v4']) expect(sm.architecture_cost_ms[cls]).toBeDefined();
    expect(sm.read_factor.gguf_kquant.value).toBeGreaterThanOrEqual(1);
    expect(sm.read_factor.gguf_kquant.value).toBeLessThanOrEqual(1.4);
  });

  it('a chip profile is a share of the spec bandwidth, so it transfers across GPU bins', () => {
    const e = factors.efficiency.by_chip['M4 Max'];
    const top = resolveChipProfile(m4max40, factors)!;
    expect(top.source).toBe('measured');
    expect(top.bEffRatio).toBe(e.b_eff_ratio);
    expect(top.t0Ms).toBe(e.t0_ms);
    expect(top.bEffGbs).toBeCloseTo(e.b_eff_ratio! * 546, 3);
    expect(top.fitN).toBe(e.fit_n);
    const bin = resolveChipProfile(m4max32, factors)!;
    expect(bin.bEffRatio).toBe(top.bEffRatio);
    expect(bin.bEffGbs).toBeCloseTo(e.b_eff_ratio! * 410, 3);
  });

  it('assumed and tier profiles follow the same order as the multiplier lookup', () => {
    const u = resolveChipProfile(ultra, factors)!;
    expect(u.source).toBe('assumed');
    expect(u.bEffRatio).toBe(factors.efficiency.assumed['M5 Ultra'].b_eff_ratio);
    expect(u.bEffGbs).toBeCloseTo(factors.efficiency.assumed['M5 Ultra'].b_eff_ratio! * 1200, 3);
    const m6 = resolveChipProfile(machineById('mac-mini-m6'), factors)!;
    expect(m6.source).toBe('assumed');
    const custom: Machine = { id: 'x', kind: 'mac', family: 'Custom', chip: 'M9 Pro', memory_options_gb: [64], bandwidth_gbs: 400, platform: 'apple', price_usd: null, custom: true };
    const t = resolveChipProfile(custom, factors)!;
    expect(t.source).toBe('tier');
    expect(t.bEffRatio).toBe(factors.efficiency.by_tier.pro.b_eff_ratio);
    expect(resolveChipProfile({ ...custom, platform: 'rocm', chip: 'Ryzen AI Max+ 395' }, factors)).toBeNull();
    const s = resolveChipProfile(spark, factors)!;
    expect(s.source).toBe('measured');
    expect(s.key).toBe('DGX Spark');
  });

  it('architecture classes: dense, MoE, hybrid MoE, latent-attention MoE and the DeepSeek override', () => {
    expect(archClassOf(qwen)).toBe('dense');
    expect(archClassOf(oss)).toBe('moe');
    expect(archClassOf(loadModel('qwen3.8-flash-next'))).toBe('moe_hybrid');
    expect(archClassOf(glm)).toBe('moe_latent');
    expect(archClassOf(loadModel('glm-5.2'))).toBe('moe_latent');
    expect(archClassOf(loadModel('kimi-k3'))).toBe('moe_latent');
    expect(archClassOf(loadModel('deepseek-v4-flash-0731'))).toBe('deepseek_v4');
    expect(archClassOf(loadModel('gemma-4-26b-a4b'))).toBe('moe');
    expect(archClassOf({ architecture: null })).toBe('dense');
  });

  it('quant families and MLX groups', () => {
    expect(quantFamilyOf(quantByLabel(glm, 'UD-Q4_K_XL'))).toBe('gguf_kquant');
    expect(quantFamilyOf(quantByLabel(glm, 'Q8_0'))).toBe('q8_0');
    expect(quantFamilyOf(quantByLabel(qwen, 'Q4_0'))).toBe('q4_0');
    expect(quantFamilyOf(quantByLabel(qwen, 'MLX-4bit'))).toBe('mlx');
    expect(quantFamilyOf(quantByLabel(oss, 'MXFP4'))).toBe('mxfp4');
    expect(mlxGroupOf('dense', 27, m4max40)).toBe('dense');
    expect(mlxGroupOf('moe', 3.6, m4max40)).toBe('small_active_moe_m4plus');
    expect(mlxGroupOf('moe', 3.6, machineById('macbook-pro-m1-max'))).toBe('small_active_moe_pre_m4');
    expect(mlxGroupOf('moe', 3.6, spark)).toBe('small_active_moe_pre_m4');
    expect(mlxGroupOf('moe_latent', 18, m4max40)).toBe('moe_other');
  });

  it('GLM-5.3 Flash Q4 on the M5 Ultra: a read at the effective bandwidth plus the fixed costs', () => {
    const q4 = quantByLabel(glm, 'UD-Q4_K_XL');
    const s = estimateSpeed({ model: glm, quant: q4, machine: ultra, contextTokens: 8192, kvBits: 16, factors: floor });
    expect(s.modelKind).toBe('floor');
    expect(s.efficiency).toBeNull();
    expect(s.efficiencySource).toBe('assumed');
    expect(s.archClass).toBe('moe_latent');
    const p = resolveChipProfile(ultra, factors)!;
    const parts = floorMs(p, 1200, s.gbPerToken, 'gguf_kquant', 'moe_latent', 'gguf', 'moe_other', 'mla', 8192, sm);
    expect(s.tokS).toBeCloseTo(1000 / parts.ms, 6);
    expect(s.readMs).toBeCloseTo(parts.readMs, 6);
    expect(s.archCostMs).toBe(sm.architecture_cost_ms.moe_latent.value);
    expect(s.t0Ms).toBe(p.t0Ms);
    expect(s.bEffGbs).toBeCloseTo(p.bEffGbs, 6);
    expect(s.tokS!).toBeGreaterThan(25);
    expect(s.tokS!).toBeLessThan(45);
    expect(s.notes.join(' ')).toMatch(/no measured rows/);
  });

  it('the K-quant read factor and the context term move the milliseconds the way the fit says', () => {
    const p = resolveChipProfile(m4max40, factors)!;
    const plain = floorMs(p, 546, 10, 'q4_0', 'dense', 'gguf', 'dense', 'gqa', 0, sm);
    const kq = floorMs(p, 546, 10, 'gguf_kquant', 'dense', 'gguf', 'dense', 'gqa', 0, sm);
    expect(kq.readMs / plain.readMs).toBeCloseTo(sm.read_factor.gguf_kquant.value, 6);
    // the K-quant penalty is a Metal measurement; the Spark reads K-quants at the plain rate until someone measures one
    expect(sm.read_factor.gguf_kquant.platforms).toEqual(['apple']);
    const cudaKq = floorMs(p, 546, 10, 'gguf_kquant', 'dense', 'gguf', 'dense', 'gqa', 0, sm, 'cuda');
    expect(cudaKq.readMs).toBeCloseTo(plain.readMs, 6);
    expect(plain.fixedMs).toBeCloseTo(p.t0Ms, 6);
    expect(plain.attnMs).toBe(0);
    const latent = floorMs(p, 546, 10, 'q4_0', 'moe_latent', 'gguf', 'moe_other', 'mla', 32768, sm);
    expect(latent.attnMs).toBeCloseTo(sm.attention_ms_per_32k.mla.value, 6);
    expect(latent.fixedMs).toBeCloseTo(p.t0Ms + sm.architecture_cost_ms.moe_latent.value, 6);
    const far = estimateSpeed({ model: glm, quant: quantByLabel(glm, 'UD-Q4_K_XL'), machine: ultra, contextTokens: 131072, kvBits: 16, factors: floor });
    const near = estimateSpeed({ model: glm, quant: quantByLabel(glm, 'UD-Q4_K_XL'), machine: ultra, contextTokens: 8192, kvBits: 16, factors: floor });
    expect(far.tokS!).toBeLessThan(near.tokS!);
  });

  it('MLX scales the fixed cost, never the read; off Apple silicon the GGUF path runs', () => {
    const mlx4 = quantByLabel(qwen, 'MLX-4bit');
    const s = estimateSpeed({ model: qwen, quant: mlx4, machine: m4max40, contextTokens: 4096, kvBits: 16, factors: floor });
    expect(s.runtimeApplied).toBe('mlx');
    expect(s.runtimeFactor).toBe(sm.mlx.overhead_factor.dense.value);
    const p = resolveChipProfile(m4max40, factors)!;
    const parts = floorMs(p, 546, s.gbPerToken, 'mlx', 'dense', 'mlx', 'dense', 'gqa', 4096, sm);
    expect(s.tokS).toBeCloseTo(1000 / parts.ms, 6);
    expect(parts.readMs).toBeCloseTo((s.gbPerToken / p.bEffGbs) * 1000, 6);
    expect(parts.attnMs).toBeCloseTo(sm.mlx.attention_ms_per_32k.pre_m5.value * (4096 / 32768), 6);
    const onSpark = estimateSpeed({ model: qwen, quant: mlx4, machine: spark, contextTokens: 4096, kvBits: 16, factors: floor });
    expect(onSpark.runtimeApplied).toBe('gguf');
    expect(onSpark.runtimeFactor).toBe(1);
    expect(onSpark.notes.join(' ')).toMatch(/MLX runs only on Apple silicon/);
    const small = estimateSpeed({ model: oss, quant: quantByLabel(oss, 'MLX-8bit'), machine: m4max40, contextTokens: 4096, kvBits: 16, factors: floor });
    expect(small.runtimeFactor).toBe(sm.mlx.overhead_factor.small_active_moe_m4plus.value);
  });

  it('the M5 generation pays its own MLX long-context cost: the M5 Max ladder is met, and the Ultra keeps its bandwidth lead at 256K', () => {
    expect(mlxContextGroupOf({ chip: 'M5 Max (40-core GPU)' })).toBe('m5plus');
    expect(mlxContextGroupOf({ chip: 'M6' })).toBe('m5plus');
    expect(mlxContextGroupOf({ chip: 'M4 Max' })).toBe('pre_m5');
    expect(mlxContextGroupOf({ chip: 'Ryzen AI Max+ 395' })).toBe('pre_m5');
    expect(sm.mlx.attention_ms_per_32k.m5plus.value).toBeLessThan(sm.mlx.attention_ms_per_32k.pre_m5.value);
    const m5max = machineById('macbook-pro-m5-max-40c');
    const mlx4 = quantByLabel(qwen, 'MLX-4bit');
    // the oMLX M5 Max ladder (accelerations off): 28.6 tok/s at 32K, 23.9 at 64K
    const at32 = estimateSpeed({ model: qwen, quant: mlx4, machine: m5max, contextTokens: 32768, kvBits: 16, factors: floor });
    const at64 = estimateSpeed({ model: qwen, quant: mlx4, machine: m5max, contextTokens: 65536, kvBits: 16, factors: floor });
    expect(at32.tokS!).toBeGreaterThan(25);
    expect(at32.tokS!).toBeLessThan(31);
    expect(at64.tokS!).toBeGreaterThan(20);
    expect(at64.tokS!).toBeLessThan(27);
    expect(at64.attnMs).toBeCloseTo(sm.mlx.attention_ms_per_32k.m5plus.value * 2, 6);
    // the same file on the M4 Max is charged the pre-M5 cost
    const m4 = estimateSpeed({ model: qwen, quant: mlx4, machine: m4max40, contextTokens: 65536, kvBits: 16, factors: floor });
    expect(m4.attnMs).toBeCloseTo(sm.mlx.attention_ms_per_32k.pre_m5.value * 2, 6);
    // twice the bandwidth stays close to twice the speed at 256K, because the context cost no longer swamps the read
    const mlx8 = quantByLabel(qwen, 'MLX-8bit');
    const maxFar = estimateSpeed({ model: qwen, quant: mlx8, machine: m5max, contextTokens: 262144, kvBits: 16, factors: floor });
    const ultraFar = estimateSpeed({ model: qwen, quant: mlx8, machine: ultra, contextTokens: 262144, kvBits: 16, factors: floor });
    expect(ultraFar.tokS! / maxFar.tokS!).toBeGreaterThan(1.7);
    expect(ultraFar.tokS! / maxFar.tokS!).toBeLessThan(2);
    expect(maxFar.tokS!).toBeGreaterThan(9);
    expect(ultraFar.tokS!).toBeGreaterThan(17);
  });

  it('a chip with no profile gets no number; the multiplier branch is untouched by the block', () => {
    const halo: Machine = { id: 'h', kind: 'pc', family: 'Custom', chip: 'Ryzen AI Max+ 395', memory_options_gb: [128], bandwidth_gbs: 256, platform: 'rocm', price_usd: null, custom: true };
    const s = estimateSpeed({ model: glm, quant: quantByLabel(glm, 'UD-Q4_K_XL'), machine: halo, contextTokens: 8192, kvBits: 16, factors: floor });
    expect(s.tokS).toBeNull();
    expect(s.efficiencySource).toBe('none');
    expect(s.notes.join(' ')).toMatch(/no measured profile/);
    const mult = estimateSpeed({ model: glm, quant: quantByLabel(glm, 'UD-Q4_K_XL'), machine: ultra, contextTokens: 8192, kvBits: 16, factors });
    const legacy = estimateSpeed({ model: glm, quant: quantByLabel(glm, 'UD-Q4_K_XL'), machine: ultra, contextTokens: 8192, kvBits: 16, factors: loadMultiplierFactors() });
    expect(mult.modelKind).toBe(speedModelKind(factors));
    if (speedModelKind(factors) === 'multiplier') expect(mult.tokS).toBeCloseTo(legacy.tokS!, 6);
  });
});
