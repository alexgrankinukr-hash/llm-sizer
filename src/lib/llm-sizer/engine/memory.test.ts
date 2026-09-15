import { describe, expect, it } from 'vitest';
import { availableMemory, breakdown, kvCacheBytes, neededMemory, verdictFor } from './memory';
import { loadModel, machineById, quantByLabel } from './__fixtures__/load';

const GB = 1e9;

describe('availableMemory (METHODOLOGY §4, §7)', () => {
  const mini = machineById('mac-mini-m6');
  const ultra = machineById('mac-studio-m5-ultra');
  const spark = machineById('nvidia-dgx-spark');

  it('32 GB mini: the cap binds with apps off, the apps bind with the work toggle on', () => {
    // 32 GB is 32 GiB = 34.36 decimal GB; two thirds of it is 22.9 GB
    const off = availableMemory(mini, { memoryGb: 32, workApps: 0, override: false });
    expect(off.availableGb).toBeCloseTo(22.91, 2);
    expect(off.binding).toBe('cap');
    expect(off.capReserveGb).toBeCloseTo(34.36 - 22.91 - 6, 2);
    const on = availableMemory(mini, { memoryGb: 32, workApps: 16, override: false });
    expect(on.availableGb).toBeCloseTo(12.36, 2);
    expect(on.binding).toBe('apps');
    expect(on.capReserveGb).toBe(0);
  });

  it('256 GB Studio: 206 by default (three quarters of 274.9), 259.8 with the override (macOS keeps 5.5 %); 512 GB → 519.5', () => {
    expect(availableMemory(ultra, { memoryGb: 256, workApps: 0, override: false }).availableGb).toBeCloseTo(206.16, 2);
    expect(availableMemory(ultra, { memoryGb: 256, workApps: 0, override: 1 }).availableGb).toBeCloseTo(259.76, 2);
    expect(availableMemory(ultra, { memoryGb: 512, workApps: 0, override: 1 }).availableGb).toBeCloseTo(519.52, 2);
    expect(availableMemory(ultra, { memoryGb: 256, workApps: 16, override: 1 }).availableGb).toBeCloseTo(243.76, 2);
  });

  it('the GPU limit is two thirds under 36 GB and three quarters from 36 GB', () => {
    const at = (gb: number) => availableMemory(mini, { memoryGb: gb, workApps: 0, override: false });
    expect(at(24).availableGb).toBeCloseTo(24 * 1.073741824 * (2 / 3), 6);
    expect(at(36).availableGb).toBeCloseTo(36 * 1.073741824 * 0.75, 6);
    expect(at(48).availableGb).toBeCloseTo(38.65, 2);
  });

  it('the override never lifts the OS + apps floor', () => {
    const a = availableMemory(mini, { memoryGb: 32, workApps: 16, override: 1 });
    expect(a.availableGb).toBeCloseTo(12.36, 2);
  });

  it('prebuilt unified boxes reserve 8 GB and have no cap', () => {
    const a = availableMemory(spark, { memoryGb: 128, workApps: 0, override: false });
    expect(a.availableGb).toBeCloseTo(129.44, 2);
    expect(a.capPct).toBeNull();
    expect(availableMemory(spark, { memoryGb: 128, workApps: 16, override: false }).availableGb).toBeCloseTo(113.44, 2);
  });
});

describe('kvCacheBytes mirrors arch.py', () => {
  it('Qwen 3.8 27B: 32 KB per token on 16 of 64 layers plus a constant linear state', () => {
    const a = loadModel('qwen3.8-27b').architecture!;
    expect(a.full_bytes_per_token_8bit).toBe(32768);
    expect(kvCacheBytes(a, 32768, 8) / GB).toBeCloseTo(1.149, 2);
    expect(kvCacheBytes(a, 131072, 8) / GB).toBeCloseTo(4.37, 2);
    // 4-bit halves the growing term only; the linear state is fp16 and stays
    expect(kvCacheBytes(a, 131072, 4)).toBe(2147483648 + a.linear_state_bytes);
    expect(kvCacheBytes(a, 131072, 16)).toBe(8589934592 + a.linear_state_bytes);
  });

  it('GLM-5.2 (MLA): ≈ 5.9 GB at 128K; GLM-5.3 Flash ≈ 0.8 GB', () => {
    expect(kvCacheBytes(loadModel('glm-5.2').architecture, 131072, 8) / GB).toBeCloseTo(5.89, 2);
    expect(kvCacheBytes(loadModel('glm-5.3-flash').architecture, 131072, 8) / GB).toBeCloseTo(0.81, 2);
  });

  it('Gemma 4 26B-A4B: 5 global layers grow, 25 sliding layers stop at the 1,024-token window; K and V both stored', () => {
    const a = loadModel('gemma-4-26b-a4b').architecture!;
    expect(kvCacheBytes(a, 131072, 8)).toBe(1447034880);
    expect(kvCacheBytes(a, 131072, 4)).toBe(723517440);
    expect(kvCacheBytes(a, 131072, 16)).toBe(2894069760);
    expect(kvCacheBytes(a, 512, 8)).toBe(10240 * 512 + 102400 * 512);
  });

  it('no architecture → no cache', () => {
    expect(kvCacheBytes(null, 131072, 8)).toBe(0);
  });
});

describe('neededMemory + verdict (METHODOLOGY §5, §9)', () => {
  it('example A: Qwen 27B Q4 on a 32 GB mini, 32K, FP16 cache', () => {
    const model = loadModel('qwen3.8-27b');
    const q4 = quantByLabel(model, 'Q4_K_M', 'lmstudio-community/Qwen3.8-27B-GGUF');
    expect(q4.size_gb).toBeCloseTo(16.81, 2);
    const off = neededMemory(q4, model.architecture, 32768, 16, 22.91);
    expect(off.totalGb).toBeCloseTo(20.70, 2);
    expect(off.cacheGb).toBeCloseTo(2.22, 2);
    expect(off.buffersGb).toBeCloseTo(1.668, 3); // 1.5 GB + 1 % of 16.8
    expect(off.ratio).toBeCloseTo(0.9036, 3);
    expect(verdictFor(off)).toBe('tight');
    const on = neededMemory(q4, model.architecture, 32768, 16, 12.36);
    expect(verdictFor(on)).toBe('no');
  });

  it('example B: GLM-5.3 Flash Q4 at 128K needs ≈ 204.8 GB; tight at 99 % without the override, runs with it', () => {
    const model = loadModel('glm-5.3-flash');
    const q4 = quantByLabel(model, 'UD-Q4_K_XL');
    const need = neededMemory(q4, model.architecture, 131072, 16, 206.16);
    expect(need.totalGb).toBeCloseTo(204.75, 1);
    expect(need.buffersGb).toBeCloseTo(3.50, 2);
    expect(verdictFor(need)).toBe('tight');
    const withOverride = neededMemory(q4, model.architecture, 131072, 16, 259.76);
    expect(withOverride.ratio).toBeCloseTo(0.788, 3);
    expect(verdictFor(withOverride)).toBe('runs');
    expect(neededMemory(q4, model.architecture, 131072, 16, 243.76).ratio).toBeCloseTo(0.840, 3); // with the apps budget
  });

  it('example C: GLM-5.2 Q4 at 128K on 512 GB with the override is tight (buffers 1.5 GB + 1 %, macOS keeps 30 GB)', () => {
    const model = loadModel('glm-5.2');
    const q4 = quantByLabel(model, 'UD-Q4_K_XL');
    const need = neededMemory(q4, model.architecture, 131072, 16, 519.52);
    expect(need.buffersGb).toBeCloseTo(6.173, 3);
    expect(need.cacheGb).toBeCloseTo(11.78, 2);
    expect(need.totalGb).toBeCloseTo(485.2, 1);
    expect(need.ratio).toBeCloseTo(0.9340, 3);
    expect(verdictFor(need)).toBe('tight');
  });

  it('a special build counts its resident size, with buffers on that size', () => {
    const model = loadModel('qwen3.8-flash-next');
    const build = model.special_builds![0];
    const need = neededMemory(build, model.architecture, 32768, 16, 48);
    expect(need.weightsGb).toBe(45.8);
    expect(need.buffersGb).toBeCloseTo(1.958, 3);
    expect(need.totalGb).toBeCloseTo(48.62, 2);
    expect(need.ssdGb).toBe(39.1); // the n-gram table on the SSD: beside the total, never in it
    expect(verdictFor(need)).toBe('no');
    expect(verdictFor(neededMemory(build, model.architecture, 32768, 16, 58))).toBe('runs');
    expect(neededMemory(model.quants[0], model.architecture, 32768, 16, 58).ssdGb).toBe(0);
  });

  it('breakdown sums to RAM when the model fits and reports the excess when it does not', () => {
    const mini = machineById('mac-mini-m6');
    const model = loadModel('qwen3.8-27b');
    const q4 = quantByLabel(model, 'Q4_K_M', 'lmstudio-community/Qwen3.8-27B-GGUF');
    const av = availableMemory(mini, { memoryGb: 32, workApps: 0, override: false });
    const need = neededMemory(q4, model.architecture, 32768, 16, av.availableGb);
    const b = breakdown(32, av, need);
    // the bar is in decimal GB: a 32 GB (GiB) machine is 34.36 GB
    expect(b.ramGb).toBeCloseTo(34.36, 2);
    expect(b.os + b.apps + b.capReserve + b.weights + b.cache + b.buffers + b.free).toBeCloseTo(b.ramGb, 6);
    expect(b.over).toBe(0);
    const tight = availableMemory(mini, { memoryGb: 32, workApps: 16, override: false });
    const b2 = breakdown(32, tight, neededMemory(q4, model.architecture, 32768, 16, tight.availableGb));
    expect(b2.over).toBeCloseTo(8.34, 2);
    expect(b2.free).toBe(0);
  });
});

describe('memory policy margins (METHODOLOGY §4, §5): explicit policies, dormant defaults', async () => {
  const { buffersGb, scratchGb } = await import('./memory');
  const { MEMORY_POLICY, MEMORY_POLICY_V08 } = await import('./constants');
  const qwen = loadModel('qwen3.8-27b');
  const glm = loadModel('glm-5.2');
  const mlx4 = quantByLabel(qwen, 'MLX-4bit');
  const q4 = quantByLabel(qwen, 'Q4_0');
  const next = { ...MEMORY_POLICY_V08, bufferMinGb: 0, bufferBaseGb: 1.5, bufferPct: 0.01, bufferCapGb: null, mlxScratchGbPer1k: 0.15, osReservePctOnOverride: 0.055 };

  it('buffers: 1 GB or 5 % capped at 16 GB under the 0.8 policy; 1.5 GB plus 1 % of the weights under the next one', () => {
    expect(buffersGb(16.1, MEMORY_POLICY_V08)).toBeCloseTo(1, 6);
    expect(buffersGb(100, MEMORY_POLICY_V08)).toBeCloseTo(5, 6);
    expect(buffersGb(500, MEMORY_POLICY_V08)).toBeCloseTo(16, 6);
    expect(buffersGb(16.1, next)).toBeCloseTo(1.661, 3);
    expect(buffersGb(470, next)).toBeCloseTo(6.2, 6);
    expect(buffersGb(0, next)).toBeCloseTo(1.5, 6);
  });

  it('MLX prompt scratch grows above 8K of context and is zero for GGUF files', () => {
    expect(scratchGb(mlx4, 32768, next)).toBeCloseTo(3.6, 6);
    expect(scratchGb(mlx4, 131072, next)).toBeCloseTo(18, 6);
    expect(scratchGb(mlx4, 8192, next)).toBe(0);
    expect(scratchGb(q4, 131072, next)).toBe(0);
    expect(scratchGb(mlx4, 131072, MEMORY_POLICY_V08)).toBe(0);
    const need = neededMemory(mlx4, qwen.architecture, 32768, 16, 100, next);
    expect(need.scratchGb).toBeCloseTo(3.6, 6);
    expect(need.totalGb).toBeCloseTo(need.weightsGb + need.cacheGb + need.buffersGb + need.scratchGb, 9);
    const v08 = neededMemory(mlx4, qwen.architecture, 32768, 16, 100, MEMORY_POLICY_V08);
    expect(v08.scratchGb).toBe(0);
    expect(breakdown(64, availableMemory(machineById('macbook-pro-m4-max-40c'), { memoryGb: 64, workApps: 0, override: false }), need).scratch).toBeCloseTo(3.6, 6);
  });

  it('with the GPU limit lifted, macOS keeps the larger of 6 GB and its share of the memory', () => {
    const ultra = machineById('mac-studio-m5-ultra');
    const lifted = availableMemory(ultra, { memoryGb: 512, workApps: 0, override: 1 }, next);
    expect(lifted.osGb).toBeCloseTo(0.055 * 549.76, 1);
    expect(lifted.availableGb).toBeCloseTo(549.76 - 0.055 * 549.76, 1);
    expect(availableMemory(ultra, { memoryGb: 512, workApps: 0, override: 1 }, MEMORY_POLICY_V08).osGb).toBe(6);
    // the default GPU limit already holds back a quarter, so the base reserve applies there
    expect(availableMemory(ultra, { memoryGb: 512, workApps: 0, override: false }, next).osGb).toBe(6);
    expect(availableMemory(machineById('mac-mini-m6'), { memoryGb: 32, workApps: 0, override: 1 }, next).osGb).toBe(6);
  });

  it('the policy in force is the one the tests above pin', () => {
    expect(MEMORY_POLICY).toEqual(next);
    expect(neededMemory(glm.quants[0], glm.architecture, 8192, 16, 100).scratchGb).toBe(0);
  });
});
