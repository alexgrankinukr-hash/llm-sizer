import { describe, expect, it } from 'vitest';
import { clusterMs, clusterOf, dispatchCaveat, linkClassOf, prefillClusterFactor, tensorAllowed, tensorCapable } from './cluster';
import { availableMemory, breakdown, neededMemory } from './memory';
import { evaluateCell } from './index';
import { loadFactors, loadModel, machineById } from './__fixtures__/load';
import type { ClusterFactors, Factors, Settings } from './types';

/** A hand-built cluster block, so these tests do not move when calibrate.py refits the published series. */
const HAND: ClusterFactors = {
  hop_ms: { mac: { value: 4.0, n: 4, source: 'fitted', range: [2.2, 9.1] }, spark: { value: 3.9, n: 2, source: 'fitted', range: [3.7, 4.2] }, gpu: { value: 1.0, n: 0, source: 'assumed' } },
  tensor_c_ms: { dense: { value: 4.5, n: 4, source: 'fitted', range: [3.9, 5.2] }, moe: { value: 10.9, n: 5, source: 'fitted', range: [9.3, 12.5] }, gpu: { value: 1.5, n: 0, source: 'assumed' } },
  prefill: { layer_factor: { value: 1.0, n: 1, source: 'assumed' }, tensor_exponent: { value: 0.75, n: 0, source: 'assumed' } },
  dispatch_caveat: { min_layers: 60 },
};
const base = loadFactors();
const factors: Factors = { ...base, cluster: HAND };
const withoutCluster: Factors = { ...base, cluster: undefined };

const s = (memoryGb: number, extra: Partial<Settings> = {}): Settings => ({ memoryGb, workApps: 0, override: false, contextTokens: 32768, kvBits: 16, runtime: 'gguf', qualityFloorBits: 2, ...extra });

describe('the cluster helpers (METHODOLOGY §4, §6.1)', () => {
  it('clusterOf: one machine is null, counts clamp to four, the split defaults to layer', () => {
    expect(clusterOf({})).toBeNull();
    expect(clusterOf({ machines: 1 })).toBeNull();
    expect(clusterOf({ machines: 2 })).toEqual({ n: 2, split: 'layer' });
    expect(clusterOf({ machines: 9, split: 'tensor' })).toEqual({ n: 4, split: 'tensor' });
  });

  it('link class and tensor eligibility follow the catalog', () => {
    expect(linkClassOf(machineById('mac-studio-m5-ultra'))).toBe('mac');
    expect(linkClassOf(machineById('nvidia-dgx-spark'))).toBe('spark');
    expect(linkClassOf({ platform: 'cuda', family: 'RTX PRO 6000 box', chip: 'RTX PRO 6000' })).toBe('gpu');
    expect(tensorCapable(machineById('mac-studio-m5-ultra'))).toBe(true);
    expect(tensorCapable(machineById('mac-mini-m6'))).toBe(false);
    expect(tensorCapable(machineById('nvidia-dgx-spark'))).toBe(true);
    // a custom Apple machine has no interconnect field: the chip rule decides
    expect(tensorCapable({ platform: 'apple', chip: 'M4 Pro' })).toBe(true);
    expect(tensorCapable({ platform: 'apple', chip: 'M3 Ultra' })).toBe(true);
    expect(tensorCapable({ platform: 'apple', chip: 'M2 Ultra' })).toBe(false);
    expect(tensorAllowed(2)).toBe(true);
    expect(tensorAllowed(3)).toBe(false);
    expect(tensorAllowed(4)).toBe(true);
  });

  it('clusterMs: a layer split adds a hop per extra machine, tensor parallel divides and adds C × log2 N', () => {
    expect(clusterMs(60, 2, 'layer', 'spark', 'moe', HAND).ms).toBeCloseTo(63.9, 6);
    expect(clusterMs(60, 4, 'layer', 'mac', 'dense', HAND).ms).toBeCloseTo(72, 6);
    expect(clusterMs(60, 2, 'tensor', 'mac', 'dense', HAND).ms).toBeCloseTo(34.5, 6);
    expect(clusterMs(60, 4, 'tensor', 'spark', 'moe', HAND).ms).toBeCloseTo(15 + 21.8, 6);
    expect(clusterMs(60, 2, 'tensor', 'gpu', 'moe', HAND).cMs).toBe(1.5);
    expect(prefillClusterFactor(2, 'layer', HAND).factor).toBe(1);
    expect(prefillClusterFactor(2, 'tensor', HAND).factor).toBeCloseTo(Math.pow(2, 0.75), 9);
  });

  it('the dispatch caveat: tensor parallel on a deep mixture of experts only', () => {
    expect(dispatchCaveat('moe_latent', 93, 'tensor', HAND)).toBe(true);
    expect(dispatchCaveat('moe_latent', 93, 'layer', HAND)).toBe(false);
    expect(dispatchCaveat('dense', 93, 'tensor', HAND)).toBe(false);
    expect(dispatchCaveat('moe', 48, 'tensor', HAND)).toBe(false);
  });
});

describe('pooled memory (METHODOLOGY §4, §5)', () => {
  const spark = machineById('nvidia-dgx-spark');
  const studio = machineById('mac-studio-m5-max-40c');

  it('two Sparks: each keeps its 8 GB, the link takes 2 GB each, the apps reserve is taken once', () => {
    const one = availableMemory(spark, s(128));
    const two = availableMemory(spark, s(128, { machines: 2 }));
    expect(one.availableGb).toBeCloseTo(129.44, 2);
    expect(two.availableGb).toBeCloseTo(254.88, 2);
    expect(two.osGb).toBe(16);
    expect(two.clusterGb).toBe(4);
    expect(two.machines).toBe(2);
    expect(availableMemory(spark, s(128, { machines: 2, workApps: 16 })).availableGb).toBeCloseTo(238.88, 2);
  });

  it('two 128 GB M5 Max Studios: the GPU limit is judged per machine, and the override lifts each', () => {
    const two = availableMemory(studio, s(128, { machines: 2 }));
    expect(two.availableGb).toBeCloseTo(202.16, 2);
    expect(two.binding).toBe('cap');
    const lifted = availableMemory(studio, s(128, { machines: 2, override: 1 }));
    expect(lifted.availableGb).toBeCloseTo(255.76, 2);
    // 2 × 24 GB stays at two thirds each: the threshold compares the machine, not the pool
    const minis = availableMemory(machineById('mac-mini-m5-pro'), s(24, { machines: 2 }));
    expect(minis.capPct).toBeCloseTo(2 / 3, 9);
    expect(minis.availableGb).toBeCloseTo(2 * 24 * 1.073741824 * (2 / 3) - 4, 2);
  });

  it('a single machine is byte-identical to before', () => {
    const a = availableMemory(studio, s(128));
    const b = availableMemory(studio, s(128, { machines: 1 }));
    expect(b).toEqual({ ...a, machines: 1, clusterGb: 0 });
    expect(a.machines).toBe(1);
    expect(a.clusterGb).toBe(0);
  });

  it('needs across the pool: looser shards, buffers per machine, the cache duplicated only under tensor parallel on latent attention', () => {
    const glm = loadModel('glm-5.3-flash');
    const q = glm.quants.find((x) => x.label === 'UD-Q4_K_XL')!;
    const one = neededMemory(q, glm.architecture, 32768, 16, 254.88);
    const layer = neededMemory(q, glm.architecture, 32768, 16, 254.88, undefined, { n: 2, split: 'layer' });
    const tensor = neededMemory(q, glm.architecture, 32768, 16, 254.88, undefined, { n: 2, split: 'tensor' });
    expect(layer.weightsGb).toBeCloseTo(one.weightsGb * 1.05, 6);
    expect(tensor.weightsGb).toBeCloseTo(one.weightsGb * 1.08, 6);
    expect(layer.cacheGb).toBeCloseTo(one.cacheGb, 9);
    expect(tensor.cacheGb).toBeCloseTo(one.cacheGb * 2, 9);
    expect(layer.buffersGb).toBeCloseTo(1.5 + 0.01 * layer.weightsGb + 1.5, 6);
    expect(layer.totalGb).toBeGreaterThan(one.totalGb);
    const qwen = loadModel('qwen3.8-27b');
    const q4 = qwen.quants.find((x) => x.label === 'Q4_K_M')!;
    const dense = neededMemory(q4, qwen.architecture, 32768, 16, 100, undefined, { n: 2, split: 'tensor' });
    expect(dense.cacheGb).toBeCloseTo(neededMemory(q4, qwen.architecture, 32768, 16, 100).cacheGb, 9); // grouped-query: the shards sum to one cache
  });

  it('the breakdown bar is the pool', () => {
    const av = availableMemory(studio, s(128, { machines: 2 }));
    const need = neededMemory(loadModel('qwen3.8-27b').quants[0], loadModel('qwen3.8-27b').architecture, 32768, 16, av.availableGb, undefined, { n: 2, split: 'layer' });
    const b = breakdown(128, av, need);
    expect(b.ramGb).toBeCloseTo(2 * 128 * 1.073741824, 6);
    expect(b.machines).toBe(2);
    expect(b.cluster).toBe(4);
    expect(b.os + b.apps + b.capReserve + b.cluster + b.weights + b.cache + b.buffers + b.scratch + b.free).toBeCloseTo(b.ramGb, 3);
  });
});

describe('worked example H: linked machines end to end', () => {
  const spark = machineById('nvidia-dgx-spark');
  const studio = machineById('mac-studio-m5-max-40c');
  const glm = loadModel('glm-5.3-flash');

  it('H1: GLM-5.3 Flash Q4 at 32K runs on two Sparks, on a layer split at one Spark\'s time plus the hop, in tensor parallel at half plus C', () => {
    const one = evaluateCell(glm, spark, s(128), factors);
    expect(one.verdict).toBe('compromise'); // the 3-bit file on one Spark, as before
    const layer = evaluateCell(glm, spark, s(128, { machines: 2 }), factors);
    expect(layer.verdict).toBe('runs');
    expect(layer.need.ratio).toBeGreaterThan(0.8);
    expect(layer.need.ratio).toBeLessThan(0.9);
    expect(layer.flags).toContain('linked-layer');
    expect(layer.flags).not.toContain('linked-dispatch');
    expect(layer.speed.cluster?.split).toBe('layer');
    expect(layer.speed.cluster?.link).toBe('spark');
    expect(layer.speed.cluster?.source).toBe('fitted');
    const single = layer.speed.cluster!.singleMs;
    expect(layer.speed.tokS!).toBeCloseTo(1000 / (single + 3.9), 6);
    expect(layer.speed.tokS!).toBeGreaterThan(12);
    expect(layer.speed.tokS!).toBeLessThan(18);
    expect(layer.prefill?.parts?.cluster).toBe(1);
    const tensor = evaluateCell(glm, spark, s(128, { machines: 2, split: 'tensor' }), factors);
    expect(tensor.verdict).toBe('runs');
    expect(tensor.flags).toContain('linked-tensor');
    expect(tensor.speed.tokS!).toBeCloseTo(1000 / (single / 2 + 10.9), 6);
    expect(tensor.speed.tokS!).toBeGreaterThan(layer.speed.tokS!);
    expect(tensor.prefill?.parts?.cluster).toBeCloseTo(Math.pow(2, 0.75), 9);
    expect(tensor.prefill!.tokS!).toBeCloseTo(layer.prefill!.tokS! * Math.pow(2, 0.75), 6);
    expect(tensor.speed.notes.some((n) => /tensor parallel/.test(n))).toBe(true);
  });

  it('H2: two 128 GB M5 Max Studios need the override on a layer split; the fix carries the pool', () => {
    const cell = evaluateCell(glm, studio, s(128, { machines: 2 }), factors);
    expect(cell.verdict).toBe('compromise');
    expect(cell.fix?.changes.map((c) => c.kind)).toEqual(['override']);
    expect(cell.fix?.availability.machines).toBe(2);
    expect(cell.fix?.availability.availableGb).toBeCloseTo(255.76, 2);
    expect(cell.fix?.need.ratio).toBeGreaterThan(0.8);
    expect(cell.fix?.need.ratio).toBeLessThan(0.9);
    expect(cell.speed.cluster?.link).toBe('mac');
    expect(cell.breakdown.machines).toBe(2);
  });

  it('H3: the everyday model on two 24 GB minis runs, slower than one 48 GB mini writes it', () => {
    const qwen = loadModel('qwen3.8-27b');
    const mini = machineById('mac-mini-m5-pro');
    const pair = evaluateCell(qwen, mini, s(24, { machines: 2 }), factors);
    expect(pair.verdict).toBe('runs');
    const one48 = evaluateCell(qwen, mini, s(48), factors);
    expect(one48.verdict).toBe('runs');
    expect(pair.speed.tokS!).toBeLessThan(one48.speed.tokS!);
    expect(pair.speed.notes.some((n) => /fits one machine writes slower/.test(n))).toBe(true);
    const tensor = evaluateCell(qwen, mini, s(24, { machines: 2, split: 'tensor' }), factors);
    expect(tensor.speed.tokS!).toBeGreaterThan(one48.speed.tokS!);
  });

  it('the dispatch caveat on Kimi K3 in tensor parallel, and no speed without cluster terms', () => {
    const kimi = loadModel('kimi-k3');
    const ultra = machineById('mac-studio-m5-ultra');
    const t = evaluateCell(kimi, ultra, s(512, { machines: 4, split: 'tensor', override: 1 }), factors);
    expect(t.flags).toContain('linked-dispatch');
    const l = evaluateCell(kimi, ultra, s(512, { machines: 4, split: 'layer', override: 1 }), factors);
    expect(l.flags).not.toContain('linked-dispatch');
    const none = evaluateCell(glm, spark, s(128, { machines: 2 }), withoutCluster);
    expect(none.verdict).toBe('runs');
    expect(none.speed.tokS).toBeNull();
    expect(none.speed.notes).toContain('no linked-machine terms in this data set');
  });
});
