import { describe, expect, it } from 'vitest';
import { loadMachines } from '../engine/__fixtures__/load';
import { customGroup, findRow, groupLabel, groupTensorCapable, linkedLabel, machineGroups, parseRowKey, pooledGb, pooledPrice, rowKey, rowLabel, slimFactors, slimMachine } from './machines';

const groups = machineGroups(loadMachines());

describe('machine groups', () => {
  it('merges bandwidth bins into one group per family × chip', () => {
    const mini = groups.find((g) => g.id === 'mac-mini-m6')!;
    expect(mini.rows.map((r) => [r.gb, r.machineId])).toEqual([[16, 'mac-mini-m6-16gb'], [24, 'mac-mini-m6'], [32, 'mac-mini-m6']]);
    expect(mini.rows[0].bandwidthGbs).toBe(153);
    expect(mini.rows[2].bandwidthGbs).toBe(170);
    expect(mini.rows[0].priceUsd).toBe(899);
    expect(groupLabel(mini)).toBe('Mac mini · M6');
    const max = groups.find((g) => g.id === 'macbook-pro-m5-max')!;
    expect(max.rows.map((r) => [r.gb, r.machineId])).toEqual([[36, 'macbook-pro-m5-max-32c'], [48, 'macbook-pro-m5-max-40c'], [64, 'macbook-pro-m5-max-40c'], [128, 'macbook-pro-m5-max-40c']]);
    expect(max.rows.every((r) => !r.binLabel)).toBe(true);
  });

  it('keeps one-row groups for the Spark, sets status from the rows, and preserves catalog order', () => {
    const spark = groups.find((g) => g.id === 'nvidia-dgx-spark')!;
    expect(spark.rows).toHaveLength(1);
    expect(spark.silhouette).toBe('spark');
    expect(spark.status).toBe('current');
    expect(groups.find((g) => g.id === 'macbook-air-m1')?.status).toBe('discontinued');
    expect(groups[0].id).toBe('macbook-air-m1');
    expect(groups.length).toBeGreaterThan(30);
    expect(groups.length).toBeLessThan(loadMachines().length);
  });

  it('labels duplicate sizes with the bin', () => {
    const catalog = loadMachines().filter((m) => m.id.startsWith('macbook-pro-m5-max'));
    const twin = { ...catalog[1], memory_options_gb: [36, 48] };
    const [g] = machineGroups([catalog[0], twin]);
    const thirtySix = g.rows.filter((r) => r.gb === 36);
    expect(thirtySix).toHaveLength(2);
    expect(thirtySix.map((r) => r.binLabel)).toEqual(['32-core GPU', '40-core GPU']);
  });

  it('row keys round-trip and custom machines become one-row groups', () => {
    const key = rowKey('mac-mini-m6', 32);
    expect(parseRowKey(key)).toEqual({ groupId: 'mac-mini-m6', gb: 32, linked: null });
    expect(findRow(groups, key)?.row.machineId).toBe('mac-mini-m6');
    expect(findRow(groups, 'nope:1')).toBeNull();
    const g = customGroup({ name: 'My Halo box', memoryGb: 128, bandwidthGbs: 256, platform: 'rocm' });
    expect(g.id).toBe('custom-my-halo-box');
    expect(g.rows[0].gb).toBe(128);
    expect(g.custom).toBe(true);
  });
});

describe('slim island props', () => {
  it('groups and cells are identical with the slimmed machine rows and factors', async () => {
    const { loadFactors, loadMachines, loadModel } = await import('../engine/__fixtures__/load');
    const { evaluateCell } = await import('../engine/index');
    const full = loadMachines();
    const slim = full.map(slimMachine);
    expect(JSON.stringify(slim).includes('"sources"')).toBe(false);
    expect(JSON.stringify(slim).includes('"notes"')).toBe(false);
    expect(JSON.stringify(slim).length).toBeLessThan(JSON.stringify(full).length * 0.7);
    const strip = (g: ReturnType<typeof machineGroups>) => g.map((x) => ({ id: x.id, rows: x.rows.map((r) => [r.gb, r.machineId, r.bandwidthGbs, r.priceUsd, r.binLabel]) }));
    expect(strip(machineGroups(slim))).toEqual(strip(machineGroups(full)));
    const factors = loadFactors();
    const lean = slimFactors(factors);
    expect(JSON.stringify(lean).includes('examples')).toBe(false);
    expect(JSON.stringify(lean).includes('"note"')).toBe(false);
    if (factors.speed_model) {
      expect(lean.speed_model?.kind).toBe(factors.speed_model.kind);
      expect(lean.speed_model?.architecture_cost_ms.moe_latent.value).toBe(factors.speed_model.architecture_cost_ms.moe_latent.value);
      expect(lean.speed_model?.mlx.overhead_factor.dense.value).toBe(factors.speed_model.mlx.overhead_factor.dense.value);
      expect(JSON.stringify(lean.speed_model).includes('_formula')).toBe(false);
      expect(lean.efficiency.by_chip['M4 Max'].b_eff_ratio).toBe(factors.efficiency.by_chip['M4 Max'].b_eff_ratio);
      expect(lean.efficiency.by_tier.pro.t0_ms).toBe(factors.efficiency.by_tier.pro.t0_ms);
    }
    const model = loadModel('glm-5.3-flash');
    const settings = { memoryGb: 256, workApps: 0, override: false as const, contextTokens: 32768, kvBits: 8 as const, runtime: 'gguf' as const, qualityFloorBits: 2 };
    for (const id of ['mac-studio-m5-ultra', 'nvidia-dgx-spark', 'mac-mini-m6']) {
      const a = evaluateCell(model, full.find((m) => m.id === id)!, { ...settings, memoryGb: id === 'mac-mini-m6' ? 32 : id === 'nvidia-dgx-spark' ? 128 : 256 }, factors);
      const b = evaluateCell(model, slim.find((m) => m.id === id)!, { ...settings, memoryGb: id === 'mac-mini-m6' ? 32 : id === 'nvidia-dgx-spark' ? 128 : 256 }, lean);
      const noExamples = (x: unknown) => JSON.parse(JSON.stringify(x, (k, v) => (k === 'examples' ? undefined : v)));
      expect(noExamples(b)).toEqual(noExamples(a));
    }
  });
});

describe('linked pools in keys and labels (METHODOLOGY §4)', () => {
  const groups = machineGroups(loadMachines());
  const spark = groups.find((g) => g.id === 'nvidia-dgx-spark')!;
  const studio = groups.find((g) => g.id === 'mac-studio-m5-max')!;
  const two = { count: 2, split: 'layer' } as const;
  const fourT = { count: 4, split: 'tensor' } as const;

  it('the key keeps the per-machine size and carries the pool as a suffix', () => {
    expect(rowKey('nvidia-dgx-spark', 128, two)).toBe('nvidia-dgx-spark:128:x2');
    expect(rowKey('mac-studio-m5-max', 128, fourT)).toBe('mac-studio-m5-max:128:x4t');
    expect(parseRowKey('nvidia-dgx-spark:128:x2')).toEqual({ groupId: 'nvidia-dgx-spark', gb: 128, linked: { count: 2, split: 'layer' } });
    expect(parseRowKey('mac-studio-m5-max:128:x4t')).toEqual({ groupId: 'mac-studio-m5-max', gb: 128, linked: { count: 4, split: 'tensor' } });
    expect(parseRowKey('mac-studio-m5-max:128:x9')).toBeNull(); // not a pool token: the tail must be a size
    expect(findRow(groups, 'nvidia-dgx-spark:128:x2')?.linked).toEqual({ count: 2, split: 'layer' });
    expect(findRow(groups, 'nvidia-dgx-spark:128')?.linked).toBeNull();
  });

  it('pooled size and price, and the labels a viewer reads', () => {
    expect(pooledGb(128, two)).toBe(256);
    expect(pooledGb(128, null)).toBe(128);
    expect(pooledPrice(4699, two)).toBe(9398);
    expect(pooledPrice(null, two)).toBeNull();
    expect(linkedLabel(spark, two)).toBe('2 × NVIDIA DGX Spark');
    expect(linkedLabel(spark, null)).toBe('NVIDIA DGX Spark');
    expect(rowLabel(spark, spark.rows[0], two)).toBe('2 × NVIDIA DGX Spark · 128 GB each · 256 GB pooled');
    expect(rowLabel(studio, studio.rows.find((r) => r.gb === 128)!, null)).toBe('Mac Studio · M5 Max 128 GB');
  });

  it('tensor parallel is offered on Thunderbolt 5 Macs, the Spark, and not on the M6 mini', () => {
    expect(groupTensorCapable(studio)).toBe(true);
    expect(groupTensorCapable(spark)).toBe(true);
    expect(groupTensorCapable(groups.find((g) => g.id === 'mac-mini-m6')!)).toBe(false);
  });
});
