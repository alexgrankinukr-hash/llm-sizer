import { describe, expect, it } from 'vitest';
import { loadFactors, loadMachines, loadModel } from '../engine/__fixtures__/load';
import { evaluateCell } from '../engine/index';
import type { Machine } from '../engine/types';
import { bestBySize, cellId, chipGrid, chipOf, shownConfig, sizesText, waitBandIndex, writeBandIndex, type ChipGrid } from './chip-grid';
import { defaultState } from './defaults';
import { buildFor, fitColumn } from './machine-fit';
import { machineGroups } from './machines';
import { settingsFor } from './matrix';

const factors = loadFactors();
const machines = loadMachines();
const groups = machineGroups(machines);
const qwen = loadModel('qwen3.8-27b');
const base = defaultState();
const grid8k = chipGrid(qwen, 'Q4', 8192, groups, base, factors);
const cell = (g: ChipGrid, tier: Parameters<typeof cellId>[0], gen: number) => {
  const c = g.cells.get(cellId(tier, gen));
  if (!c) throw new Error(`no ${tier} M${gen}`);
  return c;
};

describe('the chips', () => {
  it('reads tier and generation from the catalog chip name, bins and memory variants stripped', () => {
    expect(chipOf({ platform: 'apple', chip: 'M5 Max (40-core GPU)', family: 'MacBook Pro' })).toEqual({ tier: 'max', gen: 5, name: 'M5 Max' });
    expect(chipOf({ platform: 'apple', chip: 'M6 (16 GB)', family: 'Mac mini' })).toEqual({ tier: 'base', gen: 6, name: 'M6' });
    expect(chipOf({ platform: 'apple', chip: 'M3 Ultra', family: 'Mac Studio' })).toEqual({ tier: 'ultra', gen: 3, name: 'M3 Ultra' });
    expect(chipOf({ platform: 'cuda', chip: 'GB10 Grace Blackwell', family: 'NVIDIA DGX Spark' })).toBeNull();
  });

  it('lays out every Apple chip in the catalog: six generations, four tiers, twenty chips, no M4 Ultra', () => {
    expect(grid8k.gens).toEqual([1, 2, 3, 4, 5, 6]);
    expect(grid8k.tiers).toEqual(['base', 'pro', 'max', 'ultra']);
    expect(grid8k.cells.size).toBe(20);
    expect(grid8k.cells.has(cellId('ultra', 4))).toBe(false);
    expect(grid8k.cells.has(cellId('pro', 6))).toBe(false);
    expect(grid8k.genYears).toMatchObject({ 1: 2020, 5: 2025, 6: 2026 });
    expect(grid8k.tierFamilies.base).toEqual(['MacBook Air', 'MacBook Pro', 'Mac mini', 'iMac']);
    expect(grid8k.tierFamilies.ultra).toEqual(['Mac Studio', 'Mac Pro']);
  });

  it('gives a chip the catalog adds later its own cell and column', () => {
    const m6Pro: Machine = { ...machines.find((m) => m.id === 'macbook-pro-m5-pro')!, id: 'macbook-pro-m6-pro', chip: 'M6 Pro', generation: 'M6', year: 2026 };
    const m7: Machine = { ...machines.find((m) => m.id === 'mac-mini-m6')!, id: 'mac-mini-m7', chip: 'M7', generation: 'M7', year: 2027 };
    const g = chipGrid(qwen, 'Q4', 8192, machineGroups([...machines, m6Pro, m7]), base, factors);
    expect(g.cells.has(cellId('pro', 6))).toBe(true);
    expect(g.gens).toEqual([1, 2, 3, 4, 5, 6, 7]);
    expect(cell(g, 'pro', 6).configs.map((c) => c.gb)).toEqual([24, 48, 64]);
  });
});

describe('memory sizes', () => {
  it('merges families that sell the same configuration, and keeps the cheapest list price', () => {
    const m5max = cell(grid8k, 'max', 5);
    expect(sizesText(m5max)).toBe('36, 48, 64 or 128 GB');
    const c128 = m5max.configs.find((c) => c.gb === 128)!;
    expect(c128.families).toEqual(['MacBook Pro', 'Mac Studio']);
    expect(c128.bandwidthGbs).toBe(614);
  });

  it('the biggest memory is the chip at its best; the smallest that runs it can be a slower version', () => {
    const m5max = cell(grid8k, 'max', 5);
    const biggest = shownConfig(m5max, 'max')!;
    const smallest = shownConfig(m5max, 'min')!;
    expect(biggest.gb).toBe(128);
    expect(smallest.gb).toBe(36);
    expect(smallest.bandwidthGbs).toBe(460);
    expect(smallest.cores).toBe(32);
    expect(smallest.status).toBe('runs');
    // the 36 GB M5 Max is the 32-core chip: it writes at 460 GB/s, not 614
    expect(smallest.writeTokS!).toBeLessThan(biggest.writeTokS! * 0.85);
    expect(smallest.readTokS!).toBeLessThan(biggest.readTokS!);
  });

  it('memory size alone does not change writing speed: the 256 and 512 GB M5 Ultra write alike, the 256 reads on the 64-core GPU', () => {
    const m5ultra = cell(grid8k, 'ultra', 5);
    const at = (gb: number) => m5ultra.configs.find((c) => c.gb === gb)!;
    expect(at(256).writeTokS).toBeCloseTo(at(512).writeTokS!, 6);
    expect(at(256).cores).toBe(64);
    expect(at(512).cores).toBe(80);
    expect(at(512).readTokS! / at(256).readTokS!).toBeCloseTo(80 / 64, 6);
  });

  it('a chip that cannot hold the model reads "doesn\'t fit" in both modes, at its biggest size', () => {
    const m1 = cell(grid8k, 'base', 1);
    expect(shownConfig(m1, 'max')!.status).toBe('no-fit');
    expect(shownConfig(m1, 'min')!).toBe(shownConfig(m1, 'max'));
    expect(shownConfig(m1, 'min')!.gb).toBe(16);
    expect(shownConfig(m1, 'min')!.writeTokS).toBeNull();
  });

  it('a 24 GB base M2 runs the 27B only with the memory limit lifted; with the override set it simply runs', () => {
    const m2 = shownConfig(cell(grid8k, 'base', 2), 'max')!;
    expect(m2.gb).toBe(24);
    expect(m2.status).toBe('lifted');
    expect(m2.lifted).toEqual(['override']);
    expect(m2.writeTokS).not.toBeNull();
    const lifted = chipGrid(qwen, 'Q4', 8192, groups, { ...base, cap: 1 }, factors);
    expect(shownConfig(cell(lifted, 'base', 2), 'max')!.status).toBe('runs');
  });

  it('smallest-that-runs prefers a fit as asked over a smaller size that needs the limit lifted', () => {
    const m4 = cell(grid8k, 'base', 4);
    const sizes = bestBySize(m4);
    expect(sizes.map((c) => c.gb)).toEqual([16, 24, 32]);
    const shown = shownConfig(m4, 'min')!;
    expect(shown.status).toBe('runs');
    expect(sizes.filter((c) => c.gb < shown.gb).every((c) => c.status !== 'runs')).toBe(true);
  });
});

describe('the numbers are the table\'s numbers', () => {
  it('every configuration is evaluateCell through settingsFor', () => {
    const c = cell(grid8k, 'max', 4).configs.find((x) => x.gb === 128)!;
    const quant = buildFor(qwen, 'Q4', base.runtime)!;
    const direct = evaluateCell(qwen, c.row.machine, settingsFor(base, c.row, fitColumn(qwen, 'Q4', 8192), quant), factors);
    expect(c.writeTokS).toBe(direct.speed.tokS);
    expect(c.readTokS).toBe(direct.prefill!.tokS);
  });

  it('a longer context costs writing speed and makes the read wait grow faster than the prompt', () => {
    const at128 = chipGrid(qwen, 'Q4', 131072, groups, base, factors);
    const short = shownConfig(cell(grid8k, 'max', 5), 'max')!;
    const long = shownConfig(cell(at128, 'max', 5), 'max')!;
    expect(long.writeTokS!).toBeLessThan(short.writeTokS!);
    expect(long.waitS! / short.waitS!).toBeGreaterThan(16);
  });

  it('a model without a file in the bucket evaluates nothing, and the grid still lists the chips', () => {
    const g = chipGrid(qwen, 'Q2', 8192, groups, { ...base, runtime: 'mlx' }, factors);
    if (g.quant === null) {
      expect(g.cells.size).toBe(20);
      expect([...g.cells.values()].every((c) => c.configs.length === 0)).toBe(true);
    }
  });

  it('bands follow the engine: an edge belongs to the faster band for writing, to the shorter wait for reading', () => {
    expect(writeBandIndex(4.9, factors)).toBe(0);
    expect(writeBandIndex(5, factors)).toBe(1);
    expect(writeBandIndex(200, factors)).toBe(factors.feels_like.length - 1);
    expect(waitBandIndex(3, factors)).toBe(0);
    expect(waitBandIndex(3.1, factors)).toBe(1);
    expect(waitBandIndex(10_000, factors)).toBe(factors.prefill!.feels_like.length - 1);
  });
});
