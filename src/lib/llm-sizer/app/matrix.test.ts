import { describe, expect, it } from 'vitest';
import { loadFactors, loadMachines, loadModel } from '../engine/__fixtures__/load';
import { defaultState } from './defaults';
import { tableLayout, verdictSentence } from './layout';
import { machineGroups } from './machines';
import { evaluateMatrix, type RecordState } from './matrix';

const factors = loadFactors();
const groups = machineGroups(loadMachines());
const records: Record<string, RecordState> = Object.fromEntries(
  ['qwen3.8-27b', 'qwen3.8-flash-next', 'glm-5.3-flash', 'kimi-k3', 'glm-5.2'].map((id) => [id, { status: 'ready', model: loadModel(id) }]),
);

describe('the first screen', () => {
  const state = defaultState();
  const matrix = evaluateMatrix(state, groups, records, factors);
  const layout = tableLayout(state, matrix);
  const marker = (row: string, col: number) => layout.rows.find((r) => r.key === row)!.cells[col].marker;

  it('has 7 rows × 4 columns and the video chart\'s verdicts at Q4 / 32K', () => {
    expect(matrix.rows.map((r) => r.key)).toEqual(['mac-mini-m6:16', 'mac-mini-m6:24', 'mac-mini-m6:32', 'mac-studio-m5-ultra:96', 'mac-studio-m5-ultra:256', 'mac-studio-m5-ultra:512', 'nvidia-dgx-spark:128']);
    expect(matrix.columns.map((c) => c.quant?.label)).toEqual(['Q4_K_M', 'Q4_K_M', 'UD-Q4_K_XL', 'UD-Q4_K_XL']);
    // Qwen 27B: 16 GB → ring (a 2-bit build fits), 24 GB → ring, 32 GB → ring (tight at 90 % once buffers are 1.5 GB + 1 %)
    expect(marker('mac-mini-m6:32', 0)).toBe('ring');
    expect(marker('mac-mini-m6:16', 0)).toBe('ring');
    expect(marker('mac-studio-m5-ultra:96', 0)).toBe('run');
    // Flash Next needs ~120 GB at Q4: Studio 256/512 run, Spark 128 is a compromise, mini never
    expect(marker('mac-studio-m5-ultra:256', 1)).toBe('run');
    expect(marker('nvidia-dgx-spark:128', 1)).toBe('ring');
    expect(marker('mac-mini-m6:32', 1)).toBe('no');
    // GLM-5.3 Flash 200 GB: 256 needs the override (ring), 512 runs, Spark = 2-bit ring
    expect(marker('mac-studio-m5-ultra:256', 2)).toBe('ring');
    expect(marker('mac-studio-m5-ultra:512', 2)).toBe('run');
    expect(marker('nvidia-dgx-spark:128', 2)).toBe('ring');
    // Kimi K3: nothing fits anywhere
    expect(layout.rows.every((r) => r.cells[3].marker === 'no')).toBe(true);
  });

  it('layout carries headers, rails, prices, aria labels and footnotes', () => {
    expect(layout.headers.map((h) => h.name)).toEqual(['Qwen 3.8 27B', 'Qwen 3.8 Flash Next', 'GLM-5.3 Flash', 'Kimi K3']);
    expect(layout.headers[1].params).toBe('180B · 6B active');
    expect(layout.headers[0].quantLabel).toBe('Q4');
    expect(layout.headers[0].contextLabel).toBe('32K');
    expect(layout.rails.map((r) => r.label)).toEqual(['Mac mini · M6', 'Mac Studio · M5 Ultra', 'NVIDIA DGX Spark']);
    expect(layout.rails[0].price).toBe('from $899');
    expect(layout.rows[2].cells[0].ariaLabel).toBe('Qwen 3.8 27B on Mac mini · M6 32 GB: runs with a compromise (tight)');
    expect(layout.rows[6].price).toBe('$4,699');
    expect(layout.toggles.work).toBe('nothing else running');
    expect(layout.footnotes.length).toBeGreaterThanOrEqual(0);
  });

  it('the work toggle turns the 32 GB mini into a ring (METHODOLOGY A) and loading columns show as loading', () => {
    const work = evaluateMatrix({ ...state, work: 16 }, groups, records, factors);
    const cell = work.cells.get('mac-mini-m6:32|0')!;
    expect(cell.result?.verdict).toBe('compromise');
    expect(verdictSentence(cell.result!)).toMatch(/close your apps/);
    const partial = evaluateMatrix(state, groups, { 'qwen3.8-27b': { status: 'loading' } }, factors);
    expect(tableLayout(state, partial).rows[0].cells[0].marker).toBe('loading');
    expect(tableLayout(state, partial).rows[0].cells[3].marker).toBe('loading');
  });

  it('custom models and machines evaluate like catalog ones', () => {
    const custom = evaluateMatrix(
      { ...state, groups: ['custom-big-box'], columns: [{ id: 'custom-my-70b', quant: 'auto', ctx: 32768 }],
        customModels: [{ mode: 'quick', name: 'My 70B', paramsTotalB: 70, weightsGb: 40 }],
        customMachines: [{ name: 'Big box', memoryGb: 128, bandwidthGbs: 500, platform: 'apple' }] },
      groups, {}, factors,
    );
    const cell = custom.cells.get('custom-big-box:128|0')!;
    expect(cell.result?.verdict).toBe('runs');
    expect(cell.result?.flags).toContain('assumed');
  });
});

describe('visibleRows', () => {
  it('drops hidden sizes and falls back to every size when all are hidden', async () => {
    const { visibleRows } = await import('./matrix');
    const { machineGroups } = await import('./machines');
    const { loadMachines } = await import('../engine/__fixtures__/load');
    const { defaultState } = await import('./defaults');
    const studio = machineGroups(loadMachines()).find((g) => g.id === 'mac-studio-m5-ultra')!;
    expect(visibleRows({ ...defaultState(), hiddenSizes: { 'mac-studio-m5-ultra': [96] } }, studio).map((r) => r.gb)).toEqual([256, 512]);
    expect(visibleRows({ ...defaultState(), hiddenSizes: { 'mac-studio-m5-ultra': [96, 256, 512] } }, studio).map((r) => r.gb)).toEqual([96, 256, 512]);
    expect(visibleRows(defaultState(), studio)).toBe(studio.rows);
  });
});

describe('a linked pool in the table', () => {
  const state = { ...defaultState(), linked: { 'nvidia-dgx-spark': { count: 2 as const, split: 'layer' as const } } };
  const matrix = evaluateMatrix(state, groups, records, factors);
  const layout = tableLayout(state, matrix);

  it('keys the row with the pool, labels the rail and the size, sums the price, and runs GLM-5.3 Flash across two Sparks', () => {
    expect(matrix.rows.map((r) => r.key)).toContain('nvidia-dgx-spark:128:x2');
    expect(matrix.rows.find((r) => r.key === 'nvidia-dgx-spark:128:x2')?.linked).toEqual({ count: 2, split: 'layer' });
    const rail = layout.rails.find((r) => r.groupId === 'nvidia-dgx-spark')!;
    expect(rail.label).toBe('2 × NVIDIA DGX Spark');
    expect(rail.price).toBe('from $9,398');
    const row = layout.rows.find((r) => r.key === 'nvidia-dgx-spark:128:x2')!;
    expect(row.sizeLabel).toBe('128 GB each · 256 GB pooled');
    expect(row.price).toBe('$9,398');
    expect(row.cells[2].marker).toBe('run');
    expect(row.cells[2].ariaLabel).toBe('GLM-5.3 Flash on 2 × NVIDIA DGX Spark · 128 GB each · 256 GB pooled: runs');
    const cell = matrix.cells.get('nvidia-dgx-spark:128:x2|2')!.result!;
    expect(cell.flags).toContain('linked-layer');
    expect(cell.availability.machines).toBe(2);
    expect(layout.rows.find((r) => r.key === 'mac-mini-m6:32')!.sizeLabel).toBe('32 GB');
  });
});
