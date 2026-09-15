import { describe, expect, it } from 'vitest';
import { loadFactors, loadMachines, loadModel } from '../engine/__fixtures__/load';
import type { ModelIndexEntry } from '../engine/types';
import { defaultState } from './defaults';
import { findRow, machineGroups } from './machines';
import { buildFor, fitSettingsKey, modelGrid, modelSummary, orderCatalog, orderedIds, pickSections, resolvePick, syntheticSheetTarget } from './machine-fit';
import type { MatrixRow } from './matrix';

const factors = loadFactors();
const groups = machineGroups(loadMachines());
const base = defaultState();
const pick = (key: string): MatrixRow => {
  const f = findRow(groups, key);
  if (!f) throw new Error(`no row ${key}`);
  return { key, group: f.group, row: f.row, linked: f.linked };
};
const mini32 = pick('mac-mini-m6:32');
const studio256 = pick('mac-studio-m5-ultra:256');
const summary = (id: string, row: MatrixRow, state = base) => modelSummary(loadModel(id), row, state, factors);

describe('the summary rule: best build at 32K, longest context at Q4', () => {
  it('32 GB mini: a 27B dense model is tight at Q4, small models run at Q8, big ones do not fit', () => {
    const qwen = summary('qwen3.8-27b', mini32);
    expect(qwen.status).toBe('asked');
    expect(qwen.marker).toBe('ring');
    expect(qwen.best).toMatchObject({ bucket: 'Q4', verdict: 'tight' });
    expect(qwen.best!.quant.label).toBe('Q4_K_M');
    expect(qwen.best!.tokS!).toBeGreaterThan(6);
    expect(qwen.best!.tokS!).toBeLessThan(7);
    expect(qwen.longest).toMatchObject({ ctx: 32768, verdict: 'tight' });
    expect(qwen.headline).toMatch(/^Q4 · 6\.\d tok\/s · usable · tight$/);
    expect(qwen.contextLine).toBe('up to 32K at Q4 · tight');
    expect(qwen.ariaLabel).toMatch(/^Qwen 3\.8 27B on Mac mini · M6 32 GB: runs with a compromise; Q4/);

    const moe = summary('qwen3.6-35b-a3b', mini32);
    expect(moe.best).toMatchObject({ bucket: 'Q3', verdict: 'runs' });
    expect(moe.best!.quant.label).toBe('UD-Q3_K_XL');
    expect(moe.longest).toBeNull();
    expect(moe.asked?.verdict).toBe('compromise');
    expect(moe.contextLine).toBe('Q4 with a compromise');
    expect(moe.asked?.fix?.reason).toMatch(/memory-limit override/);

    expect(summary('gemma-4-e4b', mini32)).toMatchObject({ status: 'asked', marker: 'run', longest: { ctx: 131072, verdict: 'runs' } });
    expect(summary('gemma-4-e4b', mini32).best).toMatchObject({ bucket: 'Q8', verdict: 'runs' });
    const oss = summary('gpt-oss-20b', mini32);
    expect(oss.best).toMatchObject({ bucket: 'Q8', verdict: 'runs', feels: 'fast' });
    expect(oss.best!.quant.label).toBe('Q8_0');
    expect(oss.longest?.ctx).toBe(131072);
    const g31 = summary('gemma-4-31b', mini32);
    expect(g31.best).toMatchObject({ bucket: 'Q3', verdict: 'runs' });
    expect(g31.longest).toMatchObject({ ctx: 8192, verdict: 'tight' });
    expect(g31.contextLine).toBe('up to 8K at Q4 · tight');
  });

  it('32 GB mini: a compromise row says what runs; a dash carries the nearest miss', () => {
    const coder = summary('qwen3-coder-next', mini32);
    expect(coder.status).toBe('compromise');
    expect(coder.marker).toBe('ring');
    expect(coder.best).toBeNull();
    expect(coder.headline).toMatch(/^Q2 · \d+ tok\/s · with a compromise$/);
    expect(coder.detail).toMatch(/IQ2_XXS.*quality loss/);
    const big = summary('gpt-oss-120b', mini32);
    expect(big.status).toBe('no-fit');
    expect(big.marker).toBe('no');
    expect(big.headline).toBe("Doesn't fit");
    expect(big.detail).toMatch(/^the smallest allowed build, Q3_K_S \(3-bit\) at 62\.6 GB/);
    expect(big.ariaLabel).toMatch(/doesn't fit; Doesn't fit \(the smallest allowed build/);
    expect(big.contextLine).toBe('not at Q4');
  });

  it('the work toggle turns rows into "close your apps" compromises; a small model is unmoved', () => {
    const work = { ...base, work: 16 };
    expect(summary('qwen3.8-27b', mini32, work)).toMatchObject({ status: 'compromise', marker: 'ring' });
    expect(summary('qwen3.8-27b', mini32, work).detail).toMatch(/close your apps/);
    expect(summary('gpt-oss-20b', mini32, work).headline).toMatch(/^Q4 · \d+ tok\/s · with a compromise$/);
    expect(summary('gemma-4-e4b', mini32, work).best).toMatchObject({ bucket: 'Q8', verdict: 'runs' });
    expect(fitSettingsKey(work)).not.toBe(fitSettingsKey(base));
  });

  it('256 GB Studio: Q8 all round, GLM-5.3 Flash tight at Q4, the 397B at Q3, GLM-5.2 a compromise, Kimi K3 a dash', () => {
    const qwen = summary('qwen3.8-27b', studio256);
    expect(qwen.best).toMatchObject({ bucket: 'Q8', verdict: 'runs' });
    expect(qwen.best!.quant.label).toBe('Q8_0');
    expect(qwen.best!.tokS!).toBeGreaterThan(29);
    expect(qwen.best!.tokS!).toBeLessThan(34);
    expect(qwen.longest?.ctx).toBe(262144);
    const glm = summary('glm-5.3-flash', studio256);
    expect(glm.best).toMatchObject({ bucket: 'Q4', verdict: 'tight' });
    expect(glm.best!.quant.label).toBe('UD-Q4_K_XL');
    expect(glm.longest).toMatchObject({ ctx: 131072, verdict: 'tight' });
    expect(glm.marker).toBe('ring');
    const big = summary('qwen3.5-397b-a17b', studio256);
    expect(big.best).toMatchObject({ bucket: 'Q3', verdict: 'runs' });
    expect(big.longest).toBeNull();
    expect(big.contextLine).toBe('Q4 with a compromise');
    expect(summary('minimax-m3', studio256).best).toMatchObject({ bucket: 'Q3', verdict: 'tight' });
    const glm52 = summary('glm-5.2', studio256);
    expect(glm52.status).toBe('compromise');
    expect(glm52.headline).toMatch(/^Q2 · \d+ tok\/s · with a compromise$/);
    expect(glm52.detail).toMatch(/UD-IQ2_M/);
    expect(summary('kimi-k3', studio256)).toMatchObject({ status: 'no-fit', marker: 'no', gaps: ['Q6', 'Q3'] });
    expect(summary('qwen3.8-flash-next', studio256).best).toMatchObject({ bucket: 'Q8', verdict: 'tight' });
  });

  it('the runtime and the memory limit flow through settingsFor', () => {
    const mlx = summary('qwen3.8-27b', studio256, { ...base, runtime: 'mlx' });
    expect(mlx.best!.quant.label).toBe('MLX-8bit');
    expect(summary('glm-5.2', studio256, { ...base, runtime: 'mlx' }).detail).toMatch(/MLX-MXFP4/);
    const lifted = summary('glm-5.3-flash', studio256, { ...base, cap: 1 });
    expect(lifted.best).toMatchObject({ bucket: 'Q4', verdict: 'runs' });
    expect(lifted.longest?.ctx).toBe(262144);
  });
});

describe('gaps and the grid', () => {
  it('a bucket without a file is a gap, never a substitute; Q4 and Q8 always have one', () => {
    expect(buildFor(loadModel('kimi-k3'), 'Q3', 'gguf')).toBeNull();
    expect(buildFor(loadModel('kimi-k3'), 'Q6', 'gguf')).toBeNull();
    expect(summary('deepseek-v4-flash-0731', studio256).gaps).toEqual(['Q6']);
    expect(summary('kimi-k2.6', studio256).gaps).toEqual(['Q6']);
    for (const id of ['qwen3.8-27b', 'glm-5.3-flash', 'glm-5.2', 'kimi-k3', 'deepseek-v4-flash-0731', 'qwen3.8-flash-next', 'gemma-4-26b-a4b', 'gpt-oss-20b', 'gemma-4-e4b', 'qwen3-coder-next', 'gemma-4-31b', 'gpt-oss-120b', 'kimi-k2.6', 'qwen3.5-397b-a17b', 'qwen3.6-35b-a3b', 'minimax-m3']) {
      expect(buildFor(loadModel(id), 'Q4', 'gguf'), id).not.toBeNull();
      expect(buildFor(loadModel(id), 'Q8', 'gguf'), id).not.toBeNull();
    }
  });

  it('the grid is bucket rows by context columns, gaps as single cells, speeds falling with context', () => {
    const kimi = modelGrid(loadModel('kimi-k3'), studio256, base, factors);
    expect(kimi.chips).toEqual([8192, 32768, 131072, 262144]);
    expect(kimi.rows.map((r) => r.bucket)).toEqual(['Q2', 'Q3', 'Q4', 'Q6', 'Q8']);
    expect(kimi.rows[1].cells).toEqual([{ kind: 'gap', bucket: 'Q3' }]);
    expect(kimi.rows[3].cells).toEqual([{ kind: 'gap', bucket: 'Q6' }]);
    expect(kimi.rows[0].cells.map((c) => (c.kind === 'cell' ? c.marker : 'gap'))).toEqual(['no', 'no', 'no', 'no']);
    const e4b = modelGrid(loadModel('gemma-4-e4b'), mini32, base, factors);
    expect(e4b.chips).toEqual([8192, 32768, 131072]);
    const q4 = e4b.rows[2];
    expect(q4.bucket).toBe('Q4');
    expect(q4.cells.map((c) => (c.kind === 'cell' ? c.marker : 'gap'))).toEqual(['run', 'run', 'run']);
    const speeds = q4.cells.map((c) => (c.kind === 'cell' ? c.tokS! : 0));
    expect(speeds[0]).toBeGreaterThan(speeds[1]);
    expect(speeds[1]).toBeGreaterThan(speeds[2]);
    expect(q4.cells[0].kind === 'cell' && q4.cells[0].ariaLabel).toMatch(/^Gemma 4 E4B at Q4 · 8K on Mac mini · M6 32 GB: runs, \d+ tok\/s$/);
  });
});

describe('ordering, the picker and the sheet target', () => {
  const entry = (id: string, over: Partial<ModelIndexEntry>): ModelIndexEntry => ({
    id, name: id, provider: 'x', hf_repo: `x/${id}`, released: null, params_total_b: null, params_active_b: null, context_max: null, featured: false, tier: null, rank: null, gated: null, status: 'active', license: null, commercial: null,
    arch: 'gqa', hybrid: false, moe: false, assumed: false, accel: { mtp: false, dflash: false, dspark: false }, kv_bytes_per_token_8bit: null, sizes: { min_gb: null, q4_gb: null, q8_gb: null, native_gb: null }, formats: ['gguf'], quant_count: 1, four_bit_native: false, detail_sha: 'a',
    ...over,
  });

  it('models fall into size bands by their Q4 file, smallest first, superseded and hidden left out', () => {
    const size = (q4: number | null, min: number | null = null) => ({ sizes: { min_gb: min, q4_gb: q4, q8_gb: null, native_gb: null } });
    const sections = orderCatalog([
      entry('mid', size(80)),
      entry('tiny', size(7)),
      entry('gone', { status: 'superseded', ...size(20) }),
      entry('huge', size(1500)),
      entry('small', size(38)),
      entry('big', size(300)),
      entry('nofile', size(null)),
      entry('large', size(200)),
      entry('hid', { status: 'hidden', ...size(1) }),
      entry('edge', size(40)),
      entry('minonly', size(null, 500)),
    ]);
    expect(sections.map((s) => s.key)).toEqual(['xs', 's', 'm', 'l', 'xl', 'unknown']);
    expect(sections[0].entries.map((m) => m.id)).toEqual(['tiny', 'small']);
    expect(sections[1].entries.map((m) => m.id)).toEqual(['edge', 'mid']);
    expect(sections[2].entries.map((m) => m.id)).toEqual(['large']);
    expect(sections[3].entries.map((m) => m.id)).toEqual(['big', 'minonly']);
    expect(sections[4].entries.map((m) => m.id)).toEqual(['huge']);
    expect(sections[5].entries.map((m) => m.id)).toEqual(['nofile']);
    expect(orderedIds(sections)).toEqual(['tiny', 'small', 'edge', 'mid', 'large', 'big', 'minonly', 'huge', 'nofile']);
  });

  it('the memory column is the memory the configuration in the headline needs', () => {
    const qwen = summary('qwen3.8-27b', studio256);
    expect(qwen.memoryGb).toBeCloseTo(qwen.best!.result.need.totalGb, 6);
    expect(qwen.memoryText).toMatch(/^\d+(\.\d)? GB$/);
    const glm52 = summary('glm-5.2', studio256);
    expect(glm52.memoryGb).toBeCloseTo(glm52.asked!.fix!.need.totalGb, 6);
    const kimi = summary('kimi-k3', studio256);
    expect(kimi.memoryGb).toBeCloseTo(kimi.asked!.need.totalGb, 6);
    expect(kimi.memoryText).toMatch(/(GB|TB) at Q4$/);
    expect(kimi.ariaLabel).toMatch(/needs [\d.]+ (GB|TB) at Q4$/);
  });

  it('the picker offers current families, the user\'s own machines, then older ones', () => {
    const sections = pickSections(groups, { ...base, customMachines: [{ name: 'My Halo box', memoryGb: 128, bandwidthGbs: 256, platform: 'rocm' }] });
    expect(sections[0].label).toBe('MacBook Air');
    expect(sections.map((s) => s.label)).toContain('Mac Studio');
    const yours = sections.find((s) => s.label === 'Your machines');
    expect(yours?.rows.map((r) => r.key)).toEqual(['custom-my-halo-box:128']);
    const older = sections[sections.length - 1];
    expect(older.label).toBe('Older machines');
    expect(older.rows.length).toBeGreaterThan(50);
    expect(older.rows.every((r) => r.group.status === 'discontinued')).toBe(true);
    const m5max = sections.find((s) => s.label === 'MacBook Pro')!.rows.filter((r) => r.group.id === 'macbook-pro-m5-max').map((r) => r.row.gb);
    expect(m5max).toEqual([36, 48, 64, 128]);
  });

  it('a stale pick falls back to the biggest table row, then to the biggest of the first current group', () => {
    const tableRows = [pick('mac-mini-m6:16'), pick('mac-mini-m6:32'), pick('mac-studio-m5-ultra:512')];
    expect(resolvePick('nope:1', groups, base, tableRows)?.key).toBe('mac-studio-m5-ultra:512');
    expect(resolvePick('macbook-pro-m4-max:128', groups, base, tableRows)?.key).toBe('macbook-pro-m4-max:128');
    expect(resolvePick('custom-my-halo-box:128', groups, base, tableRows)?.key).toBe('mac-studio-m5-ultra:512');
    expect(resolvePick('custom-my-halo-box:128', groups, { ...base, customMachines: [{ name: 'My Halo box', memoryGb: 128, bandwidthGbs: 256, platform: 'rocm' }] }, tableRows)?.key).toBe('custom-my-halo-box:128');
    expect(resolvePick(null, groups, base, [])?.key).toBe('macbook-air-m5:32');
  });

  it('the sheet target carries the model, the machine row and the evaluated cell', () => {
    const glm = loadModel('glm-5.3-flash');
    const s = summary('glm-5.3-flash', studio256);
    const t = syntheticSheetTarget(glm, studio256, 'Q4', 32768, s.asked!, base);
    expect(t.column.index).toBe(-1);
    expect(t.column.quant?.label).toBe('UD-Q4_K_XL');
    expect(t.column.column).toEqual({ id: 'glm-5.3-flash', quant: 'auto', ctx: 32768 });
    expect(t.cell.rowKey).toBe('mac-studio-m5-ultra:256');
    expect(t.cell.result?.verdict).toBe('tight');
    expect(t.row.group.id).toBe('mac-studio-m5-ultra');
    // the buying map evaluates an exact file (the column's own build): the sheet names that file, not the bucket's default
    const exact = glm.quants.find((q) => q.label === 'Q8_0')!;
    expect(syntheticSheetTarget(glm, studio256, 'Q8', 32768, s.asked!, base, exact).column.quant?.label).toBe('Q8_0');
    expect(syntheticSheetTarget(glm, studio256, 'Q4', 32768, s.asked!, base, null).column.quant).toBeNull();
  });
});

describe('the engram-on-SSD build in the machine view', () => {
  it('is the best build for Flash Next on a 64 GB MacBook Pro when no file fits as asked, and never on a 256 GB Studio', () => {
    const fn = loadModel('qwen3.8-flash-next');
    const mbp = pick('macbook-pro-m5-max:64');
    const summary = modelSummary(fn, mbp, defaultState(), factors);
    expect(summary.status).toBe('asked');
    expect(summary.best).toMatchObject({ kind: 'special', bucket: 'Q4', verdict: 'tight', tokS: null, quant: { label: 'IQ4_XS · engram on SSD' } });
    expect(summary.headline).toBe('IQ4_XS · engram on SSD · speed not estimated · tight');
    expect(summary.detail).toBe('45.8 GB in memory, 39.1 GB streamed from the SSD');
    expect(summary.memoryText).toBe('48.6 GB');
    const studio = modelSummary(fn, pick('mac-studio-m5-ultra:256'), defaultState(), factors);
    expect(studio.best).toMatchObject({ kind: 'bucket', bucket: 'Q8' });
  });
});

describe('linked pools in the machine view', () => {
  const linkedState = { ...base, linked: { 'nvidia-dgx-spark': { count: 2 as const, split: 'layer' as const } } };

  it('the picker gains a "Linked in your table" section and a pooled key resolves with its pool', () => {
    const sections = pickSections(groups, linkedState);
    const linked = sections.find((s) => s.label === 'Linked in your table')!;
    expect(linked.rows.map((r) => r.key)).toEqual(['nvidia-dgx-spark:128:x2']);
    expect(linked.rows[0].linked).toEqual({ count: 2, split: 'layer' });
    expect(pickSections(groups, base).some((s) => s.label === 'Linked in your table')).toBe(false);
    const resolved = resolvePick('nvidia-dgx-spark:128:x2', groups, linkedState, [])!;
    expect(resolved.linked).toEqual({ count: 2, split: 'layer' });
    // the fallback picks the biggest pool in the table, not the biggest single machine
    const rows: MatrixRow[] = [pick('mac-studio-m5-ultra:256'), { ...pick('nvidia-dgx-spark:128'), key: 'nvidia-dgx-spark:128:x4', linked: { count: 4, split: 'layer' } }];
    expect(resolvePick(null, groups, linkedState, rows)?.key).toBe('nvidia-dgx-spark:128:x4');
  });

  it('the summary evaluates through the pool: GLM-5.3 Flash runs at Q4 on two Sparks', () => {
    const two: MatrixRow = { ...pick('nvidia-dgx-spark:128'), key: 'nvidia-dgx-spark:128:x2', linked: { count: 2, split: 'layer' } };
    const s = summary('glm-5.3-flash', two, linkedState);
    expect(s.status).toBe('asked');
    expect(s.best?.kind).toBe('bucket');
    expect(s.best && 'bucket' in s.best ? s.best.bucket : null).toBe('Q4');
    expect(s.best?.result.availability.machines).toBe(2);
    expect(s.ariaLabel).toContain('2 × NVIDIA DGX Spark · 128 GB each · 256 GB pooled');
  });
});
