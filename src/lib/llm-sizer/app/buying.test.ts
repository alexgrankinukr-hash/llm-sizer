import { describe, expect, it } from 'vitest';
import { loadFactors, loadMachines, loadModel } from '../engine/__fixtures__/load';
import { defaultState } from './defaults';
import { machineGroups } from './machines';
import type { MatrixColumn } from './matrix';
import { resolveQuant } from './quants';
import type { AppState } from './state';
import { atlasKey, buildOptions, buyingMap, byPrice, cheapestSingleThatRuns, evaluateBuilds, joinLabels, markDetail, minSpeedPresets, misses, placeRowLabels, resolveBuild, type BuyingCriteria, type BuyingMark } from './buying';

const factors = loadFactors();
const groups = machineGroups(loadMachines());
const base = defaultState();
const toggles = { work: 'nothing else running', limit: 'macOS memory limit 67 % / 75 %' };
const ANY: BuyingCriteria = { minTokS: 0, budgetUsd: null, formFactor: 'any' };

function col(id: string, quant: MatrixColumn['column']['quant'] = 'auto', ctx = 32768, state: AppState = base): MatrixColumn {
  const model = loadModel(id);
  return { index: 0, column: { id, quant, ctx }, status: 'ready', model, quant: resolveQuant(model, quant, state.runtime) };
}
function atlasFor(id: string, state: AppState = base, quant: MatrixColumn['column']['quant'] = 'auto') {
  return evaluateBuilds(loadModel(id), col(id, quant, 32768, state), groups, state, factors, '2026-09-02');
}
const keys = (marks: BuyingMark[]) => marks.map((m) => m.key);

describe('the machine set', () => {
  it('is every row of every current machine, 30 configurations on 10 memory sizes, 29 of them priced', () => {
    const atlas = atlasFor('qwen3.8-27b');
    const marks = atlas.builds.get('Q4')!.marks;
    expect(marks).toHaveLength(30);
    expect(new Set(keys(marks as BuyingMark[])).size).toBe(30);
    expect(atlas.rows).toEqual([16, 24, 32, 36, 48, 64, 96, 128, 256, 512]);
    expect(marks.filter((m) => m.priceUsd !== null)).toHaveLength(29);
    expect(marks.find((m) => m.key === 'mac-studio-m5-ultra:512')).toMatchObject({ priceUsd: null, priceSource: 'Apple', formFactor: 'desktop' });
    expect(marks.find((m) => m.key === 'macbook-pro-m5-max:36')).toMatchObject({ label: 'MacBook Pro · M5 Max · 32-core GPU · 36 GB', short: 'MacBook Pro M5 Max 32-core', formFactor: 'laptop', priceUsd: 4099 });
    expect(marks.find((m) => m.key === 'nvidia-dgx-spark:128')).toMatchObject({ label: 'NVIDIA DGX Spark · 128 GB', short: 'NVIDIA DGX Spark', priceSource: 'NVIDIA', aboutHref: '/tools/llm-sizer/about#machine-nvidia-dgx-spark' });
    expect(marks.find((m) => m.key === 'mac-mini-m6:16')?.label).toBe('Mac mini · M6 · 16 GB');
    expect(marks.every((m) => !m.discontinued && !m.custom)).toBe(true);
    expect(atlas.hasOwnMachines).toBe(false);
  });
  it('adds the visitor\'s own machines: a custom box has no price, no form factor and no speed estimate', () => {
    const state: AppState = { ...base, customMachines: [{ name: 'Halo box', memoryGb: 128, bandwidthGbs: 256, platform: 'rocm' }] };
    const map = buyingMap(atlasFor('qwen3.8-27b', state), 'column', ANY, toggles);
    expect(map.marks).toHaveLength(31);
    const halo = map.marks.find((m) => m.custom)!;
    // no speed figure is fine when any speed will do; a minimum it cannot be checked against is a miss
    expect(halo).toMatchObject({ key: 'custom-halo-box:128', formFactor: 'any', priceUsd: null, priceSource: null, tokS: null, status: 'fits', aboutHref: null, misses: [], qualifies: true });
    expect(markDetail(halo, ANY)).toMatch(/to spare · speed not estimated$/);
    const at12 = buyingMap(atlasFor('qwen3.8-27b', state), 'column', { minTokS: 12, budgetUsd: null, formFactor: 'any' }, toggles).marks.find((m) => m.custom)!;
    expect(at12).toMatchObject({ misses: ['unknown-speed'], qualifies: false });
    expect(markDetail(at12, { minTokS: 12, budgetUsd: null, formFactor: 'any' })).toBe('no speed estimate, so your 12 tok/s cannot be checked');
    expect(map.atlas.hasOwnMachines).toBe(true);
    expect(map.captionParts).toContain('every current Mac and the NVIDIA DGX Spark, your machines');
  });
  it('adds the older catalog machines the visitor put in the table, every size of them, after the current ones', () => {
    const state: AppState = { ...base, groups: [...base.groups, 'macbook-pro-m4-max'] };
    const map = buyingMap(atlasFor('qwen3.8-27b', state), 'column', ANY, toggles);
    expect(map.marks).toHaveLength(34);
    const older = map.marks.filter((m) => m.discontinued);
    expect(keys(older)).toEqual(['macbook-pro-m4-max:36', 'macbook-pro-m4-max:48', 'macbook-pro-m4-max:64', 'macbook-pro-m4-max:128']);
    expect(older[3]).toMatchObject({ priceUsd: null, status: 'fits', qualifies: true, formFactor: 'laptop' });
    expect(map.atlas.hasOwnMachines).toBe(true);
    expect(map.captionParts).toContain('every current Mac and the NVIDIA DGX Spark, your machines');
  });
});

describe('build options', () => {
  it('Q4 and Q8 for a plain column, the column\'s own bucket as a third when it is neither', () => {
    const q = buildOptions(loadModel('qwen3.8-27b'), col('qwen3.8-27b'), 'gguf');
    expect(q.map((o) => [o.key, o.label, o.quant?.label, o.isColumn, o.note])).toEqual([
      ['Q4', 'Q4', 'Q4_K_M', true, null],
      ['Q8', 'Q8', 'Q8_0', false, null],
    ]);
    const q6 = buildOptions(loadModel('qwen3.8-27b'), col('qwen3.8-27b', { bucket: 'Q6' }), 'gguf');
    expect(q6.map((o) => [o.key, o.label, o.quant?.label, o.isColumn, o.bucket])).toEqual([
      ['Q4', 'Q4', 'Q4_K_M', false, 'Q4'],
      ['Q8', 'Q8', 'Q8_0', false, 'Q8'],
      ['column', 'Q6', 'Q6_K', true, 'Q6'],
    ]);
    expect(resolveBuild(q6, 'column').label).toBe('Q6');
    expect(resolveBuild(q6, 'Q8').label).toBe('Q8');
    const noQ8 = q6.map((o) => (o.key === 'Q8' ? { ...o, quant: null } : o));
    expect(resolveBuild(noQ8, 'Q8').label).toBe('Q6');
  });
  it('a re-packed Q8 carries the engine\'s caveat; MLX picks the MLX files; Kimi K3 has both buckets', () => {
    const oss = buildOptions(loadModel('gpt-oss-20b'), col('gpt-oss-20b'), 'gguf');
    expect(oss[1].note).toMatch(/re-packs the same weights/);
    const mlxState: AppState = { ...base, runtime: 'mlx' };
    const mlx = buildOptions(loadModel('qwen3.8-27b'), col('qwen3.8-27b', 'auto', 32768, mlxState), 'mlx');
    expect(mlx.map((o) => o.quant?.label)).toEqual(['MLX-4bit', 'MLX-8bit']);
    const kimi = buildOptions(loadModel('kimi-k3'), col('kimi-k3'), 'gguf');
    expect(kimi.map((o) => o.quant?.label)).toEqual(['UD-Q4_K_XL', 'UD-Q8_K_XL']);
  });
});

describe('Qwen 3.8 27B at Q4 · 32K', () => {
  const atlas = atlasFor('qwen3.8-27b');
  it('every 32 GB and larger current Mac fits, the 24 GB ones need a compromise, the pick is the $1,299 mini', () => {
    const map = buyingMap(atlas, 'column', ANY, toggles);
    expect(map.build.label).toBe('Q4');
    const byKey = new Map(map.marks.map((m) => [m.key, m]));
    expect(byKey.get('mac-mini-m6:32')).toMatchObject({ status: 'fits', qualifies: true, approx: true, band: 'usable', priceUsd: 1299 });
    expect(byKey.get('mac-mini-m6:32')!.result.verdict).toBe('tight');
    expect(byKey.get('mac-mini-m6:32')!.tokS!).toBeGreaterThan(6);
    expect(byKey.get('mac-mini-m6:32')!.tokS!).toBeLessThan(7);
    expect(byKey.get('mac-mini-m6:24')).toMatchObject({ status: 'compromise', qualifies: false, wouldQualify: true, misses: ['compromise'] });
    expect(byKey.get('macbook-air-m5:16')).toMatchObject({ status: 'compromise', wouldQualify: true });
    expect(byKey.get('imac-m4:32')).toMatchObject({ status: 'fits', band: 'painful' });
    expect(byKey.get('mac-studio-m5-ultra:96')).toMatchObject({ status: 'fits', band: 'fast', approx: true });
    expect(map.recommendation.pick?.key).toBe('mac-mini-m6:32');
    expect(map.recommendation.alternative).toMatchObject({ mark: { key: 'macbook-air-m5:32' }, why: ['form'] });
    expect(keys(map.recommendation.unpricedQualifying)).toEqual(['mac-studio-m5-ultra:512']);
    expect(map.recommendation.nearest).toEqual([]);
    // cheapest first, the unpriced last, and the card agrees with the list
    const prices = map.qualifying.map((m) => m.priceUsd);
    expect(prices.slice(0, -1).every((p) => p !== null)).toBe(true);
    expect(prices.at(-1)).toBeNull();
    expect(map.qualifying[0].key).toBe(map.recommendation.pick!.key);
    expect(markDetail(byKey.get('mac-mini-m6:32')!, ANY)).toMatch(/^\d+\.\d GB of \d+\.\d GB used · tight · \d\.\d GB to spare$/);
    expect(map.captionParts.slice(0, 3)).toEqual(['Qwen 3.8 27B at Q4 (Q4_K_M) · 32K', 'GGUF', 'cache FP16']);
    expect(map.captionParts).toContain('list prices as of 2026-09-02');
  });
  it('at 12 tok/s the mini misses on speed; the pick is the cheapest priced machine at 12 or more, the alternative buys headroom', () => {
    const map = buyingMap(atlas, 'column', { ...ANY, minTokS: 12 }, toggles);
    const mini = map.marks.find((m) => m.key === 'mac-mini-m6:32')!;
    expect(mini.misses).toEqual(['speed']);
    expect(markDetail(mini, map.criteria)).toBe('too slow: ~6.5 tok/s, under your 12');
    expect(map.recommendation.pick).toMatchObject({ key: 'mac-mini-m5-pro:48', priceUsd: 2299 });
    expect(map.qualifying.every((m) => m.tokS! >= 12)).toBe(true);
    expect(map.qualifying.filter((m) => m.priceUsd !== null).every((m) => m.priceUsd! >= 2299)).toBe(true);
    expect(map.recommendation.alternative).toMatchObject({ mark: { key: 'mac-mini-m5-pro:64' }, why: ['headroom'] });
    const laptop = buyingMap(atlas, 'column', { ...ANY, minTokS: 12, formFactor: 'laptop' }, toggles);
    expect(laptop.recommendation.pick).toMatchObject({ key: 'macbook-pro-m5-pro:48', formFactor: 'laptop' });
    expect(laptop.qualifying.every((m) => m.formFactor === 'laptop')).toBe(true);
    expect(laptop.marks.find((m) => m.key === 'mac-mini-m5-pro:48')!.misses).toEqual(['form']);
  });
  it('nothing qualifies under $2,000 at 12 tok/s: the nearest misses are named, no pick is made up', () => {
    const map = buyingMap(atlas, 'column', { minTokS: 12, budgetUsd: 2000, formFactor: 'any' }, toggles);
    expect(map.qualifying).toEqual([]);
    expect(map.recommendation.pick).toBeNull();
    expect(map.recommendation.alternative).toBeNull();
    const nearest = map.recommendation.nearest;
    expect(nearest.map((n) => n.kind)).toEqual(['price', 'speed', 'change']);
    expect(nearest[0]).toMatchObject({ mark: { key: 'mac-mini-m5-pro:48' }, detail: '$2,299 · $299 over your budget' });
    expect(nearest[1]).toMatchObject({ mark: { key: 'mac-mini-m6:32' }, detail: '~6.5 tok/s, under your 12' });
    expect(nearest[1].mark.priceUsd!).toBeLessThanOrEqual(2000);
    expect(nearest[2].mark.key).toBe('mac-mini-m5-pro:24');
    expect(nearest[2].detail).toMatch(/memory-limit override/);
    expect(map.marks.find((m) => m.key === 'mac-studio-m5-ultra:512')!.misses).toEqual(['unpriced']);
  });
  it('the Q8 map shares the axis and changes the answer', () => {
    const q8 = buyingMap(atlas, 'Q8', ANY, toggles);
    expect(q8.build.label).toBe('Q8');
    expect(q8.marks.find((m) => m.key === 'mac-mini-m6:32')).toMatchObject({ status: 'compromise', wouldQualify: true });
    expect(q8.recommendation.pick).toMatchObject({ key: 'mac-mini-m5-pro:48', band: 'usable' });
    expect(atlas.axisMax).toBe(80);
    for (const b of atlas.builds.values()) for (const m of b.marks) if (m.tokS !== null) expect(m.tokS).toBeLessThanOrEqual(atlas.axisMax);
    expect(atlas.axisMax % 20).toBe(0);
    expect(q8.captionParts[0]).toBe('Qwen 3.8 27B at Q8 (Q8_0) · 32K');
  });
  it('an older machine from the table follows the same rules as everything else: unpriced under a budget, never the pick', () => {
    const state: AppState = { ...base, groups: [...base.groups, 'macbook-pro-m4-max'] };
    const map = buyingMap(atlasFor('qwen3.8-27b', state), 'column', { minTokS: 0, budgetUsd: 500, formFactor: 'laptop' }, toggles);
    expect(map.qualifying).toEqual([]);
    expect(map.marks.find((m) => m.key === 'macbook-pro-m4-max:128')?.misses).toEqual(['unpriced']);
    expect(map.recommendation.pick).toBeNull();
    const open = buyingMap(atlasFor('qwen3.8-27b', state), 'column', ANY, toggles);
    // unpriced marks sort fastest first, then more memory first
    expect(keys(open.recommendation.unpricedQualifying)).toEqual(['mac-studio-m5-ultra:512', 'macbook-pro-m4-max:128', 'macbook-pro-m4-max:64', 'macbook-pro-m4-max:48', 'macbook-pro-m4-max:36']);
    expect(open.recommendation.pick?.key).toBe('mac-mini-m6:32');
  });
});

describe('GLM-5.3 Flash at Q4 · 32K and a model nothing runs', () => {
  it('the 256 GB Ultra is the pick (tight, fast), the 512 GB one qualifies without a price, the 128 GB machines would qualify with a change', () => {
    const atlas = atlasFor('glm-5.3-flash');
    const map = buyingMap(atlas, 'column', ANY, toggles);
    expect(map.recommendation.pick).toMatchObject({ key: 'mac-studio-m5-ultra:256', priceUsd: 9499 });
    expect(map.recommendation.pick!.result.verdict).toBe('tight');
    expect(map.recommendation.pick!.tokS!).toBeGreaterThan(29);
    expect(map.recommendation.pick!.tokS!).toBeLessThan(34);
    expect(keys(map.recommendation.unpricedQualifying)).toEqual(['mac-studio-m5-ultra:512']);
    expect(keys(map.withChange)).toEqual(['nvidia-dgx-spark:128', 'mac-studio-m5-max:128', 'macbook-pro-m5-max:128']);
    for (const m of map.withChange) {
      expect(m.tokS).not.toBeNull();
      expect(m.qualifies).toBe(false);
    }
    expect(map.marks.find((m) => m.key === 'mac-studio-m5-ultra:96')).toMatchObject({ status: 'no-fit', tokS: null, freeGb: null, misses: ['no-fit'] });
    const under = buyingMap(atlas, 'column', { ...ANY, budgetUsd: 6000 }, toggles);
    expect(under.qualifying).toEqual([]);
    expect(under.recommendation.nearest.map((n) => n.kind)).toEqual(['price', 'change']);
    expect(under.recommendation.nearest[0].detail).toBe('$9,499 · $3,499 over your budget');
    expect(under.recommendation.nearest[1].mark.key).toBe('nvidia-dgx-spark:128');
  });
  it('Kimi K3 fits nowhere: every mark a dash, nothing qualifies, nothing is near', () => {
    const map = buyingMap(atlasFor('kimi-k3'), 'column', ANY, toggles);
    expect(map.marks.every((m) => m.status === 'no-fit' && m.tokS === null)).toBe(true);
    expect(map.qualifying).toEqual([]);
    expect(map.withChange).toEqual([]);
    expect(map.recommendation.nearest).toEqual([]);
    expect(markDetail(map.marks[0], ANY)).toMatch(/^doesn't fit: /);
  });
});

describe('the engram-on-SSD build on the map', () => {
  it('Flash Next gets an SSD build segment: the smallest resident part, its caveat as the note, the column build unchanged', () => {
    const atlas = atlasFor('qwen3.8-flash-next');
    expect(atlas.options.map((o) => o.key)).toEqual(['Q4', 'Q8', 'ssd']);
    const ssd = atlas.options[2];
    expect(ssd).toMatchObject({ label: 'SSD build', isColumn: false, bucket: 'Q4', column: { quant: { special: 0 } } });
    expect(ssd.quant).toMatchObject({ label: 'IQ4_XS · engram on SSD', resident_gb: 45.8 });
    expect(ssd.note).toBe('IQ4_XS · engram on SSD: 45.8 GB in memory, 39.1 GB streamed from the SSD; speed not estimated');
    expect(atlasFor('qwen3.8-27b').options.map((o) => o.key)).toEqual(['Q4', 'Q8']);
  });
  it('on the SSD build the 64 GB Macs fit with no speed: they qualify at any speed, miss a minimum, and the cheapest is the pick', () => {
    const map = buyingMap(atlasFor('qwen3.8-flash-next'), 'ssd', ANY, toggles);
    expect(map.build.key).toBe('ssd');
    const mini64 = map.marks.find((m) => m.key === 'mac-mini-m5-pro:64')!;
    expect(mini64).toMatchObject({ status: 'fits', tokS: null, qualifies: true, misses: [] });
    expect(mini64.result.need.ssdGb).toBe(39.1);
    expect(map.marks.find((m) => m.key === 'mac-mini-m5-pro:48')).toMatchObject({ status: 'no-fit' });
    expect(map.recommendation.pick?.key).toBe('mac-mini-m5-pro:64');
    expect(map.recommendation.pick?.priceUsd).toBe(2699);
    expect(map.captionParts[0]).toBe('Qwen 3.8 Flash Next at SSD build (IQ4_XS · engram on SSD, 45.8 GB in memory and 39.1 GB from the SSD) · 32K');
    const at12 = buyingMap(atlasFor('qwen3.8-flash-next'), 'ssd', { minTokS: 12, budgetUsd: null, formFactor: 'any' }, toggles);
    expect(at12.marks.find((m) => m.key === 'mac-mini-m5-pro:64')).toMatchObject({ qualifies: false, misses: ['unknown-speed'] });
    expect(at12.qualifying).toEqual([]); // every mark on this build has no speed figure: a minimum cannot be met
    expect(at12.recommendation.pick).toBeNull();
  });
  it('a column that is the SSD build gets it as its own segment, named for what it is', () => {
    const atlas = atlasFor('qwen3.8-flash-next', base, { special: 0 });
    expect(atlas.options.map((o) => [o.key, o.label, o.isColumn])).toEqual([['Q4', 'Q4', false], ['Q8', 'Q8', false], ['column', 'SSD build', true]]);
  });
});

describe('the rules on their own', () => {
  const mark = (over: Partial<Pick<BuyingMark, 'status' | 'tokS' | 'priceUsd' | 'formFactor'>>) => ({ status: 'fits' as const, tokS: 20, priceUsd: 2000, formFactor: 'desktop' as const, ...over });
  it('misses, in order: no-fit stops everything; a compromise keeps going; unknown speed never meets a target', () => {
    expect(misses(mark({ status: 'no-fit', priceUsd: 99999 }), { minTokS: 60, budgetUsd: 100, formFactor: 'laptop' })).toEqual(['no-fit']);
    expect(misses(mark({ status: 'compromise', tokS: 5 }), { minTokS: 12, budgetUsd: 1000, formFactor: 'laptop' })).toEqual(['compromise', 'speed', 'price', 'form']);
    expect(misses(mark({ tokS: null }), { minTokS: 0, budgetUsd: null, formFactor: 'any' })).toEqual([]);
    expect(misses(mark({ tokS: null }), { minTokS: 5, budgetUsd: null, formFactor: 'any' })).toEqual(['unknown-speed']);
    expect(misses(mark({ priceUsd: null }), { minTokS: 0, budgetUsd: 3000, formFactor: 'any' })).toEqual(['unpriced']);
    expect(misses(mark({ priceUsd: null }), ANY)).toEqual([]);
    expect(misses(mark({ formFactor: 'any' }), { minTokS: 0, budgetUsd: null, formFactor: 'laptop' })).toEqual([]);
  });
  it('the comparator: price first with unpriced last, then faster, then bigger', () => {
    const m = (key: string, priceUsd: number | null, tokS: number | null, gb: number) => ({ key, label: key, priceUsd, tokS, gb }) as BuyingMark;
    const sorted = [m('d', null, 50, 512), m('a', 2000, 10, 32), m('c', 2000, 10, 64), m('b', 2000, 20, 32), m('e', 1000, null, 16)].sort(byPrice);
    expect(sorted.map((x) => x.key)).toEqual(['e', 'b', 'c', 'a', 'd']);
  });
  it('the minimum-speed presets are "any" and the band edges', () => {
    expect(minSpeedPresets(factors).map((p) => [p.tokS, p.label, p.opens])).toEqual([
      [0, 'any', null],
      [5, '5', 'usable'],
      [12, '12', 'comfortable'],
      [30, '30', 'fast'],
      [60, '60', 'cloud-like'],
    ]);
  });
});

describe('label placement', () => {
  const measure = (t: string) => t.length * 4.2;
  const bounds = { left: 100, right: 884 };
  it('far-apart marks label to the right; near ones take the lanes above and below; a crowd is clipped, never overlapped', () => {
    const far = placeRowLabels([{ key: 'a', x: 200, label: 'Mac mini M6' }, { key: 'b', x: 500, label: 'MacBook Air M5' }], measure, bounds);
    expect(far.map((p) => [p.keys, p.lane, p.anchor])).toEqual([[['a'], 0, 'start'], [['b'], 0, 'start']]);
    const near = placeRowLabels([{ key: 'a', x: 200, label: 'Mac mini M6' }, { key: 'b', x: 220, label: 'MacBook Air M5' }, { key: 'c', x: 240, label: 'iMac M4' }, { key: 'd', x: 260, label: 'MacBook Pro M5' }], measure, bounds);
    expect(near.map((p) => [p.keys[0], p.lane, p.hidden])).toEqual([['a', 0, false], ['b', 1, false], ['c', 2, false], ['d', 0, false]]);
    for (const p of near) for (const q of near) if (p !== q && p.lane === q.lane && !p.hidden && !q.hidden) expect(p.x1 + 8 <= q.x0 || q.x1 + 8 <= p.x0).toBe(true);
  });
  it('a stack of up to three gets a label per mark, one under another; a bigger one, or one without room beside it, a joined label', () => {
    const two = placeRowLabels([{ key: 'a', x: 400, label: 'MacBook Pro M5 Max 40-core', bold: true }, { key: 'b', x: 402, label: 'Mac Studio M5 Max 40-core' }], measure, bounds);
    expect(two.map((p) => [p.keys, p.text, p.dy, p.anchor, p.bold])).toEqual([
      [['a'], 'MacBook Pro M5 Max 40-core', -3, 'start', true],
      [['b'], 'Mac Studio M5 Max 40-core', 11, 'start', false],
    ]);
    expect(two[0].stackDy).toEqual({ a: -7, b: 7 });
    expect(two[0].x).toBe(401);
    const three = placeRowLabels([{ key: 'a', x: 400, label: 'A' }, { key: 'b', x: 400, label: 'B' }, { key: 'c', x: 400, label: 'C' }], measure, bounds);
    expect(three.map((p) => p.dy)).toEqual([-8, 4, 16]);
    expect(three[0].stackDy).toEqual({ a: -12, b: 0, c: 12 });
    const four = placeRowLabels(['a', 'b', 'c', 'd'].map((k) => ({ key: k, x: 400, label: `Machine ${k}` })), measure, bounds);
    expect(four).toHaveLength(1);
    expect(four[0]).toMatchObject({ keys: ['a', 'b', 'c', 'd'], text: 'Machine a / Machine b / Machine c / Machine d', lane: 0 });
    // no room beside a stack: the pair to its right sits 30 px away, so the labels go left of it
    const crowded = placeRowLabels([{ key: 'a', x: 400, label: 'Mac mini M5 Pro' }, { key: 'b', x: 400, label: 'MacBook Pro M5 Pro' }, { key: 'c', x: 430, label: 'Mac Studio M5 Max' }, { key: 'd', x: 430, label: 'MacBook Pro M5 Max' }], measure, bounds);
    expect(crowded.filter((p) => p.keys.includes('a') || p.keys.includes('b')).every((p) => p.anchor === 'end')).toBe(true);
    expect(crowded.filter((p) => p.keys.includes('c') || p.keys.includes('d')).every((p) => p.anchor === 'start')).toBe(true);
    // the pick sits 45 px left of a pair: its long tagged label cannot go beside it, so it takes the far lane above, whole, placed before the pair
    const spark = placeRowLabels([{ key: 's', x: 153, label: 'NVIDIA DGX Spark · cheapest that qualifies', bold: true }, { key: 'a', x: 198, label: 'Mac Studio M5 Max 40-core' }, { key: 'b', x: 198, label: 'MacBook Pro M5 Max 40-core' }], measure, { left: 68, right: 884 });
    const pick = spark.find((p) => p.keys[0] === 's')!;
    expect(pick).toMatchObject({ lane: 3, anchor: 'middle', text: 'NVIDIA DGX Spark · cheapest that qualifies', hidden: false, bold: true });
    expect(pick.dy).toBeLessThan(-14);
    expect(spark.filter((p) => p.keys[0] !== 's').every((p) => p.lane === 0 && p.anchor === 'start' && !p.hidden)).toBe(true);
    const edge = placeRowLabels([{ key: 'a', x: 870, label: 'Mac Studio M5 Ultra' }], measure, bounds);
    expect(edge[0]).toMatchObject({ lane: 0, anchor: 'end', dx: -9 });
    expect(joinLabels(['Mac mini M6', 'MacBook Air M5', 'MacBook Pro M5'])).toBe('Mac mini M6 / MacBook Air M5 / MacBook Pro M5');
    expect(joinLabels(['MacBook Air M5', 'MacBook Pro M5'])).toBe('MacBook Air / MacBook Pro M5');
  });
});

describe('linked pools on the map', () => {
  const linkedState: AppState = { ...base, linked: { 'nvidia-dgx-spark': { count: 2, split: 'layer' } } };

  it('a linked Spark adds one pooled mark on the 256 GB row at the summed price, beside the single one', () => {
    const plain = atlasFor('glm-5.3-flash');
    expect(plain.builds.get('Q4')!.marks).toHaveLength(30);
    const atlas = atlasFor('glm-5.3-flash', linkedState);
    const marks = atlas.builds.get('Q4')!.marks;
    expect(marks).toHaveLength(31);
    const pair = marks.find((m) => m.key === 'nvidia-dgx-spark:128:x2')!;
    expect(pair).toMatchObject({ gb: 256, priceUsd: 9398, label: '2 × NVIDIA DGX Spark · 128 GB each · 256 GB pooled', short: '2 × NVIDIA DGX Spark', status: 'fits', linked: { count: 2, split: 'layer' } });
    expect(pair.tokS).toBeGreaterThan(10);
    expect(marks.find((m) => m.key === 'nvidia-dgx-spark:128')).toMatchObject({ gb: 128, priceUsd: 4699, status: 'compromise' });
    expect(atlas.rows).toEqual([16, 24, 32, 36, 48, 64, 96, 128, 256, 512]);
    expect(atlas.hasOwnMachines).toBe(true);
    const map = buyingMap(atlas, 'Q4', ANY, toggles);
    expect(map.rows.find((r) => r.gb === 256)!.marks.map((m) => m.key)).toContain('nvidia-dgx-spark:128:x2');
    expect(map.captionParts.some((p) => /linked pools/.test(p))).toBe(true);
    expect(atlasKey(linkedState, col('glm-5.3-flash'))).not.toBe(atlasKey(base, col('glm-5.3-flash')));
  });

  it('the cheapest single machine that runs GLM-5.3 Flash as asked is the 256 GB M5 Ultra', () => {
    const column = col('glm-5.3-flash', 'auto', 32768, linkedState);
    const option = buildOptions(loadModel('glm-5.3-flash'), column, 'gguf').find((o) => o.key === 'Q4')!;
    const single = cheapestSingleThatRuns(loadModel('glm-5.3-flash'), option, groups, linkedState, factors);
    expect(single?.key).toBe('mac-studio-m5-ultra:256');
    expect(single?.priceUsd).toBe(9499);
    expect(single?.linked).toBeNull();
    const kimi = col('kimi-k3');
    const none = cheapestSingleThatRuns(loadModel('kimi-k3'), buildOptions(loadModel('kimi-k3'), kimi, 'gguf').find((o) => o.key === 'Q4')!, groups, base, factors);
    expect(none).toBeNull();
  });
});
