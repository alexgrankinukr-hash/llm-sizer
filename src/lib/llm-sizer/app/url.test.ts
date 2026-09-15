import { describe, expect, it } from 'vitest';
import { defaultState } from './defaults';
import type { AppState } from './state';
import { decodeState, encodeState } from './url';

const s0 = defaultState();

describe('share-link codec', () => {
  it('the default state is short, URL-safe and round-trips', () => {
    const s = encodeState(s0);
    expect(s.length).toBeLessThan(220);
    expect(encodeURIComponent(s)).toBe(s);
    expect(s.startsWith('1~g')).toBe(true);
    const back = decodeState(s, s0);
    expect(back.ok).toBe(true);
    expect(back.state).toEqual(s0);
  });

  it('a maximal state round-trips (custom entries, views, every toggle)', () => {
    const max: AppState = {
      ...s0,
      groups: ['mac-mini-m6', 'custom-my-halo-box'],
      columns: [
        { id: 'glm-5.3-flash', quant: { label: 'UD-Q4_K_XL', repo: 'unsloth' }, ctx: 131072, kvBits: 4 },
        { id: 'qwen3.8-27b', quant: 'auto', ctx: 12000 },
        { id: 'custom-my-70b', quant: 'auto', ctx: 32768 },
      ],
      work: 16,
      cap: 1,
      runtime: 'mlx',
      kvBits: 8,
      floorBits: 3,
      simpleQuants: false,
      view: { kind: 'model', column: 1, build: 'Q8', minTokS: 30 },
      budgetUsd: 3500,
      formFactor: 'laptop',
      customModels: [
        { mode: 'quick', name: 'My 70B', paramsTotalB: 70, paramsActiveB: 12, weightsGb: 40, contextMax: 65536 },
        { mode: 'full', name: 'Odd ~*! model', paramsTotalB: 9, weightsGb: 5, attention: 'mla', layers: 40, kvLayers: 10, kvHeads: 8, headDim: 128, kEqV: true, kvLoraRank: 512, ropeDim: 64, slidingLayers: 0 },
      ],
      customMachines: [{ name: 'My Halo box', memoryGb: 128, bandwidthGbs: 256, platform: 'rocm', chip: 'Ryzen AI Max+' }],
    };
    const s = encodeState(max);
    expect(s).not.toMatch(/[~*!]{2}/);
    const back = decodeState(s, s0);
    expect(back.ok).toBe(true);
    expect(back.state).toEqual(max);
    const viaSearch = new URLSearchParams(`s=${encodeURIComponent(s)}`).get('s');
    expect(decodeState(viaSearch, s0).state).toEqual(max);
  });

  it('drops unknown groups with a notice when the catalog is known, keeps unknown models for later', () => {
    const s = encodeState({ ...s0, groups: ['mac-mini-m6', 'gone-machine'], columns: [{ id: 'qwen3.8-27b', quant: 'auto', ctx: 32768 }, { id: 'gone-model', quant: 'auto', ctx: 8192 }] });
    const back = decodeState(s, s0, { knownGroups: new Set(['mac-mini-m6']) });
    expect(back.state.groups).toEqual(['mac-mini-m6']);
    expect(back.notices[0]).toMatch(/gone-machine/);
    expect(back.state.columns.map((c) => c.id)).toEqual(['qwen3.8-27b', 'gone-model']);
  });

  it('garbage, wrong versions and empty links fall back to the defaults', () => {
    expect(decodeState('', s0).ok).toBe(false);
    expect(decodeState('2~gmac-mini-m6', s0)).toMatchObject({ ok: false, state: s0 });
    expect(decodeState('1~zzz', s0)).toMatchObject({ ok: false, state: s0 });
    expect(decodeState('1~g!!~c*', s0).ok).toBe(false);
    const odd = decodeState('1~gmac-mini-m6~cqwen3.8-27b*Q4_K_M*banana*k7~w-5~o9~rzzz~k3~f0~vmodel*9', s0);
    expect(odd.ok).toBe(true);
    expect(odd.state.columns).toEqual([]);
    expect(odd.state.work).toBe(0);
    expect(odd.state.cap).toBe(1);
    expect(odd.state.runtime).toBe('gguf');
    expect(odd.state.kvBits).toBe(16);
    expect(odd.state.floorBits).toBe(1);
    expect(odd.state.view.kind).toBe('table');
  });
});

describe('bucket settings in the link', () => {
  it('round-trips a bucket as its name and keeps labels as labels', () => {
    const s = { ...defaultState(), columns: [{ id: 'qwen3.8-27b', quant: { bucket: 'Q8' as const }, ctx: 32768 }, { id: 'glm-5.3-flash', quant: { label: 'UD-Q4_K_XL' }, ctx: 32768 }] };
    const encoded = encodeState(s);
    expect(encoded).toContain('qwen3.8-27b*Q8*32');
    const back = decodeState(encoded, defaultState());
    expect(back.state.columns[0].quant).toEqual({ bucket: 'Q8' });
    expect(back.state.columns[1].quant).toEqual({ label: 'UD-Q4_K_XL' });
  });
  it('tells the reader when a link carries a custom model this browser does not have', () => {
    const back = decodeState('1~gmac-mini-m6~ccustom-my-model*-*32!qwen3.8-27b*-*32', defaultState());
    expect(back.state.columns.map((c) => c.id)).toEqual(['qwen3.8-27b']);
    expect(back.notices.some((n) => n.includes('custom model'))).toBe(true);
  });
});

describe('memory view, hidden sizes and repeated models in the link', () => {
  it('round-trips the memory view and drops it when its group is absent', () => {
    const s: AppState = { ...defaultState(), view: { kind: 'memory', row: 'mac-mini-m6:32' } };
    const encoded = encodeState(s);
    expect(encoded).toContain('~vmemory*mac-mini-m6%3A32');
    expect(decodeState(encoded, defaultState()).state.view).toEqual({ kind: 'memory', row: 'mac-mini-m6:32' });
    expect(decodeState('1~gnvidia-dgx-spark~cqwen3.8-27b*-*32~vmemory*mac-mini-m6%3A32', defaultState()).state.view).toEqual({ kind: 'table' });
  });
  it('round-trips hidden sizes on the group item and ignores garbage', () => {
    const s: AppState = { ...defaultState(), hiddenSizes: { 'mac-mini-m6': [16, 24] } };
    const encoded = encodeState(s);
    expect(encoded).toContain('gmac-mini-m6*16*24!mac-studio-m5-ultra');
    expect(decodeState(encoded, defaultState()).state.hiddenSizes).toEqual({ 'mac-mini-m6': [16, 24] });
    expect(decodeState('1~gmac-mini-m6*x*-3*24*24~cqwen3.8-27b*-*32', defaultState()).state.hiddenSizes).toEqual({ 'mac-mini-m6': [24] });
  });
  it('decodes a link from before hidden sizes existed and keeps repeated model ids', () => {
    const back = decodeState('1~gmac-mini-m6~cqwen3.8-27b*-*32!qwen3.8-27b*Q8*128', defaultState());
    expect(back.state.hiddenSizes).toEqual({});
    expect(back.state.columns.map((c) => [c.id, c.ctx])).toEqual([['qwen3.8-27b', 32768], ['qwen3.8-27b', 131072]]);
  });
});

describe('the machine view in the link', () => {
  it('round-trips the picked machine and the expanded flag', () => {
    const s: AppState = { ...defaultState(), view: { kind: 'machine', row: 'mac-studio-m5-ultra:256', expanded: true } };
    const encoded = encodeState(s);
    expect(encoded).toContain('~vmachine*mac-studio-m5-ultra%3A256*x');
    expect(decodeState(encoded, defaultState()).state.view).toEqual({ kind: 'machine', row: 'mac-studio-m5-ultra:256', expanded: true });
    const collapsed = encodeState({ ...s, view: { kind: 'machine', row: 'mac-studio-m5-ultra:256', expanded: false } });
    expect(collapsed).toContain('~vmachine*mac-studio-m5-ultra%3A256');
    expect(collapsed).not.toContain('*x');
    expect(decodeState(collapsed, defaultState()).state.view).toEqual({ kind: 'machine', row: 'mac-studio-m5-ultra:256', expanded: false });
  });
  it('keeps a machine that is not in the table, drops one the catalog does not know, keeps a custom one the link carries', () => {
    const link = '1~gmac-mini-m6~cqwen3.8-27b*-*32~vmachine*macbook-pro-m4-max%3A128';
    expect(decodeState(link, defaultState()).state.view).toEqual({ kind: 'machine', row: 'macbook-pro-m4-max:128', expanded: false });
    const known = new Set(['mac-mini-m6', 'macbook-pro-m4-max']);
    expect(decodeState(link, defaultState(), { knownGroups: known }).state.view.kind).toBe('machine');
    expect(decodeState(link, defaultState(), { knownGroups: new Set(['mac-mini-m6']) }).state.view).toEqual({ kind: 'table' });
    const custom = '1~gmac-mini-m6~cqwen3.8-27b*-*32~vmachine*custom-my-halo-box%3A128~yMy%20Halo%20box*128*256*rocm*-';
    expect(decodeState(custom, defaultState(), { knownGroups: new Set(['mac-mini-m6']) }).state.view).toEqual({ kind: 'machine', row: 'custom-my-halo-box:128', expanded: false });
    expect(decodeState('1~gmac-mini-m6~cqwen3.8-27b*-*32~vmachine*garbage', defaultState(), { knownGroups: new Set(['mac-mini-m6']) }).state.view).toEqual({ kind: 'table' });
  });
});

describe('the buying map in the link', () => {
  it('encodes the plain map as before and round-trips a build and a minimum speed', () => {
    const plain = encodeState({ ...s0, view: { kind: 'model', column: 0, build: 'column', minTokS: 0 } });
    expect(plain).toContain('~vmodel*0');
    expect(plain).not.toContain('~vmodel*0*');
    expect(decodeState(plain, s0).state.view).toEqual({ kind: 'model', column: 0, build: 'column', minTokS: 0 });
    const speedOnly = encodeState({ ...s0, view: { kind: 'model', column: 0, build: 'column', minTokS: 12 } });
    expect(speedOnly).toContain('~vmodel*0*-*12');
    expect(decodeState(speedOnly, s0).state.view).toEqual({ kind: 'model', column: 0, build: 'column', minTokS: 12 });
    const q8 = encodeState({ ...s0, view: { kind: 'model', column: 2, build: 'Q8', minTokS: 0 } });
    expect(q8).toContain('~vmodel*2*Q8');
    expect(decodeState(q8, s0).state.view).toEqual({ kind: 'model', column: 2, build: 'Q8', minTokS: 0 });
  });
  it('opens links made before the map at the map defaults, and still drops a column that is not there', () => {
    expect(decodeState('1~gmac-mini-m6~cqwen3.8-27b*-*32~vmodel*0*c', s0).state.view).toEqual({ kind: 'model', column: 0, build: 'column', minTokS: 0 });
    expect(decodeState('1~gmac-mini-m6~cqwen3.8-27b*-*32~vmodel*0*t', s0).state.view).toEqual({ kind: 'model', column: 0, build: 'column', minTokS: 0 });
    expect(decodeState('1~gmac-mini-m6~cqwen3.8-27b*-*32~vmodel*0*Q6*-4', s0).state.view).toEqual({ kind: 'model', column: 0, build: 'column', minTokS: 0 });
    expect(decodeState('1~gmac-mini-m6~cqwen3.8-27b*-*32~vmodel*9*t', s0).state.view).toEqual({ kind: 'table' });
  });
  it('budget and form factor: garbage is ignored, and the old owned-machine section still decodes to a normal link', () => {
    const s = decodeState('1~gmac-mini-m6~cqwen3.8-27b*-*32~b2000~pl~mmac-mini-m6%3A32', s0);
    expect(s.state.budgetUsd).toBe(2000);
    expect(s.state.formFactor).toBe('laptop');
    expect(s.state.groups).toEqual(['mac-mini-m6']);
    expect(s.ok).toBe(true);
    expect('ownedRow' in s.state).toBe(false);
    const junk = decodeState('1~gmac-mini-m6~cqwen3.8-27b*-*32~b-5~px~mgarbage', s0);
    expect(junk.state.budgetUsd).toBeNull();
    expect(junk.state.formFactor).toBe('any');
    expect(decodeState('1~gmac-mini-m6~cqwen3.8-27b*-*32~bzzz~pd', s0).state).toMatchObject({ budgetUsd: null, formFactor: 'desktop' });
    expect(decodeState('1~gmac-mini-m6~cqwen3.8-27b*-*32~b999999999', s0).state.budgetUsd).toBe(200000);
    expect(encodeState(s.state)).not.toContain('~m');
  });
});

describe('special builds in the link', () => {
  it('writes a special build as s<n>, reads it back, and keeps the ssd build of the map', () => {
    const state = { ...s0, columns: [{ id: 'qwen3.8-flash-next', quant: { special: 1 } as const, ctx: 32768 }], view: { kind: 'model' as const, column: 0, build: 'ssd' as const, minTokS: 0 } };
    const enc = encodeState(state);
    expect(enc).toContain('qwen3.8-flash-next*s1*32');
    expect(enc).toContain('~vmodel*0*ssd');
    const back = decodeState(enc, s0).state;
    expect(back.columns[0]).toEqual({ id: 'qwen3.8-flash-next', quant: { special: 1 }, ctx: 32768 });
    expect(back.view).toEqual({ kind: 'model', column: 0, build: 'ssd', minTokS: 0 });
    expect(decodeState('1~gmac-mini-m6~cqwen3.8-flash-next*s0*32', s0).state.columns[0].quant).toEqual({ special: 0 });
    expect(decodeState('1~gmac-mini-m6~cqwen3.8-flash-next*-*32', s0).state.columns[0].quant).toBe('auto'); // a pre-0.11 link
    expect(decodeState('1~gmac-mini-m6~cqwen3.8-flash-next*-*32~vmodel*0*zzz', s0).state.view).toMatchObject({ build: 'column' });
  });
});

describe('linked machines in the link', () => {
  it('round-trips the count and the split on the group item, beside hidden sizes', () => {
    const s: AppState = { ...defaultState(), groups: ['nvidia-dgx-spark', 'mac-mini-m6'], linked: { 'nvidia-dgx-spark': { count: 2, split: 'layer' }, 'mac-mini-m6': { count: 4, split: 'tensor' } }, hiddenSizes: { 'mac-mini-m6': [16, 24] } };
    const encoded = encodeState(s);
    expect(encoded).toContain('gnvidia-dgx-spark*x2!mac-mini-m6*x4t*16*24');
    const back = decodeState(encoded, defaultState()).state;
    expect(back.linked).toEqual(s.linked);
    expect(back.hiddenSizes).toEqual({ 'mac-mini-m6': [16, 24] });
    // the token may sit anywhere among the sizes; bad tokens are ignored; a count of 3 in tensor becomes 4
    expect(decodeState('1~gmac-mini-m6*16*x2*24~cqwen3.8-27b*-*32', defaultState()).state.linked).toEqual({ 'mac-mini-m6': { count: 2, split: 'layer' } });
    expect(decodeState('1~gmac-mini-m6*x3t~cqwen3.8-27b*-*32', defaultState()).state.linked).toEqual({ 'mac-mini-m6': { count: 4, split: 'tensor' } });
    expect(decodeState('1~gmac-mini-m6*x5*x1*xt*x*24~cqwen3.8-27b*-*32', defaultState()).state).toMatchObject({ linked: {}, hiddenSizes: { 'mac-mini-m6': [24] } });
  });

  it('a pooled row in a view round-trips, and a link from before pools carries none', () => {
    const s: AppState = { ...defaultState(), linked: { 'nvidia-dgx-spark': { count: 2, split: 'tensor' } }, view: { kind: 'speed', row: 'nvidia-dgx-spark:128:x2t' } };
    const encoded = encodeState(s);
    expect(encoded).toContain('~vspeed*nvidia-dgx-spark%3A128%3Ax2t');
    expect(decodeState(encoded, defaultState()).state.view).toEqual({ kind: 'speed', row: 'nvidia-dgx-spark:128:x2t' });
    const old = decodeState('1~gmac-mini-m6*16*24!mac-studio-m5-ultra~cqwen3.8-27b*-*32', defaultState()).state;
    expect(old.linked).toEqual({});
    expect(encodeState(old)).toContain('gmac-mini-m6*16*24!mac-studio-m5-ultra');
  });
});
