import { describe, expect, it } from 'vitest';
import { defaultState } from './defaults';
import { MAX_COLUMNS, MAX_GROUPS, modelView, normaliseLinked, quantKey, reducer, type AppState } from './state';

const s0 = defaultState();

describe('reducer', () => {
  it('adds, removes and moves groups within the cap', () => {
    let s = reducer(s0, { type: 'ADD_GROUP', id: 'macbook-pro-m5-max' });
    expect(s.groups).toHaveLength(4);
    s = reducer(s, { type: 'ADD_GROUP', id: 'macbook-pro-m5-max' }); // a repeat links a second machine, it takes no slot
    expect(s.groups).toHaveLength(4);
    expect(s.linked['macbook-pro-m5-max']).toEqual({ count: 2, split: 'layer' });
    s = reducer(s, { type: 'SET_LINKED', groupId: 'macbook-pro-m5-max', count: 1 });
    expect(s.linked['macbook-pro-m5-max']).toBeUndefined();
    for (const id of ['a', 'b', 'c', 'd']) s = reducer(s, { type: 'ADD_GROUP', id });
    expect(s.groups).toHaveLength(MAX_GROUPS);
    s = reducer(s, { type: 'MOVE_GROUP', id: 'nvidia-dgx-spark', to: 0 });
    expect(s.groups[0]).toBe('nvidia-dgx-spark');
    s = reducer({ ...s, view: { kind: 'speed', row: 'nvidia-dgx-spark:128' } }, { type: 'REMOVE_GROUP', id: 'nvidia-dgx-spark' });
    expect(s.groups).not.toContain('nvidia-dgx-spark');
    expect(s.view.kind).toBe('table');
  });

  it('columns: add with defaults, per-column settings, all-columns, remove shifts the model view', () => {
    let s = reducer(s0, { type: 'ADD_COLUMN', id: 'gemma-4-26b-a4b' });
    expect(s.columns[4]).toEqual({ id: 'gemma-4-26b-a4b', quant: 'auto', ctx: 32768 });
    s = reducer(s, { type: 'SET_COLUMN_QUANT', index: 4, quant: { label: 'Q8_0' } });
    s = reducer(s, { type: 'SET_COLUMN_CTX', index: 4, ctx: 131072 });
    s = reducer(s, { type: 'SET_COLUMN_KV', index: 4, kvBits: 4 });
    expect(s.columns[4]).toEqual({ id: 'gemma-4-26b-a4b', quant: { label: 'Q8_0' }, ctx: 131072, kvBits: 4 });
    s = reducer(s, { type: 'APPLY_ALL_COLUMNS', quant: 'auto', ctx: 8192 });
    expect(s.columns.every((c) => c.quant === 'auto' && c.ctx === 8192)).toBe(true);
    for (let i = 0; i < 10; i++) s = reducer(s, { type: 'ADD_COLUMN', id: `m${i}` });
    expect(s.columns).toHaveLength(MAX_COLUMNS);
    s = reducer({ ...s, view: { kind: 'model', column: 3, build: 'column', minTokS: 0 } }, { type: 'REMOVE_COLUMN', index: 1 });
    expect(s.view).toEqual({ kind: 'model', column: 2, build: 'column', minTokS: 0 });
    s = reducer(s, { type: 'REMOVE_COLUMN', index: 2 });
    expect(s.view.kind).toBe('table');
  });

  it('toggles clamp their values', () => {
    expect(reducer(s0, { type: 'SET_WORK', gb: 999 }).work).toBe(64);
    expect(reducer(s0, { type: 'SET_CAP', cap: 5 }).cap).toBe(1);
    expect(reducer(s0, { type: 'SET_CAP', cap: null }).cap).toBeNull();
    expect(reducer(s0, { type: 'SET_FLOOR', bits: 0 }).floorBits).toBe(1);
    expect(reducer(s0, { type: 'SET_RUNTIME', runtime: 'mlx' }).runtime).toBe('mlx');
  });

  it('custom entries add their column or group; local hydration never overrides the URL', () => {
    let s = reducer(s0, { type: 'ADD_CUSTOM_MODEL', model: { mode: 'quick', name: 'Mine', paramsTotalB: 70, weightsGb: 40 }, columnId: 'custom-mine' });
    expect(s.customModels).toHaveLength(1);
    expect(s.columns.at(-1)?.id).toBe('custom-mine');
    s = reducer(s, { type: 'ADD_CUSTOM_MACHINE', machine: { name: 'Box', memoryGb: 128, bandwidthGbs: 300, platform: 'cuda' }, groupId: 'custom-box' });
    expect(s.groups.at(-1)).toBe('custom-box');
    s = reducer(s, { type: 'HYDRATE_LOCAL', customModels: [{ mode: 'quick', name: 'Mine', paramsTotalB: 1, weightsGb: 1 }, { mode: 'quick', name: 'Other', paramsTotalB: 7, weightsGb: 4 }] });
    expect(s.customModels.find((m) => m.name === 'Mine')?.paramsTotalB).toBe(70);
    expect(s.customModels).toHaveLength(2);
    s = reducer(s, { type: 'REMOVE_CUSTOM', kind: 'model', name: 'Other' });
    expect(s.customModels).toHaveLength(1);
    const replaced: AppState = { ...s0, work: 16 };
    expect(reducer(s, { type: 'RESET', state: replaced })).toBe(replaced);
  });
});

describe('MOVE_COLUMN', () => {
  it('reorders columns and keeps the one-model view on the same model', () => {
    let s = defaultState();
    s = reducer({ ...s, view: { kind: 'model', column: 0, build: 'column', minTokS: 0 } }, { type: 'MOVE_COLUMN', index: 0, to: 2 });
    expect(s.columns.map((c) => c.id)).toEqual(['qwen3.8-flash-next', 'glm-5.3-flash', 'qwen3.8-27b', 'kimi-k3']);
    expect(s.view).toEqual({ kind: 'model', column: 2, build: 'column', minTokS: 0 });
    expect(reducer(s, { type: 'MOVE_COLUMN', index: 9, to: 0 })).toBe(s);
    expect(reducer(s, { type: 'MOVE_COLUMN', index: 1, to: 1 })).toBe(s);
    const clamped = reducer(s, { type: 'MOVE_COLUMN', index: 0, to: 99 });
    expect(clamped.columns[3].id).toBe('qwen3.8-flash-next');
  });
  it('APPLY_ALL_COLUMNS sets a bucket on every column without needing the records', () => {
    const s = reducer(defaultState(), { type: 'APPLY_ALL_COLUMNS', quant: { bucket: 'Q8' } });
    expect(s.columns.every((c) => typeof c.quant === 'object' && 'bucket' in c.quant && c.quant.bucket === 'Q8')).toBe(true);
  });
});

describe('columns at different settings and hidden sizes', () => {
  it('ADD_COLUMN allows the same model at another quant or context and refuses an exact repeat', () => {
    const s0 = defaultState();
    expect(reducer(s0, { type: 'ADD_COLUMN', id: 'qwen3.8-27b', ctx: 32768 })).toBe(s0);
    const s1 = reducer(s0, { type: 'ADD_COLUMN', id: 'qwen3.8-27b', quant: { bucket: 'Q8' }, ctx: 131072 });
    expect(s1.columns.filter((c) => c.id === 'qwen3.8-27b')).toHaveLength(2);
    expect(reducer(s1, { type: 'ADD_COLUMN', id: 'qwen3.8-27b', quant: { bucket: 'Q8' }, ctx: 131072 })).toBe(s1);
  });
  it('TOGGLE_SIZE hides and shows sizes, keeps one visible, and cleans up', () => {
    const sizes = [96, 256, 512];
    let s = defaultState();
    s = reducer(s, { type: 'TOGGLE_SIZE', groupId: 'mac-studio-m5-ultra', gb: 96, sizes });
    expect(s.hiddenSizes).toEqual({ 'mac-studio-m5-ultra': [96] });
    s = reducer(s, { type: 'TOGGLE_SIZE', groupId: 'mac-studio-m5-ultra', gb: 512, sizes });
    expect(s.hiddenSizes['mac-studio-m5-ultra']).toEqual([96, 512]);
    expect(reducer(s, { type: 'TOGGLE_SIZE', groupId: 'mac-studio-m5-ultra', gb: 256, sizes })).toBe(s); // the last visible size stays
    expect(reducer(s, { type: 'TOGGLE_SIZE', groupId: 'mac-studio-m5-ultra', gb: 999, sizes })).toBe(s);
    s = reducer(s, { type: 'TOGGLE_SIZE', groupId: 'mac-studio-m5-ultra', gb: 96, sizes });
    s = reducer(s, { type: 'TOGGLE_SIZE', groupId: 'mac-studio-m5-ultra', gb: 512, sizes });
    expect(s.hiddenSizes).toEqual({});
  });
  it('SET_VIEW stores the machine view; REMOVE_GROUP leaves it alone (its pick may be any catalog machine)', () => {
    const s = reducer(defaultState(), { type: 'SET_VIEW', view: { kind: 'machine', row: 'mac-mini-m6:32', expanded: false } });
    expect(s.view).toEqual({ kind: 'machine', row: 'mac-mini-m6:32', expanded: false });
    expect(reducer(s, { type: 'REMOVE_GROUP', id: 'mac-mini-m6' }).view).toEqual({ kind: 'machine', row: 'mac-mini-m6:32', expanded: false });
  });
  it('REMOVE_GROUP clears a memory view on that group and its hidden sizes; MOVE_GROUP keeps them', () => {
    let s = reducer(defaultState(), { type: 'TOGGLE_SIZE', groupId: 'mac-mini-m6', gb: 16, sizes: [16, 24, 32] });
    s = { ...s, view: { kind: 'memory', row: 'mac-mini-m6:24' } };
    const moved = reducer(s, { type: 'MOVE_GROUP', id: 'mac-mini-m6', to: 2 });
    expect(moved.hiddenSizes).toEqual({ 'mac-mini-m6': [16] });
    const removed = reducer(s, { type: 'REMOVE_GROUP', id: 'mac-mini-m6' });
    expect(removed.view).toEqual({ kind: 'table' });
    expect(removed.hiddenSizes).toEqual({});
  });
});

describe('the buying map settings', () => {
  it('SET_BUDGET clamps to the allowed range and clears on nonsense', () => {
    expect(reducer(s0, { type: 'SET_BUDGET', usd: 50 }).budgetUsd).toBe(100);
    expect(reducer(s0, { type: 'SET_BUDGET', usd: 1e9 }).budgetUsd).toBe(200000);
    expect(reducer(s0, { type: 'SET_BUDGET', usd: 3499.6 }).budgetUsd).toBe(3500);
    expect(reducer(s0, { type: 'SET_BUDGET', usd: Number.NaN }).budgetUsd).toBeNull();
    expect(reducer(reducer(s0, { type: 'SET_BUDGET', usd: 2000 }), { type: 'SET_BUDGET', usd: null }).budgetUsd).toBeNull();
    expect(reducer(s0, { type: 'SET_FORM_FACTOR', formFactor: 'laptop' }).formFactor).toBe('laptop');
  });
  it('modelView keeps the build and the minimum speed only when the visitor is already on the map', () => {
    expect(modelView(2)).toEqual({ kind: 'model', column: 2, build: 'column', minTokS: 0 });
    expect(modelView(2, { kind: 'table' })).toEqual({ kind: 'model', column: 2, build: 'column', minTokS: 0 });
    expect(modelView(1, { kind: 'model', column: 0, build: 'Q8', minTokS: 30 })).toEqual({ kind: 'model', column: 1, build: 'Q8', minTokS: 30 });
  });
});

describe('quantKey', () => {
  it('tells the three kinds of setting apart', () => {
    expect(quantKey('auto')).toBe('auto');
    expect(quantKey({ bucket: 'Q8' })).toBe('b:Q8');
    expect(quantKey({ label: 'Q8_0' })).toBe('l:Q8_0');
    expect(quantKey({ special: 0 })).toBe('s:0');
  });
});

describe('linked machines in the reducer', () => {
  it('adding a machine already in the table links another, up to four; SET_LINKED sets, normalises and clears', () => {
    let s = reducer(s0, { type: 'ADD_GROUP', id: 'nvidia-dgx-spark' });
    expect(s.groups).toHaveLength(3);
    expect(s.linked['nvidia-dgx-spark']).toEqual({ count: 2, split: 'layer' });
    s = reducer(s, { type: 'ADD_GROUP', id: 'nvidia-dgx-spark' });
    s = reducer(s, { type: 'ADD_GROUP', id: 'nvidia-dgx-spark' });
    expect(s.linked['nvidia-dgx-spark'].count).toBe(4);
    const capped = reducer(s, { type: 'ADD_GROUP', id: 'nvidia-dgx-spark' });
    expect(capped).toBe(s);
    s = reducer(s, { type: 'SET_LINKED', groupId: 'nvidia-dgx-spark', count: 3, split: 'tensor' });
    expect(s.linked['nvidia-dgx-spark']).toEqual({ count: 4, split: 'tensor' }); // tensor needs 2 or 4
    s = reducer(s, { type: 'SET_LINKED', groupId: 'nvidia-dgx-spark', count: 2 });
    expect(s.linked['nvidia-dgx-spark']).toEqual({ count: 2, split: 'tensor' }); // the split is kept
    s = reducer(s, { type: 'SET_LINKED', groupId: 'nvidia-dgx-spark', count: 1 });
    expect(s.linked).toEqual({});
    expect(reducer(s, { type: 'SET_LINKED', groupId: 'not-in-table', count: 2 })).toBe(s);
    expect(normaliseLinked(5)).toEqual({ count: 4, split: 'layer' });
    expect(normaliseLinked(0)).toBeNull();
  });

  it('an open view follows the pool, and removing the group clears it', () => {
    let s = reducer({ ...s0, view: { kind: 'speed', row: 'nvidia-dgx-spark:128' } }, { type: 'SET_LINKED', groupId: 'nvidia-dgx-spark', count: 2 });
    expect(s.view).toEqual({ kind: 'speed', row: 'nvidia-dgx-spark:128:x2' });
    s = reducer(s, { type: 'SET_LINKED', groupId: 'nvidia-dgx-spark', count: 4, split: 'tensor' });
    expect(s.view).toEqual({ kind: 'speed', row: 'nvidia-dgx-spark:128:x4t' });
    s = reducer(s, { type: 'SET_LINKED', groupId: 'nvidia-dgx-spark', count: 1 });
    expect(s.view).toEqual({ kind: 'speed', row: 'nvidia-dgx-spark:128' });
    s = reducer(s, { type: 'SET_LINKED', groupId: 'nvidia-dgx-spark', count: 2 });
    s = reducer(s, { type: 'REMOVE_GROUP', id: 'nvidia-dgx-spark' });
    expect(s.linked).toEqual({});
    expect(s.view.kind).toBe('table');
  });

  it('re-submitting a custom machine replaces it without linking a second', () => {
    const machine = { name: 'My Halo box', memoryGb: 128, bandwidthGbs: 256, platform: 'rocm' as const };
    let s = reducer(s0, { type: 'ADD_CUSTOM_MACHINE', machine, groupId: 'custom-my-halo-box' });
    expect(s.groups).toContain('custom-my-halo-box');
    s = reducer(s, { type: 'ADD_CUSTOM_MACHINE', machine: { ...machine, bandwidthGbs: 300 }, groupId: 'custom-my-halo-box' });
    expect(s.linked['custom-my-halo-box']).toBeUndefined();
    expect(s.customMachines.find((m) => m.name === 'My Halo box')?.bandwidthGbs).toBe(300);
    s = reducer(s, { type: 'ADD_GROUP', id: 'custom-my-halo-box' });
    expect(s.linked['custom-my-halo-box']).toEqual({ count: 2, split: 'layer' });
  });
});
