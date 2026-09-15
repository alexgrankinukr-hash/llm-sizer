/**
 * The share-link codec: AppState ⇄ the `s` query parameter.
 *
 * A short tokenized string, not compressed JSON: sections are separated by `~`, items by `!`,
 * fields by `*`; ids are `[a-z0-9.-]`, labels `[A-Za-z0-9_.-]`, so the result survives
 * `encodeURIComponent` unchanged. Sections at their defaults are omitted. Version prefix `1`. A column's build field is `-`
 * (the default), a bucket name, an exact file label, or `s<n>` for the model's n-th special build.
 *
 *   1~gmac-mini-m6!mac-studio-m5-ultra~cqwen3.8-27b*-*32!glm-5.3-flash*UD-Q4_K_XL*128*k4~w16~o1~rmlx~b3500~pl
 * Each section starts with its key letter; `(` prefixes an optional quantizer name. `b` is the budget in US
 * dollars, `p` the form-factor preference (`l` laptop, `d` desktop). An `m` section (the owned machine, 0.9.5 to 0.10.2) is ignored.
 */
import { BUCKET_RE, isBucketSetting, isLabelSetting, isSpecialSetting, type Bucket } from './quants';
import type { KvBits, Platform, Runtime } from '../engine/types';
import { customMachine, customModelFull, customModelQuick, type CustomMachineInput, type CustomModelFullInput } from '../engine/custom';
import type { AppState, CustomModel, ModelColumn, View } from './state';
import { clampBudget, clampMinTokS, MAX_COLUMNS, MAX_GROUPS, STATE_DEFAULTS } from './state';
import { parseRowKey, type LinkedInfo } from './machines';
import { normaliseLinked } from './state';

const VERSION = '1';
const SECTION = '~';
const ITEM = '!';
const FIELD = '*';
const ID_RE = /^[a-z0-9.-]{1,80}$/;
const LABEL_RE = /^[A-Za-z0-9_.-]{1,40}$/;

function encText(text: string): string {
  // custom names: percent-encode, then also the characters encodeURIComponent leaves alone that we use as separators
  return encodeURIComponent(text).replace(/[~*!'()]/g, (c) => `%${c.charCodeAt(0).toString(16).toUpperCase()}`);
}

function decText(text: string): string {
  try {
    return decodeURIComponent(text);
  } catch {
    return text;
  }
}

function encCtx(ctx: number): string {
  return ctx % 1024 === 0 ? String(ctx / 1024) : `t${ctx}`;
}

function decCtx(text: string): number | null {
  if (text.startsWith('t')) {
    const n = Number(text.slice(1));
    return Number.isFinite(n) && n > 0 ? Math.floor(n) : null;
  }
  const k = Number(text);
  return Number.isFinite(k) && k > 0 ? Math.floor(k * 1024) : null;
}

function num(text: string | undefined): number | undefined {
  if (text === undefined || text === '' || text === '-') return undefined;
  const n = Number(text);
  return Number.isFinite(n) ? n : undefined;
}

function encColumn(c: ModelColumn): string {
  // a special build is `s<index>`; its label has spaces and never enters the link
  const parts = [c.id, c.quant === 'auto' ? '-' : isBucketSetting(c.quant) ? c.quant.bucket : isSpecialSetting(c.quant) ? `s${c.quant.special}` : c.quant.label, encCtx(c.ctx)];
  if (c.kvBits) parts.push(`k${c.kvBits}`);
  if (isLabelSetting(c.quant) && c.quant.repo) parts.push(`(${encText(c.quant.repo)}`);
  return parts.join(FIELD);
}

function decColumn(text: string): ModelColumn | null {
  const [id, label, ctxText, ...rest] = text.split(FIELD);
  if (!id || !ID_RE.test(id)) return null;
  const ctx = decCtx(ctxText ?? '32');
  if (ctx === null) return null;
  const col: ModelColumn = { id, quant: 'auto', ctx };
  if (label && /^s\d{1,2}$/.test(label)) col.quant = { special: Number(label.slice(1)) };
  else if (label && label !== '-' && LABEL_RE.test(label)) col.quant = BUCKET_RE.test(label) ? { bucket: label as Bucket } : { label };
  for (const extra of rest) {
    if (/^k(4|8|16)$/.test(extra)) col.kvBits = Number(extra.slice(1)) as KvBits;
    else if (extra.startsWith('(') && isLabelSetting(col.quant)) col.quant = { ...col.quant, repo: decText(extra.slice(1)) };
  }
  return col;
}

function encCustomModel(m: CustomModel): string {
  const base = [encText(m.name), m.paramsTotalB, m.paramsActiveB ?? '-', m.weightsGb, m.contextMax ?? '-'];
  if (m.mode === 'full') {
    base.push('full', m.attention, m.layers, m.kvLayers ?? '-', m.kvHeads ?? '-', m.headDim ?? '-', m.kEqV ? 1 : 0, m.kvLoraRank ?? '-', m.ropeDim ?? '-', m.slidingLayers ?? 0, m.window ?? '-');
  }
  return base.join(FIELD);
}

function decCustomModel(text: string): CustomModel | null {
  const f = text.split(FIELD);
  const name = decText(f[0] ?? '');
  const paramsTotalB = num(f[1]);
  const weightsGb = num(f[3]);
  if (!name || !paramsTotalB || !weightsGb) return null;
  const quick = { name, paramsTotalB, paramsActiveB: num(f[2]), weightsGb, contextMax: num(f[4]) };
  if (f[5] !== 'full') return { mode: 'quick', ...quick };
  const attention = f[6] as CustomModelFullInput['attention'];
  const layers = num(f[7]);
  if (!['gqa', 'mla', 'latent'].includes(attention) || !layers) return { mode: 'quick', ...quick };
  return {
    mode: 'full',
    ...quick,
    attention,
    layers,
    kvLayers: num(f[8]),
    kvHeads: num(f[9]),
    headDim: num(f[10]),
    kEqV: f[11] === '1',
    kvLoraRank: num(f[12]),
    ropeDim: num(f[13]),
    slidingLayers: num(f[14]) ?? 0,
    window: num(f[15]),
  };
}

function encCustomMachine(m: CustomMachineInput): string {
  return [encText(m.name), m.memoryGb, m.bandwidthGbs, m.platform, m.chip ? encText(m.chip) : '-'].join(FIELD);
}

function decCustomMachine(text: string): CustomMachineInput | null {
  const f = text.split(FIELD);
  const name = decText(f[0] ?? '');
  const memoryGb = num(f[1]);
  const bandwidthGbs = num(f[2]);
  const platform = f[3] as Platform;
  if (!name || !memoryGb || !bandwidthGbs || !['apple', 'cuda', 'rocm'].includes(platform)) return null;
  const chip = f[4] && f[4] !== '-' ? decText(f[4]) : undefined;
  return { name, memoryGb, bandwidthGbs, platform, ...(chip ? { chip } : {}) };
}

function encView(v: View): string | null {
  if (v.kind === 'speed' || v.kind === 'memory') return [v.kind, encText(v.row)].join(FIELD);
  if (v.kind === 'model') {
    // trailing defaults are dropped, so the plain map still encodes as `model*0`
    const parts: (string | number)[] = ['model', v.column];
    if (v.build !== 'column' || v.minTokS > 0) parts.push(v.build === 'column' ? '-' : v.build);
    if (v.minTokS > 0) parts.push(v.minTokS);
    return parts.join(FIELD);
  }
  if (v.kind === 'machine') return ['machine', encText(v.row), ...(v.expanded ? ['x'] : [])].join(FIELD);
  return null;
}

function decView(text: string): View {
  const f = text.split(FIELD);
  if ((f[0] === 'speed' || f[0] === 'memory') && f[1]) return { kind: f[0], row: decText(f[1]) };
  if (f[0] === 'machine' && f[1]) return { kind: 'machine', row: decText(f[1]), expanded: f[2] === 'x' };
  if (f[0] === 'model') {
    const column = num(f[1]);
    // `c` / `t` are the scope letters of links made before the buying map: they open the map at its defaults
    if (column !== undefined && column >= 0) return { kind: 'model', column: Math.floor(column), build: f[2] === 'Q4' || f[2] === 'Q8' || f[2] === 'ssd' ? f[2] : 'column', minTokS: clampMinTokS(num(f[3]) ?? 0) };
  }
  return { kind: 'table' };
}

/** State → the `s` parameter value (already URL-safe). */
export function encodeState(state: AppState): string {
  const sections: string[] = [VERSION];
  if (state.groups.length) {
    // hidden sizes ride on the group item (`mac-mini-m6*16*24`): the common case encodes nothing, a size the catalog adds later shows by itself
    sections.push(`g${state.groups.map((id) => {
      const l = state.linked[id];
      const h = state.hiddenSizes[id] ?? [];
      const fields = [id, ...(l ? [`x${l.count}${l.split === 'tensor' ? 't' : ''}`] : []), ...h];
      return fields.join(FIELD);
    }).join(ITEM)}`);
  }
  if (state.columns.length) sections.push(`c${state.columns.map(encColumn).join(ITEM)}`);
  if (state.work !== STATE_DEFAULTS.work) sections.push(`w${state.work}`);
  if (state.cap !== null) sections.push(`o${state.cap}`);
  if (state.runtime !== STATE_DEFAULTS.runtime) sections.push(`r${state.runtime}`);
  if (state.kvBits !== STATE_DEFAULTS.kvBits) sections.push(`k${state.kvBits}`);
  if (state.floorBits !== STATE_DEFAULTS.floorBits) sections.push(`f${state.floorBits}`);
  if (!state.simpleQuants) sections.push('qx');
  const view = encView(state.view);
  if (view) sections.push(`v${view}`);
  if (state.customModels.length) sections.push(`x${state.customModels.map(encCustomModel).join(ITEM)}`);
  if (state.customMachines.length) sections.push(`y${state.customMachines.map(encCustomMachine).join(ITEM)}`);
  if (state.budgetUsd !== null) sections.push(`b${state.budgetUsd}`);
  if (state.formFactor !== STATE_DEFAULTS.formFactor) sections.push(`p${state.formFactor === 'laptop' ? 'l' : 'd'}`);
  return sections.join(SECTION);
}

export interface DecodeOptions {
  /** group ids that exist (catalog groups); unknown ones are dropped with a notice */
  knownGroups?: Set<string>;
}

export interface DecodeResult {
  state: AppState;
  notices: string[];
  /** false when the string was unusable and defaults were returned */
  ok: boolean;
}

/** The `s` parameter → state. Never throws; garbage yields the fallback with `ok: false`. */
export function decodeState(text: string | null | undefined, fallback: AppState, opts: DecodeOptions = {}): DecodeResult {
  if (!text) return { state: fallback, notices: [], ok: false };
  const sections = text.split(SECTION);
  if (sections[0] !== VERSION) return { state: fallback, notices: ['this link was made by another version of the tool'], ok: false };
  const notices: string[] = [];
  const state: AppState = { ...STATE_DEFAULTS, groups: [], columns: [], hiddenSizes: {}, linked: {}, customModels: [], customMachines: [], view: { kind: 'table' }, budgetUsd: null, formFactor: 'any' };
  const parsed = new Map<string, string>();
  for (const s of sections.slice(1)) {
    if (s.length > 1) parsed.set(s[0], s.slice(1));
  }
  const custom = parsed.get('x');
  if (custom) for (const item of custom.split(ITEM)) {
    const m = decCustomModel(item);
    if (m) state.customModels.push(m);
  }
  const machines = parsed.get('y');
  if (machines) for (const item of machines.split(ITEM)) {
    const m = decCustomMachine(item);
    if (m) state.customMachines.push(m);
  }
  const customGroupIds = new Set(state.customMachines.map((m) => customMachine(m).id));
  const customModelIds = new Set(state.customModels.map((m) => (m.mode === 'full' ? customModelFull(m) : customModelQuick(m)).id));

  const groups = parsed.get('g');
  if (groups) for (const item of groups.split(ITEM)) {
    const [id, ...rest] = item.split(FIELD);
    if (!id || !ID_RE.test(id)) continue;
    if (opts.knownGroups && !opts.knownGroups.has(id) && !customGroupIds.has(id)) {
      notices.push(`the link included a machine that is not in the catalog any more (${id})`);
      continue;
    }
    if (state.groups.length < MAX_GROUPS && !state.groups.includes(id)) {
      state.groups.push(id);
      // `x2` / `x4t` = a linked pool (count, tensor); every other field is a hidden size
      let linked: LinkedInfo | null = null;
      const sizes: string[] = [];
      for (const f of rest) {
        const m = /^x([234])(t?)$/.exec(f);
        if (m) linked = normaliseLinked(Number(m[1]), m[2] ? 'tensor' : 'layer');
        else sizes.push(f);
      }
      if (linked) state.linked[id] = linked;
      const hidden = [...new Set(sizes.map(Number).filter((n) => Number.isInteger(n) && n > 0))].sort((a, b) => a - b);
      if (hidden.length) state.hiddenSizes[id] = hidden;
    }
  }
  const columns = parsed.get('c');
  if (columns) for (const item of columns.split(ITEM)) {
    const c = decColumn(item);
    if (!c) continue;
    if (c.id.startsWith('custom-') && !customModelIds.has(c.id)) {
      notices.push(`the link included a custom model this browser does not have (${c.id.replace(/^custom-/, '')})`);
      continue;
    }
    if (state.columns.length < MAX_COLUMNS) state.columns.push(c); // the same model may repeat at other settings
  }
  const work = num(parsed.get('w'));
  if (work !== undefined) state.work = Math.max(0, Math.min(64, Math.round(work)));
  const cap = num(parsed.get('o'));
  if (cap !== undefined) state.cap = Math.min(1, Math.max(0.1, cap));
  const runtime = parsed.get('r');
  if (runtime === 'mlx' || runtime === 'gguf') state.runtime = runtime as Runtime;
  const kv = num(parsed.get('k'));
  if (kv === 4 || kv === 8 || kv === 16) state.kvBits = kv as KvBits;
  const floor = num(parsed.get('f'));
  if (floor !== undefined) state.floorBits = Math.max(1, Math.min(8, Math.round(floor)));
  if (parsed.get('q') === 'x') state.simpleQuants = false;
  const view = parsed.get('v');
  if (view) state.view = decView(view);
  if (state.view.kind === 'model' && state.view.column >= state.columns.length) state.view = { kind: 'table' };
  if ((state.view.kind === 'speed' || state.view.kind === 'memory') && !state.groups.some((g) => 'row' in state.view && state.view.row.startsWith(`${g}:`))) state.view = { kind: 'table' };
  if (state.view.kind === 'machine') {
    // any catalog machine may be picked, not only a table row; with the catalog known, a machine that is neither in it nor typed into the link falls back
    const p = parseRowKey(state.view.row);
    if (!p || (opts.knownGroups && !opts.knownGroups.has(p.groupId) && !customGroupIds.has(p.groupId))) state.view = { kind: 'table' };
  }
  const budget = num(parsed.get('b'));
  if (budget !== undefined && budget > 0) state.budgetUsd = clampBudget(budget);
  const pref = parsed.get('p');
  if (pref === 'l') state.formFactor = 'laptop';
  else if (pref === 'd') state.formFactor = 'desktop';

  if (!state.groups.length && !state.columns.length) return { state: fallback, notices: notices.length ? notices : ['this link had nothing to show'], ok: false };
  return { state, notices, ok: true };
}
