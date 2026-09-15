/**
 * The app's state (everything a share link must reproduce) and its reducer. Pure; no React here.
 */
import { CONTEXT_CHIPS } from '../engine/constants';
import { type CustomMachineInput, type CustomModelFullInput, type CustomModelQuickInput } from '../engine/custom';
import type { ClusterSplit, KvBits, Runtime } from '../engine/types';
import { parseRowKey, rowKey, type LinkedInfo } from './machines';
import type { ColumnQuant } from './quants';

export const MAX_GROUPS = 6;
export const MAX_COLUMNS = 8;
export const DEFAULT_CONTEXT = 32768;

export interface ModelColumn {
  id: string;
  quant: ColumnQuant;
  ctx: number;
  /** per-column cache precision; undefined = the global setting */
  kvBits?: KvBits;
}

/** The buying map's build switch: the column's own file, the Q4 / Q8 file of the same model, or its engram-on-SSD build. */
export type BuildKey = 'column' | 'Q4' | 'Q8' | 'ssd';
/** A buyer's form-factor preference (Settings › Assumptions). */
export type FormFactor = 'any' | 'laptop' | 'desktop';

export type View =
  | { kind: 'table' }
  | { kind: 'speed'; row: string }
  | { kind: 'memory'; row: string }
  /** the buying map: one column's model across every current machine, at a build, above a minimum speed */
  | { kind: 'model'; column: number; build: BuildKey; minTokS: number }
  /** one machine at one memory size against the whole catalog; `row` may name any catalog or custom machine, not only a table row */
  | { kind: 'machine'; row: string; expanded: boolean };

export type CustomModel = ({ mode: 'quick' } & CustomModelQuickInput) | ({ mode: 'full' } & CustomModelFullInput);

export interface AppState {
  v: 1;
  groups: string[];
  columns: ModelColumn[];
  /** GB reserved for the user's apps; 0 = the work toggle is off */
  work: number;
  /** null = the platform's default GPU limit; a fraction of RAM otherwise (1 = the override) */
  cap: null | number;
  runtime: Runtime;
  kvBits: KvBits;
  floorBits: number;
  simpleQuants: boolean;
  view: View;
  /** per machine group, the memory sizes left out of the table; absent = every size the machine is sold with */
  hiddenSizes: Record<string, number[]>;
  /** per machine group, how many identical machines are linked as one pool and how the model is split; absent = one machine */
  linked: Record<string, LinkedInfo>;
  customModels: CustomModel[];
  customMachines: CustomMachineInput[];
  /** the most a buyer will spend, in US dollars; null = no limit */
  budgetUsd: number | null;
  formFactor: FormFactor;
}

export const STATE_DEFAULTS: Omit<AppState, 'groups' | 'columns'> = {
  v: 1,
  work: 0,
  cap: null,
  runtime: 'gguf',
  kvBits: 16,
  floorBits: 2,
  simpleQuants: true,
  view: { kind: 'table' },
  hiddenSizes: {},
  linked: {},
  customModels: [],
  customMachines: [],
  budgetUsd: null,
  formFactor: 'any',
};

/** At most this many identical machines in one linked pool. */
export const MAX_LINKED = 4;

/** A count of 1 (or less) means one machine; tensor parallel needs 2 or 4, so a count of 3 in tensor becomes 4. */
export function normaliseLinked(count: number, split: ClusterSplit = 'layer'): LinkedInfo | null {
  const n = Math.min(MAX_LINKED, Math.floor(count));
  if (n < 2) return null;
  const c = (split === 'tensor' && n === 3 ? 4 : n) as LinkedInfo['count'];
  return { count: c, split };
}

/** The view's row key rewritten for a group's new pool, so the open view follows the setting. */
function relinkView(view: View, groupId: string, linked: LinkedInfo | null): View {
  if (view.kind !== 'speed' && view.kind !== 'memory' && view.kind !== 'machine') return view;
  if (!view.row.startsWith(`${groupId}:`)) return view;
  const p = parseRowKey(view.row);
  if (!p || p.groupId !== groupId) return view;
  return { ...view, row: rowKey(groupId, p.gb, linked) };
}

export const BUDGET_MIN_USD = 100;
export const BUDGET_MAX_USD = 200_000;
export const MIN_TOK_S_MAX = 1000;

export type Action =
  | { type: 'ADD_GROUP'; id: string }
  | { type: 'REMOVE_GROUP'; id: string }
  | { type: 'MOVE_GROUP'; id: string; to: number }
  | { type: 'TOGGLE_SIZE'; groupId: string; gb: number; /** the group's catalog sizes, so the last visible one can never be hidden */ sizes: number[] }
  | { type: 'SET_LINKED'; groupId: string; count: number; split?: ClusterSplit }
  | { type: 'ADD_COLUMN'; id: string; ctx?: number; quant?: ColumnQuant }
  | { type: 'REMOVE_COLUMN'; index: number }
  | { type: 'MOVE_COLUMN'; index: number; to: number }
  | { type: 'SET_COLUMN_QUANT'; index: number; quant: ColumnQuant }
  | { type: 'SET_COLUMN_CTX'; index: number; ctx: number }
  | { type: 'SET_COLUMN_KV'; index: number; kvBits: KvBits | undefined }
  | { type: 'APPLY_ALL_COLUMNS'; quant?: ColumnQuant; ctx?: number }
  | { type: 'SET_WORK'; gb: number }
  | { type: 'SET_CAP'; cap: null | number }
  | { type: 'SET_RUNTIME'; runtime: Runtime }
  | { type: 'SET_KV_BITS'; kvBits: KvBits }
  | { type: 'SET_FLOOR'; bits: number }
  | { type: 'SET_SIMPLE_QUANTS'; simple: boolean }
  | { type: 'SET_VIEW'; view: View }
  | { type: 'SET_BUDGET'; usd: number | null }
  | { type: 'SET_FORM_FACTOR'; formFactor: FormFactor }
  | { type: 'ADD_CUSTOM_MODEL'; model: CustomModel; columnId: string }
  | { type: 'ADD_CUSTOM_MACHINE'; machine: CustomMachineInput; groupId: string }
  | { type: 'REMOVE_CUSTOM'; kind: 'model' | 'machine'; name: string }
  | { type: 'HYDRATE_LOCAL'; customModels?: CustomModel[]; customMachines?: CustomMachineInput[] }
  | { type: 'REPLACE'; state: AppState }
  | { type: 'RESET'; state: AppState };

/** Identity of a quant setting, for duplicate checks. */
export function quantKey(q: ColumnQuant): string {
  if (q === 'auto') return 'auto';
  return 'bucket' in q ? `b:${q.bucket}` : 'special' in q ? `s:${q.special}` : `l:${q.label}`;
}

function clampCtx(ctx: number): number {
  if (!Number.isFinite(ctx) || ctx < 512) return DEFAULT_CONTEXT;
  return Math.min(Math.floor(ctx), 4_194_304);
}

function clampCap(cap: null | number): null | number {
  if (cap === null) return null;
  if (!Number.isFinite(cap)) return null;
  return Math.min(1, Math.max(0.1, cap));
}

export function clampBudget(usd: number | null): number | null {
  if (usd === null || !Number.isFinite(usd)) return null;
  return Math.min(BUDGET_MAX_USD, Math.max(BUDGET_MIN_USD, Math.round(usd)));
}

export function clampMinTokS(tokS: number): number {
  if (!Number.isFinite(tokS) || tokS < 0) return 0;
  return Math.min(MIN_TOK_S_MAX, Math.round(tokS * 10) / 10);
}

/** The buying map for a column: keeps the build and the minimum speed when the visitor is already on the map. */
export function modelView(column: number, current?: View): View {
  if (current?.kind === 'model') return { kind: 'model', column, build: current.build, minTokS: current.minTokS };
  return { kind: 'model', column, build: 'column', minTokS: 0 };
}

export function reducer(state: AppState, action: Action): AppState {
  switch (action.type) {
    case 'ADD_GROUP': {
      // a machine already in the table gains a second, linked one (up to four); a new one takes a slot
      if (state.groups.includes(action.id)) {
        const current = state.linked[action.id];
        const next = normaliseLinked((current?.count ?? 1) + 1, current?.split);
        if (!next || next.count === current?.count) return state;
        return { ...state, linked: { ...state.linked, [action.id]: next }, view: relinkView(state.view, action.id, next) };
      }
      if (state.groups.length >= MAX_GROUPS) return state;
      return { ...state, groups: [...state.groups, action.id] };
    }
    case 'SET_LINKED': {
      if (!state.groups.includes(action.groupId)) return state;
      const next = normaliseLinked(action.count, action.split ?? state.linked[action.groupId]?.split);
      const linked = { ...state.linked };
      if (next) linked[action.groupId] = next;
      else delete linked[action.groupId];
      return { ...state, linked, view: relinkView(state.view, action.groupId, next) };
    }
    case 'REMOVE_GROUP': {
      const groups = state.groups.filter((g) => g !== action.id);
      const view = (state.view.kind === 'speed' || state.view.kind === 'memory') && state.view.row.startsWith(`${action.id}:`) ? ({ kind: 'table' } as View) : state.view;
      const hiddenSizes = { ...state.hiddenSizes };
      delete hiddenSizes[action.id];
      const linked = { ...state.linked };
      delete linked[action.id];
      return { ...state, groups, view, hiddenSizes, linked };
    }
    case 'TOGGLE_SIZE': {
      if (!action.sizes.includes(action.gb)) return state;
      const hidden = new Set(state.hiddenSizes[action.groupId] ?? []);
      if (hidden.has(action.gb)) hidden.delete(action.gb);
      else hidden.add(action.gb);
      if (action.sizes.every((gb) => hidden.has(gb))) return state; // one size always stays
      const list = action.sizes.filter((gb) => hidden.has(gb));
      const hiddenSizes = { ...state.hiddenSizes };
      if (list.length) hiddenSizes[action.groupId] = list;
      else delete hiddenSizes[action.groupId];
      return { ...state, hiddenSizes };
    }
    case 'MOVE_GROUP': {
      const from = state.groups.indexOf(action.id);
      if (from === -1) return state;
      const groups = [...state.groups];
      groups.splice(from, 1);
      groups.splice(Math.max(0, Math.min(groups.length, action.to)), 0, action.id);
      return { ...state, groups };
    }
    case 'ADD_COLUMN': {
      // the same model may appear twice at different settings; only an exact repeat is refused
      const column: ModelColumn = { id: action.id, quant: action.quant ?? 'auto', ctx: clampCtx(action.ctx ?? DEFAULT_CONTEXT) };
      if (state.columns.length >= MAX_COLUMNS || state.columns.some((c) => c.id === column.id && quantKey(c.quant) === quantKey(column.quant) && c.ctx === column.ctx)) return state;
      return { ...state, columns: [...state.columns, column] };
    }
    case 'REMOVE_COLUMN': {
      if (action.index < 0 || action.index >= state.columns.length) return state;
      const columns = state.columns.filter((_, i) => i !== action.index);
      let view = state.view;
      if (view.kind === 'model') {
        if (view.column === action.index) view = { kind: 'table' };
        else if (view.column > action.index) view = { ...view, column: view.column - 1 };
      }
      return { ...state, columns, view };
    }
    case 'MOVE_COLUMN': {
      const { index } = action;
      if (index < 0 || index >= state.columns.length) return state;
      const to = Math.max(0, Math.min(state.columns.length - 1, action.to));
      if (to === index) return state;
      const columns = [...state.columns];
      const [moved] = columns.splice(index, 1);
      columns.splice(to, 0, moved);
      let view = state.view;
      if (view.kind === 'model') {
        const order = state.columns.map((_, i) => i);
        const [m] = order.splice(index, 1);
        order.splice(to, 0, m);
        view = { ...view, column: order.indexOf(view.column) };
      }
      return { ...state, columns, view };
    }
    case 'SET_COLUMN_QUANT':
      return updateColumn(state, action.index, (c) => ({ ...c, quant: action.quant }));
    case 'SET_COLUMN_CTX':
      return updateColumn(state, action.index, (c) => ({ ...c, ctx: clampCtx(action.ctx) }));
    case 'SET_COLUMN_KV':
      return updateColumn(state, action.index, (c) => {
        const next = { ...c };
        if (action.kvBits === undefined) delete next.kvBits;
        else next.kvBits = action.kvBits;
        return next;
      });
    case 'APPLY_ALL_COLUMNS':
      return {
        ...state,
        columns: state.columns.map((c) => ({ ...c, ...(action.quant !== undefined ? { quant: action.quant } : {}), ...(action.ctx !== undefined ? { ctx: clampCtx(action.ctx) } : {}) })),
      };
    case 'SET_WORK':
      return { ...state, work: Math.max(0, Math.min(64, Math.round(action.gb))) };
    case 'SET_CAP':
      return { ...state, cap: clampCap(action.cap) };
    case 'SET_RUNTIME':
      return { ...state, runtime: action.runtime };
    case 'SET_KV_BITS':
      return { ...state, kvBits: action.kvBits };
    case 'SET_FLOOR':
      return { ...state, floorBits: Math.max(1, Math.min(8, Math.round(action.bits))) };
    case 'SET_SIMPLE_QUANTS':
      return { ...state, simpleQuants: action.simple };
    case 'SET_VIEW':
      return { ...state, view: action.view };
    case 'SET_BUDGET':
      return { ...state, budgetUsd: clampBudget(action.usd) };
    case 'SET_FORM_FACTOR':
      return { ...state, formFactor: action.formFactor };
    case 'ADD_CUSTOM_MODEL': {
      const customModels = state.customModels.filter((m) => m.name !== action.model.name).concat(action.model);
      const withModel = { ...state, customModels };
      return reducer(withModel, { type: 'ADD_COLUMN', id: action.columnId });
    }
    case 'ADD_CUSTOM_MACHINE': {
      const customMachines = state.customMachines.filter((m) => m.name !== action.machine.name).concat(action.machine);
      const withMachine = { ...state, customMachines };
      if (state.groups.includes(action.groupId)) return withMachine; // re-submitting the form replaces the machine, it does not link a second
      return reducer(withMachine, { type: 'ADD_GROUP', id: action.groupId });
    }
    case 'REMOVE_CUSTOM': {
      if (action.kind === 'model') return { ...state, customModels: state.customModels.filter((m) => m.name !== action.name) };
      return { ...state, customMachines: state.customMachines.filter((m) => m.name !== action.name) };
    }
    case 'HYDRATE_LOCAL': {
      // local entries fill in what the URL did not carry; the URL wins on conflicts
      const models = [...state.customModels];
      for (const m of action.customModels ?? []) if (!models.some((x) => x.name === m.name)) models.push(m);
      const machines = [...state.customMachines];
      for (const m of action.customMachines ?? []) if (!machines.some((x) => x.name === m.name)) machines.push(m);
      return { ...state, customModels: models, customMachines: machines };
    }
    case 'REPLACE':
    case 'RESET':
      return action.state;
    default:
      return state;
  }
}

function updateColumn(state: AppState, index: number, fn: (c: ModelColumn) => ModelColumn): AppState {
  if (index < 0 || index >= state.columns.length) return state;
  return { ...state, columns: state.columns.map((c, i) => (i === index ? fn(c) : c)) };
}

/** The nearest chip at or below a context, for pickers. */
export function nearestChip(ctx: number, chips: readonly number[] = CONTEXT_CHIPS): number {
  const below = chips.filter((c) => c <= ctx);
  return below.length ? below[below.length - 1] : chips[0];
}
