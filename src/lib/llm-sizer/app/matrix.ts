/**
 * The whole table in one pass: every (machine row × model column) cell through the engine. Pure, so the
 * island memoizes one call and the image route reuses it.
 */
import { customModelFull, customModelQuick } from '../engine/custom';
import { evaluateCell } from '../engine/index';
import type { CellResult, Factors, ModelDetail, Quant, SpecialBuild, Settings } from '../engine/types';
import { customGroup, rowKey, type GroupRow, type LinkedInfo, type MachineGroup } from './machines';
import { resolveQuant } from './quants';
import type { AppState, CustomModel, ModelColumn } from './state';

export type RecordState =
  | { status: 'loading' }
  | { status: 'ready'; model: ModelDetail }
  | { status: 'missing' }
  | { status: 'error'; message?: string };

export interface MatrixRow {
  key: string;
  group: MachineGroup;
  row: GroupRow;
  /** the pool this row is, when the group is linked; null for one machine */
  linked: LinkedInfo | null;
}

export interface MatrixColumn {
  index: number;
  column: ModelColumn;
  status: RecordState['status'];
  model: ModelDetail | null;
  quant: Quant | SpecialBuild | null;
}

export interface MatrixCell {
  key: string;
  rowKey: string;
  column: number;
  status: RecordState['status'];
  result: CellResult | null;
}

export interface Matrix {
  rows: MatrixRow[];
  columns: MatrixColumn[];
  cells: Map<string, MatrixCell>;
}

export function cellKey(row: string, column: number): string {
  return `${row}|${column}`;
}

export function customModelDetail(m: CustomModel): ModelDetail {
  return m.mode === 'full' ? customModelFull(m) : customModelQuick(m);
}

/** Groups the state refers to, catalog and custom, in the state's order. */
export function stateGroups(state: AppState, catalog: MachineGroup[]): MachineGroup[] {
  const customs = state.customMachines.map(customGroup);
  const out: MachineGroup[] = [];
  for (const id of state.groups) {
    const g = catalog.find((x) => x.id === id) ?? customs.find((x) => x.id === id);
    if (g) out.push(g);
  }
  return out;
}

/** The rows a group contributes to the table; a link that hides every size (or a stale one) shows all of them. */
export function visibleRows(state: AppState, group: MachineGroup): GroupRow[] {
  const hidden = state.hiddenSizes[group.id];
  if (!hidden?.length) return group.rows;
  const rows = group.rows.filter((r) => !hidden.includes(r.gb));
  return rows.length ? rows : group.rows;
}

/** The record for a column: a custom model built locally, or the fetched export record. */
export function columnRecord(column: ModelColumn, state: AppState, records: Record<string, RecordState>): RecordState {
  if (column.id.startsWith('custom-')) {
    const m = state.customModels.find((x) => customModelDetail(x).id === column.id);
    return m ? { status: 'ready', model: customModelDetail(m) } : { status: 'missing' };
  }
  return records[column.id] ?? { status: 'loading' };
}

export function settingsFor(state: AppState, row: GroupRow, column: ModelColumn, quant: Quant | SpecialBuild | null, linked: LinkedInfo | null = null): Settings {
  return {
    memoryGb: row.gb,
    workApps: state.work,
    override: state.cap === null ? false : state.cap,
    contextTokens: column.ctx,
    kvBits: column.kvBits ?? state.kvBits,
    runtime: state.runtime,
    qualityFloorBits: state.floorBits,
    ...(quant ? { quant } : {}),
    ...(linked ? { machines: linked.count, split: linked.split } : {}),
  };
}

export function evaluateMatrix(state: AppState, catalog: MachineGroup[], records: Record<string, RecordState>, factors: Factors): Matrix {
  const groups = stateGroups(state, catalog);
  const rows: MatrixRow[] = [];
  for (const group of groups) {
    const linked = state.linked[group.id] ?? null;
    for (const row of visibleRows(state, group)) rows.push({ key: rowKey(group.id, row.gb, linked), group, row, linked });
  }
  const columns: MatrixColumn[] = state.columns.map((column, index) => {
    const rec = columnRecord(column, state, records);
    const model = rec.status === 'ready' ? rec.model : null;
    const quant = model ? resolveQuant(model, column.quant, state.runtime) : null;
    return { index, column, status: rec.status, model, quant };
  });
  const cells = new Map<string, MatrixCell>();
  for (const r of rows) {
    for (const c of columns) {
      const key = cellKey(r.key, c.index);
      if (!c.model) {
        cells.set(key, { key, rowKey: r.key, column: c.index, status: c.status, result: null });
        continue;
      }
      const result = evaluateCell(c.model, r.row.machine, settingsFor(state, r.row, c.column, c.quant, r.linked), factors);
      cells.set(key, { key, rowKey: r.key, column: c.index, status: 'ready', result });
    }
  }
  return { rows, columns, cells };
}
