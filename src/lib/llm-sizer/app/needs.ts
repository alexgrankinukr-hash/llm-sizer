/**
 * How much memory a configuration needs, independent of any machine: weights + context cache + buffers.
 * One entry per column, so the same model added twice at different builds or contexts can be compared.
 */
import { neededMemory, type Cluster } from '../engine/index';
import type { MlxContextGroup, Need } from '../engine/types';
import type { Matrix, RecordState } from './matrix';
import type { AppState } from './state';

export interface MemoryNeed {
  index: number;
  name: string;
  /** the build asked for (a quant label or a hand-maintained build) */
  buildLabel: string;
  /** the context after clamping to the model's maximum */
  contextTokens: number;
  clamped: boolean;
  status: RecordState['status'];
  need: Need | null;
}

/**
 * `cluster` = the pool the bars are compared against, so the memory view's bars and its pooled limit line agree. `group`
 * is that machine's chip group for the MLX prompt scratch (`mlxContextGroupOf`); with no machine to compare against the
 * bars are a current Mac's, M5 and later, which carry none (METHODOLOGY §5).
 */
export function memoryNeeds(state: AppState, matrix: Matrix, cluster: Cluster | null = null, group: MlxContextGroup = 'm5plus'): MemoryNeed[] {
  return matrix.columns.map((column) => {
    const model = column.model;
    const quant = column.quant ?? model?.special_builds?.[0] ?? null;
    const base = { index: column.index, name: model?.name ?? column.column.id, status: column.status };
    if (!model || !quant) return { ...base, buildLabel: '—', contextTokens: column.column.ctx, clamped: false, need: null };
    const max = model.context_max;
    const clamped = !!max && column.column.ctx > max;
    const contextTokens = clamped ? (max as number) : column.column.ctx;
    const need = neededMemory(quant, model.architecture ?? null, contextTokens, column.column.kvBits ?? state.kvBits, 0, undefined, cluster, undefined, 'mlx', group);
    return { ...base, buildLabel: quant.label, contextTokens, clamped, need };
  });
}
