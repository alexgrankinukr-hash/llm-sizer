/** Options for a column's quant and context controls (the settings list). Pure. */
import { contextChips } from '../engine/index';
import { dedupeByLabel } from '../engine/fix';
import type { ModelDetail } from '../engine/types';
import type { MatrixColumn } from './matrix';
import { availableBuckets, bucketOf, isBucketSetting, isSpecialSetting, SIMPLE_BUCKETS, type Bucket, type ColumnQuant } from './quants';
import { ssdGbOf } from '../engine/memory';
import type { SpecialBuild } from '../engine/types';
import type { AppState } from './state';

export interface QuantOption {
  value: string;
  label: string;
}

const gb = (n: number) => n.toFixed(n < 100 ? 1 : 0);

/** The model's hand-maintained builds, after the files: "Q4 · engram on SSD · 45.8 GB in memory" in simple mode, the label and both sizes otherwise. */
function specialOptions(builds: SpecialBuild[] | undefined, simple: boolean): QuantOption[] {
  return (builds ?? []).map((b, i) => ({
    value: `special:${i}`,
    label: simple ? `${bucketOf(b.label) ?? 'Q4'} · engram on SSD · ${gb(b.resident_gb)} GB in memory` : `${b.label} · ${gb(b.resident_gb)} GB in memory + ${gb(ssdGbOf(b))} GB on SSD`,
  }));
}

/** The choices for a column: buckets in simple mode, exact files otherwise; the model's special builds after either. */
export function quantOptions(col: MatrixColumn, state: AppState): QuantOption[] {
  if (!col.model) return [{ value: 'auto', label: 'Q4' }];
  if (state.simpleQuants) {
    const buckets = availableBuckets(col.model, state.runtime).filter((b) => SIMPLE_BUCKETS.includes(b));
    const current = isBucketSetting(col.column.quant) ? col.column.quant.bucket : null;
    if (current && !buckets.includes(current)) buckets.push(current);
    return [...buckets.map((b) => ({ value: `bucket:${b}`, label: b === 'Q4' ? 'Q4 (default)' : b })), ...specialOptions(col.model.special_builds, true)];
  }
  const files = dedupeByLabel(col.model.quants.filter((q) => q.format !== 'safetensors')).sort((a, b) => a.size_bytes - b.size_bytes);
  return [{ value: 'auto', label: 'Q4 (default)' }, ...files.map((q) => ({ value: `label:${q.label}`, label: `${q.label} · ${gb(q.size_gb)} GB` })), ...specialOptions(col.model.special_builds, false)];
}

/** The option value that represents the column's current setting. */
export function quantValue(col: MatrixColumn, state: AppState): string {
  const q = col.column.quant;
  if (q === 'auto') return state.simpleQuants ? 'bucket:Q4' : 'auto';
  if (isSpecialSetting(q)) return `special:${q.special}`;
  if (isBucketSetting(q)) {
    if (!state.simpleQuants) return col.quant ? `label:${col.quant.label}` : 'auto';
    return `bucket:${q.bucket}`;
  }
  if (state.simpleQuants) return `bucket:${bucketOf(q.label) ?? 'Q4'}`;
  return `label:${q.label}`;
}

/** An option value back to a column setting. */
export function parseQuantValue(value: string): ColumnQuant {
  if (value === 'auto' || value === 'bucket:Q4') return 'auto';
  const at = value.indexOf(':');
  const kind = at < 0 ? value : value.slice(0, at);
  const rest = at < 0 ? '' : value.slice(at + 1);
  if (kind === 'bucket') return { bucket: rest as Bucket };
  if (kind === 'special') return { special: Math.max(0, Math.floor(Number(rest)) || 0) };
  return { label: rest };
}

/** Context lengths to offer: the model's chips, plus the current value when it is off the chips. */
export function ctxOptions(model: Pick<ModelDetail, 'context_max'> | null, ctx: number): number[] {
  const chips = model ? contextChips(model) : [8192, 32768, 131072, 262144];
  return chips.includes(ctx) ? chips : [...chips, ctx].sort((a, b) => a - b);
}
