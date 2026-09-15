/**
 * Quant buckets for Simple mode (Q2 / Q3 / Q4 / Q6 / Q8 / FP16) and the rule that picks the file a
 * column uses: the export's reference quant for Q4 and Q8, the largest file in the bucket otherwise,
 * always in the runtime's format when one exists.
 */
import { QUANTIZER_PREFERENCE } from '../engine/constants';
import { nominalBits } from '../engine/fix';
import { defaultQuant } from '../engine/index';
import type { SpecialBuild, ModelDetail, Quant, Runtime } from '../engine/types';
import { isSpecialBuild, ssdGbOf } from '../engine/memory';

export type Bucket = 'Q1' | 'Q2' | 'Q3' | 'Q4' | 'Q5' | 'Q6' | 'Q8' | 'FP16' | 'FP32';
export const SIMPLE_BUCKETS: Bucket[] = ['Q2', 'Q3', 'Q4', 'Q6', 'Q8'];

export function bucketOf(label: string): Bucket | null {
  const nb = nominalBits(label);
  if (nb === null) return null;
  if (nb <= 1) return 'Q1';
  if (nb <= 2) return 'Q2';
  if (nb <= 3) return 'Q3';
  if (nb <= 4) return 'Q4';
  if (nb <= 5) return 'Q5';
  if (nb <= 6) return 'Q6';
  if (nb <= 8) return 'Q8';
  if (nb <= 16) return 'FP16';
  return 'FP32';
}

function formatFor(runtime: Runtime): Quant['format'] {
  return runtime === 'mlx' ? 'mlx' : 'gguf';
}

function rank(q: Quant): number {
  const i = (QUANTIZER_PREFERENCE as readonly string[]).indexOf(q.quantizer);
  return i === -1 ? 99 : i;
}

/** All files of a bucket, preferring the runtime's format (falls back to the other file format, never to raw safetensors unless nothing else exists). */
export function quantsInBucket(model: ModelDetail, bucket: Bucket, runtime: Runtime): Quant[] {
  const inBucket = model.quants.filter((q) => bucketOf(q.label) === bucket);
  const preferred = inBucket.filter((q) => q.format === formatFor(runtime));
  if (preferred.length) return preferred;
  const other = inBucket.filter((q) => q.format !== 'safetensors');
  return other.length ? other : inBucket;
}

/** The file a Simple-mode column uses for a bucket. */
export function pickQuant(model: ModelDetail, bucket: Bucket, runtime: Runtime): Quant | null {
  if (bucket === 'Q4') return defaultQuant(model, runtime, 'q4');
  const pool = quantsInBucket(model, bucket, runtime);
  if (bucket === 'Q8' && !(runtime === 'mlx' && pool.some((q) => q.format === 'mlx'))) {
    const q8 = defaultQuant(model, runtime, 'q8');
    if (q8 && bucketOf(q8.label) === 'Q8') return q8;
  }
  if (!pool.length) return null;
  // the largest file of the bucket from the preferred quantizer: highest quality the bucket offers
  return [...pool].sort((a, b) => rank(a) - rank(b) || b.size_bytes - a.size_bytes)[0];
}

/** A column's quant setting: the default (Q4), a bucket resolved per model at evaluation time, an exact file label, or one of the
 * model's hand-maintained special builds by its index in `special_builds` (hand-ordered in the overrides, so the index is stable). */
export type ColumnQuant = 'auto' | { bucket: Bucket } | { label: string; repo?: string } | { special: number };
export const BUCKET_RE = /^(Q[1-8]|FP16|FP32)$/;

export function isBucketSetting(q: ColumnQuant): q is { bucket: Bucket } {
  return q !== 'auto' && 'bucket' in q;
}

export function isLabelSetting(q: ColumnQuant): q is { label: string; repo?: string } {
  return q !== 'auto' && 'label' in q;
}

export function isSpecialSetting(q: ColumnQuant): q is { special: number } {
  return q !== 'auto' && 'special' in q;
}

/** One description for either kind of build, so the views never reach into `repo` or `size_gb` themselves. */
export interface BuildInfo {
  label: string;
  /** what must be in memory */
  sizeGb: number;
  /** what streams from the SSD (0 for a plain file) */
  ssdGb: number;
  /** stable identity: the file's repo and label, or the special build's label */
  key: string;
  isSpecial: boolean;
}

export function buildInfo(q: Quant | SpecialBuild): BuildInfo {
  if (isSpecialBuild(q)) return { label: q.label, sizeGb: q.resident_gb, ssdGb: ssdGbOf(q), key: `special:${q.label}`, isSpecial: true };
  return { label: q.label, sizeGb: q.size_gb, ssdGb: 0, key: `${q.repo}/${q.label}`, isSpecial: false };
}

/** Resolve a column's quant setting to a concrete file, or to one of the model's special builds. */
export function resolveQuant(model: ModelDetail, setting: ColumnQuant, runtime: Runtime): Quant | SpecialBuild | null {
  if (setting === 'auto') return pickQuant(model, 'Q4', runtime) ?? model.quants[0] ?? null;
  // a special build that the record no longer carries falls back to the reference file, like an unknown label does
  if (isSpecialSetting(setting)) return model.special_builds?.[setting.special] ?? pickQuant(model, 'Q4', runtime) ?? model.quants[0] ?? null;
  if (isBucketSetting(setting)) return pickQuant(model, setting.bucket, runtime) ?? pickQuant(model, 'Q4', runtime) ?? model.quants[0] ?? null;
  const exact = model.quants.find((q) => q.label === setting.label && (!setting.repo || q.repo === setting.repo || q.quantizer === setting.repo));
  if (exact) return exact;
  const sameLabel = model.quants.filter((q) => q.label === setting.label).sort((a, b) => rank(a) - rank(b));
  if (sameLabel.length) return sameLabel[0];
  const bucket = bucketOf(setting.label);
  return (bucket && pickQuant(model, bucket, runtime)) ?? pickQuant(model, 'Q4', runtime) ?? model.quants[0] ?? null;
}

/** Buckets this model actually has files for, in size order. */
export function availableBuckets(model: ModelDetail, _runtime?: Runtime): Bucket[] {
  const present = new Set<Bucket>();
  for (const q of model.quants) {
    const b = bucketOf(q.label);
    if (b && q.format !== 'safetensors') present.add(b);
  }
  const order: Bucket[] = ['Q1', 'Q2', 'Q3', 'Q4', 'Q5', 'Q6', 'Q8', 'FP16', 'FP32'];
  return order.filter((b) => present.has(b));
}
