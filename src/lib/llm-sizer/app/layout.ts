/**
 * What the table shows, computed once for the DOM, presentation mode and the PNG: marker per cell,
 * header and rail text, footnotes. Keeps the three renderings from drifting apart.
 */
import { isSpecialBuild } from '../engine/memory';
import { formatContext, formatParams, formatPrice } from '../engine/format';
import type { CellFlag, CellResult, Verdict } from '../engine/types';
import { linkedLabel, pooledGb, pooledPrice, rowLabel, type LinkedInfo } from './machines';
import { bucketOf, isBucketSetting, isSpecialSetting } from './quants';
import type { Matrix, MatrixCell, MatrixColumn, MatrixRow } from './matrix';
import type { AppState } from './state';

export type Marker = 'run' | 'ring' | 'no' | 'loading' | 'missing';

export function markerFor(cell: MatrixCell): Marker {
  if (cell.status === 'loading' || cell.status === 'error') return 'loading';
  if (cell.status === 'missing' || !cell.result) return 'missing';
  return markerForVerdict(cell.result.verdict);
}

export function markerForVerdict(v: Verdict): Marker {
  if (v === 'runs') return 'run';
  if (v === 'no-fit') return 'no';
  return 'ring';
}

export const MARKER_TEXT: Record<Marker, string> = {
  run: 'runs',
  ring: 'runs with a compromise',
  no: "doesn't fit",
  loading: 'loading',
  missing: 'not in the catalog',
};

export const FLAG_TEXT: Record<CellFlag, string> = {
  assumed: 'architecture assumed conventional: the model family is unfamiliar, so the context-cache math is the pessimistic default',
  'clamped-context': 'context reduced to the model\'s maximum',
  'no-mlx-file': 'no MLX file for this model, so the GGUF file is used',
  'no-gguf-mlx-build': 'no GGUF or MLX build exists yet; the fit uses the raw weights',
  gated: 'gated on Hugging Face: you need to accept the license before downloading',
  'four-bit-native': 'ships in 4-bit natively; higher-precision files are re-packs of the same weights',
  'q8-is-repack': 'this model ships in 4-bit: the "Q8" file re-packs the same weights, so it costs the extra memory without the extra quality',
  'unmeasured-chip': 'speed is an estimate: this chip has no measured rows yet',
  'kv-precision-note': '4-bit cache on compressed attention may be unsupported by your runtime',
  'ssd-build': 'this build keeps part of the model in memory and streams its n-gram table from the SSD; the bar shows the part in memory and the SSD part is listed beside it. It needs that much free SSD space, a llama.cpp with qwen4exp support, mmap on, -fit off and --jinja',
  'linked-layer': 'linked machines on a layer split (llama.cpp RPC on any link, MLX pipeline): memory pools, each machine keeps its own reserve and GPU limit, the pool writes at one machine\'s speed minus a hop per extra machine and reads at one machine\'s rate. An estimate fitted on a handful of published cluster runs; a model that fits one machine writes slower across two',
  'linked-tensor': 'linked machines in tensor parallel (MLX over Thunderbolt 5 RDMA on macOS 26.2+, vLLM or TensorRT-LLM over the Spark\'s 200 Gb/s ports, PCIe on a GPU box): needs 2 or 4 machines, a direct fast link between every pair and the raised memory limit on each; every machine keeps the whole download on disk; the context cache is duplicated on latent-attention models. Speed fitted on a handful of published runs',
  'linked-dispatch': 'a very deep mixture of experts in tensor parallel can run far below the bandwidth maths: one measured run (Kimi K3 on four M3 Ultras) wrote 2 tok/s where the estimate says 36, from per-kernel dispatch overhead. Treat the speed as a ceiling',
};

export interface HeaderLayout {
  index: number;
  name: string;
  provider: string;
  params: string;
  quantLabel: string;
  contextLabel: string;
  status: MatrixColumn['status'];
  /** notes that hold for every cell of the column, so they are said once here instead of in each cell */
  footnotes: number[];
}

export interface RailLayout {
  groupId: string;
  label: string;
  chip: string;
  price: string | null;
  silhouette: string;
  /** apple | cuda | rocm: which mark the rail shows */
  platform: string;
  rowKeys: string[];
}

export interface RowLayout {
  key: string;
  groupId: string;
  /** the machine's own size */
  gb: number;
  /** "128 GB", or "128 GB each · 256 GB pooled" for a linked pool */
  sizeLabel: string;
  linked: LinkedInfo | null;
  binLabel?: string;
  price: string | null;
  cells: { key: string; column: number; marker: Marker; ariaLabel: string; footnotes: number[] }[];
}

export interface TableLayout {
  headers: HeaderLayout[];
  rails: RailLayout[];
  rows: RowLayout[];
  footnotes: string[];
  toggles: { work: string; limit: string };
}

export function quantLabelFor(column: MatrixColumn, simple: boolean): string {
  if (isSpecialSetting(column.column.quant)) {
    const q = column.quant;
    if (q && isSpecialBuild(q)) return simple ? `${bucketOf(q.label) ?? 'Q4'} · SSD` : q.label;
    return 'SSD build';
  }
  if (isBucketSetting(column.column.quant)) {
    if (!column.quant) return column.column.quant.bucket;
    return simple ? bucketOf(column.quant.label) ?? column.quant.label : column.quant.label;
  }
  if (column.column.quant !== 'auto') {
    const label = column.column.quant.label;
    return simple ? bucketOf(label) ?? label : label;
  }
  if (column.quant) return simple ? bucketOf(column.quant.label) ?? column.quant.label : column.quant.label;
  return 'Q4';
}

export function paramsText(model: MatrixColumn['model']): string {
  if (!model) return '';
  const total = formatParams(model.params_total);
  const active = model.params_active && model.params_total && model.params_active < model.params_total * 0.9 ? ` · ${formatParams(model.params_active)} active` : '';
  return `${total}${active}`;
}

/**
 * The build and context a cell is really running: for a compromise that is the fix's, not the one asked for,
 * so a chart never labels a bar with a file that does not fit.
 */
export function runningConfig(result: CellResult | null, asked: { quantLabel: string | null; ctx: number }): { quantLabel: string; ctx: number; atTheFix: boolean } {
  const fix = result?.verdict === 'compromise' ? result.fix : null;
  const quantChange = fix?.changes.find((c) => c.kind === 'quant');
  const contextChange = fix?.changes.find((c) => c.kind === 'context');
  const build = fix?.changes.find((c) => c.kind === 'ssdPaged');
  return {
    quantLabel: build ? build.build.label : quantChange ? quantChange.quant.label : (asked.quantLabel ?? '—'),
    ctx: contextChange ? contextChange.tokens : asked.ctx,
    atTheFix: !!fix,
  };
}

export function verdictSentence(result: CellResult): string {
  const pct = Math.round(result.need.ratio * 100);
  const avail = result.availability.availableGb.toFixed(result.availability.availableGb < 100 ? 1 : 0);
  if (result.verdict === 'runs') return `Runs: ${pct} % of the ${avail} GB available`;
  if (result.verdict === 'tight') return `Runs, tight: ${pct} % of the ${avail} GB available`;
  if (result.verdict === 'compromise' && result.fix) return `Runs with a compromise: ${result.fix.reason}`;
  return `Doesn't fit${result.nearestMiss ? `: ${result.nearestMiss}` : ''}`;
}

/**
 * Only notes that change what a reader would do earn a number under the table: the architecture is a guess,
 * the context was cut, nothing runnable exists yet, the licence blocks the download, a bigger file buys no
 * quality, the cache precision may not be supported. Everything else (which file format was used, the fact
 * that a model is 4-bit native when you already asked for 4-bit) is detail for the cell sheet.
 */
const TABLE_FOOTNOTE_FLAGS: ReadonlySet<CellFlag> = new Set(['assumed', 'clamped-context', 'no-gguf-mlx-build', 'gated', 'q8-is-repack', 'kv-precision-note', 'linked-dispatch']);

export function tableLayout(state: AppState, matrix: Matrix): TableLayout {
  const footnotes: string[] = [];
  const footnoteIndex = new Map<CellFlag, number>();
  const note = (flag: CellFlag): number => {
    let i = footnoteIndex.get(flag);
    if (i === undefined) {
      footnotes.push(FLAG_TEXT[flag]);
      i = footnotes.length;
      footnoteIndex.set(flag, i);
    }
    return i;
  };
  // a flag every cell of a column carries is a fact about the model, not about a machine: it goes in the header
  const columnFlags: CellFlag[][] = matrix.columns.map((c) => {
    const perCell: CellFlag[][] = [];
    for (const r of matrix.rows) {
      const result = matrix.cells.get(`${r.key}|${c.index}`)?.result;
      if (result) perCell.push(result.flags.filter((f) => TABLE_FOOTNOTE_FLAGS.has(f)));
    }
    if (!perCell.length) return [];
    return perCell[0].filter((flag) => perCell.every((flags) => flags.includes(flag)));
  });
  const headers: HeaderLayout[] = matrix.columns.map((c, i) => ({
    index: c.index,
    name: c.model?.name ?? c.column.id,
    provider: c.model?.provider ?? '',
    params: paramsText(c.model),
    quantLabel: quantLabelFor(c, state.simpleQuants),
    contextLabel: formatContext(c.column.ctx),
    status: c.status,
    footnotes: columnFlags[i].map(note).sort((a, b) => a - b),
  }));
  const rails: RailLayout[] = [];
  const rows: RowLayout[] = [];
  for (const r of matrix.rows) {
    let rail = rails.find((x) => x.groupId === r.group.id);
    if (!rail) {
      const cheapest = r.group.rows.map((x) => pooledPrice(x.priceUsd, r.linked)).filter((p): p is number => p !== null).sort((a, b) => a - b)[0];
      rail = { groupId: r.group.id, label: linkedLabel(r.group, r.linked), chip: r.group.chip, price: cheapest !== undefined ? `from ${formatPrice(cheapest)}` : null, silhouette: r.group.silhouette, platform: r.group.platform, rowKeys: [] };
      rails.push(rail);
    }
    rail.rowKeys.push(r.key);
    const cells = matrix.columns.map((c) => {
      const cell = matrix.cells.get(`${r.key}|${c.index}`)!;
      const marker = markerFor(cell);
      const flags = cell.result?.flags ?? [];
      const notes = flags
        .filter((f) => TABLE_FOOTNOTE_FLAGS.has(f) && !columnFlags[c.index].includes(f))
        .map(note)
        .sort((a, b) => a - b);
      const name = c.model?.name ?? c.column.id;
      const what = cell.result ? MARKER_TEXT[marker] + (cell.result.verdict === 'tight' ? ' (tight)' : '') : MARKER_TEXT[marker];
      return { key: cell.key, column: c.index, marker, ariaLabel: `${name} on ${rowLabel(r.group, r.row, r.linked)}: ${what}`, footnotes: notes };
    });
    const price = pooledPrice(r.row.priceUsd, r.linked);
    const sizeLabel = r.linked ? `${r.row.gb} GB each · ${pooledGb(r.row.gb, r.linked)} GB pooled` : `${r.row.gb} GB`;
    rows.push({ key: r.key, groupId: r.group.id, gb: r.row.gb, sizeLabel, linked: r.linked, binLabel: r.row.binLabel, price: price !== null ? formatPrice(price) : null, cells });
  }
  const limitPct = state.cap !== null ? `${Math.round(state.cap * 100)} % (override)` : '67 % / 75 %';
  return {
    headers,
    rails,
    rows,
    footnotes,
    toggles: {
      work: state.work > 0 ? `apps open (${state.work} GB reserved)` : 'nothing else running',
      limit: `macOS memory limit ${limitPct}`,
    },
  };
}

export { MARKER_TEXT as markerText };
export type { MatrixRow };
