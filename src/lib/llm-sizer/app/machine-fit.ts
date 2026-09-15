/**
 * "What can it run?": one machine at one memory size against the whole catalog. Pure, so the view memoises per
 * model and the future per-machine pages can reuse it. The rules:
 *  - the summary names the highest-quality build that runs as asked at 32K (Q8 → Q6 → Q4 → Q3 → Q2) and the
 *    longest context that runs at Q4; when nothing runs as asked, the cheapest compromise the engine finds at
 *    Q4 / 32K, or the nearest miss;
 *  - a bucket the model has no file for is a gap, shown as one, never a substitute file;
 *  - the grid is every bucket × every context chip the model allows, with the table's three markers.
 */
import { contextChips, evaluateCell } from '../engine/index';
import { isSpecialBuild } from '../engine/memory';
import { formatContext, formatGb, formatTokS } from '../engine/format';
import type { CellResult, Factors, ModelDetail, ModelIndexEntry, Quant, Runtime, SpecialBuild } from '../engine/types';
import { customGroup, findRow, pooledGb, rowKey, rowLabel, type LinkedInfo, type MachineGroup } from './machines';
import { settingsFor, type MatrixCell, type MatrixColumn, type MatrixRow } from './matrix';
import { markerForVerdict, MARKER_TEXT, paramsText, runningConfig, verdictSentence, type Marker } from './layout';
import { bucketOf, pickQuant, SIMPLE_BUCKETS, type Bucket } from './quants';
import { DEFAULT_CONTEXT, type AppState, type ModelColumn } from './state';

/** Buckets from the highest quality down: the order the summary tries them in. */
export const BEST_FIRST: Bucket[] = [...SIMPLE_BUCKETS].reverse();

/** The settings the view depends on, as one string, so per-model memos survive unrelated state changes. */
export function fitSettingsKey(state: AppState): string {
  return `${state.work}|${state.cap ?? 'd'}|${state.runtime}|${state.kvBits}|${state.floorBits}`;
}

/** The file a bucket really has, or null: Q4's silent fall-through to the first file never reaches the grid. */
export function buildFor(model: ModelDetail, bucket: Bucket, runtime: Runtime): Quant | null {
  const q = pickQuant(model, bucket, runtime);
  return q && bucketOf(q.label) === bucket ? q : null;
}

/** A synthetic column so `settingsFor` applies the toggles exactly as the table does. */
export function fitColumn(model: ModelDetail, bucket: Bucket, ctx: number): ModelColumn {
  return { id: model.id, quant: bucket === 'Q4' ? 'auto' : { bucket }, ctx };
}

/** One evaluation, or null when the bucket has no file. */
export function evaluateFit(model: ModelDetail, pick: MatrixRow, bucket: Bucket, ctx: number, state: AppState, factors: Factors): CellResult | null {
  const quant = buildFor(model, bucket, state.runtime);
  if (!quant) return null;
  return evaluateCell(model, pick.row.machine, settingsFor(state, pick.row, fitColumn(model, bucket, ctx), quant, pick.linked), factors);
}

export type FitVerdict = 'asked' | 'compromise' | 'no-fit';

export interface ModelSummary {
  modelId: string;
  name: string;
  provider: string;
  params: string;
  status: FitVerdict;
  marker: Marker;
  /** the highest-quality build that runs as asked at 32K; `kind: 'special'` when it is an engram-on-SSD build, tried after every bucket */
  best: { kind: 'bucket' | 'special'; bucket: Bucket; quant: Quant | SpecialBuild; verdict: 'runs' | 'tight'; tokS: number | null; feels: string | null; ctx: number; result: CellResult } | null;
  /** the longest context chip that runs as asked at Q4 */
  longest: { ctx: number; verdict: 'runs' | 'tight'; tokS: number | null; result: CellResult } | null;
  /** the Q4 / 32K evaluation: what the compromise and no-fit copy describe, and what the marker opens */
  asked: CellResult | null;
  /** memory the configuration in the headline needs: the best build's, the fix's, or (for a dash) what Q4 at 32K would need */
  memoryGb: number | null;
  memoryText: string;
  headline: string;
  detail: string | null;
  contextLine: string;
  ariaLabel: string;
  gaps: Bucket[];
}

function fits(r: CellResult | null): r is CellResult & { verdict: 'runs' | 'tight' } {
  return !!r && (r.verdict === 'runs' || r.verdict === 'tight');
}

/** The machine in words: "Mac Studio · M5 Max 128 GB", or the pool ("2 × NVIDIA DGX Spark · 128 GB each · 256 GB pooled"). */
export function machineText(pick: MatrixRow): string {
  return rowLabel(pick.group, pick.row, pick.linked);
}

export function modelSummary(model: ModelDetail, pick: MatrixRow, state: AppState, factors: Factors): ModelSummary {
  const gaps: Bucket[] = [];
  const evaluated = new Map<Bucket, CellResult>();
  let best: ModelSummary['best'] = null;
  for (const bucket of BEST_FIRST) {
    const quant = buildFor(model, bucket, state.runtime);
    if (!quant) {
      gaps.push(bucket);
      continue;
    }
    const r = evaluateFit(model, pick, bucket, DEFAULT_CONTEXT, state, factors);
    if (!r) continue;
    evaluated.set(bucket, r);
    if (fits(r)) {
      best = { kind: 'bucket', bucket, quant, verdict: r.verdict, tokS: r.speed.tokS, feels: r.speed.feelsLike?.label ?? null, ctx: DEFAULT_CONTEXT, result: r };
      break;
    }
  }
  // no file fits as asked: a build that keeps its n-gram table on the SSD may, and that is a real way to run the model
  if (!best) {
    (model.special_builds ?? []).forEach((build, i) => {
      if (best) return;
      const col: ModelColumn = { id: model.id, quant: { special: i }, ctx: DEFAULT_CONTEXT };
      const r = evaluateCell(model, pick.row.machine, settingsFor(state, pick.row, col, build, pick.linked), factors);
      if (fits(r)) best = { kind: 'special', bucket: bucketOf(build.label) ?? 'Q4', quant: build, verdict: r.verdict, tokS: r.speed.tokS, feels: null, ctx: DEFAULT_CONTEXT, result: r };
    });
  }
  // the buckets the loop never reached are still gaps or files; only their evaluation was skipped
  for (const bucket of BEST_FIRST) if (!gaps.includes(bucket) && !buildFor(model, bucket, state.runtime)) gaps.push(bucket);
  const q4 = buildFor(model, 'Q4', state.runtime);
  const asked = q4 ? (evaluated.get('Q4') ?? evaluateFit(model, pick, 'Q4', DEFAULT_CONTEXT, state, factors)) : null;
  let longest: ModelSummary['longest'] = null;
  if (q4) {
    for (const ctx of [...contextChips(model)].reverse()) {
      const r = ctx === DEFAULT_CONTEXT && asked ? asked : evaluateFit(model, pick, 'Q4', ctx, state, factors);
      if (fits(r)) {
        longest = { ctx, verdict: r.verdict, tokS: r.speed.tokS, result: r };
        break;
      }
    }
  }
  const status: FitVerdict = best ? 'asked' : asked?.verdict === 'compromise' ? 'compromise' : 'no-fit';
  const marker = markerForVerdict(best ? best.verdict : (asked?.verdict ?? 'no-fit'));
  // the row says one thing per column; the reason behind a ring or a dash is the detail (tooltip, aria, the sheet)
  let headline: string;
  let detail: string | null = null;
  if (best && best.kind === 'special') {
    headline = `${best.quant.label} · speed not estimated${best.verdict === 'tight' ? ' · tight' : ''}`;
    detail = `${formatGb(best.result.need.weightsGb)} in memory, ${formatGb(best.result.need.ssdGb)} streamed from the SSD`;
  } else if (best) {
    headline = `${best.bucket} · ${formatTokS(best.tokS)}${best.feels ? ` · ${best.feels}` : ''}${best.verdict === 'tight' ? ' · tight' : ''}`;
  } else if (asked?.verdict === 'compromise' && asked.fix) {
    const running = runningConfig(asked, { quantLabel: q4?.label ?? null, ctx: DEFAULT_CONTEXT });
    headline = `${bucketOf(running.quantLabel) ?? running.quantLabel} · ${formatTokS(asked.speed.tokS)} · with a compromise`;
    detail = asked.fix.reason;
  } else if (asked) {
    headline = "Doesn't fit";
    detail = asked.nearestMiss;
  } else {
    headline = "Doesn't fit";
    detail = 'no Q4 file for this runtime';
  }
  const contextLine = longest
    ? `up to ${formatContext(longest.ctx)} at Q4${longest.verdict === 'tight' ? ' · tight' : ''}`
    : asked?.verdict === 'compromise' && asked.fix
      ? 'Q4 with a compromise'
      : 'not at Q4';
  const memoryGb = best ? best.result.need.totalGb : asked?.verdict === 'compromise' && asked.fix ? asked.fix.need.totalGb : asked ? asked.need.totalGb : null;
  const memoryText = memoryGb === null ? '—' : status === 'no-fit' ? `${formatGb(memoryGb)} at Q4` : formatGb(memoryGb);
  const ariaLabel = `${model.name} on ${machineText(pick)}: ${MARKER_TEXT[marker]}; ${headline}${detail ? ` (${detail})` : ''}; ${contextLine}${memoryGb !== null ? `; needs ${memoryText}` : ''}`;
  return { modelId: model.id, name: model.name, provider: model.provider, params: paramsText(model), status, marker, best, longest, asked, memoryGb, memoryText, headline, detail, contextLine, ariaLabel, gaps };
}

export type FitCell =
  | { kind: 'gap'; bucket: Bucket }
  | { kind: 'cell'; bucket: Bucket; ctx: number; result: CellResult; marker: Marker; tokS: number | null; atTheFix: boolean; ariaLabel: string };

export interface ModelGrid {
  chips: number[];
  rows: { bucket: Bucket; quant: Quant | null; cells: FitCell[] }[];
}

/** Every bucket × every context chip the model allows. Gaps are never evaluated. */
export function modelGrid(model: ModelDetail, pick: MatrixRow, state: AppState, factors: Factors): ModelGrid {
  const chips = contextChips(model);
  const rows: ModelGrid['rows'] = [];
  for (const bucket of SIMPLE_BUCKETS) {
    const quant = buildFor(model, bucket, state.runtime);
    if (!quant) {
      rows.push({ bucket, quant: null, cells: [{ kind: 'gap', bucket }] });
      continue;
    }
    const cells: FitCell[] = [];
    for (const ctx of chips) {
      const result = evaluateCell(model, pick.row.machine, settingsFor(state, pick.row, fitColumn(model, bucket, ctx), quant, pick.linked), factors);
      const marker = markerForVerdict(result.verdict);
      const running = runningConfig(result, { quantLabel: quant.label, ctx });
      const what =
        result.verdict === 'compromise' && result.fix
          ? `runs with a compromise: ${result.fix.reason}`
          : result.verdict === 'no-fit'
            ? MARKER_TEXT.no
            : `${MARKER_TEXT[marker]}${result.verdict === 'tight' ? ' (tight)' : ''}, ${formatTokS(result.speed.tokS)}`;
      cells.push({
        kind: 'cell',
        bucket,
        ctx,
        result,
        marker,
        tokS: result.verdict === 'no-fit' ? null : result.speed.tokS,
        atTheFix: running.atTheFix,
        ariaLabel: `${model.name} at ${bucket} · ${formatContext(ctx)} on ${machineText(pick)}: ${what}`,
      });
    }
    rows.push({ bucket, quant, cells });
  }
  return { chips, rows };
}

// ---- ordering, the machine picker and the sheet ------------------------------------------------------------

/** Size bands by the Q4 file (the index carries it, so the list is ordered before any record loads): the memory class a model belongs to. */
export const SIZE_BANDS: { key: string; label: string; maxGb: number }[] = [
  { key: 'xs', label: 'Under 40 GB at Q4', maxGb: 40 },
  { key: 's', label: '40 to 120 GB at Q4', maxGb: 120 },
  { key: 'm', label: '120 to 256 GB at Q4', maxGb: 256 },
  { key: 'l', label: '256 to 512 GB at Q4', maxGb: 512 },
  { key: 'xl', label: 'Over 512 GB at Q4', maxGb: Number.POSITIVE_INFINITY },
];

export interface CatalogSection {
  key: string;
  label: string;
  entries: ModelIndexEntry[];
}

function q4SizeGb(m: ModelIndexEntry): number {
  return m.sizes.q4_gb ?? m.sizes.min_gb ?? Number.POSITIVE_INFINITY;
}

/** Active models only, in size bands, smallest first inside a band; superseded and hidden entries are left out. */
export function orderCatalog(index: ModelIndexEntry[]): CatalogSection[] {
  const active = index.filter((m) => m.status !== 'superseded' && m.status !== 'hidden').sort((a, b) => q4SizeGb(a) - q4SizeGb(b) || a.name.localeCompare(b.name));
  const sections: CatalogSection[] = [];
  let lower = 0;
  for (const band of SIZE_BANDS) {
    const entries = active.filter((m) => q4SizeGb(m) >= lower && q4SizeGb(m) < band.maxGb);
    if (entries.length) sections.push({ key: band.key, label: band.label, entries });
    lower = band.maxGb;
  }
  const unknown = active.filter((m) => !Number.isFinite(q4SizeGb(m)));
  if (unknown.length) sections.push({ key: 'unknown', label: 'Size not known yet', entries: unknown });
  return sections;
}

export function orderedIds(sections: CatalogSection[]): string[] {
  return sections.flatMap((s) => s.entries.map((m) => m.id));
}

export interface PickSection {
  label: string;
  rows: MatrixRow[];
}

function toRows(group: MachineGroup, linked: LinkedInfo | null = null): MatrixRow[] {
  return group.rows.map((row) => ({ key: rowKey(group.id, row.gb, linked), group, row, linked }));
}

/** Every machine the picker offers: current families in catalog order, then the user's own, then older machines. */
export function pickSections(groups: MachineGroup[], state: AppState): PickSection[] {
  const out: PickSection[] = [];
  const families = new Map<string, MatrixRow[]>();
  for (const g of groups) {
    if (g.status !== 'current') continue;
    const rows = families.get(g.family) ?? [];
    rows.push(...toRows(g));
    families.set(g.family, rows);
  }
  for (const [family, rows] of families) out.push({ label: family, rows });
  const customGroups = state.customMachines.map(customGroup);
  const customs = customGroups.flatMap((g) => toRows(g));
  if (customs.length) out.push({ label: 'Your machines', rows: customs });
  const linkedRows = state.groups.flatMap((id) => {
    const linked = state.linked[id];
    const group = linked ? (groups.find((g) => g.id === id) ?? customGroups.find((g) => g.id === id)) : undefined;
    return group && linked ? toRows(group, linked) : [];
  });
  if (linkedRows.length) out.push({ label: 'Linked in your table', rows: linkedRows });
  const older = groups.filter((g) => g.status === 'discontinued').flatMap((g) => toRows(g));
  if (older.length) out.push({ label: 'Older machines', rows: older });
  return out;
}

/** The row a key names, or the biggest machine in the table, or the biggest of the first current group. */
export function resolvePick(key: string | null, groups: MachineGroup[], state: AppState, tableRows: MatrixRow[]): MatrixRow | null {
  if (key) {
    const found = findRow([...groups, ...state.customMachines.map(customGroup)], key);
    if (found) return { key, group: found.group, row: found.row, linked: found.linked };
  }
  const size = (r: MatrixRow) => pooledGb(r.row.gb, r.linked);
  const biggest = tableRows.reduce<MatrixRow | null>((acc, r) => (acc === null || size(r) > size(acc) ? r : acc), null);
  if (biggest) return biggest;
  const first = groups.find((g) => g.status === 'current') ?? groups[0];
  if (!first) return null;
  const rows = toRows(first);
  return rows.reduce<MatrixRow | null>((acc, r) => (acc === null || r.row.gb > acc.row.gb ? r : acc), null);
}

/** What the cell sheet needs for a model that is not a table column. `quant` names the exact file when the caller evaluated one (the buying map's own-build option). */
export function syntheticSheetTarget(model: ModelDetail, pick: MatrixRow, bucket: Bucket, ctx: number, result: CellResult, state: AppState, quant?: Quant | SpecialBuild | null): { cell: MatrixCell; row: MatrixRow; column: MatrixColumn } {
  const column = fitColumn(model, bucket, ctx);
  const key = `${pick.key}|fit:${model.id}:${bucket}:${ctx}`;
  return {
    row: pick,
    column: { index: -1, column: quant && isSpecialBuild(quant) && model.special_builds ? { ...column, quant: { special: model.special_builds.indexOf(quant) } } : column, status: 'ready', model, quant: quant === undefined ? buildFor(model, bucket, state.runtime) : quant },
    cell: { key, rowKey: pick.key, column: -1, status: 'ready', result },
  };
}
