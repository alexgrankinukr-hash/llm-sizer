/**
 * "Which machine?": the buying map. One model at one build and context across every current machine
 * configuration, placed by memory size and estimated writing speed, then filtered by what a buyer asked for
 * (a minimum speed, a budget, a form factor) and turned into a recommendation. Pure, so the view memoises one
 * atlas per column and the share image renders the same list. The rules:
 *  - a configuration qualifies when it fits as asked (runs or tight), has a speed estimate at or above the
 *    minimum, costs no more than the budget (priced machines only) and matches the form factor; the machine
 *    the visitor already owns is exempt from the budget and the form factor, because it is not being bought;
 *  - a compromise (the engine's fix: a smaller build, a shorter context, apps closed, the limit lifted) never
 *    qualifies: it is a ring at the fix's speed and a separate alternative;
 *  - a machine that does not fit is never placed at a speed; a fit with no speed estimate never qualifies;
 *  - the cheapest qualifying configuration with a current list price is the pick; an unpriced one is never
 *    called cheapest; the same comparator orders every list, so the card never disagrees with the list.
 * No formula or constant changed: every cell is `evaluateCell` through `settingsFor`, as in the table.
 */
import { Q8_REPACK_BITS } from '../engine/constants';
import { nominalBits } from '../engine/fix';
import { evaluateCell } from '../engine/index';
import { formatContext, formatDate, formatGb, formatPrice, formatTokS } from '../engine/format';
import type { CellResult, Factors, ModelDetail, Quant, Runtime, SpecialBuild } from '../engine/types';
import { isSpecialBuild } from '../engine/memory';
import { FLAG_TEXT } from './layout';
import { buildFor, fitColumn, fitSettingsKey } from './machine-fit';
import { customGroup, groupLabel, pooledGb, pooledPrice, rowKey, type GroupRow, type LinkedInfo, type MachineGroup } from './machines';
import { settingsFor, type MatrixColumn } from './matrix';
import { ABOUT_PATH, machineAnchor } from './provenance';
import { bucketOf, buildInfo, quantsInBucket, type Bucket } from './quants';
import { quantKey, type AppState, type BuildKey, type FormFactor, type ModelColumn } from './state';

export type { BuildKey, FormFactor };

export interface BuildOption {
  key: BuildKey;
  /** the segment text: "Q4", "Q8", "SSD build", or the column's own bucket or file label when it is none of those */
  label: string;
  /** null when the model has no file in this bucket for the runtime: the segment is disabled */
  quant: Quant | SpecialBuild | null;
  /** the column evaluated through settingsFor: carries the table column's context and cache precision */
  column: ModelColumn;
  /** the bucket the sheet opens at */
  bucket: Bucket;
  /** true on the option that is the table column's own build */
  isColumn: boolean;
  /** why it is disabled, or a caveat worth a hover */
  note: string | null;
}

export type MarkStatus = 'fits' | 'compromise' | 'no-fit';
export type Miss = 'no-fit' | 'compromise' | 'unknown-speed' | 'speed' | 'unpriced' | 'price' | 'form';

export interface BuyingCriteria {
  minTokS: number;
  budgetUsd: number | null;
  formFactor: FormFactor;
}

/** One exact configuration (a machine row at one memory size) evaluated at one build; the criteria come later. */
export interface EvalMark {
  key: string;
  group: MachineGroup;
  row: GroupRow;
  gb: number;
  /** "MacBook Pro · M5 Max · 40-core GPU · 128 GB" */
  label: string;
  /** "MacBook Pro M5 Max 40-core": the chart label (the row already says the memory) */
  short: string;
  formFactor: FormFactor;
  priceUsd: number | null;
  priceSource: string | null;
  discontinued: boolean;
  custom: boolean;
  /** the pool this mark is, when the visitor linked the machine; null for one machine */
  linked: LinkedInfo | null;
  result: CellResult;
  status: MarkStatus;
  /** where the mark sits: the ask's speed for a fit, the fix's for a compromise, null for a dash or an unknown */
  tokS: number | null;
  approx: boolean;
  band: string | null;
  bandIndex: number;
  /** memory to spare after the configuration that fits, null for a dash */
  freeGb: number | null;
  aboutHref: string | null;
}

export interface BuyingMark extends EvalMark {
  misses: Miss[];
  qualifies: boolean;
  /** a compromise whose only miss is being a compromise: it would qualify at its fix */
  wouldQualify: boolean;
}

export interface BuildEval {
  option: BuildOption;
  marks: EvalMark[];
}

/** Every build evaluated once; switching builds or criteria never re-runs the engine. */
export interface BuildAtlas {
  model: ModelDetail;
  column: MatrixColumn;
  ctx: number;
  kvBits: number;
  runtime: Runtime;
  options: BuildOption[];
  builds: Map<BuildKey, BuildEval>;
  /** the x-axis ceiling over every enabled build, so the map never rescales on a switch */
  axisMax: number;
  /** the distinct memory sizes, ascending: the map's rows */
  rows: number[];
  machinesAsOf: string | null;
  /** the machine set has the visitor's own machines or older machines from the table beside the current catalog */
  hasOwnMachines: boolean;
}

export interface Recommendation {
  /** the cheapest qualifying configuration with a current list price */
  pick: BuyingMark | null;
  alternative: { mark: BuyingMark; why: ('headroom' | 'band' | 'form')[] } | null;
  unpricedQualifying: BuyingMark[];
  /** filled only when nothing qualifies: at most one nearest miss per kind */
  nearest: { kind: 'price' | 'form' | 'speed' | 'change'; mark: BuyingMark; detail: string }[];
}

export interface BuyingRow {
  gb: number;
  marks: BuyingMark[];
}

export interface BuyingMap {
  atlas: BuildAtlas;
  build: BuildOption;
  criteria: BuyingCriteria;
  marks: BuyingMark[];
  rows: BuyingRow[];
  qualifying: BuyingMark[];
  withChange: BuyingMark[];
  rest: BuyingMark[];
  recommendation: Recommendation;
  captionParts: string[];
}

// ---- the machine set ---------------------------------------------------------------------------------------

export interface MapRow {
  group: MachineGroup;
  row: GroupRow;
  discontinued: boolean;
  linked: LinkedInfo | null;
}

/**
 * Every row of every current catalog group, then the older catalog machines the visitor put in the table, then the
 * visitor's own machines, then the pools the visitor linked in the table (the single rows stay on the map beside them).
 */
export function mapRows(groups: MachineGroup[], state: AppState, includeLinked = true): MapRow[] {
  const out: MapRow[] = [];
  for (const group of groups) {
    if (group.status !== 'current') continue;
    for (const row of group.rows) out.push({ group, row, discontinued: false, linked: null });
  }
  for (const id of state.groups) {
    const group = groups.find((g) => g.id === id);
    if (!group || group.status === 'current') continue;
    for (const row of group.rows) out.push({ group, row, discontinued: true, linked: null });
  }
  const customs = state.customMachines.map(customGroup);
  for (const group of customs) {
    for (const row of group.rows) out.push({ group, row, discontinued: false, linked: null });
  }
  if (includeLinked) {
    for (const id of state.groups) {
      const linked = state.linked[id];
      if (!linked) continue;
      const group = groups.find((g) => g.id === id) ?? customs.find((g) => g.id === id);
      if (!group) continue;
      for (const row of group.rows) out.push({ group, row, discontinued: !group.custom && group.status !== 'current', linked });
    }
  }
  return out;
}

export function formFactorFor(group: Pick<MachineGroup, 'custom' | 'silhouette'>): FormFactor {
  if (group.custom) return 'any';
  return group.silhouette === 'laptop' ? 'laptop' : 'desktop';
}

/** The GPU bin a row belongs to ("40-core GPU"), when the catalog names one; a memory-only variant is not a bin. */
export function variantOf(row: GroupRow): string | null {
  if (row.binLabel) return row.binLabel;
  const inParens = /\(([^)]+)\)/.exec(row.machine.chip)?.[1];
  return inParens && !/\bGB\b/.test(inParens) ? inParens : null;
}

export function markLabels(group: MachineGroup, row: GroupRow, linked: LinkedInfo | null = null): { label: string; short: string } {
  const variant = variantOf(row);
  const name = group.kind === 'prebuilt' || group.family === group.chip ? group.family : `${group.family} ${group.chip}`;
  const shortOne = `${name}${variant ? ` ${variant.replace(/\s*GPU$/i, '')}` : ''}`;
  if (linked) {
    return {
      label: `${linked.count} × ${groupLabel(group)}${variant ? ` · ${variant}` : ''} · ${row.gb} GB each · ${pooledGb(row.gb, linked)} GB pooled`,
      short: `${linked.count} × ${shortOne}`,
    };
  }
  const label = `${groupLabel(group)}${variant ? ` · ${variant}` : ''} · ${row.gb} GB`;
  return { label, short: shortOne };
}

export function priceSourceFor(group: Pick<MachineGroup, 'kind' | 'family' | 'custom'>): string | null {
  if (group.custom) return null;
  if (group.kind === 'mac') return 'Apple';
  return group.family.split(' ')[0] || null;
}

// ---- builds ------------------------------------------------------------------------------------------------

function sameFile(a: Quant | SpecialBuild | null, b: Quant | SpecialBuild | null): boolean {
  return !!a && !!b && buildInfo(a).key === buildInfo(b).key;
}

/** The segment's caveat for a build that streams from the SSD. */
function ssdNote(build: SpecialBuild): string {
  const info = buildInfo(build);
  return `${build.label}: ${formatGb(info.sizeGb)} in memory, ${formatGb(info.ssdGb)} streamed from the SSD; speed not estimated`;
}

function runtimeName(runtime: Runtime): string {
  return runtime === 'mlx' ? 'MLX' : 'GGUF';
}

/** Q4, Q8, the model's engram-on-SSD build when it has one, then the column's own build when it is none of those. A bucket without a file is disabled, never another bucket's file. */
export function buildOptions(model: ModelDetail, column: MatrixColumn, runtime: Runtime): BuildOption[] {
  const ctx = column.column.ctx;
  const kv = column.column.kvBits ? { kvBits: column.column.kvBits } : {};
  const options: BuildOption[] = [];
  for (const bucket of ['Q4', 'Q8'] as const) {
    const quant = buildFor(model, bucket, runtime);
    const isColumn = sameFile(quant, column.quant);
    let note: string | null = null;
    if (!quant) note = `no ${bucket} file for ${runtimeName(runtime)}`;
    else if (bucket === 'Q8' && quant.bits !== null && quant.bits < Q8_REPACK_BITS && (nominalBits(quant.label) ?? 0) >= 8) note = FLAG_TEXT['q8-is-repack'];
    options.push({ key: bucket, label: bucket, quant, column: isColumn ? column.column : { ...fitColumn(model, bucket, ctx), ...kv }, bucket, isColumn, note });
  }
  const special = model.special_builds ?? [];
  const columnIsSpecial = !!column.quant && isSpecialBuild(column.quant);
  if (special.length && !columnIsSpecial) {
    // the build with the smallest part in memory: it fits the most machines
    const build = special.reduce((a, b) => (b.resident_gb < a.resident_gb ? b : a));
    const col: ModelColumn = { id: model.id, quant: { special: special.indexOf(build) }, ctx, ...kv };
    options.push({ key: 'ssd', label: 'SSD build', quant: build, column: col, bucket: bucketOf(build.label) ?? 'Q4', isColumn: false, note: ssdNote(build) });
  }
  if (!options.some((o) => o.isColumn) && column.quant) {
    const label = column.quant.label;
    const bucket = bucketOf(label) ?? 'Q4';
    options.push({ key: 'column', label: columnIsSpecial ? 'SSD build' : (bucketOf(label) ?? label), quant: column.quant, column: column.column, bucket, isColumn: true, note: columnIsSpecial ? ssdNote(column.quant as SpecialBuild) : null });
  }
  return options;
}

/** The option a key names when it is enabled; else the column's own; else the first enabled one. */
export function resolveBuild(options: BuildOption[], key: BuildKey): BuildOption {
  const wanted = key === 'column' ? options.find((o) => o.isColumn) : options.find((o) => o.key === key && o.quant);
  return wanted ?? options.find((o) => o.isColumn) ?? options.find((o) => o.quant) ?? options[0];
}

// ---- evaluation --------------------------------------------------------------------------------------------

function evaluateMark(model: ModelDetail, option: BuildOption, r: MapRow, state: AppState, factors: Factors): EvalMark {
  const result = evaluateCell(model, r.row.machine, settingsFor(state, r.row, option.column, option.quant, r.linked), factors);
  const status: MarkStatus = result.verdict === 'no-fit' ? 'no-fit' : result.verdict === 'compromise' ? 'compromise' : 'fits';
  const tokS = status === 'no-fit' ? null : result.speed.tokS;
  const band = status === 'no-fit' || tokS === null ? null : (result.speed.feelsLike?.label ?? null);
  const { label, short } = markLabels(r.group, r.row, r.linked);
  return {
    key: rowKey(r.group.id, r.row.gb, r.linked),
    group: r.group,
    row: r.row,
    gb: pooledGb(r.row.gb, r.linked),
    label,
    short,
    formFactor: formFactorFor(r.group),
    priceUsd: pooledPrice(r.row.priceUsd, r.linked),
    priceSource: priceSourceFor(r.group),
    discontinued: r.discontinued,
    custom: !!r.group.custom,
    linked: r.linked,
    result,
    status,
    tokS,
    approx: result.speed.efficiencySource !== 'measured' || result.speed.cluster?.source === 'assumed',
    band,
    bandIndex: band ? factors.feels_like.findIndex((b) => b.label === band) : -1,
    freeGb: status === 'no-fit' ? null : Math.max(0, result.availability.availableGb - result.need.totalGb),
    aboutHref: r.group.custom ? null : `${ABOUT_PATH}#${machineAnchor(r.row.machineId)}`,
  };
}

/** The x-axis ceiling rule the speed chart uses: at least 80, 15 % of room, rounded up to 20. */
export function axisMaxFor(speeds: number[]): number {
  return Math.max(80, Math.ceil((Math.max(1, ...speeds) * 1.15) / 20) * 20);
}

/** The engine pass: every enabled build across the whole machine set. */
export function evaluateBuilds(model: ModelDetail, column: MatrixColumn, groups: MachineGroup[], state: AppState, factors: Factors, machinesAsOf: string | null = null): BuildAtlas {
  const options = buildOptions(model, column, state.runtime);
  const rows = mapRows(groups, state);
  const builds = new Map<BuildKey, BuildEval>();
  const speeds: number[] = [];
  for (const option of options) {
    if (!option.quant) continue;
    const marks = rows.map((r) => evaluateMark(model, option, r, state, factors));
    for (const m of marks) if (m.tokS !== null) speeds.push(m.tokS);
    builds.set(option.key, { option, marks });
  }
  return {
    model,
    column,
    ctx: column.column.ctx,
    kvBits: column.column.kvBits ?? state.kvBits,
    runtime: state.runtime,
    options,
    builds,
    axisMax: axisMaxFor(speeds),
    rows: [...new Set(rows.map((r) => pooledGb(r.row.gb, r.linked)))].sort((a, b) => a - b),
    machinesAsOf,
    hasOwnMachines: rows.some((r) => r.group.custom || r.discontinued || r.linked),
  };
}

/** The settings an atlas depends on, as one string, so the view's memo survives unrelated state changes. */
export function atlasKey(state: AppState, column: MatrixColumn): string {
  const c = column.column;
  const linked = Object.entries(state.linked).map(([id, l]) => `${id}x${l.count}${l.split[0]}`).join(',');
  return [fitSettingsKey(state), column.index, c.id, quantKey(c.quant), c.ctx, c.kvBits ?? '', column.quant ? buildInfo(column.quant).key : '-', state.customMachines.map((m) => m.name).join(','), state.groups.join(','), linked].join('|');
}

/** The sheet's entry point: the column's own build (or the first with a file) through `cheapestSingleThatRuns`. */
export function cheapestSingleForColumn(model: ModelDetail, column: MatrixColumn, groups: MachineGroup[], state: AppState, factors: Factors): EvalMark | null {
  const options = buildOptions(model, column, state.runtime);
  const option = options.find((o) => o.isColumn && o.quant) ?? options.find((o) => o.quant);
  return option ? cheapestSingleThatRuns(model, option, groups, state, factors) : null;
}

/**
 * The cheapest single catalog machine that runs this model as asked (runs or tight, a list price): the comparison a
 * linked pool's sheet names, so two boxes never quietly beat one bigger box that fits.
 */
export function cheapestSingleThatRuns(model: ModelDetail, option: BuildOption, groups: MachineGroup[], state: AppState, factors: Factors): EvalMark | null {
  if (!option.quant) return null;
  const marks = mapRows(groups, state, false)
    .filter((r) => !r.discontinued && !r.group.custom)
    .map((r) => evaluateMark(model, option, r, state, factors))
    .filter((m) => m.status === 'fits' && m.priceUsd !== null)
    .map((m) => ({ ...m, misses: [] as Miss[], qualifies: true, wouldQualify: false }))
    .sort(byPrice);
  return marks[0] ?? null;
}

// ---- the criteria ------------------------------------------------------------------------------------------

/** Why a mark does not qualify, in the order the rules are applied; empty = it qualifies. */
export function misses(m: Pick<EvalMark, 'status' | 'tokS' | 'priceUsd' | 'formFactor'>, c: BuyingCriteria): Miss[] {
  if (m.status === 'no-fit') return ['no-fit'];
  const out: Miss[] = [];
  if (m.status === 'compromise') out.push('compromise');
  // no speed figure (a custom platform, a build that streams from the SSD): fine when any speed will do, never enough for a minimum
  if (m.tokS === null) {
    if (c.minTokS > 0) out.push('unknown-speed');
  } else if (m.tokS < c.minTokS) out.push('speed');
  if (c.budgetUsd !== null) {
    if (m.priceUsd === null) out.push('unpriced');
    else if (m.priceUsd > c.budgetUsd) out.push('price');
  }
  if (c.formFactor !== 'any' && m.formFactor !== 'any' && m.formFactor !== c.formFactor) out.push('form');
  return out;
}

export function qualifies(m: Pick<EvalMark, 'status' | 'tokS' | 'priceUsd' | 'formFactor'>, c: BuyingCriteria): boolean {
  return misses(m, c).length === 0;
}

/** Price ascending with unpriced last, then faster first, then more memory first, then the label. */
export const byPrice = (a: BuyingMark, b: BuyingMark): number => {
  if (a.priceUsd !== b.priceUsd) {
    if (a.priceUsd === null) return 1;
    if (b.priceUsd === null) return -1;
    return a.priceUsd - b.priceUsd;
  }
  if (a.tokS !== b.tokS) {
    if (a.tokS === null) return 1;
    if (b.tokS === null) return -1;
    return b.tokS - a.tokS;
  }
  return b.gb - a.gb || a.label.localeCompare(b.label);
};

function speedText(m: Pick<EvalMark, 'tokS' | 'approx'>): string {
  return `${m.approx && m.tokS !== null ? '~' : ''}${formatTokS(m.tokS)}`;
}

/** The recommendation under the map. `marks` may come in any order; every list here is cheapest first. */
export function recommend(marks: BuyingMark[], c: BuyingCriteria): Recommendation {
  const qualifying = marks.filter((m) => m.qualifies).sort(byPrice);
  const buyable = qualifying;
  const pick = buyable.find((m) => m.priceUsd !== null) ?? null;
  const unpricedQualifying = buyable.filter((m) => m.priceUsd === null);
  let alternative: Recommendation['alternative'] = null;
  if (pick) {
    for (const m of buyable.slice(buyable.indexOf(pick) + 1)) {
      if (m.priceUsd === null) continue;
      const why: ('headroom' | 'band' | 'form')[] = [];
      if (m.freeGb !== null && pick.freeGb !== null && m.freeGb >= pick.freeGb * 1.25 && m.freeGb - pick.freeGb >= 1) why.push('headroom');
      if (m.bandIndex > pick.bandIndex) why.push('band');
      if (c.formFactor === 'any' && m.formFactor !== 'any' && pick.formFactor !== 'any' && m.formFactor !== pick.formFactor) why.push('form');
      if (why.length) {
        alternative = { mark: m, why };
        break;
      }
    }
  }
  const nearest: Recommendation['nearest'] = [];
  if (!qualifying.length) {
    const only = (m: BuyingMark, kind: Miss) => m.misses.length === 1 && m.misses[0] === kind;
    const price = marks.filter((m) => only(m, 'price')).sort(byPrice)[0];
    if (price && c.budgetUsd !== null && price.priceUsd !== null) nearest.push({ kind: 'price', mark: price, detail: `${formatPrice(price.priceUsd)} · ${formatPrice(price.priceUsd - c.budgetUsd)} over your budget` });
    const form = marks.filter((m) => only(m, 'form')).sort(byPrice)[0];
    if (form) nearest.push({ kind: 'form', mark: form, detail: form.formFactor === 'laptop' ? 'a laptop' : 'a desktop' });
    const speed = marks.filter((m) => m.misses.length > 0 && m.misses.every((x) => x === 'speed')).sort((a, b) => (b.tokS ?? 0) - (a.tokS ?? 0) || byPrice(a, b))[0];
    if (speed) nearest.push({ kind: 'speed', mark: speed, detail: `${speedText(speed)}, under your ${c.minTokS}` });
    const change = marks.filter((m) => m.wouldQualify).sort(byPrice)[0];
    if (change && change.result.fix) nearest.push({ kind: 'change', mark: change, detail: change.result.fix.reason });
  }
  return { pick, alternative, unpricedQualifying, nearest };
}

function captionFor(atlas: BuildAtlas, build: BuildOption, toggles: { work: string; limit: string }): string[] {
  const file = build.quant?.label ?? null;
  const info = build.quant ? buildInfo(build.quant) : null;
  const ssd = info?.isSpecial ? `, ${formatGb(info.sizeGb)} in memory and ${formatGb(info.ssdGb)} from the SSD` : '';
  const at = file && file !== build.label ? `${build.label} (${file}${ssd})` : `${build.label}${ssd}`;
  const current = [...new Set(atlas.builds.get(build.key)?.marks.filter((m) => !m.custom && !m.discontinued).map((m) => m.group) ?? [])];
  const others = [...new Set(current.filter((g) => g.kind !== 'mac').map((g) => g.family))];
  const linkedSets = atlas.builds.get(build.key)?.marks.some((m) => m.linked) ?? false;
  const set = `every current Mac${others.length ? ` and the ${others.join(', ')}` : ''}${atlas.hasOwnMachines ? (linkedSets ? ', your machines and linked pools' : ', your machines') : ''}`;
  const parts = [
    `${atlas.model.name} at ${at} · ${formatContext(atlas.ctx)}`,
    runtimeName(atlas.runtime),
    `cache ${atlas.kvBits === 16 ? 'FP16' : `${atlas.kvBits}-bit`}`,
    toggles.work,
    toggles.limit,
    set,
    'speeds are estimates, ~ = no measured rows for the chip',
  ];
  if (atlas.machinesAsOf) parts.push(`list prices as of ${formatDate(atlas.machinesAsOf)}`);
  return parts;
}

/** The map for one build under the buyer's criteria. */
export function buyingMap(atlas: BuildAtlas, buildKey: BuildKey, criteria: BuyingCriteria, toggles: { work: string; limit: string }): BuyingMap {
  const build = resolveBuild(atlas.options, buildKey);
  const evalMarks = atlas.builds.get(build.key)?.marks ?? [];
  const marks: BuyingMark[] = evalMarks.map((m) => {
    const miss = misses(m, criteria);
    return { ...m, misses: miss, qualifies: miss.length === 0, wouldQualify: m.status === 'compromise' && miss.every((x) => x === 'compromise') };
  });
  const rows: BuyingRow[] = atlas.rows.map((gb) => ({
    gb,
    marks: marks.filter((m) => m.gb === gb).sort((a, b) => (a.tokS === null ? -1 : b.tokS === null ? 1 : a.tokS - b.tokS) || a.label.localeCompare(b.label)),
  }));
  const qualifying = marks.filter((m) => m.qualifies).sort(byPrice);
  const withChange = marks.filter((m) => m.wouldQualify).sort(byPrice);
  const rest = marks.filter((m) => !m.qualifies && !m.wouldQualify).sort(byPrice);
  return { atlas, build, criteria, marks, rows, qualifying, withChange, rest, recommendation: recommend(marks, criteria), captionParts: captionFor(atlas, build, toggles) };
}

/** The minimum-speed presets: "any", then every finite feels-like band edge (5, 12, 30, 60 tok/s). */
export function minSpeedPresets(factors: Factors): { tokS: number; label: string; opens: string | null }[] {
  const out: { tokS: number; label: string; opens: string | null }[] = [{ tokS: 0, label: 'any', opens: null }];
  factors.feels_like.forEach((band, i) => {
    if (band.max_tok_s !== null) out.push({ tokS: band.max_tok_s, label: String(band.max_tok_s), opens: factors.feels_like[i + 1]?.label ?? null });
  });
  return out;
}

/** One line for a mark in the list: what it gives, or why it misses. */
export function markDetail(m: BuyingMark, c: BuyingCriteria): string {
  if (m.status === 'no-fit') return `doesn't fit: ${m.result.nearestMiss ?? 'no build small enough'}`;
  if (m.status === 'compromise' && m.result.fix) return m.result.fix.reason;
  const first = m.misses.find((x) => x !== 'compromise');
  if (first === 'unknown-speed') return `no speed estimate, so your ${c.minTokS} tok/s cannot be checked`;
  if (first === 'speed') return `too slow: ${speedText(m)}, under your ${c.minTokS}`;
  if (first === 'price') return `over budget: ${formatPrice(m.priceUsd)}`;
  if (first === 'unpriced') return 'no current price';
  if (first === 'form') return m.formFactor === 'laptop' ? 'a laptop' : 'a desktop';
  const spare = m.freeGb !== null ? `${formatGb(m.freeGb)} to spare` : '';
  const noSpeed = m.tokS === null ? ' · speed not estimated' : '';
  return `${formatGb(m.result.need.totalGb)} of ${formatGb(m.result.availability.availableGb)} used${m.result.verdict === 'tight' ? ' · tight' : ''}${spare ? ` · ${spare}` : ''}${noSpeed}`;
}

/** A model's Q8 and Q4 files at a glance, for the caption of a disabled segment. */
export function hasBucket(model: ModelDetail, bucket: Bucket, runtime: Runtime): boolean {
  return quantsInBucket(model, bucket, runtime).length > 0;
}

// ---- label placement ---------------------------------------------------------------------------------------

export interface RowLabelInput {
  key: string;
  x: number;
  label: string;
  /** drawn heavier: the pick, the machine you own */
  bold?: boolean;
}

export interface PlacedLabel {
  /** one key for a label of its own; every key of a stack for a joined label */
  keys: string[];
  x: number;
  text: string;
  /** 0 = beside the mark, 1 = above the stack, 2 = below it, 3 = a second line above, 4 = a second line below */
  lane: 0 | 1 | 2 | 3 | 4;
  anchor: 'start' | 'end' | 'middle';
  dx: number;
  dy: number;
  hidden: boolean;
  /** vertical offsets of every mark in the stack this label belongs to */
  stackDy: Record<string, number>;
  bold: boolean;
  /** the label's horizontal extent, for hit targets */
  x0: number;
  x1: number;
}

const GAP = 8;
const LANE_DX = 9;
/** marks closer than this share a stack: two circles of radius 6.5 this close read as one */
const STACK_PX = 5;
/** the largest mark radius the chart draws: labels keep this clear of a neighbour's circle */
const MARK_R = 7;
/** a stack up to this size gets a label per mark, one under another; a bigger one gets one joined label */
const OWN_LABELS_MAX = 3;

/** Vertical offsets for marks that share an x: far enough apart that each mark can carry its own 10 px label. */
export function stackOffsets(n: number): number[] {
  if (n === 1) return [0];
  if (n === 2) return [-7, 7];
  if (n === 3) return [-12, 0, 12];
  return Array.from({ length: n }, (_, i) => -14 + (28 * i) / (n - 1));
}

type Lane = 0 | 1 | 2 | 3 | 4;

/** the label lanes: beside the mark, above it, below it, and a second line above and below; a stack pushes the outer lanes past its outermost circle */
function laneDy(lane: Lane, n: number): number {
  const spread = Math.max(0, ...stackOffsets(n));
  if (lane === 0) return 4;
  if (lane === 1) return -(spread + 9);
  if (lane === 2) return spread + 15;
  if (lane === 3) return -(spread + 20);
  return spread + 26;
}

/** "MacBook Pro / Mac Studio M5 Max 40-core": the labels of a stack, with the words they share said once. */
export function joinLabels(labels: string[]): string {
  if (labels.length === 1) return labels[0];
  const words = labels.map((l) => l.split(' '));
  let common = 0;
  while (words.every((w) => w.length > common + 1) && words.every((w) => w[w.length - 1 - common] === words[0][words[0].length - 1 - common])) common += 1;
  if (!common) return labels.join(' / ');
  const suffix = words[0].slice(words[0].length - common).join(' ');
  return `${words.map((w) => w.slice(0, w.length - common).join(' ')).join(' / ')} ${suffix}`;
}

/**
 * Labels for one row's marks. Marks that share an x form a stack; a stack of up to three gets a label per mark,
 * one under another beside the circles (to the right, else to the left); a bigger stack, or one with no room
 * beside it, gets one joined label beside it, else above or below it, else clipped. Labels never overlap each
 * other or a neighbouring stack's circles. Pure numbers, so it is testable without a DOM.
 */
export function placeRowLabels(marks: RowLabelInput[], measure: (text: string) => number, bounds: { left: number; right: number }): PlacedLabel[] {
  const sorted = [...marks].sort((a, b) => a.x - b.x);
  const stacks: { keys: string[]; x: number; labels: string[]; bold: boolean[] }[] = [];
  for (const m of sorted) {
    const last = stacks[stacks.length - 1];
    if (last && Math.abs(m.x - last.x) < STACK_PX) {
      last.keys.push(m.key);
      last.labels.push(m.label);
      last.bold.push(!!m.bold);
      last.x = (last.x * (last.keys.length - 1) + m.x) / last.keys.length;
    } else stacks.push({ keys: [m.key], x: m.x, labels: [m.label], bold: [!!m.bold] });
  }
  const lanes: Record<Lane, { x0: number; x1: number }[]> = { 0: [], 1: [], 2: [], 3: [], 4: [] };
  const free = (lane: Lane, x0: number, x1: number) => lanes[lane].every((p) => x1 + GAP <= p.x0 || x0 >= p.x1 + GAP);
  // a stack's outer circles reach into the lanes above and below: a label there keeps clear of them when its
  // text box (10 px type on the lane's baseline) would cross the circles, not merely share their column
  const tall = stacks.filter((s) => s.keys.length > 1).map((s) => ({ x0: s.x - MARK_R - GAP, x1: s.x + MARK_R + GAP, reach: Math.max(0, ...stackOffsets(s.keys.length)) + MARK_R, s }));
  const clearOfStacks = (self: (typeof stacks)[number], lane: Lane, n: number, x0: number, x1: number) => {
    const dy = laneDy(lane, n);
    const top = dy - 8;
    const bottom = dy + 2;
    return tall.every((t) => t.s === self || x1 <= t.x0 || x0 >= t.x1 || bottom <= -t.reach || top >= t.reach);
  };
  const out: PlacedLabel[] = [];
  // the tagged labels (the pick, the machine you own) are placed first, so they get the best lane in a crowd
  const order = stacks.map((s, i) => i).sort((a, b) => Number(stacks[b].bold.some(Boolean)) - Number(stacks[a].bold.some(Boolean)) || a - b);
  stackLoop: for (const i of order) {
    const s = stacks[i];
    const n = s.keys.length;
    const offsets = stackOffsets(n);
    const stackDy = Object.fromEntries(s.keys.map((k, j) => [k, offsets[j]]));
    const nextX = stacks[i + 1]?.x ?? Number.POSITIVE_INFINITY;
    const prevX = stacks[i - 1]?.x ?? Number.NEGATIVE_INFINITY;
    const anyBold = s.bold.some(Boolean);
    // a label per mark beside the circles, one under another
    if (n <= OWN_LABELS_MAX) {
      const widths = s.labels.map(measure);
      const maxW = Math.max(...widths);
      const right = { x0: s.x + LANE_DX, x1: s.x + LANE_DX + maxW };
      const left = { x0: s.x - LANE_DX - maxW, x1: s.x - LANE_DX };
      const side = right.x1 <= bounds.right && right.x1 + GAP + MARK_R <= nextX && free(0, right.x0, right.x1) ? 'start' : left.x0 >= bounds.left && left.x0 >= prevX + GAP + MARK_R && free(0, left.x0, left.x1) ? 'end' : null;
      if (side) {
        const span = side === 'start' ? right : left;
        lanes[0].push(span);
        s.keys.forEach((k, j) => {
          const w = widths[j];
          const x0 = side === 'start' ? span.x0 : span.x1 - w;
          out.push({ keys: [k], x: s.x, text: s.labels[j], lane: 0, anchor: side, dx: side === 'start' ? LANE_DX : -LANE_DX, dy: offsets[j] + 4, hidden: false, stackDy, bold: s.bold[j], x0, x1: x0 + w });
        });
        continue;
      }
    }
    // one joined label for the stack: beside it, else above or below it
    const text = joinLabels(s.labels);
    const width = measure(text);
    const base = { keys: s.keys, x: s.x, stackDy, hidden: false, bold: anyBold };
    let x0 = s.x + LANE_DX;
    let x1 = x0 + width;
    if (x1 <= bounds.right && x1 + GAP + MARK_R <= nextX && free(0, x0, x1)) {
      lanes[0].push({ x0, x1 });
      out.push({ ...base, text, lane: 0, anchor: 'start', dx: LANE_DX, dy: laneDy(0, n), x0, x1 });
      continue;
    }
    x1 = s.x - LANE_DX;
    x0 = x1 - width;
    if (x0 >= bounds.left && x0 >= prevX + GAP + MARK_R && free(0, x0, x1)) {
      lanes[0].push({ x0, x1 });
      out.push({ ...base, text, lane: 0, anchor: 'end', dx: -LANE_DX, dy: laneDy(0, n), x0, x1 });
      continue;
    }
    for (const lane of [1, 2, 3, 4] as const) {
      const cx0 = Math.max(bounds.left, Math.min(bounds.right - width, s.x - width / 2));
      const cx1 = cx0 + width;
      if (cx1 <= bounds.right && free(lane, cx0, cx1) && clearOfStacks(s, lane, n, cx0, cx1)) {
        lanes[lane].push({ x0: cx0, x1: cx1 });
        const dx = cx0 + width / 2 - s.x;
        out.push({ ...base, text, lane, anchor: 'middle', dx, dy: laneDy(lane, n), x0: cx0, x1: cx1 });
        continue stackLoop;
      }
    }
    // nothing fits: clip to the free gap on the right
    const room = Math.max(0, Math.min(bounds.right, nextX - GAP - MARK_R) - (s.x + LANE_DX));
    const perChar = Math.max(1, measure('a'));
    const chars = Math.floor(room / perChar);
    const clipped = chars >= text.length ? text : chars > 1 ? `${text.slice(0, chars - 1).trimEnd()}…` : '';
    x0 = s.x + LANE_DX;
    x1 = x0 + measure(clipped);
    lanes[0].push({ x0, x1 });
    out.push({ ...base, text: clipped, lane: 0, anchor: 'start', dx: LANE_DX, dy: laneDy(0, n), hidden: chars < 4, x0, x1 });
  }
  // placed in priority order, returned in reading order
  return out.sort((a, b) => a.x - b.x);
}
