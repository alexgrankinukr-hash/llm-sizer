/**
 * "How fast is every Mac?": one model at one build and context on every Apple chip in the catalog, as a grid of
 * generations (M1, M2, …) by tier (base, Pro, Max, Ultra). A chip is every catalog row that carries it, at every memory
 * size it was sold with; the grid shows one size per chip (the biggest, or the smallest that runs the model) and the
 * size list under it shows them all. Memory size moves speed only through the chip version that comes with it: writing
 * follows memory bandwidth (the 36 GB M5 Max is the 32-core chip at 460 GB/s), reading follows GPU cores (the 256 GB
 * M5 Ultra reads on the 64-core GPU its list price buys), and the size decides what fits. A chip the catalog adds later
 * (an M6 Pro, an M7) gets its cell, and its column, by itself.
 * No formula or constant changed: every configuration is `evaluateCell` through `settingsFor`, as in the table.
 */
import { evaluateCell } from '../engine/index';
import { firstWordWaitS, prefillCoresOf } from '../engine/prefill';
import { chipKey, generationNumberOf, tierOf } from '../engine/speed';
import type { CellResult, Factors, ModelDetail, Quant, SpecialBuild } from '../engine/types';
import { buildFor, fitColumn } from './machine-fit';
import { rowKey, type GroupRow, type MachineGroup } from './machines';
import { settingsFor } from './matrix';
import { bucketOf, SIMPLE_BUCKETS, type Bucket } from './quants';
import { DEFAULT_CONTEXT, nearestChip, type AppState, type MemoryPick, type ModelColumn, type View } from './state';

export type ChipTier = 'base' | 'pro' | 'max' | 'ultra';
export const CHIP_TIERS: ChipTier[] = ['base', 'pro', 'max', 'ultra'];
export const TIER_NAMES: Record<ChipTier, string> = { base: 'Base', pro: 'Pro', max: 'Max', ultra: 'Ultra' };
export type { MemoryPick };
/** runs as asked; runs only with the macOS memory limit lifted (or the work apps closed); does not fit as asked */
export type ChipStatus = 'runs' | 'lifted' | 'no-fit';

/** Families in the order a buyer thinks of them: laptops, then the small desktops, then the big ones. */
const FAMILY_ORDER = ['MacBook Air', 'MacBook Pro', 'Mac mini', 'iMac', 'Mac Studio', 'Mac Pro'];

/** One memory size of one chip version, merged across the families that sell it (same bandwidth, same GPU cores). */
export interface ChipConfig {
  /** the row key of the representative machine row, for the sheet */
  key: string;
  gb: number;
  bandwidthGbs: number;
  /** the GPU cores reading runs on at this size: the version the list price buys */
  cores: number | null;
  families: string[];
  group: MachineGroup;
  row: GroupRow;
  /** the cheapest list price among the families, null when none is on sale */
  priceUsd: number | null;
  result: CellResult;
  status: ChipStatus;
  /** what it took to run, for a `lifted` configuration */
  lifted: ('override' | 'closeApps')[];
  /** writing speed, tokens per second; null when it does not fit or has no estimate */
  writeTokS: number | null;
  /** reading speed at a short prompt, tokens per second */
  readTokS: number | null;
  /** the wait before the first word for a prompt as long as the context, pasted in one go */
  waitS: number | null;
  /** the chip has measured runs of its own; otherwise the number is extrapolated */
  writeMeasured: boolean;
  readMeasured: boolean;
}

export interface ChipCell {
  tier: ChipTier;
  gen: number;
  /** "M5 Max" */
  name: string;
  /** the year the chip first shipped */
  year: number | null;
  /** every configuration, smallest memory first, the faster version first at a size */
  configs: ChipConfig[];
}

export interface ChipGrid {
  model: ModelDetail;
  bucket: Bucket;
  /** the file every configuration was evaluated with; null when the model has no file in this bucket for the runtime */
  quant: Quant | null;
  ctx: number;
  /** generations present in the catalog, oldest first */
  gens: number[];
  /** the year each generation first shipped */
  genYears: Record<number, number>;
  tiers: ChipTier[];
  /** the families each tier comes in, for the row labels */
  tierFamilies: Record<ChipTier, string[]>;
  cells: Map<string, ChipCell>;
}

export function cellId(tier: ChipTier, gen: number): string {
  return `${tier}|${gen}`;
}

/** The tier and generation of an Apple machine row; null for anything else. */
export function chipOf(machine: Pick<GroupRow['machine'], 'platform' | 'chip' | 'family'>): { tier: ChipTier; gen: number; name: string } | null {
  if (machine.platform !== 'apple') return null;
  const name = chipKey(machine.chip);
  const gen = generationNumberOf(name);
  const tier = tierOf(machine) as ChipTier | null;
  if (gen === null || !tier || !CHIP_TIERS.includes(tier)) return null;
  return { tier, gen, name };
}

export function statusOf(result: CellResult): { status: ChipStatus; lifted: ChipConfig['lifted'] } {
  if (result.verdict === 'runs' || result.verdict === 'tight') return { status: 'runs', lifted: [] };
  const changes = result.fix?.changes ?? [];
  if (result.verdict === 'compromise' && changes.length > 0 && changes.every((c) => c.kind === 'override' || c.kind === 'closeApps')) {
    return { status: 'lifted', lifted: [...new Set(changes.map((c) => c.kind as 'override' | 'closeApps'))] };
  }
  return { status: 'no-fit', lifted: [] };
}

function familyRank(family: string): number {
  const i = FAMILY_ORDER.indexOf(family);
  return i === -1 ? FAMILY_ORDER.length : i;
}

/** Faster first at a size: bandwidth (writing), then cores (reading). */
function fasterFirst(a: ChipConfig, b: ChipConfig): number {
  return a.gb - b.gb || b.bandwidthGbs - a.bandwidthGbs || (b.cores ?? 0) - (a.cores ?? 0);
}

/** The engine pass: every memory size of every Apple chip in the catalog, at one build and context. */
export function chipGrid(model: ModelDetail, bucket: Bucket, ctx: number, groups: MachineGroup[], state: AppState, factors: Factors): ChipGrid {
  const quant = buildFor(model, bucket, state.runtime);
  const column = fitColumn(model, bucket, ctx);
  const cells = new Map<string, ChipCell>();
  const families: Record<ChipTier, Set<string>> = { base: new Set(), pro: new Set(), max: new Set(), ultra: new Set() };
  for (const group of groups) {
    if (group.custom) continue;
    for (const row of group.rows) {
      const chip = chipOf(row.machine);
      if (!chip) continue;
      families[chip.tier].add(group.family);
      const id = cellId(chip.tier, chip.gen);
      let cell = cells.get(id);
      if (!cell) {
        cell = { tier: chip.tier, gen: chip.gen, name: chip.name, year: null, configs: [] };
        cells.set(id, cell);
      }
      const year = row.machine.year ?? null;
      if (year !== null && (cell.year === null || year < cell.year)) cell.year = year;
      if (!quant) continue;
      const cores = prefillCoresOf(row.machine, row.gb);
      const same = cell.configs.find((c) => c.gb === row.gb && c.bandwidthGbs === row.bandwidthGbs && c.cores === cores);
      if (same) {
        if (!same.families.includes(group.family)) same.families.push(group.family);
        if (row.priceUsd !== null && (same.priceUsd === null || row.priceUsd < same.priceUsd)) same.priceUsd = row.priceUsd;
        continue;
      }
      const result = evaluateCell(model, row.machine, settingsFor(state, row, column, quant), factors);
      const { status, lifted } = statusOf(result);
      const fits = status !== 'no-fit';
      const p = result.prefill;
      const readTokS = fits && p?.tokS ? p.tokS : null;
      cell.configs.push({
        key: rowKey(group.id, row.gb),
        gb: row.gb,
        bandwidthGbs: row.bandwidthGbs,
        cores,
        families: [group.family],
        group,
        row,
        priceUsd: row.priceUsd,
        result,
        status,
        lifted,
        writeTokS: fits ? result.speed.tokS : null,
        readTokS,
        waitS: readTokS !== null && p?.d0 ? firstWordWaitS(readTokS, p.d0, result.contextTokens) : null,
        writeMeasured: result.speed.efficiencySource === 'measured',
        readMeasured: p?.source === 'measured' || (factors.prefill?.chips_with_runs ?? []).includes(chip.name),
      });
    }
  }
  const genYears: Record<number, number> = {};
  for (const cell of cells.values()) {
    cell.configs.sort(fasterFirst);
    for (const c of cell.configs) c.families.sort((a, b) => familyRank(a) - familyRank(b));
    if (cell.year !== null && (genYears[cell.gen] === undefined || cell.year < genYears[cell.gen])) genYears[cell.gen] = cell.year;
  }
  const gens = [...new Set([...cells.values()].map((c) => c.gen))].sort((a, b) => a - b);
  const tierFamilies = Object.fromEntries(CHIP_TIERS.map((t) => [t, [...families[t]].sort((a, b) => familyRank(a) - familyRank(b))])) as Record<ChipTier, string[]>;
  return { model, bucket, quant, ctx, gens, genYears, tiers: CHIP_TIERS.filter((t) => tierFamilies[t].length > 0), tierFamilies, cells };
}

/** The fastest configuration at each memory size, smallest first. */
export function bestBySize(cell: ChipCell): ChipConfig[] {
  const out: ChipConfig[] = [];
  for (const c of cell.configs) if (!out.some((o) => o.gb === c.gb)) out.push(c);
  return out;
}

/**
 * The configuration a cell shows. `max`: the biggest memory the chip was sold with (its fastest version too).
 * `min`: the smallest that runs the model as asked; failing that, the smallest that runs with the limit lifted; failing
 * that, the biggest, which then reads "doesn't fit".
 */
export function shownConfig(cell: ChipCell, pick: MemoryPick): ChipConfig | null {
  const sizes = bestBySize(cell);
  if (!sizes.length) return null;
  const biggest = sizes[sizes.length - 1];
  if (pick === 'max') return biggest;
  return sizes.find((c) => c.status === 'runs') ?? sizes.find((c) => c.status === 'lifted') ?? biggest;
}

/** The memory sizes a chip was sold with, ascending: "36, 48, 64 or 128 GB". */
export function sizesText(cell: ChipCell): string {
  const sizes = [...new Set(cell.configs.map((c) => c.gb))];
  if (sizes.length <= 1) return `${sizes[0] ?? ''} GB`;
  return `${sizes.slice(0, -1).join(', ')} or ${sizes[sizes.length - 1]} GB`;
}

/** Which band a speed falls in (the feels-like bands; the last band is open-ended). */
export function writeBandIndex(tokS: number, factors: Pick<Factors, 'feels_like'>): number {
  const i = factors.feels_like.findIndex((b) => b.max_tok_s === null || tokS < b.max_tok_s);
  return i === -1 ? factors.feels_like.length - 1 : i;
}

/** Which wait band a wait falls in, the way the engine names it (an edge belongs to the band below it). */
export function waitBandIndex(seconds: number, factors: Pick<Factors, 'prefill'>): number {
  const bands = factors.prefill?.feels_like ?? [];
  const i = bands.findIndex((b) => b.max_s === null || seconds <= b.max_s);
  return i === -1 ? bands.length - 1 : i;
}

/**
 * Where the grid opens from another view: the model of the column the visitor was looking at (the first column
 * otherwise), at that column's build when it is one of the simple buckets (Q4 otherwise) and its context chip.
 */
export function chipsViewFor(column: { column: ModelColumn; quant: Quant | SpecialBuild | null } | null, fallbackModel: string): Extract<View, { kind: 'chips' }> {
  const b = column?.quant ? bucketOf(column.quant.label) : null;
  return {
    kind: 'chips',
    model: column?.column.id ?? fallbackModel,
    bucket: b && SIMPLE_BUCKETS.includes(b) ? b : 'Q4',
    ctx: nearestChip(column?.column.ctx ?? DEFAULT_CONTEXT),
    mode: 'write',
    memory: 'max',
  };
}

/** The builds a model can be shown at: the simple buckets that have a file for the runtime. */
export function chipBuckets(model: ModelDetail, runtime: AppState['runtime']): Bucket[] {
  return SIMPLE_BUCKETS.filter((b) => buildFor(model, b, runtime) !== null);
}
