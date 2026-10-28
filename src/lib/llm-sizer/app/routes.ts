/**
 * "Every way to get 128 GB" (METHODOLOGY §6.6): one model at one build across every route to about 128 GB of memory for
 * local AI, from a gaming PC with the experts in system memory to a 96 GB workstation card, with the price, the memory
 * speed, the memory the AI can use, whether the model fits, the writing speed and the time for a 1,000-word answer.
 *
 * Every number comes from `public/data/llm-sizer/routes-128gb.json` (hand-curated, dated, sourced) or from the engine:
 *  - Macs, the DGX Spark and the linked Sparks are catalog machines; their estimate is the engine's own (`floorMs`
 *    with the chip profile, the linked-machine rule for the pair), so it equals what every other view shows.
 *  - Strix Halo and the RTX PRO 6000 have no catalog profile; theirs is fitted here, the same straight line of
 *    milliseconds against gigabytes the chip profiles use, on published rows of other models with the engine's
 *    architecture costs taken off. The comparison's own models are never in those rows: they are the check.
 *  - A gaming card with the experts in system memory, the RTX Spark laptop and four-card PCs have no calibrated rule,
 *    so they get no estimate; the offload routes show the ceiling memory speed sets instead, labelled as one.
 * A published plain run always wins over the estimate; where the two differ by more than 15 % the reason is in the data.
 * Speed-up numbers (MTP, draft models) are their own row and never stand in for a plain number. Reading speed is shown
 * only where a run measured it. Pure; no React here.
 */
import { BYTES_PER_GB } from '../engine/constants';
import { neededMemory, ngramTableShare, tableOnSsdGb, verdictFor } from '../engine/memory';
import { activeParams, archClassOf, attendedTokens, expertLayersOf, floorMs, mlxContextGroupOf, mlxGroupOf, quantFamilyOf, readFactor, resolveChipProfile, sparseBudgetOf, type ChipProfile } from '../engine/speed';
import { clusterMs, linkClassOf } from '../engine/cluster';
import type { ArchClass, Factors, KvBits, Machine, ModelDetail, Need, Platform, Quant, QuantFormat, SpecialBuild, SpeedModel } from '../engine/types';
import { bucketOf, pickQuant, quantsInBucket, type Bucket } from './quants';

// ---------------------------------------------------------------- the data file

export type RouteGroup = 'pc' | 'box' | 'mac' | 'multi' | 'big';
export const ROUTE_GROUPS: { key: RouteGroup; label: string }[] = [
  { key: 'pc', label: 'The PC you already own' },
  { key: 'box', label: 'Shared-memory boxes and laptops' },
  { key: 'mac', label: 'Macs, new and used' },
  { key: 'multi', label: 'Two boxes linked' },
  { key: 'big', label: 'More than 128 GB, or more money' },
];

export interface Source {
  usd?: number;
  name?: string;
  stock?: string;
  as_of?: string;
  source?: string;
}

export interface Measured {
  model_id: string;
  bucket: Bucket;
  kind: 'plain' | 'speedup';
  /** the short name the table's cell gives a speed-up run (the engine's name otherwise) */
  label?: string;
  /** a plain run too thin or too contradictory to stand as the number (shown in the detail, never as the headline) */
  unsettled?: boolean;
  write: number;
  write_range?: [number, number];
  n?: number;
  context: number;
  engine: string;
  /** which runtime the run used, when it is not the one `facts.runtime` names */
  engine_family?: 'vllm' | 'mlx' | 'gguf';
  build: string;
  variant?: string;
  /** what the engine needs to estimate the same build: active billions, bits over the total parameters, class */
  facts?: { active_b: number; bits: number; quant_label: string; cls: ArchClass; runtime: 'mlx' | 'gguf' | 'vllm' };
  read?: { tok_s?: number; tok_s_range?: [number, number]; context: number; engine: string; note?: string };
  date: string;
  grade: string;
  sources: string[];
  note?: string;
}

export interface CardSpec {
  name: string;
  vram_gb: number;
  bandwidth_gbs: number;
}

export interface RouteSpeed {
  rule: 'engine' | 'cluster' | 'profile' | 'offload' | 'none';
  /** catalog machine id (engine, cluster) */
  machine?: string;
  memory_gb?: number;
  machines?: number;
  split?: 'layer' | 'tensor';
  /** fitted route profile key (profile) */
  profile?: string;
  runtime?: 'mlx' | 'gguf';
  /** the n-gram table of a model that has one stays out of memory on this route whatever the file */
  table_off_memory?: boolean;
  card?: CardSpec;
  ram?: { gb: number; bandwidth_gbs: number };
  none_reason?: string;
}

export interface RouteSpec {
  id: string;
  group: RouteGroup;
  name: string;
  detail: string;
  condition: 'new' | 'used' | 'preorder';
  boxes?: Source[];
  price: { usd: number | null; usd_high?: number; what: string; as_of: string; grade: string; sources: string[] };
  memory_speed: { gbs: number; text: string; note?: string; source: string | null };
  usable: { gb: number; text: string; note?: string; alt?: { gb: number; text: string; when: string }; source: string | null };
  speed: RouteSpeed;
  measured: Measured[];
  /** another route whose measurements stand for this one (the same chip) */
  measured_from?: string;
  measured_note?: string;
  /** why the estimate misses on this route, by architecture class, `mixtures` (any mixture of experts) or `all` */
  misses?: Record<string, string>;
  misses_from?: string;
}

export interface ProfileRow {
  model: string;
  quant: string;
  active_b: number;
  bits: number;
  cls: ArchClass;
  tg: number;
  source: string;
  date: string;
  note?: string;
}

export interface ProfileSpec {
  name: string;
  platform: Platform;
  chip: string;
  bandwidth_gbs: number;
  note: string;
  rows: ProfileRow[];
}

export interface RoutesFile {
  schema: string;
  data_as_of: string;
  prices_checked: string;
  answer: { words: number; tokens: number };
  document: { pages: number; tokens: number };
  contexts: number[];
  default_model: string;
  featured_models: string[];
  model_facts: Record<string, { kv8: number; attention: string; moe_layers?: number | null; model_type?: string } | string>;
  /** what each model's n-gram table is, in words; its share of a file is the engine's (`NGRAM_TABLES`) */
  ngram_tables: Record<string, { note: string; source: string }>;
  profiles: Record<string, ProfileSpec>;
  routes: RouteSpec[];
}

/** The share of a model's estimate error the comparison accepts before a published run replaces it (METHODOLOGY §6.6). */
export const CHECK_TOLERANCE = 0.15;
/** tg128 and short rows: the context a profile row was measured at, in tokens */
const PROFILE_ROW_CONTEXT = 128;

// ---------------------------------------------------------------- route profiles

export interface FittedProfile extends ChipProfile {
  platform: Platform;
  /** rows the line rests on */
  n: number;
  rows: { model: string; quant: string; measured: number; predicted: number }[];
  /** the fit's worst row, percent */
  maxErrPct: number;
  /** the intercept came out negative and the line was refitted through zero */
  throughZero: boolean;
}

/**
 * A route's profile from published rows of other models: the straight line of milliseconds per token against gigabytes
 * read that the chip profiles use (METHODOLOGY §6.1), with each mixture's architecture cost taken off first. The slope
 * is the effective bandwidth, the intercept the fixed cost; a negative intercept is refitted through zero.
 */
export function fitRouteProfile(spec: ProfileSpec, sm: SpeedModel): FittedProfile {
  const pts = spec.rows.map((r) => {
    const family = quantFamilyOf({ label: r.quant, format: 'gguf' } as Quant);
    const x = ((r.active_b * r.bits) / 8) * readFactor(sm, family, spec.platform);
    const y = 1000 / r.tg - (sm.architecture_cost_ms[r.cls]?.value ?? 0);
    return { r, x, y };
  });
  const n = pts.length;
  const mx = pts.reduce((s, p) => s + p.x, 0) / n;
  const my = pts.reduce((s, p) => s + p.y, 0) / n;
  const sxx = pts.reduce((s, p) => s + (p.x - mx) ** 2, 0);
  const sxy = pts.reduce((s, p) => s + (p.x - mx) * (p.y - my), 0);
  let slope = sxy / sxx;
  let t0 = my - slope * mx;
  let throughZero = false;
  if (t0 < 0) {
    slope = pts.reduce((s, p) => s + p.x * p.y, 0) / pts.reduce((s, p) => s + p.x * p.x, 0);
    t0 = 0;
    throughZero = true;
  }
  // the slope is milliseconds per GB read: the effective bandwidth is its inverse, in GB/s
  const bEffGbs = 1000 / slope;
  const profile: ChipProfile = { t0Ms: round(t0, 2), bEffRatio: round(bEffGbs / spec.bandwidth_gbs, 3), bEffGbs: 0, source: 'measured', key: spec.chip, fitN: n };
  profile.bEffGbs = profile.bEffRatio * spec.bandwidth_gbs;
  const rows = pts.map(({ r }) => {
    const gb = (r.active_b * r.bits) / 8;
    const parts = floorMs(profile, spec.bandwidth_gbs, gb, quantFamilyOf({ label: r.quant, format: 'gguf' } as Quant), r.cls, 'gguf', 'dense', 'gqa', PROFILE_ROW_CONTEXT, sm, spec.platform);
    return { model: r.model, quant: r.quant, measured: r.tg, predicted: round(1000 / parts.ms, 1) };
  });
  const maxErrPct = round(Math.max(...rows.map((x) => Math.abs(x.predicted / x.measured - 1) * 100)), 1);
  return { ...profile, platform: spec.platform, n, rows, maxErrPct, throughZero };
}

function round(x: number, d: number): number {
  const k = 10 ** d;
  return Math.round(x * k) / k;
}

export function fittedProfiles(file: RoutesFile, factors: Factors): Record<string, FittedProfile> {
  const sm = factors.speed_model;
  if (!sm) return {};
  return Object.fromEntries(Object.entries(file.profiles).map(([k, spec]) => [k, fitRouteProfile(spec, sm)]));
}

// ---------------------------------------------------------------- the estimate

/** What the speed formula needs about a build, from a model record and a file, or from a measured run's facts. */
export interface BuildFacts {
  activeB: number;
  bits: number;
  quantLabel: string;
  format: QuantFormat;
  cls: ArchClass;
  /** cache bytes per token at 8-bit, all growing layers */
  kv8: number;
  attention: string;
  /** the layers that carry experts (the expert-layer rule on MLX); null for a dense model */
  moeLayers?: number | null;
  /** the cached tokens a sparse-attention model reads per token (engine `sparseBudgetOf`), null for the rest */
  sparseBudget: number | null;
}

export function buildFactsOf(model: ModelDetail, quant: Quant): BuildFacts | null {
  const active = activeParams(model);
  if (active === null || quant.bits === null) return null;
  return {
    activeB: active / BYTES_PER_GB,
    bits: quant.bits,
    quantLabel: quant.label,
    format: quant.format,
    cls: archClassOf(model),
    kv8: model.architecture?.full_bytes_per_token_8bit ?? 0,
    attention: model.architecture?.attention ?? 'gqa',
    moeLayers: expertLayersOf(model),
    sparseBudget: sparseBudgetOf(model.architecture),
  };
}

export interface Estimate {
  tokS: number;
  /** where the number comes from, for the detail line */
  how: string;
  runtime: 'mlx' | 'gguf';
  /** ms per token: read, fixed, context (one machine), and the link for a pool */
  parts: { readMs: number; fixedMs: number; attnMs: number; linkMs: number };
}

export interface Context {
  factors: Factors;
  machines: Machine[];
  profiles: Record<string, FittedProfile>;
}

export function catalogMachine(route: RouteSpec, machines: Machine[]): Machine | null {
  return route.speed.machine ? (machines.find((m) => m.id === route.speed.machine) ?? null) : null;
}

/** The runtime a route writes this file with: MLX on a Mac when the file is MLX, llama.cpp otherwise. */
export function routeRuntime(route: RouteSpec, format: QuantFormat): 'mlx' | 'gguf' {
  return route.speed.runtime === 'mlx' && format === 'mlx' ? 'mlx' : 'gguf';
}

/**
 * The tool's estimate of writing speed on a route, or null when the route has no calibrated rule (offload, none) or
 * no profile. Engine and cluster routes run exactly `estimateSpeed`'s fixed-cost branch (a test holds them equal).
 */
export function routeEstimate(route: RouteSpec, b: BuildFacts, contextTokens: number, kvBits: KvBits, ctx: Context): Estimate | null {
  const sm = ctx.factors.speed_model;
  if (!sm || sm.kind !== 'floor') return null;
  // GB read per token, as estimateSpeed computes it: the active weights plus the cache re-read over the attended tokens
  const gbAt = (runtime: 'mlx' | 'gguf') => (b.activeB * b.bits) / 8 + (b.kv8 * kvBits * attendedTokens(contextTokens, runtime, b.sparseBudget)) / 8 / BYTES_PER_GB;
  const family = quantFamilyOf({ label: b.quantLabel, format: b.format } as Quant);
  const rule = route.speed.rule;
  if (rule === 'engine' || rule === 'cluster') {
    const machine = catalogMachine(route, ctx.machines);
    if (!machine) return null;
    const profile = resolveChipProfile(machine, ctx.factors);
    if (!profile) return null;
    const runtime = machine.platform === 'apple' ? routeRuntime(route, b.format) : 'gguf';
    const group = mlxGroupOf(b.cls, b.activeB, machine);
    const p = floorMs(profile, machine.bandwidth_gbs, gbAt(runtime), family, b.cls, runtime, group, b.attention, contextTokens, sm, machine.platform, mlxContextGroupOf(machine), { chip: machine.chip, moeLayers: b.moeLayers }, b.sparseBudget);
    const fixedText = p.expertLayerMs !== undefined ? `${b.moeLayers} expert layers at ${p.expertLayerMs.toFixed(2)} ms` : `${profile.t0Ms} ms fixed`;
    const how = `${runtime === 'mlx' ? 'MLX' : 'llama.cpp'} on the ${machine.chip} profile (${Math.round(profile.bEffGbs)} GB/s effective, ${fixedText})`;
    if (rule === 'engine') return { tokS: 1000 / p.ms, how, runtime, parts: { readMs: p.readMs, fixedMs: p.fixedMs, attnMs: p.attnMs, linkMs: 0 } };
    const cf = ctx.factors.cluster;
    const n = route.speed.machines ?? 2;
    if (!cf) return null;
    const c = clusterMs(p.ms, n, route.speed.split ?? 'layer', linkClassOf(machine), b.cls === 'dense' ? 'dense' : 'moe', cf);
    return { tokS: 1000 / c.ms, how: `${how}, ${n} boxes on a layer split (+${c.hopMs ?? c.cMs} ms per hop)`, runtime, parts: { readMs: p.readMs, fixedMs: p.fixedMs, attnMs: p.attnMs, linkMs: c.ms - p.ms } };
  }
  if (rule === 'profile' && route.speed.profile) {
    const profile = ctx.profiles[route.speed.profile];
    if (!profile) return null;
    const p = floorMs(profile, route.memory_speed.gbs, gbAt('gguf'), family, b.cls, 'gguf', 'dense', b.attention, contextTokens, sm, profile.platform, 'pre_m5');
    return { tokS: 1000 / p.ms, how: `llama.cpp on the route's own profile (${Math.round(profile.bEffGbs)} GB/s effective, ${profile.t0Ms} ms fixed, fitted on ${profile.n} published rows)`, runtime: 'gguf', parts: { readMs: p.readMs, fixedMs: p.fixedMs, attnMs: p.attnMs, linkMs: 0 } };
  }
  return null;
}

/** Everything the estimate reads: the factors, the machine catalog and the routes' fitted profiles. */
export function routesContext(file: RoutesFile, factors: Factors, machines: Machine[]): Context {
  return { factors, machines, profiles: fittedProfiles(file, factors) };
}

// ---------------------------------------------------------------- memory and the file a route runs

export type RouteVerdict = 'runs' | 'tight' | 'spills' | 'no-fit' | 'no-file';

/** A plain file seen as the part that must be in memory, when the model's n-gram table stays out of it on this route. */
function residentView(quant: Quant, share: number): SpecialBuild {
  return { label: quant.label, resident_gb: quant.size_gb * (1 - share), ssd_gb: quant.size_gb * share, format: quant.format, bits: quant.bits ?? undefined };
}

/**
 * The memory a file needs on a route. An MLX file leaves the n-gram table on the SSD in the engine itself, as in every
 * other view (`tableOnSsdGb`); a route whose engine keeps it out of memory whatever the file says so in the data. The
 * prompt scratch follows the route's catalog machine (`scratchGb`); a route without one is charged the pre-M5 scratch.
 */
function needOn(route: RouteSpec, model: ModelDetail, quant: Quant, contextTokens: number, kvBits: KvBits, availableGb: number, machine: Pick<Machine, 'chip'> | null): { need: Need; tableOutGb: number } {
  const loader = route.speed.runtime === 'mlx' ? 'mlx' : 'gguf';
  const share = route.speed.table_off_memory && tableOnSsdGb(quant, model.architecture, loader) === 0 ? ngramTableShare(model.architecture) : 0;
  const build: Quant | SpecialBuild = share > 0 ? residentView(quant, share) : quant;
  const cluster = route.speed.rule === 'cluster' ? { n: route.speed.machines ?? 2, split: route.speed.split ?? 'layer' } : null;
  const need = neededMemory(build, model.architecture, contextTokens, kvBits, availableGb, undefined, cluster, undefined, loader, machine ? mlxContextGroupOf(machine) : 'pre_m5');
  return { need, tableOutGb: need.ssdGb };
}

export interface FilePick {
  quant: Quant | null;
  verdict: RouteVerdict;
  need: Need | null;
  /** the GB of an n-gram table that stays on the SSD or in system memory on this route */
  tableOutGb: number;
  /** when nothing in the build fits: the smallest file's need */
  nearest: Need | null;
}

/**
 * The file a route runs at a build, in the route's own format (MLX on a Mac, GGUF elsewhere): the file every other view
 * uses for that build when it fits, else the largest smaller file of the same build that fits with room (90 %), else the
 * largest that fits at all, the way the fix search steps down. An owner runs the best copy the machine can hold, so two
 * routes can run different files of one build. A card route that cannot hold the model on the card spills the rest into
 * the PC's memory. `machine` is the route's catalog machine (`catalogMachine`), whose chip sets the MLX prompt scratch.
 */
export function pickRouteFile(route: RouteSpec, model: ModelDetail, bucket: Bucket, contextTokens: number, kvBits: KvBits, machine: Pick<Machine, 'chip'> | null = null): FilePick {
  const runtime = route.speed.runtime === 'mlx' ? 'mlx' : 'gguf';
  const pool = quantsInBucket(model, bucket, runtime).filter((q) => q.format !== 'safetensors' && bucketOf(q.label) === bucket);
  if (!pool.length) return { quant: null, verdict: 'no-file', need: null, tableOutGb: 0, nearest: null };
  const usual = pickQuant(model, bucket, runtime);
  const first = usual && pool.includes(usual) ? usual : null;
  const bySize = [...(first ? [first] : []), ...pool.filter((q) => q !== first && (!first || q.size_bytes < first.size_bytes)).sort((a, b) => b.size_bytes - a.size_bytes)];
  const usable = route.usable.gb;
  const evals = bySize.map((q) => ({ q, ...needOn(route, model, q, contextTokens, kvBits, usable, machine) }));
  const runs = evals.find((e) => verdictFor(e.need) === 'runs');
  const tight = evals.find((e) => verdictFor(e.need) === 'tight');
  const hit = runs ?? tight;
  if (hit) return { quant: hit.q, verdict: runs ? 'runs' : 'tight', need: hit.need, tableOutGb: hit.tableOutGb, nearest: null };
  // a card with the PC's memory behind it: the model spills over, at the PC's memory speed
  if (route.speed.card && route.speed.ram && route.speed.rule === 'profile') {
    const total = usable + route.speed.ram.gb;
    const spill = [...evals].reverse().find((e) => e.need.totalGb <= total * 0.9) ?? [...evals].reverse().find((e) => e.need.totalGb <= total);
    if (spill) return { quant: spill.q, verdict: 'spills', need: { ...spill.need, ratio: spill.need.totalGb / total }, tableOutGb: spill.tableOutGb, nearest: null };
  }
  const smallest = evals[evals.length - 1];
  return { quant: null, verdict: 'no-fit', need: null, tableOutGb: 0, nearest: smallest.need };
}

// ---------------------------------------------------------------- the offload ceiling

/**
 * The most a card-plus-system-memory route could write if both memories ran at their full speed and nothing else cost
 * time: the weights on the card read at the card's speed, the experts left in the PC's memory at that memory's speed.
 * A ceiling, never an estimate: real runs land well below it, and no plain run says how far.
 */
export function offloadCeiling(route: RouteSpec, model: ModelDetail, quant: Quant, contextTokens: number, kvBits: KvBits): number | null {
  const card = route.speed.card;
  const ram = route.speed.ram;
  const a = model.architecture;
  const total = model.params_total;
  const active = activeParams(model);
  if (!card || !ram || !a || !total || !active || quant.bits === null) return null;
  const bytesPerParam = quant.bits / 8;
  const tableShare = ngramTableShare(a);
  const cacheGb = (a.full_bytes_per_token_8bit * kvBits * contextTokens) / 8 / BYTES_PER_GB;
  let gpuGb: number;
  let ramGb: number;
  const perExpert = 3 * (a.hidden_size ?? 0) * (a.moe_intermediate ?? 0);
  const layers = a.moe_layers ?? a.layers;
  if ((a.experts_total ?? 0) > 1 && perExpert > 0) {
    const routedTotal = layers * (a.experts_total ?? 0) * perExpert;
    const routedActive = Math.min(active, layers * (a.experts_active ?? 0) * perExpert);
    const fixedTotal = Math.max(0, total * (1 - tableShare) - routedTotal);
    const roomForExperts = Math.max(0, card.vram_gb - cacheGb - 1 - (fixedTotal * bytesPerParam) / BYTES_PER_GB);
    const onCard = Math.min(1, roomForExperts / ((routedTotal * bytesPerParam) / BYTES_PER_GB));
    gpuGb = ((active - routedActive) * bytesPerParam + routedActive * bytesPerParam * onCard) / BYTES_PER_GB + cacheGb;
    ramGb = (routedActive * bytesPerParam * (1 - onCard)) / BYTES_PER_GB;
  } else {
    const weights = (active * bytesPerParam) / BYTES_PER_GB;
    const onCard = Math.min(1, Math.max(0, card.vram_gb - cacheGb - 1) / weights);
    gpuGb = weights * onCard + cacheGb;
    ramGb = weights * (1 - onCard);
  }
  const ms = (gpuGb / card.bandwidth_gbs + ramGb / ram.bandwidth_gbs) * 1000;
  return ms > 0 ? 1000 / ms : null;
}

// ---------------------------------------------------------------- one model across every route

export interface ReadingOut {
  /** tokens a second, or a range */
  tokS: number | [number, number];
  /** seconds for the document, or a range (the slow end first) */
  seconds: number | [number, number];
  engine: string;
  context: number;
  note?: string;
}

export interface RouteResult {
  route: RouteSpec;
  /** the catalog machine, when the route is one */
  machine: Machine | null;
  pick: FilePick;
  /** the same model with the alternative memory setting (Windows on Strix Halo, the Surface after its setting change) */
  altVerdict: RouteVerdict | null;
  estimate: Estimate | null;
  /** the published plain run for this model and build, when one stands */
  measured: Measured | null;
  /** a plain run too thin to stand (the M5 Ultra) */
  unsettled: Measured | null;
  speedups: Measured[];
  headline: { kind: 'measured' | 'estimate' | 'none'; tokS: number | null; range: [number, number] | null; answerS: number | null };
  /** estimate against the published run, when both exist */
  check: CheckResult | null;
  /** why the estimate misses here, when the check fails or a run of the same kind missed on this route */
  caution: string | null;
  /** why there is no estimate at all */
  noEstimate: string | null;
  ceiling: number | null;
  reading: ReadingOut | null;
}

export interface CheckResult {
  estimate: number;
  measured: number;
  /** estimate ÷ measured − 1 */
  err: number;
  /** the run used another runtime than the route's estimate (vLLM against a llama.cpp profile) */
  otherRuntime: boolean;
  ok: boolean;
}

/** Why the estimate misses on a route for a class: the exact class, then any mixture of experts, then everything. */
export function missReason(route: RouteSpec, cls: ArchClass, routes: RouteSpec[]): string | null {
  const src = route.misses ?? (route.misses_from ? routes.find((r) => r.id === route.misses_from)?.misses : undefined);
  if (!src) return null;
  return src[cls] ?? (cls !== 'dense' ? src.mixtures : undefined) ?? src.all ?? null;
}

export function measurementsOf(route: RouteSpec, routes: RouteSpec[]): Measured[] {
  if (route.measured.length || !route.measured_from) return route.measured;
  return routes.find((r) => r.id === route.measured_from)?.measured ?? [];
}

function readingOf(entries: Measured[], docTokens: number): ReadingOut | null {
  const withRead = entries.find((m) => m.read && (m.read.tok_s || m.read.tok_s_range));
  const r = withRead?.read;
  if (!r) return null;
  if (r.tok_s_range) {
    const [lo, hi] = r.tok_s_range;
    return { tokS: [lo, hi], seconds: [docTokens / lo, docTokens / hi], engine: r.engine, context: r.context, note: r.note };
  }
  const t = r.tok_s as number;
  return { tokS: t, seconds: docTokens / t, engine: r.engine, context: r.context, note: r.note };
}

function runtimeOfRun(m: Measured): 'mlx' | 'gguf' | 'vllm' {
  return m.engine_family ?? m.facts?.runtime ?? 'gguf';
}

/** The tool's estimate for a published run's own build at the run's own context, from the run's facts (null without them). */
export function estimateForRun(route: RouteSpec, m: Measured, file: RoutesFile, ctx: Context): Estimate | null {
  const mf = file.model_facts[m.model_id];
  const facts = m.facts;
  if (!facts || !mf || typeof mf !== 'object' || route.speed.rule === 'offload' || route.speed.rule === 'none') return null;
  const format: QuantFormat = facts.runtime === 'mlx' ? 'mlx' : 'gguf';
  return routeEstimate(route, { activeB: facts.active_b, bits: facts.bits, quantLabel: facts.quant_label, format, cls: facts.cls, kv8: mf.kv8, attention: mf.attention, moeLayers: mf.moe_layers ?? null, sparseBudget: sparseBudgetOf(mf) }, m.context, 16, ctx);
}

export function checkOf(estimate: number, m: Measured, routeRuntimeName: 'mlx' | 'gguf'): CheckResult {
  const err = estimate / m.write - 1;
  const otherRuntime = runtimeOfRun(m) !== routeRuntimeName;
  return { estimate, measured: m.write, err, otherRuntime, ok: !otherRuntime && Math.abs(err) <= CHECK_TOLERANCE };
}

/**
 * Every route for one model at one build and context. `model` is the catalog record; the measured runs are matched
 * by model id and build. The estimate is computed for the file each route would run (`pickRouteFile`).
 */
export function routesTable(file: RoutesFile, model: ModelDetail, bucket: Bucket, contextTokens: number, kvBits: KvBits, ctx: Context): RouteResult[] {
  const cls = archClassOf(model);
  return file.routes.map((route): RouteResult => {
    const machine = catalogMachine(route, ctx.machines);
    const pick = pickRouteFile(route, model, bucket, contextTokens, kvBits, machine);
    let altVerdict: RouteVerdict | null = null;
    if (route.usable.alt) altVerdict = pickRouteFile({ ...route, usable: { ...route.usable, gb: route.usable.alt.gb } }, model, bucket, contextTokens, kvBits, machine).verdict;
    const all = measurementsOf(route, file.routes).filter((m) => m.model_id === model.id && m.bucket === bucket);
    const measured = all.find((m) => m.kind === 'plain' && !m.unsettled) ?? null;
    const unsettled = all.find((m) => m.kind === 'plain' && m.unsettled) ?? null;
    const speedups = all.filter((m) => m.kind === 'speedup');
    const fits = pick.verdict === 'runs' || pick.verdict === 'tight';
    const facts = pick.quant ? buildFactsOf(model, pick.quant) : null;
    const estimate = fits && facts && route.speed.rule !== 'offload' && route.speed.rule !== 'none' ? routeEstimate(route, facts, contextTokens, kvBits, ctx) : null;
    // the check is like for like: the estimate for the run's own build at its own context, as in the calibration table
    const runEstimate = measured ? estimateForRun(route, measured, file, ctx) : null;
    const check = runEstimate && measured ? checkOf(runEstimate.tokS, measured, routeRuntime(route, measured.facts?.runtime === 'mlx' ? 'mlx' : 'gguf')) : null;
    const reason = missReason(route, cls, file.routes);
    const caution = check ? (check.ok ? null : reason) : estimate ? reason : null;
    let noEstimate: string | null = null;
    if (!estimate) {
      if (pick.verdict === 'no-file') noEstimate = 'no file of this build in the catalog';
      else if (!fits && pick.verdict !== 'spills') noEstimate = null;
      else noEstimate = route.speed.none_reason ?? 'no profile for this machine yet';
    }
    const ceiling = pick.quant && (route.speed.rule === 'offload' || pick.verdict === 'spills') ? offloadCeiling(route, model, pick.quant, contextTokens, kvBits) : null;
    const tokens = file.answer.tokens;
    const headline: RouteResult['headline'] = measured
      ? { kind: 'measured', tokS: measured.write, range: measured.write_range ?? null, answerS: tokens / measured.write }
      : estimate
        ? { kind: 'estimate', tokS: estimate.tokS, range: null, answerS: tokens / estimate.tokS }
        : { kind: 'none', tokS: null, range: null, answerS: null };
    // reading: only what a run measured, the plain run's first, then a speed-up run's (a draft does not change reading much)
    const reading = readingOf([...(measured ? [measured] : []), ...(unsettled ? [unsettled] : []), ...speedups], file.document.tokens);
    return { route, machine, pick, altVerdict, estimate, measured, unsettled, speedups, headline, check, caution, noEstimate, ceiling, reading };
  });
}

// ---------------------------------------------------------------- the calibration table

export interface CalibrationRow {
  routeId: string;
  route: string;
  modelId: string;
  build: string;
  variant: string | null;
  measured: number;
  range: [number, number] | null;
  estimate: number | null;
  err: number | null;
  status: 'within' | 'off' | 'other-runtime' | 'unsettled' | 'no-estimate';
  reason: string | null;
  engine: string;
}

/** Predicted against measured for every published plain run in the file, from the runs' own facts (METHODOLOGY §6.6). */
export function calibrationTable(file: RoutesFile, ctx: Context): CalibrationRow[] {
  const out: CalibrationRow[] = [];
  for (const route of file.routes) {
    for (const m of route.measured) {
      if (m.kind !== 'plain') continue;
      const mf = file.model_facts[m.model_id];
      const facts = m.facts;
      const estimate = mf && typeof mf === 'object' ? estimateForRun(route, m, file, ctx) : null;
      const runtime = routeRuntime(route, facts?.runtime === 'mlx' ? 'mlx' : 'gguf');
      const check = estimate ? checkOf(estimate.tokS, m, runtime) : null;
      const status: CalibrationRow['status'] = m.unsettled ? 'unsettled' : !check ? 'no-estimate' : check.otherRuntime ? 'other-runtime' : check.ok ? 'within' : 'off';
      out.push({
        routeId: route.id,
        route: route.name,
        modelId: m.model_id,
        build: m.build,
        variant: m.variant ?? null,
        measured: m.write,
        range: m.write_range ?? null,
        estimate: estimate ? round(estimate.tokS, 1) : null,
        err: check ? round(check.err, 3) : null,
        status,
        reason: status === 'off' || status === 'other-runtime' ? missReason(route, facts?.cls ?? 'dense', file.routes) : null,
        engine: m.engine,
      });
    }
  }
  return out;
}

/** "about 19 s", "1.1 min": the answer and document timers. */
export function secondsText(s: number): string {
  if (s < 10) return `${s.toFixed(1)} s`;
  if (s < 90) return `${Math.round(s)} s`;
  return `${(s / 60).toFixed(1)} min`;
}
