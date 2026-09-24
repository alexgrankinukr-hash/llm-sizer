/**
 * Prefill (prompt processing): the read of the prompt before the first word. Compute-bound, so tokens/s × active
 * parameters is the constant, per GPU core of a generation on Apple silicon (fitted on the community llama.cpp table)
 * or per chip elsewhere; K-quant files, the architecture class and MLX scale it; the context already read slows it.
 * Mirrors speed_model.py (prefill_constant, prefill_tok_s, prefill_d0, first_word_wait_s). METHODOLOGY section 6.4.
 */
import { archClassOf, generationNumberOf, mlxContextGroupOf, quantFamilyOf } from './speed';
import { isSpecialBuild } from './memory';
import { prefillClusterFactor, termStatus, type Cluster } from './cluster';
import type { Factors, Machine, ModelDetail, Prefill, PrefillFactors, Quant, Runtime, SpecialBuild } from './types';

const PLAIN_QUANTS = new Set(['Q4_0', 'Q8_0', 'F16', 'BF16', 'MXFP4']);

/** "M5 Max (40-core GPU)" → "M5 Max": the rows name the chip without its bin. */
function bareChip(chip: string): string {
  return chip.replace(/\s*\(.*\)\s*$/, '').trim();
}

/**
 * The GPU cores a machine row runs prefill on at a memory size: the bin its list price buys there. Where the catalog
 * merges two bins on one bandwidth, that is the smaller one unless `gpu_cores_by_gb` says the size comes only with
 * the larger (a 64 GB MacBook Pro M5 Pro is 20-core); without a size, the smaller bin.
 */
export function prefillCoresOf(machine: Pick<Machine, 'gpu_cores' | 'gpu_cores_by_gb'>, memoryGb?: number): number | null {
  const cores = machine.gpu_cores ?? [];
  if (!cores.length) return null;
  const priced = memoryGb === undefined ? undefined : machine.gpu_cores_by_gb?.[String(memoryGb)];
  return priced ?? Math.min(...cores);
}

/** The larger GPU bin of the same chip, when the row reads on a smaller one: its cores, how much faster it reads, and Apple's price for it at this size where known. */
export function gpuUpgradeOf(machine: Pick<Machine, 'gpu_cores' | 'gpu_cores_by_gb' | 'gpu_upgrade_usd'>, memoryGb?: number): { cores: number; ratio: number; usd: number | null } | null {
  const cores = prefillCoresOf(machine, memoryGb);
  const largest = machine.gpu_cores?.length ? Math.max(...machine.gpu_cores) : null;
  if (!cores || !largest || largest <= cores) return null;
  const usd = memoryGb === undefined ? undefined : machine.gpu_upgrade_usd?.[String(memoryGb)];
  return { cores: largest, ratio: largest / cores, usd: usd ?? null };
}

/** K and where it comes from (speed_model.py:prefill_constant). */
export function prefillConstant(pf: PrefillFactors, machine: Pick<Machine, 'chip' | 'gpu_cores' | 'gpu_cores_by_gb' | 'platform' | 'family'>, memoryGb?: number): { k: number | null; source: Prefill['source']; cores: number | null; perCore: number | null; generation: string | null } {
  if (machine.platform === 'apple') {
    const gen = generationNumberOf(machine.chip);
    const cores = prefillCoresOf(machine, memoryGb);
    if (gen === null || !cores) return { k: null, source: 'none', cores, perCore: null, generation: null };
    const key = `M${gen}`;
    const row = pf.per_core[key];
    if (row) return { k: row.value * cores, source: pf.chips_with_rows.includes(bareChip(machine.chip)) ? 'measured' : 'generation', cores, perCore: row.value, generation: key };
    const newest = Object.keys(pf.per_core).sort((a, b) => Number(b.slice(1)) - Number(a.slice(1)))[0];
    if (!newest) return { k: null, source: 'none', cores, perCore: null, generation: null };
    return { k: pf.per_core[newest].value * cores, source: 'assumed', cores, perCore: pf.per_core[newest].value, generation: newest };
  }
  // other platforms carry one constant per machine, keyed the way the rows name it (the chip, or the family: "DGX Spark")
  const names = [machine.chip, machine.family ?? '', (machine.family ?? '').replace(/^NVIDIA\s+/i, '')];
  const row = names.map((n) => pf.by_chip[n]).find((r) => r);
  if (row) return { k: row.value, source: 'measured', cores: null, perCore: null, generation: null };
  return { k: null, source: 'none', cores: null, perCore: null, generation: null };
}

/** Short-prompt tokens/s for one build (speed_model.py:prefill_tok_s). */
export function prefillTokS(pf: PrefillFactors, k: number, quantLabel: string, format: 'gguf' | 'mlx', cls: string, runtime: Runtime, activeB: number, mlxGroup: 'pre_m5' | 'm5plus'): number {
  const kquant = runtime !== 'mlx' && format !== 'mlx' && !PLAIN_QUANTS.has(quantLabel.toUpperCase()) ? pf.kquant.factor : 1;
  const classFactor = (pf.class[cls] ?? pf.class.dense).factor;
  const mlx = runtime === 'mlx' ? pf.mlx[mlxGroup].factor : 1;
  return (k * kquant * classFactor * mlx) / activeB;
}

export function prefillClassOf(model: ModelDetail, pf: PrefillFactors): string {
  return pf.class_by_model[model.id] ?? archClassOf(model);
}

/** The context depth at which the rate has halved (speed_model.py:prefill_d0). */
export function prefillD0(pf: PrefillFactors, model: Pick<ModelDetail, 'architecture'>): { d0: number; assumed: boolean } {
  const a = model.architecture;
  const ctx = pf.context;
  const share = a?.kv_layers && a.layers ? Math.max(0.05, a.kv_layers / a.layers) : ctx.attention_share_ref;
  return { d0: (ctx.d0_tokens * ctx.attention_share_ref) / share, assumed: Math.abs(share - ctx.attention_share_ref) > 1e-9 };
}

export function firstWordWaitS(tokS: number, d0: number, promptTokens: number): number {
  return (promptTokens + (promptTokens * promptTokens) / (2 * d0)) / tokS;
}

export function prefillRateAt(tokS: number, d0: number, depth: number): number {
  return tokS / (1 + depth / d0);
}

export function feelsLikeWait(pf: PrefillFactors, seconds: number): string {
  for (const b of pf.feels_like) if (b.max_s === null || seconds <= b.max_s) return b.label;
  return pf.feels_like[pf.feels_like.length - 1]?.label ?? '';
}

export interface PrefillInput {
  model: ModelDetail;
  quant: Quant | SpecialBuild;
  machine: Machine;
  /** the machine's memory size (per machine in a pool): picks the GPU bin the list price buys */
  memoryGb?: number;
  runtime: Runtime;
  /** the context the column runs at: the wait for a prompt that fills it is one of the numbers shown */
  contextTokens: number;
  factors: Factors;
  /** linked machines pooled as one; absent or null for one machine */
  cluster?: Cluster | null;
}

export function estimatePrefill({ model, quant, machine, memoryGb, runtime, contextTokens, factors, cluster = null }: PrefillInput): Prefill {
  const pf = factors.prefill;
  const notes: string[] = [];
  const empty: Prefill = { tokS: null, d0: null, waits: [], atContext: null, source: 'none', parts: null, upgrade: null, notes };
  if (!pf) return { ...empty, notes: ['no prefill factors in this data set'] };
  const active = (model.params_active ?? model.params_total ?? 0) / 1e9;
  if (!active) return { ...empty, notes: ['no parameter count for this model'] };
  if (isSpecialBuild(quant)) return { ...empty, notes: ['no prefill estimate for a build that streams from the SSD'] };
  const c = prefillConstant(pf, machine, memoryGb);
  if (c.k === null) return { ...empty, notes: [machine.platform === 'apple' ? 'no prefill rate for this chip yet' : 'no prefill rate for this platform yet'] };
  const family = quantFamilyOf(quant);
  const kquant = runtime !== 'mlx' && family !== 'mlx' && !PLAIN_QUANTS.has(quant.label.toUpperCase()) ? pf.kquant.factor : 1;
  const cls = prefillClassOf(model, pf);
  const classFactor = (pf.class[cls] ?? pf.class.dense).factor;
  const mlxGroup = mlxContextGroupOf(machine);
  const mlx = runtime === 'mlx' ? pf.mlx[mlxGroup].factor : 1;
  const single = (c.k * kquant * classFactor * mlx) / active;
  // linked machines: one machine's rate on a layer split (measured flat), scaled by an assumed exponent under tensor parallel
  const pooled = cluster && factors.cluster ? prefillClusterFactor(cluster.n, cluster.split, factors.cluster) : null;
  const clusterFactor = pooled?.factor ?? 1;
  const tokS = single * clusterFactor;
  const { d0, assumed: d0Assumed } = prefillD0(pf, model);
  const waits = pf.waits_tokens.map((tokens) => ({ tokens, seconds: firstWordWaitS(tokS, d0, tokens), feelsLike: feelsLikeWait(pf, firstWordWaitS(tokS, d0, tokens)) }));
  const atContext = { tokens: contextTokens, seconds: firstWordWaitS(tokS, d0, contextTokens), rateTokS: prefillRateAt(tokS, d0, contextTokens) };
  // the larger GPU bin of the same chip reads in proportion to its cores (the constant is per core)
  const bin = machine.platform === 'apple' ? gpuUpgradeOf(machine, memoryGb) : null;
  const upgrade = bin ? { ...bin, tokS: tokS * bin.ratio, waits: pf.waits_tokens.map((tokens) => ({ tokens, seconds: firstWordWaitS(tokS * bin.ratio, d0, tokens) })) } : null;
  if (c.source === 'generation') notes.push(`no measured prefill row for the ${machine.chip}: its generation's rate per GPU core on ${c.cores} cores`);
  if (c.source === 'assumed') notes.push(`no measured prefill row for the ${machine.chip} or its generation: the ${c.generation} rate per GPU core on ${c.cores} cores`);
  const clsRow = pf.class[cls];
  if (clsRow && clsRow.n === 0) notes.push(`the ${cls.replace('_', ' ')} prefill factor is assumed, no measured row yet`);
  if (runtime === 'mlx' && pf.mlx[mlxGroup].n === 0) notes.push('the MLX prefill factor for this generation is assumed');
  if (d0Assumed) notes.push('how the rate falls with context is scaled from one published sweep by this model\'s share of attention layers');
  if (cluster && !factors.cluster) notes.push('no linked-machine terms in this data set: one machine\'s reading speed shown');
  if (pooled && cluster) {
    if (cluster.split === 'layer') notes.push(`reading speed is one machine's across ${cluster.n} linked machines on a layer split (measured flat: the machines read in turn)`);
    else notes.push(`reading speed × ${cluster.n}^${pooled.term.value} = ${clusterFactor.toFixed(2)} across ${cluster.n} linked machines in tensor parallel (${termStatus(pooled.term) === 'fitted' ? 'fitted' : 'assumed: compute-bound, one short-prompt measurement of 1.56× on two machines'})`);
  }
  return {
    tokS,
    d0,
    waits,
    atContext,
    source: c.source,
    parts: { k: c.k, cores: c.cores, perCore: c.perCore, generation: c.generation, kquant, cls, classFactor, mlx, activeB: active, cluster: clusterFactor },
    upgrade,
    notes,
  };
}
