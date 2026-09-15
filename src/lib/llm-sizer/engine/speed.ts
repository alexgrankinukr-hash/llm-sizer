/**
 * Speed estimates (METHODOLOGY §6): decode tokens per second for one configuration.
 *
 * Two models live here until the multiplier one is retired:
 *  - fixed-cost ("floor") model, `factors.speed_model.kind === 'floor'`: milliseconds per token = a read of the bytes
 *    per token at the chip's effective bandwidth + a fixed cost per token (chip + architecture class, scaled for MLX)
 *    + a context term; tok/s = 1000 ÷ ms. scripts/llm-sizer/speed_model.py mirrors it function for function.
 *  - multiplier model, anything else: bandwidth × efficiency × runtime factor ÷ bytes per token.
 */
import { BYTES_PER_GB, MLX_DENSE_SPLIT_B, MLX_FADE_TOKENS, SMALL_ACTIVE_B } from './constants';
import { isSpecialBuild, kvBytesPerToken } from './memory';
import { clusterMs, linkClassOf, termStatus, type Cluster } from './cluster';
import type { MlxContextGroup, Acceleration, ArchClass, ClusterSpeed, Factors, KvBits, Machine, MlxGroup, ModelDetail, Platform, Quant, QuantFamily, Runtime, SpecialBuild, Speed, SpeedModel } from './types';

/** `M5 Max (40-core GPU)` → `M5 Max`: the chip name without its bin. */
export function chipKey(chip: string): string {
  return chip.replace(/\s*\(.*\)\s*$/, '').trim();
}

/** The efficiency tier a machine falls into when it has no measured row of its own. */
export function tierOf(machine: Pick<Machine, 'platform' | 'chip' | 'family'>): string | null {
  if (machine.platform === 'cuda') return 'spark'; // the only CUDA box in v1 is the DGX Spark
  if (machine.platform !== 'apple') return null;
  const c = chipKey(machine.chip).toLowerCase();
  if (c.includes('ultra')) return 'ultra';
  if (c.includes('max')) return 'max';
  if (c.includes('pro')) return 'pro';
  if (/^m\d+$/.test(c)) return 'base';
  return null;
}

/** Names to try in `factors.efficiency`, most specific first. */
function efficiencyKeys(machine: Pick<Machine, 'platform' | 'chip' | 'family'>): string[] {
  const keys = [chipKey(machine.chip)];
  const text = `${machine.family} ${machine.chip}`.toLowerCase();
  if (machine.platform === 'cuda' && (text.includes('spark') || text.includes('gb10'))) keys.push('DGX Spark');
  return keys;
}

export interface EfficiencyResolution {
  value: number | null;
  source: Speed['efficiencySource'];
  key: string | null;
}

/** Multiplier model: by_chip (measured) → assumed → tier median → nothing. */
export function resolveEfficiency(machine: Pick<Machine, 'platform' | 'chip' | 'family'>, factors: Factors): EfficiencyResolution {
  for (const key of efficiencyKeys(machine)) {
    const measured = factors.efficiency.by_chip[key];
    if (measured && measured.value !== null) return { value: measured.value, source: measured.measured ? 'measured' : 'assumed', key };
    const assumed = factors.efficiency.assumed[key];
    if (assumed && assumed.value !== null) return { value: assumed.value, source: 'assumed', key };
  }
  const tier = tierOf(machine);
  const t = tier ? factors.efficiency.by_tier[tier] : undefined;
  if (t && t.value !== null) return { value: t.value, source: 'tier', key: tier };
  return { value: null, source: 'none', key: null };
}

/** A chip's fixed-cost profile: the share of its spec bandwidth a read reaches, and its fixed milliseconds per token. */
export interface ChipProfile {
  t0Ms: number;
  bEffRatio: number;
  /** `bEffRatio × machine.bandwidth_gbs`: the ratio transfers across GPU bins of one chip, the GB/s does not */
  bEffGbs: number;
  source: Speed['efficiencySource'];
  key: string | null;
  fitN?: number;
}

/** Fixed-cost model: by_chip (fitted) → assumed → tier median → nothing. Same order as `resolveEfficiency`. */
export function resolveChipProfile(machine: Pick<Machine, 'platform' | 'chip' | 'family' | 'bandwidth_gbs'>, factors: Factors): ChipProfile | null {
  const bw = machine.bandwidth_gbs;
  const make = (ratio: number, t0: number, source: Speed['efficiencySource'], key: string | null, fitN?: number): ChipProfile => ({
    t0Ms: t0,
    bEffRatio: ratio,
    bEffGbs: ratio * bw,
    source,
    key,
    fitN,
  });
  for (const key of efficiencyKeys(machine)) {
    const e = factors.efficiency.by_chip[key];
    if (e && e.b_eff_ratio !== undefined && e.t0_ms !== undefined) return make(e.b_eff_ratio, e.t0_ms, e.measured ? 'measured' : 'assumed', key, e.fit_n);
    const a = factors.efficiency.assumed[key];
    if (a && a.b_eff_ratio !== undefined && a.t0_ms !== undefined) return make(a.b_eff_ratio, a.t0_ms, 'assumed', key);
  }
  const tier = tierOf(machine);
  const t = tier ? factors.efficiency.by_tier[tier] : undefined;
  if (t && t.b_eff_ratio !== undefined && t.t0_ms !== undefined) return make(t.b_eff_ratio, t.t0_ms, 'tier', tier);
  return null;
}

/** Which speed model the factor file asks for. */
export function speedModelKind(factors: Factors): 'multiplier' | 'floor' {
  return factors.speed_model?.kind === 'floor' ? 'floor' : 'multiplier';
}

/** Where this machine's speed number comes from under the active model (drives the `unmeasured-chip` flag). */
export function speedSource(machine: Pick<Machine, 'platform' | 'chip' | 'family' | 'bandwidth_gbs'>, factors: Factors): Speed['efficiencySource'] {
  if (speedModelKind(factors) === 'floor') return resolveChipProfile(machine, factors)?.source ?? 'none';
  return resolveEfficiency(machine, factors).source;
}

export function isMoe(model: ModelDetail): boolean {
  const a = model.architecture;
  return !!(a && a.experts_total && a.experts_active);
}

export function activeParams(model: ModelDetail): number | null {
  return model.params_active ?? model.params_total ?? null;
}

/** The MLX runtime factor for this model at this context (1.0 for GGUF, and for MLX above the fade). Multiplier model only. */
export function runtimeFactor(quant: Quant | SpecialBuild, model: ModelDetail, machine: Pick<Machine, 'platform'>, contextTokens: number, factors: Factors): { factor: number; applied: Runtime } {
  if (isSpecialBuild(quant) || quant.format !== 'mlx' || machine.platform !== 'apple') return { factor: factors.runtime.gguf.factor, applied: 'gguf' };
  const totalB = (model.params_total ?? 0) / BYTES_PER_GB;
  const base = isMoe(model)
    ? factors.runtime.mlx.moe.factor
    : totalB >= MLX_DENSE_SPLIT_B
      ? factors.runtime.mlx.dense_14b_and_up.factor
      : factors.runtime.mlx.dense_under_14b.factor;
  const [from, to] = MLX_FADE_TOKENS;
  if (contextTokens <= from) return { factor: base, applied: 'mlx' };
  if (contextTokens >= to) return { factor: 1, applied: 'mlx' };
  const t = (contextTokens - from) / (to - from);
  return { factor: base + (1 - base) * t, applied: 'mlx' };
}

// ---- fixed-cost model pieces (mirrored in scripts/llm-sizer/speed_model.py) ----

/** Quant labels that read at full speed; everything else in GGUF (K-quants, IQ, UD) carries the K-quant read factor. */
const PLAIN_FAMILIES: Record<string, QuantFamily> = {
  Q4_0: 'q4_0', Q4_1: 'q4_0', Q5_0: 'q4_0', Q5_1: 'q4_0',
  Q8_0: 'q8_0', FP8: 'q8_0',
  F16: 'f16', BF16: 'f16', FP16: 'f16', F32: 'f16',
  MXFP4: 'mxfp4', MXFP4_MOE: 'mxfp4', NVFP4: 'mxfp4', FP4: 'mxfp4',
};
export const ARCH_CLASSES: readonly ArchClass[] = ['dense', 'moe', 'moe_hybrid', 'moe_latent', 'deepseek_v4'];
const CONTEXT_UNIT = 32768;

/** mlx | q4_0 | q8_0 | f16 | mxfp4 | gguf_kquant (speed_model.py:quant_family). */
export function quantFamilyOf(quant: Quant | SpecialBuild): QuantFamily {
  const format = 'format' in quant ? quant.format : 'gguf';
  const label = quant.label.toUpperCase();
  if (format === 'mlx' || label.startsWith('MLX')) return 'mlx';
  return PLAIN_FAMILIES[label] ?? 'gguf_kquant';
}

/** dense | moe | moe_hybrid | moe_latent | deepseek_v4 (speed_model.py:arch_class). A hand-set class on the record wins. */
export function archClassOf(model: Pick<ModelDetail, 'architecture'>): ArchClass {
  const a = model.architecture;
  const override = a?.arch_cost_class;
  if (override && (ARCH_CLASSES as readonly string[]).includes(override)) return override as ArchClass;
  if (!a) return 'dense';
  if ((a.experts_total ?? 0) <= 1) return 'dense';
  if (a.attention === 'mla' || a.attention === 'latent') return 'moe_latent';
  if ((a.linear_layers ?? 0) > 0) return 'moe_hybrid';
  return 'moe';
}

export function generationNumberOf(chip: string): number | null {
  return generationNumber(chip);
}

function generationNumber(chip: string): number | null {
  const m = /^M(\d+)/i.exec(chip.trim());
  return m ? Number(m[1]) : null;
}

/** Which MLX long-context cost applies (speed_model.py:mlx_context_group): the M5 GPU generation pays a different one; a chip whose generation cannot be read (a custom machine) is charged the pre-M5 cost, the conservative one. */
export function mlxContextGroupOf(machine: Pick<Machine, 'chip'>): MlxContextGroup {
  const g = generationNumber(machine.chip);
  return g !== null && g >= 5 ? 'm5plus' : 'pre_m5';
}

/** Which MLX fixed-cost factor applies (speed_model.py:mlx_group). */
export function mlxGroupOf(cls: ArchClass, activeB: number, machine: Pick<Machine, 'chip'>): MlxGroup {
  if (cls === 'dense') return 'dense';
  if (activeB < SMALL_ACTIVE_B) {
    const g = generationNumber(machine.chip);
    return g !== null && g >= 4 ? 'small_active_moe_m4plus' : 'small_active_moe_pre_m4';
  }
  return 'moe_other';
}

export interface FloorParts {
  readMs: number;
  fixedMs: number;
  attnMs: number;
  ms: number;
  archCostMs: number;
  mlxFactor: number;
}

/** A read factor applies on the platforms it was fitted on (`platforms`); absent means everywhere (speed_model.py:read_factor). */
export function readFactor(sm: SpeedModel, family: QuantFamily, platform: Platform): number {
  const t = sm.read_factor[family];
  if (!t) return 1;
  return t.platforms === undefined || t.platforms.includes(platform) ? t.value : 1;
}

/** Milliseconds per generated token under the fixed-cost model (speed_model.py:decode_ms). */
export function floorMs(
  profile: Pick<ChipProfile, 't0Ms' | 'bEffRatio'>,
  bandwidthGbs: number,
  gbPerToken: number,
  family: QuantFamily,
  cls: ArchClass,
  runtime: Runtime,
  group: MlxGroup,
  attention: string | null | undefined,
  contextTokens: number,
  sm: SpeedModel,
  platform: Platform = 'apple',
  mlxCtx: MlxContextGroup = 'pre_m5',
): FloorParts {
  const bEff = profile.bEffRatio * bandwidthGbs;
  const readMs = (gbPerToken / bEff) * 1000 * readFactor(sm, family, platform);
  const isMlx = runtime === 'mlx';
  const archCostMs = sm.architecture_cost_ms[cls]?.value ?? 0;
  const mlxFactor = isMlx ? (sm.mlx.overhead_factor[group]?.value ?? 1) : 1;
  const fixedMs = (profile.t0Ms + archCostMs) * mlxFactor;
  const attnKey = attention && attention in sm.attention_ms_per_32k ? attention : 'gqa';
  const mlxCtxMs = isMlx ? (sm.mlx.attention_ms_per_32k[mlxCtx] ?? sm.mlx.attention_ms_per_32k.pre_m5).value : 0;
  const attnMs = ((sm.attention_ms_per_32k[attnKey]?.value ?? 0) + mlxCtxMs) * (contextTokens / CONTEXT_UNIT);
  return { readMs, fixedMs, attnMs, ms: readMs + fixedMs + attnMs, archCostMs, mlxFactor };
}

function accelerationEntry(factors: Factors, key: string): { min: number; median: number; max: number } | null {
  const v = factors.acceleration[key];
  return v && typeof v === 'object' ? v : null;
}

/** Which accelerations this model supports, each as a measured range (never a headline number). */
export function accelerations(model: ModelDetail, factors: Factors): Acceleration[] {
  const a = model.architecture;
  if (!a) return [];
  const out: Acceleration[] = [];
  if (a.accel_mtp) {
    const e = accelerationEntry(factors, 'mtp');
    if (e) out.push({ kind: 'mtp', ...e, via: 'native multi-token-prediction heads' });
  }
  const drafts = [a.accel_dflash_repo, a.accel_dspark_repo].filter((r): r is string => !!r);
  if (drafts.length) {
    const active = (activeParams(model) ?? 0) / BYTES_PER_GB;
    const key = isMoe(model) && active < SMALL_ACTIVE_B ? 'draft_small_active_moe' : 'draft_dense';
    const e = accelerationEntry(factors, key);
    if (e) out.push({ kind: 'draft', ...e, via: drafts.join(', ') });
  }
  return out;
}

export function feelsLike(tokS: number | null, factors: Factors): Speed['feelsLike'] {
  if (tokS === null || !Number.isFinite(tokS)) return null;
  for (const band of factors.feels_like) {
    if (band.max_tok_s === null || tokS < band.max_tok_s) return { label: band.label, description: band.description ?? band.label };
  }
  return null;
}

export interface SpeedInput {
  model: ModelDetail;
  quant: Quant | SpecialBuild;
  machine: Machine;
  contextTokens: number;
  kvBits: KvBits;
  factors: Factors;
  /** linked machines pooled as one; absent or null for one machine */
  cluster?: Cluster | null;
}

/**
 * The pool's milliseconds per token from one machine's, with the sheet's parts and the notes (METHODOLOGY §6.1).
 * Returns null when the data set has no cluster terms: the pool then has no speed.
 */
function applyCluster(singleMs: number, cluster: Cluster, machine: Machine, cls: ArchClass, factors: Factors, notes: string[]): { tokS: number; info: ClusterSpeed } | null {
  const cf = factors.cluster;
  if (!cf) {
    notes.push('no linked-machine terms in this data set');
    return null;
  }
  const link = linkClassOf(machine);
  const kind = cls === 'dense' ? 'dense' : 'moe';
  const c = clusterMs(singleMs, cluster.n, cluster.split, link, kind, cf);
  const status = termStatus(c.term);
  const basis = status === 'fitted' ? `fitted on ${c.term.n} published cluster measurements` : 'assumed, no cluster measurement for this link';
  if (cluster.split === 'layer') notes.push(`estimated for ${cluster.n} linked machines on a layer split: one machine's ${singleMs.toFixed(1)} ms per token for the whole model plus ${c.hopMs} ms per extra machine (${basis})`);
  else notes.push(`estimated for ${cluster.n} linked machines in tensor parallel: one machine's ${singleMs.toFixed(1)} ms per token ÷ ${cluster.n} plus ${c.cMs} ms × log2(${cluster.n}) (${basis})`);
  notes.push('a model that fits one machine writes slower across a layer split; only tensor parallel on a fast link adds speed');
  return { tokS: 1000 / c.ms, info: { machines: cluster.n, split: cluster.split, link, singleMs, hopMs: c.hopMs, cMs: c.cMs, source: status, n: c.term.n ?? 0 } };
}

/** Decode speed for one configuration. `tokS` is null when the machine has no profile or efficiency factor, or no bandwidth. */
export function estimateSpeed({ model, quant, machine, contextTokens, kvBits, factors, cluster = null }: SpeedInput): Speed {
  const notes: string[] = [];
  const kind = speedModelKind(factors);
  const active = activeParams(model);
  const bits = isSpecialBuild(quant) ? (quant.bits ?? null) : quant.bits;
  const accel = accelerations(model, factors);

  if (isSpecialBuild(quant)) {
    // a build that streams its n-gram table from the SSD has no per-token read the model can reason from, and one published
    // measurement on one chip cannot be carried to the others: it is quoted as a note, never used as a number
    notes.push(
      quant.measured_tok_s
        ? `the publisher measured ${quant.measured_tok_s} tok/s on ${quant.measured_chip ? `an ${quant.measured_chip} (${quant.measured_on ?? 'memory size not stated'})` : `a ${quant.measured_on ?? 'machine they did not describe'}`}; no speed is estimated for a build that streams from the SSD`
        : 'no speed figure for this build',
    );
    return { tokS: null, gbPerToken: 0, efficiency: null, efficiencySource: 'none', runtimeFactor: 1, runtimeApplied: 'gguf', feelsLike: null, accelerations: accel, notes, modelKind: kind };
  }

  if (kind === 'floor') return estimateFloor({ model, quant, machine, contextTokens, kvBits, factors, cluster }, active, bits, accel, notes);

  const eff = resolveEfficiency(machine, factors);
  const runtime = runtimeFactor(quant, model, machine, contextTokens, factors);
  if (active === null || bits === null) {
    notes.push('no parameter count or bits per weight for this build');
    return { tokS: null, gbPerToken: 0, efficiency: eff.value, efficiencySource: eff.source, runtimeFactor: runtime.factor, runtimeApplied: runtime.applied, feelsLike: null, accelerations: accel, notes, modelKind: 'multiplier' };
  }
  const weightsBytes = (active * bits) / 8;
  const cacheBytes = kvBytesPerToken(model.architecture, kvBits) * contextTokens;
  const gbPerToken = (weightsBytes + cacheBytes) / BYTES_PER_GB;

  if (eff.value === null) notes.push('no measured efficiency for this platform yet');
  if (eff.source === 'assumed' || eff.source === 'tier') notes.push('estimate: this chip has no measured rows yet');
  if (runtime.applied === 'gguf' && quant.format === 'mlx') notes.push('MLX runs only on Apple silicon; GGUF speed shown');
  if (runtime.applied === 'mlx' && runtime.factor < factors.runtime.mlx.moe.factor && contextTokens > MLX_FADE_TOKENS[0]) notes.push('MLX advantage fades at long context');

  const bandwidth = machine.bandwidth_gbs;
  let tokS = eff.value !== null && bandwidth > 0 && gbPerToken > 0 ? (bandwidth * eff.value * runtime.factor) / gbPerToken : null;
  let clusterInfo: ClusterSpeed | undefined;
  if (cluster && tokS !== null) {
    const pooled = applyCluster(1000 / tokS, cluster, machine, archClassOf(model), factors, notes);
    tokS = pooled?.tokS ?? null;
    clusterInfo = pooled?.info;
  }
  return {
    tokS,
    gbPerToken,
    efficiency: eff.value,
    efficiencySource: eff.source,
    runtimeFactor: runtime.factor,
    runtimeApplied: runtime.applied,
    feelsLike: feelsLike(tokS, factors),
    accelerations: accel,
    notes,
    modelKind: 'multiplier',
    ...(clusterInfo ? { cluster: clusterInfo } : {}),
  };
}

/** The fixed-cost branch of `estimateSpeed`. */
function estimateFloor(
  { model, quant, machine, contextTokens, kvBits, factors, cluster = null }: Omit<SpeedInput, 'memoryGb'> & { quant: Quant },
  active: number | null,
  bits: number | null,
  accel: Acceleration[],
  notes: string[],
): Speed {
  const sm = factors.speed_model as SpeedModel;
  const profile = resolveChipProfile(machine, factors);
  const runtime: Runtime = quant.format === 'mlx' && machine.platform === 'apple' ? 'mlx' : 'gguf';
  const cls = archClassOf(model);
  const activeB = (active ?? 0) / BYTES_PER_GB;
  const group = mlxGroupOf(cls, activeB, machine);
  const family = quantFamilyOf(quant);
  const base: Speed = { tokS: null, gbPerToken: 0, efficiency: null, efficiencySource: profile?.source ?? 'none', runtimeFactor: 1, runtimeApplied: runtime, feelsLike: null, accelerations: accel, notes, modelKind: 'floor', archClass: cls };

  if (active === null || bits === null) {
    notes.push('no parameter count or bits per weight for this build');
    return base;
  }
  const weightsBytes = (active * bits) / 8;
  const cacheBytes = kvBytesPerToken(model.architecture, kvBits) * contextTokens;
  const gbPerToken = (weightsBytes + cacheBytes) / BYTES_PER_GB;
  base.gbPerToken = gbPerToken;

  if (profile === null) {
    notes.push('no measured profile for this platform yet');
    return base;
  }
  if (profile.source === 'assumed' || profile.source === 'tier') notes.push('estimate: this chip has no measured rows yet');
  if (runtime === 'gguf' && quant.format === 'mlx') notes.push('MLX runs only on Apple silicon; GGUF speed shown');
  if (machine.bandwidth_gbs <= 0 || gbPerToken <= 0) return base;

  const parts = floorMs(profile, machine.bandwidth_gbs, gbPerToken, family, cls, runtime, group, model.architecture?.attention, contextTokens, sm, machine.platform, mlxContextGroupOf(machine));
  if (runtime === 'mlx' && parts.mlxFactor < 1) notes.push('MLX trims the fixed cost per token on this model');
  if (runtime === 'mlx' && parts.mlxFactor > 1) notes.push('MLX adds fixed cost per token on this model');
  let tokS = parts.ms > 0 ? 1000 / parts.ms : null;
  let clusterInfo: ClusterSpeed | undefined;
  if (cluster && tokS !== null) {
    const pooled = applyCluster(parts.ms, cluster, machine, cls, factors, notes);
    tokS = pooled?.tokS ?? null;
    clusterInfo = pooled?.info;
  }
  return {
    ...base,
    tokS,
    runtimeFactor: parts.mlxFactor,
    feelsLike: feelsLike(tokS, factors),
    t0Ms: profile.t0Ms,
    bEffGbs: profile.bEffGbs,
    readMs: parts.readMs,
    archCostMs: parts.archCostMs,
    attnMs: parts.attnMs,
    ...(clusterInfo ? { cluster: clusterInfo } : {}),
  };
}
