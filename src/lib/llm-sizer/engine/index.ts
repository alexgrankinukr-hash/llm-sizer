/**
 * LLM Sizer engine — public API. Pure functions over the exported model records, the machine catalog
 * and the calibration factors; every rule is explained in METHODOLOGY.md.
 */
import { CONTEXT_CHIPS, MEMORY_POLICY, Q8_REPACK_BITS, REF_4BIT, REF_8BIT, REF_MLX_4BIT, QUANTIZER_PREFERENCE, type MemoryPolicy } from './constants';
import { findFixes, nominalBits } from './fix';
import { availableMemory, breakdown, isSpecialBuild, neededMemory, verdictFor } from './memory';
import { estimatePrefill } from './prefill';
import { archClassOf, estimateSpeed, speedSource } from './speed';
import { clusterOf, dispatchCaveat } from './cluster';
import type { CellFlag, CellResult, Factors, Machine, ModelDetail, Quant, Runtime, Settings, SpecialBuild } from './types';

export * from './types';
export * from './constants';
export { availableMemory, breakdown, buffersGb, defaultCap, isSpecialBuild, kvBytesPerToken, kvCacheBytes, neededMemory, ramDecimalGb, residentGb, scratchGb, verdictFor } from './memory';
export { accelerations, activeParams, archClassOf, chipKey, estimateSpeed, feelsLike, floorMs, isMoe, mlxContextGroupOf, mlxGroupOf, quantFamilyOf, resolveChipProfile, resolveEfficiency, runtimeFactor, speedModelKind, speedSource, tierOf } from './speed';
export { describeFix, fixCandidates, findFixes, nominalBits } from './fix';
export { clusterMs, clusterOf, dispatchCaveat, linkClassOf, prefillClusterFactor, tensorAllowed, tensorCapable, termStatus } from './cluster';
export type { Cluster } from './cluster';
export * from './format';
export * from './custom';

/** The context chips this model can use. */
export function contextChips(model: Pick<ModelDetail, 'context_max'>): number[] {
  const max = model.context_max ?? Number.POSITIVE_INFINITY;
  const chips = CONTEXT_CHIPS.filter((c) => c <= max);
  return chips.length ? chips : [max];
}

function pickByLabels(quants: Quant[], labels: readonly string[]): Quant | null {
  for (const label of labels) {
    const rows = quants.filter((q) => q.label === label);
    if (rows.length) {
      rows.sort((a, b) => {
        const ra = (QUANTIZER_PREFERENCE as readonly string[]).indexOf(a.quantizer);
        const rb = (QUANTIZER_PREFERENCE as readonly string[]).indexOf(b.quantizer);
        return (ra === -1 ? 99 : ra) - (rb === -1 ? 99 : rb);
      });
      return rows[0];
    }
  }
  return null;
}

/**
 * The quant a fresh column starts with: the export's reference pick for the tier (MLX 4-bit when the
 * runtime is MLX and such a file exists), or the same rule applied locally for custom models.
 */
export function defaultQuant(model: ModelDetail, runtime: Runtime, tier: 'q4' | 'q8' | 'native' = 'q4'): Quant | null {
  const ref = model.reference;
  if (ref) {
    const want = tier === 'q4' && runtime === 'mlx' ? ref.mlx4 ?? ref.q4 : ref[tier];
    if (want) {
      const q = model.quants.find((x) => x.repo === want.repo && x.label === want.label);
      if (q) return q;
    }
  }
  if (tier === 'native') return model.quants.find((q) => q.repo === model.hf_repo && q.format === 'safetensors') ?? null;
  const labels = tier === 'q8' ? REF_8BIT : runtime === 'mlx' ? [...REF_MLX_4BIT, ...REF_4BIT] : REF_4BIT;
  return pickByLabels(model.quants, labels) ?? model.quants[0] ?? null;
}

function clampContext(model: ModelDetail, tokens: number): { tokens: number; clamped: boolean } {
  const max = model.context_max;
  if (max && tokens > max) return { tokens: max, clamped: true };
  return { tokens: Math.max(1, Math.floor(tokens)), clamped: false };
}

function emptySpeed(): CellResult['speed'] {
  return { tokS: null, gbPerToken: 0, efficiency: null, efficiencySource: 'none', runtimeFactor: 1, runtimeApplied: 'gguf', feelsLike: null, accelerations: [], notes: [], modelKind: 'multiplier' };
}

/** Evaluate one cell: a model (at a quant and context) on a machine at a memory size, with the toggles. `policy` is the memory margins (METHODOLOGY §4, §5); the default is the one in force. */
export function evaluateCell(model: ModelDetail, machine: Machine, settings: Settings, factors: Factors, policy: MemoryPolicy = MEMORY_POLICY): CellResult {
  const flags: CellFlag[] = [];
  const { tokens: ctx, clamped } = clampContext(model, settings.contextTokens);
  if (clamped) flags.push('clamped-context');
  const arch = model.architecture;
  if (!arch || arch.conventional_assumed || arch.attention === 'assumed') flags.push('assumed');
  if (model.gated && model.gated !== 'false') flags.push('gated');
  if (model.four_bit_native) flags.push('four-bit-native');
  if (settings.kvBits === 4 && arch && (arch.attention === 'mla' || arch.attention === 'latent')) flags.push('kv-precision-note');
  if (!model.quants.some((q) => q.format === 'gguf' || q.format === 'mlx') && !(model.special_builds?.length)) flags.push('no-gguf-mlx-build');
  const src = speedSource(machine, factors);
  if (src === 'assumed' || src === 'tier') flags.push('unmeasured-chip');
  const cluster = clusterOf(settings);
  if (cluster) flags.push(cluster.split === 'tensor' ? 'linked-tensor' : 'linked-layer');
  if (cluster && factors.cluster && dispatchCaveat(archClassOf(model), arch?.layers, cluster.split, factors.cluster)) flags.push('linked-dispatch');

  const availability = availableMemory(machine, settings, policy);
  const quant: Quant | SpecialBuild | null = settings.quant ?? defaultQuant(model, settings.runtime) ?? model.special_builds?.[0] ?? null;
  if (!quant) {
    const need = { weightsGb: 0, cacheGb: 0, buffersGb: 0, scratchGb: 0, ssdGb: 0, totalGb: 0, ratio: 0 };
    return { verdict: 'no-fit', availability, need, quant: null, fix: null, alternatives: [], belowFloor: null, nearestMiss: 'no downloadable build of this model is known', speed: emptySpeed(), prefill: null, breakdown: breakdown(settings.memoryGb, availability, need), flags, contextTokens: ctx };
  }
  if (!isSpecialBuild(quant)) {
    if (settings.runtime === 'mlx' && quant.format !== 'mlx') flags.push('no-mlx-file');
    if ((nominalBits(quant.label) ?? 0) >= 8 && quant.bits !== null && quant.bits < Q8_REPACK_BITS) flags.push('q8-is-repack');
  } else flags.push('ssd-build');

  const need = neededMemory(quant, arch, ctx, settings.kvBits, availability.availableGb, policy, cluster);
  const v = verdictFor(need);
  if (v !== 'no') {
    const speed = estimateSpeed({ model, quant, machine, contextTokens: ctx, kvBits: settings.kvBits, factors, cluster });
    const prefill = estimatePrefill({ model, quant, machine, runtime: settings.runtime, contextTokens: ctx, factors, cluster });
    return { verdict: v, availability, need, quant, fix: null, alternatives: [], belowFloor: null, nearestMiss: null, speed, prefill, breakdown: breakdown(settings.memoryGb, availability, need), flags, contextTokens: ctx };
  }

  const found = findFixes(model, machine, settings, quant, ctx, policy);
  if (found.fix) {
    const fixQuant = found.fix.changes.reduce<Quant | SpecialBuild>((q, c) => (c.kind === 'quant' ? c.quant : c.kind === 'ssdPaged' ? c.build : q), quant);
    const speed = estimateSpeed({ model, quant: fixQuant, machine, contextTokens: found.fix.contextTokens, kvBits: settings.kvBits, factors, cluster });
    if (speed.tokS !== null) speed.notes.unshift('speed shown for the configuration that fits'); // a build with no number keeps its own note first
    const prefill = estimatePrefill({ model, quant: fixQuant, machine, runtime: settings.runtime, contextTokens: found.fix.contextTokens, factors, cluster });
    return {
      verdict: 'compromise',
      availability: found.fix.availability,
      need: found.fix.need,
      quant,
      fix: found.fix,
      alternatives: found.alternatives,
      belowFloor: null,
      nearestMiss: null,
      speed,
      prefill,
      breakdown: breakdown(settings.memoryGb, found.fix.availability, found.fix.need),
      flags,
      contextTokens: found.fix.contextTokens,
    };
  }
  const speed = estimateSpeed({ model, quant, machine, contextTokens: ctx, kvBits: settings.kvBits, factors, cluster });
  if (speed.tokS !== null) speed.notes.unshift('does not fit; speed shown for information only');
  return { verdict: 'no-fit', availability, need, quant, fix: null, alternatives: [], belowFloor: found.belowFloor, nearestMiss: found.nearestMiss, speed, prefill: null, breakdown: breakdown(settings.memoryGb, availability, need), flags, contextTokens: ctx };
}

export interface MachineRow {
  machine: Machine;
  settings: Settings;
}

export interface MachineRowResult extends CellResult {
  machine: Machine;
  memoryGb: number;
  priceUsd: number | null;
}

/** One model across many machine configurations, cheapest first (unpriced rows last). */
export function evaluateModelAcrossMachines(model: ModelDetail, rows: MachineRow[], factors: Factors): MachineRowResult[] {
  const results = rows.map((row) => {
    const cell = evaluateCell(model, row.machine, row.settings, factors);
    const price = row.machine.price_usd?.[String(row.settings.memoryGb)] ?? null;
    return { ...cell, machine: row.machine, memoryGb: row.settings.memoryGb, priceUsd: price };
  });
  results.sort((a, b) => {
    if (a.priceUsd === null && b.priceUsd === null) return a.memoryGb - b.memoryGb;
    if (a.priceUsd === null) return 1;
    if (b.priceUsd === null) return -1;
    return a.priceUsd - b.priceUsd || a.memoryGb - b.memoryGb;
  });
  return results;
}

/** Every (machine, memory option) row of a catalog with shared settings — the buying view's input. */
export function catalogRows(machines: Machine[], base: Omit<Settings, 'memoryGb'>, onlyCurrent = true): MachineRow[] {
  const rows: MachineRow[] = [];
  for (const machine of machines) {
    if (onlyCurrent && machine.status === 'discontinued') continue;
    for (const gb of machine.memory_options_gb) rows.push({ machine, settings: { ...base, memoryGb: gb } });
  }
  return rows;
}
