/**
 * The compromise search (METHODOLOGY §5): when the asked-for configuration does not fit, find the
 * cheapest change (or combination) that does — close apps, the memory-limit override, a shorter
 * context, a smaller quant above the quality floor, or a hand-maintained special build.
 */
import { CONTEXT_CHIPS, FIX_COST, MEMORY_POLICY, QUANTIZER_PREFERENCE, type MemoryPolicy } from './constants';
import { formatContext, formatGb } from './format';
import { clusterOf } from './cluster';
import { availableMemory, isSpecialBuild, ssdGbOf, neededMemory, residentGb, verdictFor } from './memory';
import type { Availability, Change, Fix, Machine, ModelDetail, Need, Quant, Settings, SpecialBuild } from './types';

/** Nominal bits per weight from a quant label: IQ1_S → 1, UD-Q2_K_XL → 2, Q4_K_M → 4, MLX-8bit → 8, FP8 → 8. */
export function nominalBits(label: string): number | null {
  const l = label.toUpperCase().replace(/^UD-/, '');
  let m = /^MLX-(\d+(?:\.\d+)?)BIT/.exec(l);
  if (m) return parseFloat(m[1]);
  m = /^I?Q(\d)/.exec(l);
  if (m) return parseInt(m[1], 10);
  if (/^(MLX-)?(MXFP4|NVFP4|FP4)/.test(l)) return 4;
  if (/^(MLX-)?(MXFP8|FP8)/.test(l)) return 8;
  if (/^(MLX-)?(BF16|F16|FP16)/.test(l)) return 16;
  if (/^(MLX-)?(F32|FP32)/.test(l)) return 32;
  return null;
}

function quantizerRank(q: Quant): number {
  const i = (QUANTIZER_PREFERENCE as readonly string[]).indexOf(q.quantizer);
  return i === -1 ? QUANTIZER_PREFERENCE.length : i;
}

/** One quant per label (the preferred quantizer, then the smaller file). */
export function dedupeByLabel(quants: Quant[]): Quant[] {
  const best = new Map<string, Quant>();
  for (const q of quants) {
    const cur = best.get(q.label);
    if (!cur || quantizerRank(q) < quantizerRank(cur) || (quantizerRank(q) === quantizerRank(cur) && q.size_bytes < cur.size_bytes)) best.set(q.label, q);
  }
  return [...best.values()];
}

/** Smaller quants of the same format that the search may propose. */
export function fixCandidates(model: ModelDetail, requested: Quant | SpecialBuild, floorBits: number): Quant[] {
  const format = isSpecialBuild(requested) ? requested.format ?? 'gguf' : requested.format;
  const requestedSize = residentGb(requested);
  const pool = model.quants.filter((q) => {
    if (q.format !== format) return false;
    if (q.size_gb >= requestedSize) return false;
    const nb = nominalBits(q.label);
    return nb !== null && nb >= floorBits;
  });
  return dedupeByLabel(pool).sort((a, b) => b.size_bytes - a.size_bytes);
}

interface Candidate {
  changes: Change[];
  cost: number;
  verdict: 'runs' | 'tight';
  need: Need;
  availability: Availability;
  contextTokens: number;
  bits: number;
  effectiveBits: number;
  quant: Quant | SpecialBuild;
}

function changeKey(c: Change): string {
  return c.kind === 'quant' ? `quant:${c.quant.label}` : c.kind === 'ssdPaged' ? `ssd:${c.build.label}` : c.kind === 'context' ? `context:${c.tokens}` : c.kind;
}

function changeSignature(changes: Change[]): string {
  return changes.map((c) => (c.kind === 'quant' ? `quant:${c.quant.label}` : c.kind === 'ssdPaged' ? `ssd:${c.build.label}` : c.kind)).join('|');
}

export function describeFix(changes: Change[], requested: Quant | SpecialBuild, requestedContext: number, verdict: 'runs' | 'tight'): string {
  const parts: string[] = [];
  for (const c of changes) {
    if (c.kind === 'closeApps') parts.push('runs if you close your apps');
    else if (c.kind === 'override') parts.push('needs the macOS memory-limit override');
    else if (c.kind === 'context') parts.push(`at ${formatContext(c.tokens)} context instead of ${formatContext(requestedContext)}`);
    else if (c.kind === 'quant') {
      const nb = nominalBits(c.quant.label);
      parts.push(`at ${c.quant.label}${nb ? ` (${nb}-bit)` : ''} instead of ${requested.label}, quality loss`);
    } else if (c.kind === 'ssdPaged') parts.push(`with the engram-on-SSD build (${formatGb(c.build.resident_gb)} in memory, ${formatGb(ssdGbOf(c.build))} streamed from the SSD)`);
  }
  const text = parts.join('; ');
  return verdict === 'tight' ? `${text}; tight: under 10 % headroom` : text;
}

export interface FixSearchResult {
  fix: Fix | null;
  alternatives: Fix[];
  belowFloor: Fix | null;
  nearestMiss: string | null;
}

function toFix(c: Candidate, requested: Quant | SpecialBuild, requestedContext: number): Fix {
  return {
    changes: c.changes,
    cost: c.cost,
    verdict: c.verdict,
    need: c.need,
    availability: c.availability,
    contextTokens: c.contextTokens,
    reason: describeFix(c.changes, requested, requestedContext, c.verdict),
  };
}

function search(model: ModelDetail, machine: Machine, settings: Settings, requested: Quant | SpecialBuild, requestedContext: number, floorBits: number, policy: MemoryPolicy): Candidate[] {
  const appsOptions: boolean[] = settings.workApps > 0 ? [false, true] : [false];
  const overrideOptions: boolean[] = machine.platform === 'apple' && settings.override === false ? [false, true] : [false];
  const contexts = [requestedContext, ...CONTEXT_CHIPS.filter((c) => c < requestedContext)].sort((a, b) => b - a);
  const contextStep = (ctx: number) => contexts.indexOf(ctx);
  const quantOptions: (Quant | SpecialBuild)[] = [requested, ...fixCandidates(model, requested, floorBits), ...(model.special_builds ?? [])];
  const baseAvailability = availableMemory(machine, settings, policy);
  const out: Candidate[] = [];

  for (const closeApps of appsOptions) {
    for (const override of overrideOptions) {
      const av = availableMemory(machine, { memoryGb: settings.memoryGb, workApps: closeApps ? 0 : settings.workApps, override: override ? 1 : settings.override, machines: settings.machines, split: settings.split }, policy);
      if (override && !closeApps && av.availableGb === baseAvailability.availableGb) continue; // the override changes nothing here
      if (override && closeApps) {
        const withoutOverride = availableMemory(machine, { memoryGb: settings.memoryGb, workApps: 0, override: settings.override, machines: settings.machines, split: settings.split }, policy);
        if (av.availableGb === withoutOverride.availableGb) continue;
      }
      for (const ctx of contexts) {
        for (const q of quantOptions) {
          const changes: Change[] = [];
          if (closeApps) changes.push({ kind: 'closeApps' });
          if (override) changes.push({ kind: 'override', cap: 1 });
          if (ctx !== requestedContext) changes.push({ kind: 'context', tokens: ctx });
          if (q !== requested) changes.push(isSpecialBuild(q) ? { kind: 'ssdPaged', build: q } : { kind: 'quant', quant: q });
          if (changes.length === 0) continue; // the ask itself
          const need = neededMemory(q, model.architecture, ctx, settings.kvBits, av.availableGb, policy, clusterOf(settings));
          const v = verdictFor(need);
          if (v === 'no') continue;
          let cost = 0;
          for (const c of changes) {
            if (c.kind === 'closeApps') cost += FIX_COST.closeApps;
            else if (c.kind === 'override') cost += FIX_COST.override;
            else if (c.kind === 'context') cost += FIX_COST.contextStep * contextStep(c.tokens);
            else if (c.kind === 'quant') cost += FIX_COST.quant + ((nominalBits(c.quant.label) ?? 0) < 3 ? FIX_COST.quantUnder3Bits : 0);
            else if (c.kind === 'ssdPaged') cost += FIX_COST.ssdPaged;
          }
          if (v === 'tight') cost += FIX_COST.tight;
          const bits = isSpecialBuild(q) ? (q.bits ?? 0) : (nominalBits(q.label) ?? q.bits ?? 0);
          const effectiveBits = isSpecialBuild(q) ? (q.bits ?? 0) : (q.bits ?? 0);
          out.push({ changes, cost, verdict: v, need, availability: av, contextTokens: ctx, bits, effectiveBits, quant: q });
        }
      }
    }
  }
  out.sort(
    (a, b) =>
      a.cost - b.cost ||
      (a.verdict === b.verdict ? 0 : a.verdict === 'runs' ? -1 : 1) ||
      b.bits - a.bits ||
      b.effectiveBits - a.effectiveBits ||
      b.contextTokens - a.contextTokens,
  );
  return out;
}

function nearestMissText(model: ModelDetail, machine: Machine, settings: Settings, requested: Quant | SpecialBuild, requestedContext: number, floorBits: number, policy: MemoryPolicy): string | null {
  const pool: (Quant | SpecialBuild)[] = [...fixCandidates(model, requested, floorBits), ...(model.special_builds ?? [])];
  if (!isSpecialBuild(requested) && (nominalBits(requested.label) ?? 0) >= floorBits) pool.push(requested);
  if (pool.length === 0) return null;
  const smallest = pool.reduce((a, b) => (residentGb(b) < residentGb(a) ? b : a));
  const best = availableMemory(machine, { memoryGb: settings.memoryGb, workApps: 0, override: machine.platform === 'apple' ? 1 : settings.override, machines: settings.machines, split: settings.split }, policy);
  const need = neededMemory(smallest, model.architecture, requestedContext, settings.kvBits, best.availableGb, policy, clusterOf(settings));
  const label = isSpecialBuild(smallest) ? `"${smallest.label}" (${formatGb(smallest.resident_gb)} in memory)` : `${smallest.label}${nominalBits(smallest.label) ? ` (${nominalBits(smallest.label)}-bit)` : ''} at ${formatGb(residentGb(smallest))}`;
  return `the smallest allowed build, ${label}, needs about ${formatGb(need.totalGb)} at ${formatContext(requestedContext)}; this machine offers ${formatGb(best.availableGb)} at most`;
}

/** Everything the cell needs when the ask does not fit. */
export function findFixes(model: ModelDetail, machine: Machine, settings: Settings, requested: Quant | SpecialBuild, requestedContext: number, policy: MemoryPolicy = MEMORY_POLICY): FixSearchResult {
  const floor = settings.qualityFloorBits;
  const candidates = search(model, machine, settings, requested, requestedContext, floor, policy);
  const fix = candidates[0] ? toFix(candidates[0], requested, requestedContext) : null;
  const alternatives: Fix[] = [];
  if (fix) {
    // other ways in: not the winning fix plus something extra, and one entry per distinct change set
    const kept: string[][] = [candidates[0].changes.map(changeKey)];
    const seen = new Set([changeSignature(candidates[0].changes)]);
    for (const c of candidates.slice(1)) {
      const sig = changeSignature(c.changes);
      if (seen.has(sig)) continue;
      const keys = c.changes.map(changeKey);
      if (kept.some((k) => k.every((x) => keys.includes(x)))) continue; // never "an earlier option plus something extra"
      seen.add(sig);
      kept.push(keys);
      alternatives.push(toFix(c, requested, requestedContext));
      if (alternatives.length === 3) break;
    }
  }
  let belowFloor: Fix | null = null;
  if (!fix && floor > 0) {
    const relaxed = search(model, machine, settings, requested, requestedContext, 0, policy).filter((c) =>
      c.changes.some((ch) => ch.kind === 'quant' && (nominalBits(ch.quant.label) ?? 0) < floor),
    );
    belowFloor = relaxed[0] ? toFix(relaxed[0], requested, requestedContext) : null;
  }
  const nearestMiss = fix ? null : nearestMissText(model, machine, settings, requested, requestedContext, floor, policy);
  return { fix, alternatives, belowFloor, nearestMiss };
}
