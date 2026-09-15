/**
 * Linked machines (METHODOLOGY §4, §5, §6.1, §6.4): two to four identical machines pooled into one cluster.
 *
 * Two split modes, decided by the user per machine group:
 *  - layer split (llama.cpp RPC on any link, MLX pipeline, exo pipeline): memory pools, and every token passes through
 *    every machine in turn, so the pool writes at one machine's time per token for the whole model plus a hop per
 *    extra machine; reading speed stays one machine's.
 *  - tensor parallel (MLX + Thunderbolt 5 RDMA on Macs, vLLM / TensorRT-LLM over the Spark's 200 Gb/s ports, PCIe on
 *    GPU boxes): every machine reads its shard at once, so the time per token is one machine's ÷ N plus a
 *    synchronisation cost that grows with log2(N); reading speed scales with an assumed exponent.
 * The constants come from `factors.cluster` (calibrate.py, fitted on the published cluster series). `speed_model.py`
 * mirrors `clusterMs` and `prefillClusterFactor` function for function.
 */
import { CLUSTER_MAX_MACHINES } from './constants';
import type { ArchClass, ClusterFactors, ClusterLink, ClusterSplit, ClusterTerm, Machine, Settings } from './types';

export interface Cluster {
  /** machines in the pool, 2 to 4 */
  n: number;
  split: ClusterSplit;
}

/** The pool a settings object asks for; null for one machine. Counts above the cap are clamped, not refused. */
export function clusterOf(settings: Pick<Settings, 'machines' | 'split'>): Cluster | null {
  const n = Math.floor(settings.machines ?? 1);
  if (!(n > 1)) return null;
  return { n: Math.min(CLUSTER_MAX_MACHINES, n), split: settings.split ?? 'layer' };
}

/** Which hop cost applies: Macs over Thunderbolt, Sparks over their ConnectX ports, anything else over PCIe. */
export function linkClassOf(machine: Pick<Machine, 'platform' | 'family' | 'chip'>): ClusterLink {
  if (machine.platform === 'apple') return 'mac';
  const text = `${machine.family} ${machine.chip}`.toLowerCase();
  if (/spark|gb10/.test(text)) return 'spark';
  return 'gpu';
}

/**
 * Whether tensor parallel is on offer for this machine: it needs a fast link. Sparks and GPU boxes have one; a Mac
 * needs Thunderbolt 5 (the catalog's `interconnect`), which the M3 Ultra, the M4 Pro and Max and the M5 family carry.
 */
export function tensorCapable(machine: Pick<Machine, 'platform' | 'interconnect' | 'chip'>): boolean {
  if (machine.platform !== 'apple') return true;
  if (machine.interconnect) return /thunderbolt\s*5/i.test(machine.interconnect);
  const chip = machine.chip.toUpperCase();
  const gen = /M(\d+)/.exec(chip);
  const g = gen ? Number(gen[1]) : 0;
  if (g >= 5) return true;
  if (g === 4) return /PRO|MAX|ULTRA/.test(chip);
  if (g === 3) return /ULTRA/.test(chip);
  return false;
}

/** Tensor splits need an even count: 3 fails for most models (hidden sizes and head counts must divide). */
export function tensorAllowed(n: number): boolean {
  return n === 2 || n === 4;
}

/** A cluster term is fitted when it rests on measured series, assumed otherwise. */
export function termStatus(term: ClusterTerm): 'fitted' | 'assumed' {
  if (term.source === 'fitted' || term.source === 'assumed') return term.source;
  return (term.n ?? 0) > 0 ? 'fitted' : 'assumed';
}

/** Milliseconds per token for the pool, from one machine's milliseconds for the whole model. Mirrors speed_model.py:cluster_ms. */
export function clusterMs(
  singleMs: number,
  n: number,
  split: ClusterSplit,
  link: ClusterLink,
  kind: 'dense' | 'moe',
  cf: ClusterFactors,
): { ms: number; hopMs?: number; cMs?: number; term: ClusterTerm } {
  if (split === 'layer') {
    const term = cf.hop_ms[link] ?? cf.hop_ms.gpu;
    return { ms: singleMs + term.value * (n - 1), hopMs: term.value, term };
  }
  const term = cf.tensor_c_ms[link === 'gpu' ? 'gpu' : kind] ?? cf.tensor_c_ms.dense;
  return { ms: singleMs / n + term.value * Math.log2(n), cMs: term.value, term };
}

/** How the reading (prefill) rate changes across the pool. Mirrors speed_model.py:prefill_cluster_factor. */
export function prefillClusterFactor(n: number, split: ClusterSplit, cf: ClusterFactors): { factor: number; term: ClusterTerm } {
  if (split === 'layer') return { factor: cf.prefill.layer_factor.value, term: cf.prefill.layer_factor };
  return { factor: Math.pow(n, cf.prefill.tensor_exponent.value), term: cf.prefill.tensor_exponent };
}

/**
 * A very deep mixture of experts in tensor parallel can run far below the bandwidth maths (one measured run: Kimi K3
 * on four M3 Ultras wrote 2 tok/s where the estimate says 36), from per-kernel dispatch overhead. Flag, do not model.
 */
export function dispatchCaveat(cls: ArchClass, layers: number | null | undefined, split: ClusterSplit, cf: ClusterFactors): boolean {
  return split === 'tensor' && cls !== 'dense' && (layers ?? 0) > cf.dispatch_caveat.min_layers;
}
