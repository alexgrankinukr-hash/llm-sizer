/**
 * Memory math (METHODOLOGY §4 and §5): what a machine can give the model, what the model needs,
 * and the verdict. `kvCacheBytes` mirrors `scripts/llm-sizer/arch.py:kv_cache_bytes` exactly.
 */
import { BYTES_PER_GB, CLUSTER_POLICY, GIB_IN_GB, GPU_CAP_MAX_HELD_GB, GPU_CAP_TIERS, MEMORY_POLICY, NGRAM_TABLES, PREBUILT_RESERVE_GB, RUNS_HEADROOM, type ClusterPolicy, type MemoryPolicy } from './constants';
import { clusterOf, type Cluster } from './cluster';
import type { Architecture, Availability, Breakdown, KvBits, Machine, Need, Runtime, Settings, SpecialBuild, Quant } from './types';

function clamp01(x: number): number {
  return Math.min(1, Math.max(0, x));
}

/** The default GPU limit macOS 26 applies to a machine of this (nominal) size: its tier's share, with at most 48 GB held back. */
export function defaultCap(memoryGb: number): number {
  let share = GPU_CAP_TIERS[0].share;
  for (const tier of GPU_CAP_TIERS) if (memoryGb >= tier.fromGb) share = tier.share;
  return memoryGb > GPU_CAP_MAX_HELD_GB ? Math.max(share, 1 - GPU_CAP_MAX_HELD_GB / memoryGb) : share;
}

/** A machine's nominal (binary) memory size in the decimal gigabytes file sizes use: 48 GB → 51.5 GB. */
export function ramDecimalGb(memoryGb: number): number {
  return memoryGb * GIB_IN_GB;
}

/**
 * How much memory the model can actually get on this machine with these settings. Linked machines pool: each keeps its
 * own OS reserve and GPU limit (judged on its own nominal size), the work-apps reserve is taken once, and the link's
 * buffers come off the pool (METHODOLOGY §4). With one machine every number is what it was.
 */
export function availableMemory(
  machine: Machine,
  settings: Pick<Settings, 'memoryGb' | 'workApps' | 'override' | 'machines' | 'split'>,
  policy: MemoryPolicy = MEMORY_POLICY,
  clusterPolicy: ClusterPolicy = CLUSTER_POLICY,
): Availability {
  const n = clusterOf(settings)?.n ?? 1;
  const ram = ramDecimalGb(settings.memoryGb); // per machine
  const apps = Math.max(0, settings.workApps); // once: you sit at one machine
  const link = n > 1 ? n * clusterPolicy.overheadGbPerMachine : 0;
  if (machine.platform === 'apple') {
    const capPct = settings.override === false ? defaultCap(settings.memoryGb) : clamp01(settings.override);
    // with the GPU limit lifted, macOS itself is the only reserve left, and on big machines it keeps more than 6 GB
    const os = settings.override === false ? policy.osBaseGb : Math.max(policy.osBaseGb, policy.osReservePctOnOverride * ram);
    const byCap = n * ram * capPct;
    const byApps = n * (ram - os) - apps;
    const available = Math.max(0, Math.min(byCap, byApps) - link);
    const binding: Availability['binding'] = byCap <= byApps ? 'cap' : 'apps';
    const capReserve = Math.max(0, n * ram - available - n * os - apps - link);
    return { availableGb: available, osGb: n * os, appsGb: apps, capReserveGb: capReserve, capPct, binding, machines: n, clusterGb: link };
  }
  // prebuilt unified boxes (DGX Spark, AMD Halo): no GPU limit, a fixed OS + runtime reserve per machine
  const available = Math.max(0, n * (ram - PREBUILT_RESERVE_GB) - apps - link);
  return { availableGb: available, osGb: n * PREBUILT_RESERVE_GB, appsGb: apps, capReserveGb: 0, capPct: null, binding: 'apps', machines: n, clusterGb: link };
}

/**
 * Context-cache bytes at `contextTokens`: the growing term and the sliding-window term scale with the
 * cache precision; the linear-layer state is a constant in fp16 and does not.
 */
export function kvCacheBytes(arch: Architecture | null, contextTokens: number, kvBits: KvBits): number {
  if (!arch) return 0;
  const scale = kvBits / 8;
  const grow = arch.full_bytes_per_token_8bit * contextTokens;
  const window = Math.min(contextTokens, arch.window_size || contextTokens);
  const slide = arch.sliding_bytes_per_token_8bit * window;
  return Math.floor((grow + slide) * scale) + (arch.linear_state_bytes || 0);
}

/** Bytes the cache grows per token at this precision (used by the speed formula). */
export function kvBytesPerToken(arch: Architecture | null, kvBits: KvBits): number {
  if (!arch) return 0;
  return (arch.full_bytes_per_token_8bit * kvBits) / 8;
}

export function buffersGb(weightsGb: number, policy: MemoryPolicy = MEMORY_POLICY): number {
  const raw = Math.max(policy.bufferMinGb, policy.bufferBaseGb + policy.bufferPct * weightsGb);
  return policy.bufferCapGb === null ? raw : Math.min(policy.bufferCapGb, raw);
}

/** Prompt-processing scratch MLX allocates at long context; GGUF files and short prompts need none. */
export function scratchGb(quant: Quant | SpecialBuild, contextTokens: number, policy: MemoryPolicy = MEMORY_POLICY): number {
  const format = 'format' in quant ? quant.format : undefined;
  if (format !== 'mlx' || policy.mlxScratchGbPer1k <= 0) return 0;
  return (policy.mlxScratchGbPer1k * Math.max(0, contextTokens - policy.mlxScratchFromTokens)) / 1024;
}

/** Weights that must be resident: a quant's file size, or a special build's resident size. */
export function residentGb(quant: Quant | SpecialBuild): number {
  return 'resident_gb' in quant ? quant.resident_gb : quant.size_gb;
}

export function isSpecialBuild(quant: Quant | SpecialBuild): quant is SpecialBuild {
  return 'resident_gb' in quant;
}

/** What a special build streams from the SSD; records from before 0.11 carried only the download size. */
export function ssdGbOf(build: SpecialBuild): number {
  if (build.ssd_gb !== undefined) return build.ssd_gb;
  return build.disk_gb !== undefined ? Math.max(0, build.disk_gb - build.resident_gb) : 0;
}

/** The share of a file that is the model's n-gram table, for the architectures that have one (Qwen 3.8 Flash Next); 0 otherwise. */
export function ngramTableShare(arch: Pick<Architecture, 'model_type'> | null): number {
  const type = arch?.model_type ?? '';
  return NGRAM_TABLES.find((t) => type.startsWith(t.modelTypePrefix))?.share ?? 0;
}

/**
 * The GB of a plain file's n-gram table that stays on the SSD (METHODOLOGY §5): MLX files of a table architecture, as
 * oMLX, MTPLX and mlx-serve run them. `loader` is the runtime that loads the file: MLX on a Mac (the default, as the
 * memory view assumes), GGUF anywhere else, where an MLX file is counted whole as it always was. Plain GGUF files keep
 * the table inside the weight shards and need all of it in memory; the special builds that split it out carry their
 * own sizes.
 */
export function tableOnSsdGb(quant: Quant | SpecialBuild, arch: Pick<Architecture, 'model_type'> | null, loader: Runtime = 'mlx'): number {
  if (isSpecialBuild(quant) || quant.format !== 'mlx' || loader !== 'mlx') return 0;
  return quant.size_gb * ngramTableShare(arch);
}

/** The runtime that loads an MLX file on this machine: MLX on a Mac, nothing that keeps its table on the SSD elsewhere. */
export function loaderOn(machine: Pick<Machine, 'platform'>): Runtime {
  return machine.platform === 'apple' ? 'mlx' : 'gguf';
}

/**
 * What the model needs on this machine: weights + context cache + working buffers (+ MLX prompt scratch at long context).
 * The weights are what stays in memory: a special build's resident part, or the file less an n-gram table the runtime
 * leaves on the SSD (`tableOnSsdGb`); the SSD part is reported beside the total, never in it.
 * Across linked machines (METHODOLOGY §5): the shards pack looser (5 % on a layer split, 8 % under tensor parallel), the
 * cache is duplicated on every machine under tensor parallel on latent-attention models, each machine keeps its own
 * working buffers and prompt scratch.
 */
export function neededMemory(
  quant: Quant | SpecialBuild,
  arch: Architecture | null,
  contextTokens: number,
  kvBits: KvBits,
  availableGb: number,
  policy: MemoryPolicy = MEMORY_POLICY,
  cluster: Cluster | null = null,
  clusterPolicy: ClusterPolicy = CLUSTER_POLICY,
  loader: Runtime = 'mlx',
): Need {
  const n = cluster?.n ?? 1;
  const split = cluster?.split ?? 'layer';
  const table = tableOnSsdGb(quant, arch, loader);
  const resident = residentGb(quant) - table;
  const weights = n > 1 ? resident * (1 + clusterPolicy.weightsOverhead[split]) : resident;
  const cacheOne = kvCacheBytes(arch, contextTokens, kvBits) / BYTES_PER_GB;
  const latent = !!arch && (arch.attention === 'mla' || arch.attention === 'latent');
  const cache = n > 1 && split === 'tensor' && latent ? cacheOne * n : cacheOne;
  const buffers = buffersGb(weights, policy) + (n - 1) * policy.bufferBaseGb;
  const scratch = scratchGb(quant, contextTokens, policy) * n;
  const total = weights + cache + buffers + scratch;
  const ssd = isSpecialBuild(quant) ? ssdGbOf(quant) : table;
  return { weightsGb: weights, cacheGb: cache, buffersGb: buffers, scratchGb: scratch, ssdGb: ssd, totalGb: total, ratio: availableGb > 0 ? total / availableGb : Number.POSITIVE_INFINITY };
}

/** runs ≤ 90 % of available · tight ≤ 100 % · no otherwise. */
export function verdictFor(need: Need): 'runs' | 'tight' | 'no' {
  if (need.ratio <= RUNS_HEADROOM) return 'runs';
  if (need.ratio <= 1) return 'tight';
  return 'no';
}

/** The stacked bar for the breakdown panel: everything in decimal GB, so the segments sum to the machine's size. */
export function breakdown(memoryGb: number, availability: Availability, need: Need): Breakdown {
  const fits = need.totalGb <= availability.availableGb;
  const free = Math.max(0, availability.availableGb - need.totalGb);
  const over = Math.max(0, need.totalGb - availability.availableGb);
  const scale = fits ? 1 : availability.availableGb / need.totalGb;
  return {
    ramGb: ramDecimalGb(memoryGb) * availability.machines,
    os: availability.osGb,
    apps: availability.appsGb,
    capReserve: availability.capReserveGb,
    weights: need.weightsGb * scale,
    cache: need.cacheGb * scale,
    buffers: need.buffersGb * scale,
    scratch: need.scratchGb * scale,
    free,
    over,
    ssdGb: need.ssdGb,
    cluster: availability.clusterGb,
    machines: availability.machines,
  };
}
