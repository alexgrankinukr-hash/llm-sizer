/**
 * Constants of the LLM Sizer engine. Every number here is explained in METHODOLOGY.md; change both together.
 */

/** macOS itself, always (METHODOLOGY §4). */
export const OS_BASE_GB = 6;

/** OS + runtime reserve on prebuilt unified boxes such as the DGX Spark (METHODOLOGY §7). */
export const PREBUILT_RESERVE_GB = 8;

/** App budgets for the "I'll also use this machine for work" toggle (METHODOLOGY §4). */
export const APPS_GB = { off: 0, light: 8, typical: 16, heavy: 24 } as const;

/**
 * Memory sizes are quoted in binary gigabytes (a "48 GB" Mac has 48 GiB = 51.5 GB), file sizes in decimal GB.
 * Every reserve and comparison converts the machine's size with this factor first (METHODOLOGY §4).
 */
export const GIB_IN_GB = 1.073741824;

/**
 * macOS default GPU memory limit (Metal's recommended working set): ⅔ of RAM under 36 GB, ¾ from 36 GB.
 * Logged values: 16 and 32 GB machines report ⅔ (10,922 / 21,845 MiB); 36, 48, 64 and 128 GB machines report ¾
 * (28,991 / 38,655 / 49,152 / 98,304 MiB). The threshold compares the nominal size (METHODOLOGY §4).
 */
export const GPU_CAP = { small: 2 / 3, large: 3 / 4, thresholdGb: 36 } as const;

/** Working buffers: max(1 GB, 5 % of weights), capped at 16 GB (METHODOLOGY §5). */
export const BUFFER_MIN_GB = 1;
export const BUFFER_PCT = 0.05;
export const BUFFER_CAP_GB = 16;

/**
 * Memory policy margins (METHODOLOGY §4 and §5). These are safety margins, not measured facts, and are labelled as such.
 * `bufferCapGb: null` means no cap. The MLX scratch term is what MLX's prompt processing allocates at long context,
 * per 1K tokens beyond `mlxScratchFromTokens`; the override reserve is what macOS actually keeps once the GPU limit is
 * lifted, as a share of the machine's memory (the larger of it and `osBaseGb` applies).
 */
export interface MemoryPolicy {
  /** buffers = min(cap, max(min, base + pct × weights)) */
  bufferMinGb: number;
  bufferBaseGb: number;
  bufferPct: number;
  bufferCapGb: number | null;
  mlxScratchGbPer1k: number;
  mlxScratchFromTokens: number;
  osBaseGb: number;
  osReservePctOnOverride: number;
}
/** The margins the tool shipped with through METHODOLOGY 0.8: max(1 GB, 5 % of weights) capped at 16 GB, no scratch term, 6 GB for macOS. */
export const MEMORY_POLICY_V08: MemoryPolicy = { bufferMinGb: 1, bufferBaseGb: 0, bufferPct: 0.05, bufferCapGb: 16, mlxScratchGbPer1k: 0, mlxScratchFromTokens: 8192, osBaseGb: 6, osReservePctOnOverride: 0 };
/**
 * The margins in force since METHODOLOGY 0.9: buffers of 1.5 GB plus 1 % of the weights (measured working room sits
 * near 0.4 GB on a 224 GB model and near 1.6 GB on a 16 GB one), 0.15 GB of MLX prompt scratch per 1K tokens beyond 8K
 * (the growth measured on the 27B and 397B MLX sweeps), and macOS keeping 5.5 % of the memory once the GPU limit is
 * lifted (27 to 30 GB observed on a 512 GB Studio).
 */
export const MEMORY_POLICY: MemoryPolicy = { bufferMinGb: 0, bufferBaseGb: 1.5, bufferPct: 0.01, bufferCapGb: null, mlxScratchGbPer1k: 0.15, mlxScratchFromTokens: 8192, osBaseGb: 6, osReservePctOnOverride: 0.055 };

/** "Runs" needs at most this share of the available memory; up to 100 % is "tight" (METHODOLOGY §5). */
export const RUNS_HEADROOM = 0.9;

/** The automatic fix search never proposes a build under this many nominal bits per weight (METHODOLOGY §5). */
export const MIN_FIX_NOMINAL_BITS = 2;

/** Context lengths offered as chips in the table. */
export const CONTEXT_CHIPS = [8192, 32768, 131072, 262144] as const;

/** The MLX runtime factor applies in full up to the first value and fades linearly to 1.0 at the second (METHODOLOGY §6.2). */
export const MLX_FADE_TOKENS: readonly [number, number] = [24000, 36000];

/** Draft-model acceleration helps little when a mixture-of-experts model has fewer active parameters than this (billions). */
export const SMALL_ACTIVE_B = 6;

/** Dense models with at least this many parameters (billions) get the larger MLX factor. */
export const MLX_DENSE_SPLIT_B = 14;

/** Costs of the changes the fix search may propose; the cheapest combination that fits wins (METHODOLOGY §5). */
export const FIX_COST = {
  closeApps: 1,
  override: 2,
  contextStep: 4,
  quant: 8,
  quantUnder3Bits: 1,
  ssdPaged: 16,
  tight: 0.5,
} as const;

/**
 * Reference-quant preferences. The nightly export applies the same lists (`export.py`) and ships the
 * result as `reference`; the engine uses these only for custom models that have no export record.
 */
export const REF_4BIT = ['Q4_K_M', 'UD-Q4_K_XL', 'UD-Q4_K_M', 'Q4_K_S', 'IQ4_XS', 'MLX-4bit', 'Q4_0', 'MXFP4', 'MXFP4_MOE', 'MLX-MXFP4', 'NVFP4', 'MLX-NVFP4'] as const;
export const REF_8BIT = ['Q8_0', 'MLX-8bit', 'UD-Q8_K_XL', 'FP8', 'MLX-MXFP8', 'MXFP8'] as const;
export const REF_MLX_4BIT = ['MLX-4bit', 'MLX-MXFP4', 'MLX-NVFP4'] as const;
export const QUANTIZER_PREFERENCE = ['lmstudio-community', 'unsloth', 'bartowski', 'ggml-org', 'mlx-community', 'official'] as const;

/** A "Q8" file with fewer effective bits than this is a re-pack of 4-bit weights, not a higher-quality build. */
export const Q8_REPACK_BITS = 6;

/** Decimal gigabytes, as everywhere in the exported data. */
export const BYTES_PER_GB = 1e9;

/** Linked machines: at most this many identical machines in one pool (METHODOLOGY §4). */
export const CLUSTER_MAX_MACHINES = 4;

/**
 * Linked-machine memory margins (METHODOLOGY §4 and §5). Policy, like MEMORY_POLICY: 2 GB per machine for the link's
 * buffers (assumed); shards pack 5 % looser on a layer split (measured: exo split a 540 GB model into 2 × 280 GB) and
 * 8 % looser under tensor parallel (measured: Kimi K3 on four M3 Ultras held 420.8 GB per machine against a 363 GB share).
 */
export interface ClusterPolicy {
  overheadGbPerMachine: number;
  weightsOverhead: { layer: number; tensor: number };
}
export const CLUSTER_POLICY: ClusterPolicy = { overheadGbPerMachine: 2, weightsOverhead: { layer: 0.05, tensor: 0.08 } };
