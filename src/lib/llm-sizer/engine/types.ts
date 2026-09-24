/**
 * Types for the LLM Sizer engine. They mirror the JSON the tool reads:
 * `public/data/llm-sizer/machines.json`, `factors.json`, and the model records exported nightly
 * (`/api/llm-sizer/data/index.json`, `/api/llm-sizer/data/models/<id>.json`).
 * Parameter counts are raw in model records and in billions in the index; sizes are decimal gigabytes.
 */

export type Platform = 'apple' | 'cuda' | 'rocm';
export type KvBits = 4 | 8 | 16;
export type Runtime = 'gguf' | 'mlx';
export type QuantFormat = 'gguf' | 'mlx' | 'safetensors';

// ---------------------------------------------------------------- catalogs

export interface Machine {
  id: string;
  kind: 'mac' | 'prebuilt' | 'pc';
  family: string;
  chip: string;
  generation?: string;
  year?: number;
  memory_options_gb: number[];
  memory_kind?: 'unified' | 'vram';
  bandwidth_gbs: number;
  gpu_cores?: number[];
  /** the GPU bin the list price buys at a memory size, where it is not the smallest (a size that comes only with the larger bin) */
  gpu_cores_by_gb?: Record<string, number>;
  /** Apple's price for the largest GPU bin over the list price, by memory size, where both bins are sold */
  gpu_upgrade_usd?: Record<string, number>;
  platform: Platform;
  interconnect?: string;
  status?: 'current' | 'discontinued';
  price_usd: Record<string, number | null> | null;
  notes?: string;
  sources?: string[];
  /** true for machines the user typed in (Advanced) */
  custom?: boolean;
}

export interface MachinesFile {
  schema: string;
  data_as_of: string;
  machines: Machine[];
}

export interface EfficiencyEntry {
  /** the multiplier model's efficiency (null once a chip only carries a fixed-cost profile) */
  value: number | null;
  n?: number;
  measured: boolean;
  tier: string;
  generation?: string | null;
  range?: [number, number] | null;
  note?: string;
  /** the fixed-cost model's chip profile: milliseconds of fixed cost per token, and the share of the spec bandwidth a read reaches */
  t0_ms?: number;
  b_eff_ratio?: number;
  b_eff_gbs?: number;
  fit_n?: number;
  fit_max_err_pct?: number;
  fit_note?: string;
}

/** A term of the fixed-cost speed model with its provenance. */
export interface FittedTerm {
  value: number;
  n?: number;
  source: 'fitted' | 'assumed' | 'definition';
  range?: [number, number];
  /** the platforms a read factor applies on (fitted there); absent means everywhere */
  platforms?: Platform[];
}

/** How a model is split across linked machines (METHODOLOGY §4, §6.1). */
export type ClusterSplit = 'layer' | 'tensor';
/** Which hop cost a linked machine pays: Macs over Thunderbolt, Sparks over their ConnectX ports, GPU boxes over PCIe. */
export type ClusterLink = 'mac' | 'spark' | 'gpu';
/** One linked-machine constant: fitted on cluster series when `n` is positive, assumed otherwise. */
export interface ClusterTerm {
  value: number;
  n?: number;
  source?: 'fitted' | 'assumed';
  range?: [number, number] | null;
  note?: string;
}
/** The linked-machine terms calibrate.py derives (METHODOLOGY §6.1 and §6.4). */
export interface ClusterFactors {
  /** layer split: milliseconds per token per extra machine, by link */
  hop_ms: Record<ClusterLink, ClusterTerm>;
  /** tensor parallel: milliseconds per token per doubling, by architecture kind (GPU boxes have their own) */
  tensor_c_ms: Record<'dense' | 'moe' | 'gpu', ClusterTerm>;
  prefill: { layer_factor: ClusterTerm; tensor_exponent: ClusterTerm };
  /** a mixture of experts deeper than this many layers gets the dispatch caveat under tensor parallel */
  dispatch_caveat: { min_layers: number; note?: string };
}

export type ArchClass = 'dense' | 'moe' | 'moe_hybrid' | 'moe_latent' | 'deepseek_v4';
export type QuantFamily = 'mlx' | 'q4_0' | 'q8_0' | 'f16' | 'mxfp4' | 'gguf_kquant';
export type MlxGroup = 'dense' | 'small_active_moe_m4plus' | 'small_active_moe_pre_m4' | 'moe_other';

/** The fixed-cost ("floor") speed model as calibrate.py writes it (METHODOLOGY §6.1). */
export type MlxContextGroup = 'pre_m5' | 'm5plus';

export interface SpeedModel {
  kind: 'multiplier' | 'floor';
  read_factor: Record<string, FittedTerm>;
  architecture_cost_ms: Record<string, FittedTerm>;
  attention_ms_per_32k: Record<string, FittedTerm>;
  /** the long-context cost per 32K on MLX, per chip generation: M1 to M4 (`pre_m5`) and M5 and newer (`m5plus`) */
  mlx: { overhead_factor: Record<string, FittedTerm>; attention_ms_per_32k: Record<MlxContextGroup, FittedTerm> };
  validation?: {
    holdout_rule?: string | null;
    n_train: number;
    n_holdout: number;
    train_median_err?: number | null;
    holdout_median_err?: number | null;
    holdout_p90_err?: number | null;
    gate?: { passed: boolean };
    fitted_at?: string | null;
  };
}

export interface Factors {
  schema: string;
  data_as_of: string | null;
  efficiency: {
    by_chip: Record<string, EfficiencyEntry>;
    assumed: Record<string, EfficiencyEntry>;
    by_tier: Record<string, { value: number | null; n_chips: number; range: [number, number] | null; t0_ms?: number; b_eff_ratio?: number }>;
    m5_generation_lift?: number | null;
  };
  /** absent in factor files from before the fixed-cost model; the engine then runs the multiplier */
  speed_model?: SpeedModel;
  runtime: {
    gguf: { factor: number };
    mlx: {
      dense_under_14b: { factor: number; range?: [number, number] };
      dense_14b_and_up: { factor: number; range?: [number, number] };
      moe: { factor: number; range?: [number, number] };
      flatten_above_context: number;
      measured_here?: boolean;
    };
  };
  acceleration: Record<string, { min: number; median: number; max: number; n: number } | string>;
  feels_like: { max_tok_s: number | null; label: string; description?: string }[];
  /** prompt-processing terms (calibrate.py); absent in factor files from before 0.12 */
  prefill?: PrefillFactors;
  /** linked-machine terms (calibrate.py); absent in factor files from before 0.13 */
  cluster?: ClusterFactors;
}

// ---------------------------------------------------------------- models

export interface Quant {
  repo: string;
  url?: string;
  format: QuantFormat;
  label: string;
  /** effective bits per weight = file bytes × 8 ÷ total parameters (includes non-weight tensors) */
  bits: number | null;
  size_gb: number;
  size_bytes: number;
  quantizer: string;
  file_count?: number;
  qat?: boolean;
  first_seen?: string;
}

/** A hand-maintained build that is not a plain quant file: part of the model stays in memory, the rest (an n-gram table) streams from the SSD. */
export interface SpecialBuild {
  label: string;
  /** what must be in memory */
  resident_gb: number;
  /** what streams from the SSD (the n-gram table); older records without it fall back to disk_gb − resident_gb */
  ssd_gb?: number;
  /** the whole download */
  disk_gb?: number;
  format?: QuantFormat;
  bits?: number;
  url?: string;
  note?: string;
  /** the publisher's measurement of this build, when one exists; quoted as a note, never used as a cell's number */
  measured_tok_s?: number;
  measured_on?: string;
  /** the chip of that measurement, when the publisher named it */
  measured_chip?: string;
}

export interface Architecture {
  model_type?: string | null;
  attention: 'gqa' | 'mla' | 'latent' | 'assumed';
  layers: number;
  kv_layers: number;
  sliding_layers: number;
  linear_layers: number;
  kv_heads?: number | null;
  head_dim?: number | null;
  k_eq_v?: boolean;
  kv_lora_rank?: number | null;
  rope_dim?: number | null;
  window_size: number | null;
  /** per token, all layers whose cache grows with context, 8-bit cache */
  full_bytes_per_token_8bit: number;
  /** per token inside the sliding window, all sliding layers, 8-bit cache */
  sliding_bytes_per_token_8bit: number;
  /** constant state of linear/recurrent layers (fp16), not scaled by the cache precision */
  linear_state_bytes: number;
  hidden_size?: number | null;
  experts_total: number | null;
  experts_active: number | null;
  experts_shared?: number | null;
  moe_intermediate?: number | null;
  moe_layers?: number | null;
  accel_mtp: boolean;
  accel_dflash_repo?: string | null;
  accel_dspark_repo?: string | null;
  /** hand-set architecture class for the speed model's fixed cost (models.overrides.json), e.g. deepseek_v4 */
  arch_cost_class?: string | null;
  conventional_assumed: boolean;
  source: 'config' | 'override' | 'assumed';
  notes?: string[];
  kv_cache_gb_at_128k_8bit?: number;
}

export interface QuantRef {
  repo: string;
  label: string;
}

export interface ModelReference {
  q4: QuantRef | null;
  q8: QuantRef | null;
  mlx4: QuantRef | null;
  native: QuantRef | null;
}

export interface ModelDetail {
  schema?: string;
  id: string;
  name: string;
  provider: string;
  hf_repo: string;
  url?: string;
  variant?: string | null;
  released?: string | null;
  params_total: number | null;
  params_active: number | null;
  params_active_source?: string | null;
  context_max: number | null;
  native_bits?: number | null;
  license?: { name: string | null; url?: string | null; commercial?: boolean | null; note?: string | null };
  tags?: string[];
  featured: boolean;
  tier?: string | null;
  rank?: number | null;
  /** 'false' | 'auto' | 'manual' | null — a string, as Hugging Face reports it */
  gated: string | null;
  status?: string;
  superseded_by?: string | null;
  notes: string[];
  four_bit_native: boolean;
  architecture: Architecture | null;
  quants: Quant[];
  reference?: ModelReference;
  special_builds?: SpecialBuild[];
  drafts?: { dflash: string | null; dspark: string | null };
  generated_at?: string;
  sha?: string;
  /** true for models the user typed in (Advanced) */
  custom?: boolean;
}

export interface ModelIndexEntry {
  id: string;
  name: string;
  provider: string;
  hf_repo: string;
  released: string | null;
  params_total_b: number | null;
  params_active_b: number | null;
  context_max: number | null;
  featured: boolean;
  tier: string | null;
  rank: number | null;
  gated: string | null;
  status: string;
  license: string | null;
  commercial: boolean | null;
  arch: 'gqa' | 'mla' | 'latent' | 'assumed';
  hybrid: boolean;
  moe: boolean;
  assumed: boolean;
  accel: { mtp: boolean; dflash: boolean; dspark: boolean };
  kv_bytes_per_token_8bit: number | null;
  sizes: { min_gb: number | null; q4_gb: number | null; q8_gb: number | null; native_gb: number | null };
  formats: QuantFormat[];
  quant_count: number;
  four_bit_native: boolean;
  has_special_build?: boolean;
  detail_sha: string;
}

export interface ModelIndex {
  schema: string;
  generated_at: string;
  count: number;
  featured_count: number;
  models: ModelIndexEntry[];
  sha: string;
}

// ---------------------------------------------------------------- inputs

export interface Settings {
  /** the memory size of the machine row being evaluated */
  memoryGb: number;
  /** GB reserved for the user's own apps: 0 when the work toggle is off, 16 by default when on (8 / 24 / custom) */
  workApps: number;
  /** false = the platform's default GPU limit; a number = the override as a fraction of RAM (1.0 = everything) */
  override: false | number;
  contextTokens: number;
  kvBits: KvBits;
  runtime: Runtime;
  /** lowest nominal bits per weight the automatic fix search may propose (default 2) */
  qualityFloorBits: number;
  /** the build to evaluate, a quant file or a special build; defaults to the model's reference 4-bit quant for the runtime */
  quant?: Quant | SpecialBuild;
  /** identical machines linked as one pool (2 to 4); absent or 1 = one machine */
  machines?: number;
  /** how the model is split across linked machines; layer by default */
  split?: ClusterSplit;
}

// ---------------------------------------------------------------- results

export interface Availability {
  availableGb: number;
  osGb: number;
  appsGb: number;
  /** what the platform's GPU limit holds back beyond the OS and the apps (Apple only) */
  capReserveGb: number;
  /** the GPU limit in force as a fraction of RAM, null on platforms without one */
  capPct: number | null;
  binding: 'cap' | 'apps';
  /** machines in the pool (1 for one machine); the reserves above are the pool's totals */
  machines: number;
  /** communication buffers set aside across the pool (0 on one machine) */
  clusterGb: number;
}

export interface Need {
  weightsGb: number;
  cacheGb: number;
  buffersGb: number;
  /** prompt-processing scratch MLX allocates at long context (0 for GGUF files and short contexts) */
  scratchGb: number;
  /** what a special build streams from the SSD; not memory, so not in totalGb (0 for a plain file) */
  ssdGb: number;
  totalGb: number;
  /** totalGb ÷ availableGb */
  ratio: number;
}

export type Verdict = 'runs' | 'tight' | 'compromise' | 'no-fit';

export type Change =
  | { kind: 'closeApps' }
  | { kind: 'override'; cap: number }
  | { kind: 'context'; tokens: number }
  | { kind: 'quant'; quant: Quant }
  | { kind: 'ssdPaged'; build: SpecialBuild };

export interface Fix {
  changes: Change[];
  cost: number;
  verdict: 'runs' | 'tight';
  need: Need;
  availability: Availability;
  /** the settings the fix evaluates under (context, override, apps) */
  contextTokens: number;
  reason: string;
}

export interface Acceleration {
  kind: 'mtp' | 'draft';
  min: number;
  median: number;
  max: number;
  /** the draft repo(s) or 'native heads' */
  via: string;
}

/** How a linked pool's speed was made from one machine's (METHODOLOGY §6.1). */
export interface ClusterSpeed {
  machines: number;
  split: ClusterSplit;
  link: ClusterLink;
  /** one machine's milliseconds per token for the whole model */
  singleMs: number;
  hopMs?: number;
  cMs?: number;
  source: 'fitted' | 'assumed';
  /** cluster series the term rests on */
  n: number;
}

export interface Speed {
  tokS: number | null;
  gbPerToken: number;
  /** the multiplier model's efficiency; null under the fixed-cost model */
  efficiency: number | null;
  /** where the chip's factor or profile comes from */
  efficiencySource: 'measured' | 'assumed' | 'tier' | 'none';
  /** the runtime factor applied: the MLX multiplier (multiplier model) or the MLX fixed-cost factor (floor model), 1 for GGUF */
  runtimeFactor: number;
  runtimeApplied: Runtime;
  feelsLike: { label: string; description: string } | null;
  accelerations: Acceleration[];
  notes: string[];
  /** which model produced tokS */
  modelKind: 'multiplier' | 'floor';
  /** fixed-cost model parts (ms per token), for the sheet's "how this number is made" line */
  t0Ms?: number;
  bEffGbs?: number;
  readMs?: number;
  archClass?: ArchClass;
  archCostMs?: number;
  attnMs?: number;
  /** present on a linked pool */
  cluster?: ClusterSpeed;
}

export interface Breakdown {
  ramGb: number;
  os: number;
  apps: number;
  capReserve: number;
  weights: number;
  cache: number;
  buffers: number;
  scratch: number;
  free: number;
  /** how far the need exceeds the available memory (0 when it fits) */
  over: number;
  /** what streams from the SSD, drawn beside the bar, never part of it */
  ssdGb: number;
  /** communication buffers of a linked pool (0 on one machine) */
  cluster: number;
  /** machines in the pool; ramGb is the pool's size */
  machines: number;
}

export type CellFlag =
  | 'assumed'
  | 'clamped-context'
  | 'no-mlx-file'
  | 'no-gguf-mlx-build'
  | 'gated'
  | 'four-bit-native'
  | 'q8-is-repack'
  | 'unmeasured-chip'
  | 'kv-precision-note'
  | 'ssd-build'
  | 'linked-layer'
  | 'linked-tensor'
  | 'linked-dispatch';

export interface CellResult {
  verdict: Verdict;
  availability: Availability;
  need: Need;
  /** what was evaluated at the ask (a quant or a special build) */
  quant: Quant | SpecialBuild | null;
  /** the configuration a compromise cell actually reports (the fix, or the ask when it fits) */
  fix: Fix | null;
  alternatives: Fix[];
  /** the best quant-only fix ignoring the quality floor, only when nothing allowed fits */
  belowFloor: Fix | null;
  nearestMiss: string | null;
  /** speed of the configuration that fits (the fix's, when there is one) */
  speed: Speed;
  /** the prompt-processing estimate for the configuration that fits; null when nothing fits */
  prefill: Prefill | null;
  breakdown: Breakdown;
  flags: CellFlag[];
  /** the context the result was computed at (after clamping) */
  contextTokens: number;
}

/** The prefill (prompt-processing) terms calibrate.py derives; METHODOLOGY section 6.4. */
export interface PrefillFactors {
  /** tokens/s × active billions per GPU core, by generation ("M1" … "M5"), on plain llama.cpp files at 512 tokens */
  per_core: Record<string, { value: number; n: number; range: [number, number] }>;
  /** the same constant for chips where cores mean nothing (the DGX Spark) */
  by_chip: Record<string, { value: number; n: number; range: [number, number] }>;
  /** chips that have rows of their own: anything else in the generation is an extrapolation on its cores */
  chips_with_rows: string[];
  kquant: { factor: number; n: number; range?: [number, number]; note?: string };
  class: Record<string, { factor: number; n?: number; range?: [number, number]; note?: string; examples?: string[] }>;
  /** models whose prefill class is their own (from a published sweep), by model id */
  class_by_model: Record<string, string>;
  mlx: Record<'pre_m5' | 'm5plus', { factor: number; n: number; range?: [number, number]; note?: string }>;
  context: { d0_tokens: number; n_points: number; range: [number, number] | null; attention_share_ref: number; note?: string };
  waits_tokens: number[];
  feels_like: { max_s: number | null; label: string }[];
}

/** The prompt-processing estimate for one configuration: the short-prompt rate, how it falls with context, and the wait before the first word. */
export interface Prefill {
  /** prompt tokens per second at a short prompt; null when there is no constant for the chip or build */
  tokS: number | null;
  /** the context depth at which the rate has halved */
  d0: number | null;
  /** the wait before the first word for the standard prompt sizes */
  waits: { tokens: number; seconds: number; feelsLike: string }[];
  /** the same for a prompt that fills the column's context, with the rate at that depth */
  atContext: { tokens: number; seconds: number; rateTokS: number } | null;
  source: 'measured' | 'generation' | 'assumed' | 'none';
  parts: { k: number; cores: number | null; perCore: number | null; generation: string | null; kquant: number; cls: string; classFactor: number; mlx: number; activeB: number; cluster: number } | null;
  /** the same chip's larger GPU bin, when this row reads on a smaller one: its rate, its waits and Apple's price for it where known */
  upgrade: { cores: number; ratio: number; usd: number | null; tokS: number; waits: { tokens: number; seconds: number }[] } | null;
  notes: string[];
}
