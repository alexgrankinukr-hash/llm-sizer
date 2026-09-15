/**
 * Custom entries (Advanced mode): a model or a machine the user types in, turned into the same shapes
 * the engine evaluates. Quick models get a conventional-attention cache estimate and are flagged as assumed.
 */
import type { Architecture, Machine, ModelDetail, Platform, Quant, QuantFormat } from './types';

function slug(name: string): string {
  return name.toLowerCase().replace(/[^a-z0-9.]+/g, '-').replace(/^-+|-+$/g, '') || 'custom';
}

export interface CustomModelQuickInput {
  name: string;
  /** total parameters, billions */
  paramsTotalB: number;
  /** active parameters per token, billions (mixture-of-experts); defaults to the total */
  paramsActiveB?: number;
  /** size of the weights file(s) the user will load, decimal GB */
  weightsGb: number;
  contextMax?: number;
  format?: QuantFormat;
  label?: string;
}

/** A typical dense transformer of this size: layer count grows slowly with parameters (Llama-3 8B → 32, 70B → ~76). */
export function assumedLayers(paramsTotalB: number): number {
  return Math.max(8, Math.round(32 * Math.pow(Math.max(paramsTotalB, 0.1) / 8, 0.4)));
}

function customQuant(input: CustomModelQuickInput): Quant {
  const bits = input.paramsTotalB > 0 ? (input.weightsGb * 8) / input.paramsTotalB : null;
  return {
    repo: 'custom',
    format: input.format ?? 'gguf',
    label: input.label ?? (bits ? `${bits.toFixed(1)}-bit` : 'custom'),
    bits,
    size_gb: input.weightsGb,
    size_bytes: Math.round(input.weightsGb * 1e9),
    quantizer: 'custom',
  };
}

function baseDetail(input: CustomModelQuickInput, architecture: Architecture, notes: string[]): ModelDetail {
  const quant = customQuant(input);
  return {
    id: `custom-${slug(input.name)}`,
    name: input.name,
    provider: 'custom',
    hf_repo: '',
    params_total: Math.round(input.paramsTotalB * 1e9),
    params_active: Math.round((input.paramsActiveB ?? input.paramsTotalB) * 1e9),
    params_active_source: input.paramsActiveB ? 'user' : 'dense',
    context_max: input.contextMax ?? 131072,
    featured: false,
    gated: null,
    notes,
    four_bit_native: false,
    architecture,
    quants: [quant],
    reference: { q4: { repo: quant.repo, label: quant.label }, q8: null, mlx4: null, native: null },
    special_builds: [],
    custom: true,
  };
}

/** Quick custom model: enough for fit and speed, with an assumed conventional cache. */
export function customModelQuick(input: CustomModelQuickInput): ModelDetail {
  const layers = assumedLayers(input.paramsTotalB);
  const kvHeads = 8;
  const headDim = 128;
  const architecture: Architecture = {
    attention: 'assumed',
    layers,
    kv_layers: layers,
    sliding_layers: 0,
    linear_layers: 0,
    kv_heads: kvHeads,
    head_dim: headDim,
    k_eq_v: false,
    window_size: null,
    full_bytes_per_token_8bit: 2 * kvHeads * headDim * layers,
    sliding_bytes_per_token_8bit: 0,
    linear_state_bytes: 0,
    experts_total: null,
    experts_active: null,
    accel_mtp: false,
    accel_dflash_repo: null,
    accel_dspark_repo: null,
    conventional_assumed: true,
    source: 'assumed',
    notes: [`architecture assumed: ${layers} layers with grouped-query attention (8 KV heads × 128); enter the real fields in the full form for exact context math`],
  };
  return baseDetail(input, architecture, ['custom model (quick): weights and parameters from the user; attention design assumed conventional']);
}

export interface CustomModelFullInput extends CustomModelQuickInput {
  attention: 'gqa' | 'mla' | 'latent';
  layers: number;
  /** layers whose cache grows with context (defaults to all) */
  kvLayers?: number;
  kvHeads?: number;
  headDim?: number;
  kEqV?: boolean;
  kvLoraRank?: number;
  ropeDim?: number;
  slidingLayers?: number;
  window?: number;
  linearLayers?: number;
  expertsTotal?: number;
  expertsActive?: number;
  mtp?: boolean;
}

/** Full custom model: the same per-layer rules as the discovery job's classifier. */
export function customModelFull(input: CustomModelFullInput): ModelDetail {
  const kvLayers = Math.max(0, Math.min(input.layers, input.kvLayers ?? input.layers - (input.slidingLayers ?? 0) - (input.linearLayers ?? 0)));
  const kvHeads = input.kvHeads ?? 8;
  const headDim = input.headDim ?? 128;
  // keys and values are both stored even when one projection produces them (Gemma 4), so the factor is always 2
  const kv = 2;
  let perLayer: number;
  if (input.attention === 'mla') perLayer = (input.kvLoraRank ?? 512) + (input.ropeDim ?? 0);
  else if (input.attention === 'latent') perLayer = headDim + (input.ropeDim ?? 0);
  else perLayer = kv * kvHeads * headDim;
  const sliding = input.slidingLayers ?? 0;
  const architecture: Architecture = {
    attention: input.attention,
    layers: input.layers,
    kv_layers: kvLayers,
    sliding_layers: sliding,
    linear_layers: input.linearLayers ?? 0,
    kv_heads: kvHeads,
    head_dim: headDim,
    k_eq_v: !!input.kEqV,
    kv_lora_rank: input.kvLoraRank ?? null,
    rope_dim: input.ropeDim ?? null,
    window_size: sliding ? input.window ?? 1024 : null,
    full_bytes_per_token_8bit: perLayer * kvLayers,
    sliding_bytes_per_token_8bit: kv * kvHeads * headDim * sliding,
    linear_state_bytes: 0,
    experts_total: input.expertsTotal ?? null,
    experts_active: input.expertsActive ?? null,
    accel_mtp: !!input.mtp,
    accel_dflash_repo: null,
    accel_dspark_repo: null,
    conventional_assumed: false,
    source: 'override',
    notes: ['architecture fields entered by the user'],
  };
  return baseDetail(input, architecture, ['custom model (full): architecture fields from the user']);
}

export interface CustomMachineInput {
  name: string;
  memoryGb: number;
  bandwidthGbs: number;
  platform: Platform;
  chip?: string;
}

/** A machine the user describes: one memory size, its bandwidth, and the platform that picks the rules. */
export function customMachine(input: CustomMachineInput): Machine {
  return {
    id: `custom-${slug(input.name)}`,
    kind: input.platform === 'apple' ? 'mac' : 'prebuilt',
    family: input.name,
    chip: input.chip ?? input.name,
    memory_options_gb: [input.memoryGb],
    memory_kind: 'unified',
    bandwidth_gbs: input.bandwidthGbs,
    platform: input.platform,
    status: 'current',
    price_usd: null,
    custom: true,
  };
}
