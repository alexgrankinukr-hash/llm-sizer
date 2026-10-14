/**
 * The charts that aired in the Apple-Macs video (2026-08-29), as data, so the tool can be held to them.
 * `chart-crosscheck.test.ts` reproduces both matrices through the same evaluateMatrix → tableLayout path the
 * page uses and fails on any cell that disagrees with the chart unless a row in CHART_CORRECTIONS explains it.
 * The table is test support: the reasons are summarised in METHODOLOGY.md's changelog, not rendered.
 */
import type { Marker } from './layout';
import { STATE_DEFAULTS, type AppState } from './state';

export type ChartId = 'v2-128k' | 'realistic';
/** sphere, ring, dash: the three glyphs as drawn in the b-roll */
export type AiredMarker = 'S' | 'R' | 'D';

/** Column order of the aired grid. Qwen 3.8 2.4T is skipped: no official Hugging Face release, so no record. */
export const CHART_MODELS = ['qwen3.8-27b', 'qwen3.8-flash-next', 'deepseek-v4-flash-0731', 'glm-5.3-flash', 'glm-5.2', 'kimi-k3'] as const;
export type ChartModel = (typeof CHART_MODELS)[number];

/** The nine memory rows of the aired grid, mapped to the catalog rows the video meant. */
export const CHART_ROWS = [
  { gb: 16, groupId: 'mac-mini-m6', machineId: 'mac-mini-m6-16gb' },
  { gb: 24, groupId: 'mac-mini-m6', machineId: 'mac-mini-m6' },
  { gb: 32, groupId: 'mac-mini-m6', machineId: 'mac-mini-m6' },
  { gb: 48, groupId: 'mac-mini-m5-pro', machineId: 'mac-mini-m5-pro' },
  { gb: 64, groupId: 'mac-mini-m5-pro', machineId: 'mac-mini-m5-pro' },
  { gb: 96, groupId: 'mac-studio-m5-ultra', machineId: 'mac-studio-m5-ultra' },
  { gb: 128, groupId: 'mac-studio-m5-max', machineId: 'mac-studio-m5-max-40c' },
  { gb: 256, groupId: 'mac-studio-m5-ultra', machineId: 'mac-studio-m5-ultra' },
  { gb: 512, groupId: 'mac-studio-m5-ultra', machineId: 'mac-studio-m5-ultra' },
] as const;

/** What aired: nine markers per model in CHART_ROWS order (16 … 512 GB). Source: broll/fit-matrix-128k.html and fit-matrix-realistic.html. */
export const AIRED: Record<ChartId, Record<ChartModel, string>> = {
  'v2-128k': {
    'qwen3.8-27b': 'DDSSSSSSS',
    'qwen3.8-flash-next': 'DDDDRSSSS',
    'deepseek-v4-flash-0731': 'DDDDDDRSS',
    'glm-5.3-flash': 'DDDDDDRSS',
    'glm-5.2': 'DDDDDDDRS',
    'kimi-k3': 'DDDDDDDDD',
  },
  realistic: {
    'qwen3.8-27b': 'DDRSSSSSS',
    'qwen3.8-flash-next': 'DDDDDDRSS',
    'deepseek-v4-flash-0731': 'DDDDDDDSS',
    'glm-5.3-flash': 'DDDDDDDRS',
    'glm-5.2': 'DDDDDDDDR',
    'kimi-k3': 'DDDDDDDDD',
  },
};

export interface ChartCorrection {
  chart: ChartId;
  model: ChartModel;
  gb: number;
  aired: AiredMarker;
  tool: Exclude<Marker, 'loading' | 'missing'>;
  /** one sentence a reader could check */
  reason: string;
}

export const CHART_CORRECTIONS: ChartCorrection[] = [
  // v2 chart (Q4 + 128K, empty machine). Memory sizes are binary (32 GB = 34.4 GB), the cache is FP16, the macOS 26 GPU limit (74 % at 24 GB, 78 % at 32 and 48 GB, 81 % at 64 and 96 GB, 84 % at 128 GB, 87 % at 256 GB, all but 48 GB at 512 GB),
  // buffers are 1.5 GB + 1 % of the weights, and with the override macOS keeps the larger of 6 GB and 5.5 % of the memory.
  { chart: 'v2-128k', model: 'qwen3.8-27b', gb: 16, aired: 'D', tool: 'ring', reason: 'the chart drew "does not fit at Q4" as a dash; the tool shows the cheapest compromise (a 2-bit build at 32K, tight) as a ring' },
  { chart: 'v2-128k', model: 'qwen3.8-27b', gb: 24, aired: 'D', tool: 'ring', reason: 'a 2-bit build (UD-IQ2_S, 8.4 GB) fits at 128K (18.6 of 19.1 GB), tight; the chart only considered Q4, where nothing fits' },
  { chart: 'v2-128k', model: 'qwen3.8-27b', gb: 32, aired: 'S', tool: 'ring', reason: 'Q4 + 128K with an FP16 cache needs 27.1 GB against 26.8 GB at the macOS 26 limit (78 %); fits with the override (28.4 GB), tight' },
  { chart: 'v2-128k', model: 'qwen3.8-flash-next', gb: 96, aired: 'S', tool: 'ring', reason: 'the Q4 file is 119 GB (the n-gram table is part of it), not the ≈ 85 GB the chart assumed; a 2-bit build (IQ2_S, 77.8 GB) fits at the default limit (83.4 of 83.5 GB), tight' },
  { chart: 'v2-128k', model: 'qwen3.8-flash-next', gb: 128, aired: 'S', tool: 'ring', reason: 'the Q4 file is 119 GB against 115.4 GB at the macOS 26 limit (84 %); fits with the override (130 GB), tight' },
  { chart: 'v2-128k', model: 'deepseek-v4-flash-0731', gb: 96, aired: 'D', tool: 'ring', reason: 'with the override (97 GB) a 2-bit build (90.9 GB) fits at 32K, tight; the chart stopped at Q4 and 128K' },
  { chart: 'v2-128k', model: 'glm-5.2', gb: 512, aired: 'S', tool: 'ring', reason: '485 GB against 498.2 GB at the macOS 26 limit (all but 48 GB): fits at 97 %, tight, which renders as a ring' },
  // realistic chart (16 GB reserved for apps)
  { chart: 'realistic', model: 'qwen3.8-27b', gb: 16, aired: 'D', tool: 'ring', reason: 'the chart treated your apps as fixed; the tool lists closing them plus a 2-bit build at 32K as the compromise' },
  { chart: 'realistic', model: 'qwen3.8-27b', gb: 24, aired: 'D', tool: 'ring', reason: 'the chart treated your apps as fixed; the tool lists closing them plus a 2-bit build (UD-IQ2_S) as the compromise' },
  { chart: 'realistic', model: 'qwen3.8-27b', gb: 48, aired: 'S', tool: 'ring', reason: 'Q4 at 128K needs 27.1 GB (16.8 weights + 8.7 FP16 cache + 1.7 buffers) against 29.5 GB with your apps open: fits at 92 %, tight, which renders as a ring' },
  { chart: 'realistic', model: 'qwen3.8-flash-next', gb: 64, aired: 'D', tool: 'ring', reason: 'the hand-maintained SSD-paged build (45.8 GB resident, 51.0 GB with cache and buffers) fits after closing your apps (55.7 GB at the macOS 26 limit), tight at 92 %; the chart knew only the regular files' },
  { chart: 'realistic', model: 'qwen3.8-flash-next', gb: 96, aired: 'D', tool: 'ring', reason: 'a 2-bit build (IQ2_XXS, 75.2 GB) fits with your apps open (80.7 of 81.1 GB), tight; the chart stopped at Q4' },
  { chart: 'realistic', model: 'deepseek-v4-flash-0731', gb: 96, aired: 'D', tool: 'ring', reason: 'closing your apps, the override, 32K and a 2-bit build (90.9 GB) fit, tight; the chart stopped at Q4 and 128K' },
  { chart: 'realistic', model: 'deepseek-v4-flash-0731', gb: 128, aired: 'D', tool: 'ring', reason: 'a 3-bit build (UD-IQ3_XXS, 104.2 GB) fits at 128K with your apps open (113.2 of 115.4 GB), tight; the chart stopped at Q4' },
  { chart: 'realistic', model: 'glm-5.3-flash', gb: 128, aired: 'D', tool: 'ring', reason: 'a 2-bit build (UD-Q2_K_XL, 108.7 GB) fits with your apps open (112.9 of 115.4 GB), tight; the chart stopped at Q4' },
  { chart: 'realistic', model: 'glm-5.3-flash', gb: 256, aired: 'R', tool: 'run', reason: 'Q4 at 128K needs 204.8 GB against 239.1 GB at the macOS 26 limit (87 %), which your open apps do not lower on a machine this size: runs at 86 %; the chart used three quarters (206.2 GB) and drew a ring' },
  { chart: 'realistic', model: 'glm-5.2', gb: 256, aired: 'D', tool: 'ring', reason: 'closing your apps, the override and a 2-bit build (238.6 GB) fit (254 of 260 GB), tight; the chart stopped at Q4' },
];

/** The Mac-vs-Spark table (broll/mac-vs-spark.html): what aired, and what the calibrated engine says. */
export const MAC_VS_SPARK = {
  aired: { macTokS: 60, sparkTokS: 14, method: 'bandwidth × 0.48 ÷ 9.6 GB per token, GLM-5.3 Flash at Q4' },
  tool: { mac8k: [30, 40], mac128k: [17, 24], sparkQ4Fits: false, sparkFix: '3-bit, tight', sparkTokS: [13, 18] },
} as const;

export function airedToMarker(a: AiredMarker): 'run' | 'ring' | 'no' {
  return a === 'S' ? 'run' : a === 'R' ? 'ring' : 'no';
}

/** The app state that reproduces a chart in the tool: Q4 at 128K, GGUF, the macOS default limit; the realistic chart reserves 16 GB for apps. */
export function chartState(chart: ChartId): AppState {
  return {
    ...STATE_DEFAULTS,
    groups: ['mac-mini-m6', 'mac-mini-m5-pro', 'mac-studio-m5-max', 'mac-studio-m5-ultra'],
    columns: CHART_MODELS.map((id) => ({ id, quant: 'auto' as const, ctx: 131072 })),
    work: chart === 'realistic' ? 16 : 0,
  };
}
