import { describe, expect, it } from 'vitest';
import { dedupeByLabel, fixCandidates, nominalBits } from './fix';
import { loadModel, quantByLabel } from './__fixtures__/load';

describe('nominalBits', () => {
  it('reads the label, not the file size', () => {
    const cases: [string, number | null][] = [
      ['UD-IQ1_S', 1], ['UD-Q1_0', 1], ['IQ2_XXS', 2], ['UD-Q2_K_XL', 2], ['Q2_K_S', 2], ['UD-IQ3_XXS', 3], ['Q3_K_M', 3],
      ['Q4_K_M', 4], ['IQ4_XS', 4], ['UD-Q4_K_XL', 4], ['Q4_0', 4], ['MXFP4', 4], ['MXFP4_MOE', 4], ['NVFP4', 4], ['FP4', 4],
      ['MLX-4bit', 4], ['MLX-2bit', 2], ['MLX-8bit', 8], ['MLX-MXFP4', 4], ['MLX-NVFP4', 4], ['Q5_K_M', 5], ['Q6_K', 6],
      ['Q8_0', 8], ['UD-Q8_K_XL', 8], ['FP8', 8], ['MLX-MXFP8', 8], ['BF16', 16], ['MLX-BF16', 16], ['F16', 16], ['F32', 32], ['weird', null],
    ];
    for (const [label, bits] of cases) expect(nominalBits(label), label).toBe(bits);
  });
});

describe('fix candidates', () => {
  it('Kimi K3: nothing at or above 2 bits is smaller than 700 GB; the 1-bit builds are excluded', () => {
    const kimi = loadModel('kimi-k3');
    const q4 = quantByLabel(kimi, 'UD-Q4_K_XL');
    const allowed = fixCandidates(kimi, q4, 2);
    expect(allowed.every((q) => (nominalBits(q.label) ?? 0) >= 2)).toBe(true);
    expect(Math.min(...allowed.map((q) => q.size_gb))).toBeGreaterThan(700);
    const relaxed = fixCandidates(kimi, q4, 0);
    expect(relaxed.some((q) => q.label === 'UD-Q1_0')).toBe(true);
  });

  it('candidates keep the requested format and are strictly smaller', () => {
    const qwen = loadModel('qwen3.8-27b');
    const mlx = quantByLabel(qwen, 'MLX-4bit', 'lmstudio-community/Qwen3.8-27B-MLX-4bit');
    const c = fixCandidates(qwen, mlx, 2);
    expect(c.every((q) => q.format === 'mlx' && q.size_gb < mlx.size_gb)).toBe(true);
  });

  it('one quant per label, preferring the LM Studio build', () => {
    const qwen = loadModel('qwen3.8-27b');
    const d = dedupeByLabel(qwen.quants.filter((q) => q.label === 'Q4_K_M'));
    expect(d).toHaveLength(1);
    expect(d[0].quantizer).toBe('lmstudio-community');
  });
});
