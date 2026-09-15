import { describe, expect, it } from 'vitest';
import type { Quant } from '../engine/types';
import { buildInfo } from './quants';
import { loadModel } from '../engine/__fixtures__/load';
import { availableBuckets, bucketOf, pickQuant, resolveQuant } from './quants';

describe('quant buckets', () => {
  it('maps every label family to a bucket', () => {
    const cases: [string, string | null][] = [
      ['UD-IQ1_S', 'Q1'], ['IQ2_XXS', 'Q2'], ['UD-Q2_K_XL', 'Q2'], ['UD-IQ3_XXS', 'Q3'], ['Q3_K_M', 'Q3'], ['Q4_K_M', 'Q4'], ['IQ4_XS', 'Q4'],
      ['MLX-4bit', 'Q4'], ['MXFP4', 'Q4'], ['MLX-MXFP4', 'Q4'], ['NVFP4', 'Q4'], ['Q5_K_M', 'Q5'], ['Q6_K', 'Q6'], ['MLX-6bit', 'Q6'],
      ['Q8_0', 'Q8'], ['MLX-8bit', 'Q8'], ['FP8', 'Q8'], ['BF16', 'FP16'], ['MLX-BF16', 'FP16'], ['F32', 'FP32'], ['weird', null],
    ];
    for (const [label, bucket] of cases) expect(bucketOf(label), label).toBe(bucket);
  });

  it('picks the export reference for Q4 and Q8, the largest preferred file otherwise', () => {
    const glm = loadModel('glm-5.2');
    expect(pickQuant(glm, 'Q4', 'gguf')?.label).toBe('UD-Q4_K_XL');
    expect(pickQuant(glm, 'Q2', 'gguf')?.label).toBe('UD-Q2_K_XL');
    expect(pickQuant(glm, 'Q3', 'gguf')?.label).toBe('UD-Q3_K_XL');
    const qwen = loadModel('qwen3.8-27b');
    expect(pickQuant(qwen, 'Q4', 'mlx')?.label).toBe('MLX-4bit');
    expect(pickQuant(qwen, 'Q8', 'gguf')?.label).toBe('Q8_0');
    expect(pickQuant(qwen, 'Q8', 'mlx')?.format).toBe('mlx');
    expect(availableBuckets(qwen, 'gguf')).toEqual(['Q1', 'Q2', 'Q3', 'Q4', 'Q5', 'Q6', 'Q8', 'FP16']);
  });

  it('resolves auto, exact, same-label and fallback settings', () => {
    const qwen = loadModel('qwen3.8-27b');
    expect((resolveQuant(qwen, 'auto', 'gguf') as Quant | null)?.repo).toBe('lmstudio-community/Qwen3.8-27B-GGUF');
    expect((resolveQuant(qwen, { label: 'Q4_K_M', repo: 'bartowski' }, 'gguf') as Quant | null)?.quantizer).toBe('bartowski');
    expect((resolveQuant(qwen, { label: 'Q4_K_M' }, 'gguf') as Quant | null)?.quantizer).toBe('lmstudio-community');
    expect((resolveQuant(qwen, { label: 'Q7_NOPE' }, 'gguf') as Quant | null)?.label).toBe('Q8_0');   // unknown 7-bit label → the Q8 bucket
    expect((resolveQuant(qwen, { label: 'weird' }, 'gguf') as Quant | null)?.label).toBe('Q4_K_M');   // unreadable label → the Q4 reference
    expect((resolveQuant(qwen, { label: 'IQ2_XS' }, 'gguf') as Quant | null)?.label).toBe('IQ2_XS');
  });
});

describe('bucket settings', () => {
  it('resolve through pickQuant at evaluation time', async () => {
    const { loadModel } = await import('../engine/__fixtures__/load');
    const model = loadModel('qwen3.8-27b');
    expect(resolveQuant(model, { bucket: 'Q8' }, 'gguf')).toEqual(pickQuant(model, 'Q8', 'gguf'));
    expect(resolveQuant(model, { bucket: 'Q2' }, 'mlx')).toEqual(pickQuant(model, 'Q2', 'mlx'));
  });
});

describe('special builds as a column setting', () => {
  it('resolves {special} to the build by index, and to the Q4 reference when the record has no such build', () => {
    const fn = loadModel('qwen3.8-flash-next');
    expect(resolveQuant(fn, { special: 0 }, 'gguf')).toMatchObject({ label: 'IQ4_XS · engram on SSD', resident_gb: 45.8, ssd_gb: 39.1 });
    expect(resolveQuant(fn, { special: 2 }, 'gguf')).toMatchObject({ label: 'Q5_K_M · engram on SSD' });
    expect((resolveQuant(fn, { special: 9 }, 'gguf') as Quant | null)?.label).toBe((resolveQuant(fn, 'auto', 'gguf') as Quant | null)?.label);
    expect((resolveQuant(loadModel('qwen3.8-27b'), { special: 0 }, 'gguf') as Quant | null)?.label).toBe('Q4_K_M');
  });
  it('buildInfo describes a file and a special build the same way', () => {
    const fn = loadModel('qwen3.8-flash-next');
    expect(buildInfo(fn.special_builds![0])).toEqual({ label: 'IQ4_XS · engram on SSD', sizeGb: 45.8, ssdGb: 39.1, key: 'special:IQ4_XS · engram on SSD', isSpecial: true });
    const q4 = resolveQuant(fn, 'auto', 'gguf') as Quant;
    expect(buildInfo(q4)).toMatchObject({ label: q4.label, sizeGb: q4.size_gb, ssdGb: 0, key: `${q4.repo}/${q4.label}`, isSpecial: false });
  });
});
