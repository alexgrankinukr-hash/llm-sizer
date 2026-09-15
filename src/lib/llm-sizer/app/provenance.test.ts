import { readFileSync } from 'node:fs';
import { describe, expect, it } from 'vitest';
import { loadFactors, loadModel, loadMultiplierFactors, machineById, quantByLabel } from '../engine/__fixtures__/load';
import { customMachine } from '../engine/custom';
import { ABOUT_ANCHORS, PAGE_OWNED_ANCHORS, aboutHref, cellSources, efficiencyProvenance, machineAnchor, type AboutAnchor } from './provenance';

/** github-slugger for the characters our headings use */
function slugify(text: string): string {
  return text
    .toLowerCase()
    .replace(/[^\p{L}\p{N}\s_-]/gu, '')
    .trim()
    .replace(/\s/g, '-')
    .replace(/-+$/, '');
}

const factors = loadFactors();

describe('about-page anchors', () => {
  it('every methodology anchor is the slug of a real heading in METHODOLOGY.md', () => {
    const md = readFileSync(new URL('../METHODOLOGY.md', import.meta.url), 'utf8');
    const slugs = new Set(
      md
        .split('\n')
        .map((l) => /^#{2,3} (.+)$/.exec(l)?.[1])
        .filter((t): t is string => !!t)
        .map(slugify),
    );
    for (const [key, slug] of Object.entries(ABOUT_ANCHORS)) {
      if (PAGE_OWNED_ANCHORS.has(key as AboutAnchor)) continue;
      expect(slugs.has(slug), `${key} → ${slug}`).toBe(true);
    }
  });
  it('builds hrefs', () => {
    expect(aboutHref('speed')).toBe('/tools/llm-sizer/about#6-speed');
    expect(machineAnchor('mac-mini-m6')).toBe('machine-mac-mini-m6');
  });
});

describe('efficiencyProvenance', () => {
  it('names the measurement, the assumption or the tier (multiplier model, retired in 0.9)', () => {
    const factors = loadMultiplierFactors();
    expect(efficiencyProvenance(machineById('macbook-pro-m5-max-40c'), factors)).toMatchObject({ source: 'measured' });
    expect(efficiencyProvenance(machineById('macbook-pro-m5-max-40c'), factors).text).toMatch(/^measured efficiency 0\.\d\d \(M5 Max, \d rows/);
    expect(efficiencyProvenance(machineById('mac-studio-m5-ultra'), factors).text).toMatch(/^assumed efficiency 0\.55 \(M5 Ultra: /);
    expect(efficiencyProvenance(machineById('nvidia-dgx-spark'), factors).text).toMatch(/^measured efficiency 0\.75 \(DGX Spark, 5 rows, 0\.60–0\.87\)/);
    expect(efficiencyProvenance(machineById('mac-mini-m6'), factors).text).toMatch(/^assumed efficiency 0\.79 \(M6: /);
    const m6pro = efficiencyProvenance(customMachine({ name: 'Next Mac mini', memoryGb: 48, bandwidthGbs: 300, platform: 'apple', chip: 'M7 Pro' }), factors);
    expect(m6pro.source).toBe('tier');
    expect(m6pro.text).toMatch(/^pro-tier estimate 0\.\d\d \(median of \d measured chips; no measured M7 Pro row yet\)/);
    const rocm = efficiencyProvenance(customMachine({ name: 'Halo box', memoryGb: 128, bandwidthGbs: 256, platform: 'rocm' }), factors);
    expect(rocm).toMatchObject({ source: 'none', value: null });
  });

  it('under the fixed-cost model the sentence carries the effective bandwidth and the fixed cost per token', () => {
    const floor = { ...factors, speed_model: { ...factors.speed_model!, kind: 'floor' as const } };
    expect(factors.speed_model?.kind).toBe('floor');
    const m5max = efficiencyProvenance(machineById('macbook-pro-m5-max-40c'), floor);
    expect(m5max.source).toBe('measured');
    expect(m5max.text).toMatch(/^measured profile: \d{3} GB\/s effective \(\d\d % of 614 GB\/s\) and \d\.\d ms per token \(M5 Max, fitted from \d rows\)$/);
    expect(m5max.value).toBeCloseTo(factors.efficiency.by_chip['M5 Max'].b_eff_ratio! * 614, 6);
    const bin = efficiencyProvenance(machineById('macbook-pro-m4-max-32c'), floor);
    expect(bin.text).toMatch(/\(\d\d % of 410 GB\/s\)/);
    expect(efficiencyProvenance(machineById('mac-studio-m5-ultra'), floor).text).toMatch(/^assumed profile: 1020 GB\/s effective \(85 % of 1200 GB\/s\) and 1\.5 ms per token \(M5 Ultra: /);
    expect(efficiencyProvenance(machineById('nvidia-dgx-spark'), floor).text).toMatch(/^measured profile: \d{3} GB\/s effective \(\d\d % of 273 GB\/s\) and \d\.\d ms per token \(DGX Spark, fitted from 2 rows\)$/);
    const m7pro = efficiencyProvenance(customMachine({ name: 'Next Mac mini', memoryGb: 48, bandwidthGbs: 300, platform: 'apple', chip: 'M7 Pro' }), floor);
    expect(m7pro.source).toBe('tier');
    expect(m7pro.text).toMatch(/^pro-tier estimate: \d{3} GB\/s effective \(\d{2,3} % of 300 GB\/s\) and \d\.\d ms per token \(median of \d measured chips; no measured M7 Pro row yet\)$/);
    expect(efficiencyProvenance(customMachine({ name: 'Halo box', memoryGb: 128, bandwidthGbs: 256, platform: 'rocm' }), floor)).toMatchObject({ source: 'none', value: null });
  });
});

describe('cellSources', () => {
  it('links the model, the file, the machine and the methodology sections', () => {
    const model = loadModel('qwen3.8-27b');
    const quant = quantByLabel(model, 'Q4_K_M', 'lmstudio-community/Qwen3.8-27B-GGUF');
    const s = cellSources({ model, quant, machine: machineById('mac-mini-m6'), factors });
    expect(s.model?.href).toBe('https://huggingface.co/Qwen/Qwen3.8-27B');
    expect(s.weights?.href).toBe('https://huggingface.co/lmstudio-community/Qwen3.8-27B-GGUF');
    expect(s.weights?.label).toContain('Q4_K_M');
    expect(s.machine?.href).toBe('/tools/llm-sizer/about#machine-mac-mini-m6');
    expect(s.cache.href).toBe('/tools/llm-sizer/about#5-how-much-memory-a-model-needs');
    expect(s.available.href).toBe('/tools/llm-sizer/about#4-how-much-memory-is-actually-available-for-ai');
  });
  it('has no external links for custom entries', () => {
    const model = { ...loadModel('qwen3.8-27b'), custom: true };
    const machine = customMachine({ name: 'My PC', memoryGb: 24, bandwidthGbs: 1008, platform: 'cuda' });
    const s = cellSources({ model, quant: null, machine, factors });
    expect(s.model).toBeNull();
    expect(s.weights).toBeNull();
    expect(s.machine).toBeNull();
  });
});
