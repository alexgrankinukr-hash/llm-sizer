import { describe, expect, it } from 'vitest';
import { loadFactors, loadMachines, loadModel } from '../engine/__fixtures__/load';
import { defaultState } from './defaults';
import { machineGroups } from './machines';
import { evaluateMatrix, type RecordState } from './matrix';
import { memoryNeeds } from './needs';
import type { AppState } from './state';

const factors = loadFactors();
const groups = machineGroups(loadMachines());
const records: Record<string, RecordState> = { 'qwen3.8-27b': { status: 'ready', model: loadModel('qwen3.8-27b') } };

function needsFor(columns: AppState['columns']): ReturnType<typeof memoryNeeds> {
  const state: AppState = { ...defaultState(), columns };
  return memoryNeeds(state, evaluateMatrix(state, groups, records, factors));
}

describe('memoryNeeds', () => {
  it('is the same on any machine: weights plus cache plus buffers', () => {
    const [n] = needsFor([{ id: 'qwen3.8-27b', quant: 'auto', ctx: 32768 }]);
    expect(n.need).not.toBeNull();
    expect(n.need!.totalGb).toBeCloseTo(n.need!.weightsGb + n.need!.cacheGb + n.need!.buffersGb, 6);
    const state: AppState = { ...defaultState(), columns: [{ id: 'qwen3.8-27b', quant: 'auto', ctx: 32768 }], groups: ['nvidia-dgx-spark'] };
    const onSpark = memoryNeeds(state, evaluateMatrix(state, groups, records, factors));
    expect(onSpark[0].need!.totalGb).toBeCloseTo(n.need!.totalGb, 6);
  });

  it('separates the same model at two builds and two contexts', () => {
    const needs = needsFor([
      { id: 'qwen3.8-27b', quant: 'auto', ctx: 32768 },
      { id: 'qwen3.8-27b', quant: { bucket: 'Q8' }, ctx: 32768 },
      { id: 'qwen3.8-27b', quant: 'auto', ctx: 131072 },
    ]);
    expect(needs).toHaveLength(3);
    expect(needs[1].need!.weightsGb).toBeGreaterThan(needs[0].need!.weightsGb); // a bigger build
    expect(needs[2].need!.cacheGb).toBeGreaterThan(needs[0].need!.cacheGb); // a longer context
    expect(needs[2].need!.weightsGb).toBeCloseTo(needs[0].need!.weightsGb, 6); // same file
  });

  it('caps the context at the model’s maximum and says so', () => {
    const [n] = needsFor([{ id: 'qwen3.8-27b', quant: 'auto', ctx: 4_194_304 }]);
    expect(n.clamped).toBe(true);
    expect(n.contextTokens).toBe(loadModel('qwen3.8-27b').context_max);
  });

  it('reports a column with no record instead of guessing', () => {
    const [n] = needsFor([{ id: 'not-a-model', quant: 'auto', ctx: 32768 }]);
    expect(n.need).toBeNull();
    expect(n.name).toBe('not-a-model');
  });
});

describe('special builds in the memory chart', () => {
  it('carries the SSD part beside the need', () => {
    const state = { ...defaultState(), columns: [{ id: 'qwen3.8-flash-next', quant: { special: 0 } as const, ctx: 32768 }] };
    const matrix = evaluateMatrix(state, groups, { 'qwen3.8-flash-next': { status: 'ready', model: loadModel('qwen3.8-flash-next') } }, factors);
    const [n] = memoryNeeds(state, matrix);
    expect(n.buildLabel).toBe('IQ4_XS · engram on SSD');
    expect(n.need?.weightsGb).toBe(45.8);
    expect(n.need?.ssdGb).toBe(39.1);
  });
});
