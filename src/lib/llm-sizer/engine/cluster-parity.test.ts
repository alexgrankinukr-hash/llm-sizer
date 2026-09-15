/**
 * The linked-machine formulas computed by the engine must reproduce the milliseconds calibrate.py emitted for the
 * same inputs (scripts/llm-sizer/tests/fixtures/cluster-cases.json). speed_model.py mirrors cluster.ts function for
 * function; this is the pin that keeps them from drifting.
 */
import { readFileSync } from 'node:fs';
import { describe, expect, it } from 'vitest';
import { clusterMs, prefillClusterFactor } from './cluster';
import { loadFactors } from './__fixtures__/load';
import type { ClusterLink, ClusterSplit } from './types';

interface Case {
  id: string;
  single_ms: number;
  n: number;
  split: ClusterSplit;
  link: ClusterLink;
  kind: 'dense' | 'moe';
  expected_ms: number;
  expected_tok_s: number;
  prefill_factor: number;
}

const file = JSON.parse(readFileSync(new URL('../../../../scripts/llm-sizer/tests/fixtures/cluster-cases.json', import.meta.url), 'utf8')) as { cases: Case[] };
const factors = loadFactors();

describe('linked-machine parity with speed_model.py', () => {
  it('the fixture covers every link, both splits, both kinds and the counts 2 to 4', () => {
    expect(file.cases.length).toBeGreaterThanOrEqual(36);
    expect(new Set(file.cases.map((c) => c.link))).toEqual(new Set(['mac', 'spark', 'gpu']));
    expect(new Set(file.cases.map((c) => c.split))).toEqual(new Set(['layer', 'tensor']));
    expect(new Set(file.cases.map((c) => c.n))).toEqual(new Set([2, 3, 4]));
  });

  it('the data set carries the cluster block calibrate.py wrote', () => {
    const cf = factors.cluster!;
    expect(cf.hop_ms.mac.value).toBeGreaterThan(2);
    expect(cf.hop_ms.spark.value).toBeGreaterThan(2);
    expect(cf.tensor_c_ms.moe.value).toBeGreaterThan(cf.tensor_c_ms.dense.value);
    expect(cf.prefill.tensor_exponent.value).toBe(0.75);
    expect(cf.dispatch_caveat.min_layers).toBe(60);
  });

  for (const c of file.cases) {
    it(`reproduces ${c.id}`, () => {
      const cf = factors.cluster!;
      const got = clusterMs(c.single_ms, c.n, c.split, c.link, c.kind, cf);
      expect(got.ms).toBeCloseTo(c.expected_ms, 6);
      expect(1000 / got.ms).toBeCloseTo(c.expected_tok_s, 6);
      expect(prefillClusterFactor(c.n, c.split, cf).factor).toBeCloseTo(c.prefill_factor, 9);
    });
  }
});
