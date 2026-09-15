import { describe, expect, it } from 'vitest';
import { catalogRows, contextChips, customMachine, customModelQuick, defaultQuant, evaluateCell, evaluateModelAcrossMachines, kvCacheBytes, nominalBits } from './index';
import { loadFactors, loadMachines, loadModel, machineById } from './__fixtures__/load';
import type { Settings } from './types';

const factors = loadFactors();
// FP16 cache, the tool's default and every runtime's: the numbers below are the ones METHODOLOGY §9 quotes
const base: Omit<Settings, 'memoryGb'> = { workApps: 0, override: false, contextTokens: 32768, kvBits: 16, runtime: 'gguf', qualityFloorBits: 2 };
const s = (memoryGb: number, over: Partial<Settings> = {}): Settings => ({ ...base, memoryGb, ...over });

describe('defaults', () => {
  it('reference quants follow the export; MLX runtime prefers the MLX 4-bit file', () => {
    const qwen = loadModel('qwen3.8-27b');
    expect(defaultQuant(qwen, 'gguf')?.repo).toBe('lmstudio-community/Qwen3.8-27B-GGUF');
    expect(defaultQuant(qwen, 'gguf')?.label).toBe('Q4_K_M');
    expect(defaultQuant(qwen, 'mlx')?.label).toBe('MLX-4bit');
    expect(defaultQuant(qwen, 'gguf', 'q8')?.label).toBe('Q8_0');
    expect(defaultQuant(qwen, 'gguf', 'native')?.repo).toBe('Qwen/Qwen3.8-27B');
    expect(contextChips(qwen)).toEqual([8192, 32768, 131072, 262144]);
    expect(contextChips({ context_max: 100000 })).toEqual([8192, 32768]);
  });
});

describe('the matrix (METHODOLOGY §9 with the data\'s numbers)', () => {
  const mini = machineById('mac-mini-m6');
  const ultra = machineById('mac-studio-m5-ultra');
  const spark = machineById('nvidia-dgx-spark');
  const mbp = machineById('macbook-pro-m5-max-40c');

  it('1a/1b — 32 GB mini, Qwen 27B Q4 at 32K: tight at 90 % with apps off, "close your apps" with the work toggle on', () => {
    const qwen = loadModel('qwen3.8-27b');
    const off = evaluateCell(qwen, mini, s(32), factors);
    expect(off.verdict).toBe('tight');
    expect(off.need.ratio).toBeCloseTo(0.9037, 3);
    expect(off.speed.modelKind).toBe('floor');
    expect(off.speed.tokS!).toBeGreaterThan(5.5);
    expect(off.speed.tokS!).toBeLessThan(7.5);
    expect(off.flags).toContain('unmeasured-chip');
    const on = evaluateCell(qwen, mini, s(32, { workApps: 16 }), factors);
    expect(on.verdict).toBe('compromise');
    expect(on.fix?.changes.map((c) => c.kind)).toEqual(['closeApps']);
    expect(on.fix?.verdict).toBe('tight');
    expect(on.fix?.reason).toMatch(/close your apps/);
    expect(on.alternatives.some((f) => f.changes.some((c) => c.kind === 'quant'))).toBe(true);
    expect(on.breakdown.apps).toBe(0); // the breakdown shows the configuration that fits
  });

  it('2a/2b — 256 GB Studio, GLM-5.3 Flash Q4 at 128K: tight at 99 % as asked; runs with the override, at 0.840 with apps on too', () => {
    const glm = loadModel('glm-5.3-flash');
    const cell = evaluateCell(glm, ultra, s(256, { contextTokens: 131072 }), factors);
    expect(cell.quant && 'label' in cell.quant && cell.quant.label).toBe('UD-Q4_K_XL');
    expect(cell.verdict).toBe('tight');
    expect(cell.fix).toBeNull();
    expect(cell.need.ratio).toBeCloseTo(0.9932, 3);
    const lifted = evaluateCell(glm, ultra, s(256, { contextTokens: 131072, override: 1 }), factors);
    expect(lifted.verdict).toBe('runs');
    expect(lifted.need.ratio).toBeCloseTo(0.7882, 3);
    const tight = evaluateCell(glm, ultra, s(256, { contextTokens: 131072, workApps: 16, override: 1 }), factors);
    expect(tight.verdict).toBe('runs');
    expect(tight.need.ratio).toBeCloseTo(0.8400, 3);
  });

  it('2c — GLM-5.3 Flash speed: ≈ 35 tok/s on the M5 Ultra; on the Spark Q4 does not fit and the 3-bit build fits tight at ≈ 15', () => {
    const glm = loadModel('glm-5.3-flash');
    const u = evaluateCell(glm, ultra, s(512, { contextTokens: 8192, override: 1 }), factors);
    expect(u.verdict).toBe('runs');
    expect(u.speed.tokS!).toBeGreaterThan(30);
    expect(u.speed.tokS!).toBeLessThan(40);
    expect(u.speed.archClass).toBe('moe_latent');
    const sp = evaluateCell(glm, spark, s(128, { contextTokens: 131072 }), factors);
    expect(sp.verdict).toBe('compromise');
    const q = sp.fix!.changes.find((c) => c.kind === 'quant');
    expect(q && q.kind === 'quant' && nominalBits(q.quant.label)).toBe(3);
    expect(sp.fix?.verdict).toBe('tight');
    expect(sp.fix?.need.ratio).toBeCloseTo(0.9628, 3);
    expect(sp.alternatives.some((f) => f.verdict === 'runs' && f.changes.some((c) => c.kind === 'quant' && nominalBits(c.quant.label) === 2))).toBe(true);
    expect(sp.speed.tokS!).toBeGreaterThan(13);
    expect(sp.speed.tokS!).toBeLessThan(18);
    expect(sp.speed.notes[0]).toMatch(/configuration that fits/);
  });

  it('3 — 256 GB Studio, GLM-5.2 Q4 at 128K: a ring; with the override (259.8 GB) the 2-bit UD-IQ2_M build fits, tight', () => {
    const glm = loadModel('glm-5.2');
    const cell = evaluateCell(glm, ultra, s(256, { contextTokens: 131072 }), factors);
    expect(cell.verdict).toBe('compromise');
    expect(cell.fix?.changes.map((c) => c.kind)).toEqual(['override', 'quant']);
    const q = cell.fix!.changes.find((c) => c.kind === 'quant');
    expect(q && q.kind === 'quant' && q.quant.label).toBe('UD-IQ2_M');
    expect(cell.fix?.verdict).toBe('tight');
    expect(cell.fix?.need.ratio).toBeCloseTo(0.9788, 3);
    expect(cell.nearestMiss).toBeNull();
    expect(cell.belowFloor).toBeNull();
  });

  it('4 — 512 GB Studio, GLM-5.2 Q4 at 128K: tight with the override; the override is the cheapest fix without it', () => {
    const glm = loadModel('glm-5.2');
    const withOverride = evaluateCell(glm, ultra, s(512, { contextTokens: 131072, override: 1 }), factors);
    expect(withOverride.verdict).toBe('tight');
    expect(withOverride.need.ratio).toBeCloseTo(0.9340, 3);
    const apps = evaluateCell(glm, ultra, s(512, { contextTokens: 131072, override: 1, workApps: 16 }), factors);
    expect(apps.verdict).toBe('tight');
    expect(apps.need.ratio).toBeCloseTo(0.9637, 3);
    const noOverride = evaluateCell(glm, ultra, s(512, { contextTokens: 131072 }), factors);
    expect(noOverride.verdict).toBe('compromise');
    expect(noOverride.fix?.changes.map((c) => c.kind)).toEqual(['override']);
    expect(noOverride.alternatives.some((f) => f.changes.some((c) => c.kind === 'quant' && c.quant.label === 'UD-IQ3_S') && f.verdict === 'runs')).toBe(true);
  });

  it('5 — 512 GB Studio, Kimi K3: a dash at the 2-bit floor; a 1.3-bit build would load at 92 %, tight', () => {
    const kimi = loadModel('kimi-k3');
    const cell = evaluateCell(kimi, ultra, s(512, { contextTokens: 131072, override: 1 }), factors);
    expect(cell.verdict).toBe('no-fit');
    expect(cell.belowFloor?.changes.some((c) => c.kind === 'quant' && c.quant.label === 'UD-Q1_0')).toBe(true);
    expect(cell.belowFloor?.verdict).toBe('tight');
    expect(cell.belowFloor?.need.ratio).toBeCloseTo(0.917, 2);
    expect(cell.nearestMiss).toMatch(/UD-IQ2_XXS/);
    expect(cell.nearestMiss).toMatch(/520 GB at most/);
    const relaxed = evaluateCell(kimi, ultra, s(512, { contextTokens: 131072, override: 1, qualityFloorBits: 1 }), factors);
    expect(relaxed.verdict).toBe('compromise');
  });

  it('6 — 64 GB MacBook Pro, Flash Next at Q4: the engram-on-SSD build is the fix; three quarters of 68.7 GB is enough without the override', () => {
    const fn = loadModel('qwen3.8-flash-next');
    const cell = evaluateCell(fn, mbp, s(64), factors);
    expect(cell.verdict).toBe('compromise');
    expect(cell.fix?.changes.map((c) => c.kind)).toEqual(['ssdPaged']);
    expect(cell.fix?.changes[0]).toMatchObject({ kind: 'ssdPaged', build: { label: 'IQ4_XS · engram on SSD' } }); // the smallest resident part wins
    expect(cell.fix?.verdict).toBe('tight');
    expect(cell.fix?.need.ratio).toBeCloseTo(0.9434, 3);
    expect(cell.fix?.need.ssdGb).toBe(39.1);
    expect(cell.fix?.reason).toBe('with the engram-on-SSD build (45.8 GB in memory, 39.1 GB streamed from the SSD); tight: under 10 % headroom');
    expect(cell.speed.tokS).toBeNull(); // one published run on one chip: quoted, never used as a number
    expect(cell.speed.notes[0]).toMatch(/^the publisher measured 36 tok\/s on an M5 Max \(MacBook Pro M5 Max 64 GB\); no speed is estimated/);
    const studio = evaluateCell(fn, machineById('mac-studio-m5-ultra'), s(256, { quant: undefined }), factors);
    expect(studio.verdict).toBe('runs'); // the regular Q4 file fits a 256 GB Studio; the special build is not consulted
    const withApps = evaluateCell(fn, mbp, s(64, { workApps: 16 }), factors);
    expect(withApps.verdict).toBe('compromise');
    expect(withApps.fix?.changes.map((c) => c.kind)).toContain('closeApps');
  });

  it('6b — the engram-on-SSD build asked for directly: a tight fit with no fix, the SSD part beside the bar, a flag, no speed', () => {
    const fn = loadModel('qwen3.8-flash-next');
    const [iq4, q4km] = fn.special_builds!;
    const cell = evaluateCell(fn, mbp, s(64, { quant: iq4 }), factors);
    expect(cell.verdict).toBe('tight');
    expect(cell.fix).toBeNull();
    expect(cell.quant).toBe(iq4);
    expect(cell.need.weightsGb).toBe(45.8);
    expect(cell.need.ssdGb).toBe(39.1);
    expect(cell.breakdown.ssdGb).toBe(39.1);
    expect(cell.breakdown.weights + cell.breakdown.cache + cell.breakdown.buffers).toBeCloseTo(cell.need.totalGb, 6); // the SSD part is not in the bar
    expect(cell.flags).toContain('ssd-build');
    expect(cell.speed.tokS).toBeNull();
    // the README's recommended file keeps 54.5 GB in memory: over the default limit on 64 GB, so the fix lifts it
    const rec = evaluateCell(fn, mbp, s(64, { quant: q4km }), factors);
    expect(rec.verdict).toBe('compromise');
    expect(rec.fix?.changes.map((c) => c.kind)).toEqual(['override']);
    expect(rec.need.ssdGb).toBe(38.4);
    // a plain file carries no SSD part
    expect(evaluateCell(fn, machineById('mac-studio-m5-ultra'), s(256), factors).need.ssdGb).toBe(0);
    // a 48 GB machine cannot hold even the smallest resident part: the nearest miss names it in quotes
    const small = evaluateCell(fn, mbp, s(48, { quant: iq4 }), factors);
    expect(small.verdict).toBe('no-fit');
    expect(small.nearestMiss).toMatch(/^the smallest allowed build, "IQ4_XS · engram on SSD" \(45\.8 GB in memory\), needs about/);
  });

  it('7 — DGX Spark, DeepSeek V4 Flash 0731 Q4 at 128K: a 3-bit build runs; the 2-bit build is the alternative', () => {
    const ds = loadModel('deepseek-v4-flash-0731');
    const cell = evaluateCell(ds, spark, s(128, { contextTokens: 131072 }), factors);
    expect(cell.quant && 'label' in cell.quant && cell.quant.label).toBe('UD-Q4_K_XL');
    expect(cell.availability.availableGb).toBeCloseTo(129.44, 2);
    expect(cell.verdict).toBe('compromise');
    const q = cell.fix!.changes.find((c) => c.kind === 'quant');
    expect(q && q.kind === 'quant' && q.quant.label).toBe('UD-IQ3_XXS');
    expect(cell.fix?.verdict).toBe('runs');
    expect(cell.fix?.need.ratio).toBeCloseTo(0.8749, 3);
    expect(cell.alternatives.some((f) => f.verdict === 'runs' && f.changes.some((c) => c.kind === 'quant' && c.quant.label === 'Q2_K_S'))).toBe(true);
    expect(cell.speed.archClass).toBe('deepseek_v4');
    expect(cell.speed.tokS!).toBeGreaterThan(10);
    expect(cell.speed.tokS!).toBeLessThan(13);
  });

  it('9 — Gemma 4 26B-A4B cache at 128K (K and V both stored), 4-bit halves only the growing and sliding terms', () => {
    const gemma = loadModel('gemma-4-26b-a4b');
    expect(kvCacheBytes(gemma.architecture, 131072, 8) / 1e9).toBeCloseTo(1.447, 3);
    expect(kvCacheBytes(gemma.architecture, 131072, 4) / 1e9).toBeCloseTo(0.7235, 3);
    const cell = evaluateCell(gemma, mini, s(32, { kvBits: 4 }), factors);
    expect(cell.flags).not.toContain('kv-precision-note'); // grouped-query attention: 4-bit cache is routine
  });

  it('11 — every featured fixture across every current machine row evaluates without throwing, cheapest first', () => {
    const rows = catalogRows(loadMachines(), base);
    for (const id of ['qwen3.8-27b', 'glm-5.3-flash', 'glm-5.2', 'kimi-k3', 'deepseek-v4-flash-0731', 'qwen3.8-flash-next', 'gemma-4-26b-a4b', 'gpt-oss-20b']) {
      const results = evaluateModelAcrossMachines(loadModel(id), rows, factors);
      expect(results.length).toBe(rows.length);
      expect(results.every((r) => ['runs', 'tight', 'compromise', 'no-fit'].includes(r.verdict))).toBe(true);
      const priced = results.filter((r) => r.priceUsd !== null).map((r) => r.priceUsd!);
      expect(priced).toEqual([...priced].sort((a, b) => a - b));
      const firstNull = results.findIndex((r) => r.priceUsd === null);
      if (firstNull !== -1) expect(results.slice(firstNull).every((r) => r.priceUsd === null)).toBe(true);
    }
  });

  it('12 — a quick custom 70B model gets a conventional cache estimate and the assumed flag', () => {
    const m = customModelQuick({ name: 'My 70B', paramsTotalB: 70, weightsGb: 40 });
    expect(m.architecture?.layers).toBe(76);
    expect(kvCacheBytes(m.architecture, 32768, 8) / 1e9).toBeCloseTo(5.1, 1);
    const box = customMachine({ name: 'Studio 128', memoryGb: 128, bandwidthGbs: 614, platform: 'apple', chip: 'M5 Max' });
    const cell = evaluateCell(m, box, s(128), factors);
    expect(cell.verdict).toBe('runs');
    expect(cell.flags).toContain('assumed');
    expect(cell.speed.efficiencySource).toBe('measured');
  });

  it('gpt-oss-20b is 4-bit native: the Q8 file is flagged as a re-pack', () => {
    const g = loadModel('gpt-oss-20b');
    const q8 = defaultQuant(g, 'gguf', 'q8');
    expect(q8).not.toBeNull();
    const cell = evaluateCell(g, mini, s(32, { quant: q8! }), factors);
    expect(cell.flags).toContain('four-bit-native');
    expect(cell.flags).toContain('q8-is-repack');
  });

  it('invariants over the fixes', () => {
    const rows = catalogRows(loadMachines(), { ...base, workApps: 16, contextTokens: 131072 });
    for (const id of ['glm-5.3-flash', 'glm-5.2', 'deepseek-v4-flash-0731', 'qwen3.8-flash-next']) {
      for (const r of evaluateModelAcrossMachines(loadModel(id), rows, factors)) {
        if (!r.fix) continue;
        expect(r.fix.need.ratio).toBeLessThanOrEqual(1);
        expect(r.fix.contextTokens).toBeLessThanOrEqual(131072);
        for (const c of r.fix.changes) {
          if (c.kind === 'quant') expect(nominalBits(c.quant.label) ?? 0).toBeGreaterThanOrEqual(2);
          if (c.kind === 'override') expect(r.machine.platform).toBe('apple');
        }
      }
    }
  });
});
