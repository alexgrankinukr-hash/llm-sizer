import { readFileSync } from 'node:fs';
import { describe, expect, it } from 'vitest';
import { loadFactors, loadMachines, loadModel, machineById, quantByLabel } from '../engine/__fixtures__/load';
import { evaluateCell } from '../engine/index';
import { availableMemory } from '../engine/memory';
import { estimateSpeed, expertLayersOf, sparseBudgetOf } from '../engine/speed';
import type { Settings } from '../engine/types';
import { defaultState } from './defaults';
import { calibrationTable, catalogMachine, CHECK_TOLERANCE, missReason, offloadCeiling, pickRouteFile, routeEstimate, routesContext, routesTable, buildFactsOf, type RoutesFile } from './routes';
import { ROUTES_VIEW_DEFAULT } from './state';
import { decodeState, encodeState } from './url';

const file = JSON.parse(readFileSync(new URL('../../../../public/data/llm-sizer/routes-128gb.json', import.meta.url), 'utf8')) as RoutesFile;
const factors = loadFactors();
const machines = loadMachines();
const ctx = routesContext(file, factors, machines);
const route = (id: string) => {
  const r = file.routes.find((x) => x.id === id);
  if (!r) throw new Error(`no route ${id}`);
  return r;
};
const flash = loadModel('qwen3.8-flash-next');
const qwen27 = loadModel('qwen3.8-27b');
const oss = loadModel('gpt-oss-120b');
const DATE = /^\d{4}-\d{2}(-\d{2})?$/;

describe('the routes data file', () => {
  it('dates and sources every price, measurement and profile row', () => {
    expect(file.data_as_of).toMatch(DATE);
    expect(file.prices_checked).toMatch(DATE);
    for (const r of file.routes) {
      expect(r.price.as_of, r.id).toMatch(DATE);
      if (r.price.usd !== null) expect(r.price.sources.length, r.id).toBeGreaterThan(0);
      for (const b of r.boxes ?? []) {
        expect(b.as_of, `${r.id} ${b.name}`).toMatch(DATE);
        expect(b.source, `${r.id} ${b.name}`).toMatch(/^https:\/\//);
      }
      expect(r.memory_speed.gbs, r.id).toBeGreaterThan(0);
      expect(r.usable.gb, r.id).toBeGreaterThan(0);
      for (const m of r.measured) {
        expect(m.date, `${r.id} ${m.model_id}`).toMatch(DATE);
        expect(m.sources.length, `${r.id} ${m.model_id}`).toBeGreaterThan(0);
        for (const s of m.sources) expect(s, `${r.id} ${m.model_id}`).toMatch(/^https:\/\//);
        // a plain run of one of the comparison's models carries what the calibration table needs
        if (m.kind === 'plain' && file.featured_models.includes(m.model_id)) expect(m.facts, `${r.id} ${m.model_id}`).toBeDefined();
      }
    }
    for (const p of Object.values(file.profiles)) for (const row of p.rows) expect(row.source).toMatch(/^https:\/\//);
  });

  it('opens on the model the data file names, which is one of the quick picks', () => {
    expect(ROUTES_VIEW_DEFAULT.model).toBe(file.default_model);
    expect(file.featured_models).toContain(file.default_model);
    expect(file.contexts).toContain(ROUTES_VIEW_DEFAULT.ctx);
  });

  it('keeps the comparison models\' cache facts equal to their catalog records', () => {
    for (const m of [flash, qwen27, oss]) {
      const f = file.model_facts[m.id];
      expect(typeof f === 'object' && f.kv8).toBe(m.architecture?.full_bytes_per_token_8bit);
      expect(typeof f === 'object' ? (f.moe_layers ?? null) : 'missing', `${m.id} expert layers`).toBe(expertLayersOf(m));
      // the sparse-attention rule is keyed on the model type, so the runs' estimates need the catalog's
      expect(typeof f === 'object' && sparseBudgetOf(f)).toBe(sparseBudgetOf(m.architecture));
    }
  });

  it('gives a 128 GB Mac the memory the engine gives it on macOS 26', () => {
    for (const r of file.routes.filter((x) => x.speed.rule === 'engine' && x.group === 'mac')) {
      const m = machineById(r.speed.machine!);
      const a = availableMemory(m, { memoryGb: r.speed.memory_gb!, workApps: 0, override: false });
      expect(r.usable.gb, r.id).toBeCloseTo(a.availableGb, 1);
    }
    const ultra = route('mac-studio-m5-ultra');
    expect(ultra.usable.gb).toBeCloseTo(availableMemory(machineById('mac-studio-m5-ultra'), { memoryGb: 256, workApps: 0, override: false }).availableGb, 1);
    // the two linked 64 GB Sparks: the engine's pool rule
    const pair = route('two-sparks-64');
    expect(pair.usable.gb).toBeCloseTo(availableMemory(machineById('nvidia-dgx-spark'), { memoryGb: 64, workApps: 0, override: false, machines: 2, split: 'layer' }).availableGb, 1);
  });
});

describe('the estimate is the engine\'s own on catalog machines', () => {
  const settings = (gb: number, ctxTokens: number, runtime: Settings['runtime'], extra: Partial<Settings> = {}): Settings => ({ memoryGb: gb, workApps: 0, override: false, contextTokens: ctxTokens, kvBits: 16, runtime, qualityFloorBits: 2, ...extra });
  it('matches estimateSpeed for MLX on a Mac, llama.cpp on the Spark and a dense model on an Ultra', () => {
    const cases = [
      { r: 'mac-studio-m5-max', model: flash, quant: quantByLabel(flash, 'MLX-4bit'), gb: 128, runtime: 'mlx' as const },
      { r: 'mac-studio-m2-ultra', model: qwen27, quant: quantByLabel(qwen27, 'MLX-4bit', 'mlx-community/Qwen3.8-27B-4bit'), gb: 128, runtime: 'mlx' as const },
      { r: 'mac-studio-m4-max', model: oss, quant: quantByLabel(oss, 'MLX-4bit'), gb: 128, runtime: 'mlx' as const },
      { r: 'dgx-spark', model: oss, quant: quantByLabel(oss, 'MXFP4', 'ggml-org/gpt-oss-120b-GGUF'), gb: 128, runtime: 'gguf' as const },
    ];
    for (const c of cases) {
      const rt = route(c.r);
      const m = machineById(rt.speed.machine!);
      for (const t of [4096, 32768]) {
        const engine = estimateSpeed({ model: c.model, quant: c.quant, machine: m, contextTokens: t, kvBits: 16, factors }).tokS!;
        const ours = routeEstimate(rt, buildFactsOf(c.model, c.quant)!, t, 16, ctx)!.tokS;
        expect(ours, `${c.r} ${c.model.id} ${t}`).toBeCloseTo(engine, 6);
      }
    }
  });

  it('matches the engine\'s linked-machine rule for the two 64 GB Sparks', () => {
    const rt = route('two-sparks-64');
    const q = quantByLabel(oss, 'MXFP4', 'ggml-org/gpt-oss-120b-GGUF');
    const engine = estimateSpeed({ model: oss, quant: q, machine: machineById('nvidia-dgx-spark'), contextTokens: 4096, kvBits: 16, factors, cluster: { n: 2, split: 'layer' } }).tokS!;
    expect(routeEstimate(rt, buildFactsOf(oss, q)!, 4096, 16, ctx)!.tokS).toBeCloseTo(engine, 6);
  });
});

describe('route profiles', () => {
  it('fits Strix Halo on other models: about 80 % of its 256 GB/s, a few ms of fixed cost', () => {
    const p = ctx.profiles['strix-halo-vulkan'];
    expect(p.n).toBe(file.profiles['strix-halo-vulkan'].rows.length);
    expect(p.bEffRatio).toBeGreaterThan(0.7);
    expect(p.bEffRatio).toBeLessThan(0.9);
    expect(p.t0Ms).toBeGreaterThan(0);
    expect(p.t0Ms).toBeLessThan(5);
  });

  it('never fits a profile on the comparison\'s own models: they are its check', () => {
    const names = ['flash next', 'qwen 3.8 27b', 'gpt-oss-120b'];
    for (const spec of Object.values(file.profiles)) for (const row of spec.rows) expect(names.some((n) => row.model.toLowerCase().includes(n)), row.model).toBe(false);
  });
});

describe('the calibration table (METHODOLOGY §6.6)', () => {
  const table = calibrationTable(file, ctx);
  const row = (routeId: string, model: string) => {
    const r = table.find((x) => x.routeId === routeId && x.modelId === model);
    if (!r) throw new Error(`no row ${routeId} ${model}`);
    return r;
  };

  it('lands within 15 % on Flash Next on the M5 Max and the M4 Max, and reads the M2 Ultra low with a reason', () => {
    for (const id of ['mac-studio-m5-max', 'mac-studio-m4-max']) {
      const r = row(id, 'qwen3.8-flash-next');
      expect(r.status, id).toBe('within');
      expect(Math.abs(r.err!), id).toBeLessThanOrEqual(CHECK_TOLERANCE);
    }
    const m2 = row('mac-studio-m2-ultra', 'qwen3.8-flash-next');
    expect(m2.status).toBe('off');
    expect(m2.err!).toBeLessThan(-CHECK_TOLERANCE);
    expect(m2.reason).toMatch(/6\.67B/);
  });

  it('checks Strix Halo\'s profile within 15 % on gpt-oss-120b and Qwen 3.8 27B', () => {
    expect(row('strix-halo', 'gpt-oss-120b').status).toBe('within');
    expect(row('strix-halo', 'qwen3.8-27b').status).toBe('within');
  });

  it('reads the 48-core M1 Ultra high, as the trap it is, and shows why', () => {
    const r = row('mac-studio-m1-ultra', 'qwen3.8-flash-next');
    expect(r.status).toBe('off');
    expect(r.err!).toBeGreaterThan(0.3);
    expect(r.reason).toMatch(/800 GB\/s on paper/);
  });

  it('marks runs on another runtime instead of scoring them', () => {
    expect(row('dgx-spark', 'qwen3.8-flash-next').status).toBe('other-runtime');
    expect(row('rtx-pro-6000', 'qwen3.8-flash-next').status).toBe('other-runtime');
    expect(row('mac-studio-m5-ultra', 'qwen3.8-flash-next').status).toBe('unsettled');
  });

  it('has a written reason for every row that misses or ran elsewhere', () => {
    for (const r of table) if (r.status === 'off' || r.status === 'other-runtime') expect(r.reason, `${r.routeId} ${r.modelId}`).toBeTruthy();
  });
});

describe('one model across every route', () => {
  const rows = routesTable(file, flash, 'Q4', 4096, 16, ctx);
  const of = (id: string) => rows.find((r) => r.route.id === id)!;

  it('shows the published plain run over the estimate, and never a speed-up number as the headline', () => {
    const studio = of('mac-studio-m5-max');
    expect(studio.headline).toMatchObject({ kind: 'measured', tokS: 68.9 });
    expect(studio.check?.ok).toBe(true);
    // the laptop has the Studio's chip and the Studio's runs
    expect(of('macbook-pro-m5-max').headline.tokS).toBe(68.9);
    for (const model of [flash, qwen27, oss])
      for (const r of routesTable(file, model, 'Q4', 4096, 16, ctx)) {
        if (r.headline.kind === 'measured') expect(r.measured?.kind, `${model.id} ${r.route.id}`).toBe('plain');
        for (const s of r.speedups) expect(s.write === r.headline.tokS && r.measured === null, `${model.id} ${r.route.id}`).toBe(false);
      }
  });

  it('keeps the n-gram table out of memory on a Mac: the 4-bit MLX file needs about 80 GB of the 115', () => {
    const studio = of('mac-studio-m5-max');
    expect(studio.pick.quant?.format).toBe('mlx');
    expect(studio.pick.verdict).toBe('runs');
    expect(studio.pick.need!.totalGb).toBeGreaterThan(78);
    expect(studio.pick.need!.totalGb).toBeLessThan(85);
    expect(studio.pick.tableOutGb).toBeGreaterThan(25);
    // one rule: the fit table counts the same file the same way on the same machine
    const cell = evaluateCell(flash, catalogMachine(studio.route, ctx.machines)!, { memoryGb: 128, workApps: 0, override: false, contextTokens: 4096, kvBits: 16, runtime: 'mlx', qualityFloorBits: 2 }, factors);
    expect(cell.quant).toBe(studio.pick.quant);
    expect(cell.need.totalGb).toBeCloseTo(studio.pick.need!.totalGb, 6);
    expect(cell.need.ssdGb).toBeCloseTo(studio.pick.tableOutGb, 6);
    // llama.cpp on Strix Halo keeps the whole file: a smaller 4-bit file than the usual one is what fits with room
    const strix = of('strix-halo');
    expect(strix.pick.tableOutGb).toBe(0);
    expect(strix.pick.quant!.size_gb).toBeLessThan(110);
  });

  it('gives no estimate where no rule is calibrated, and says why', () => {
    for (const id of ['four-mi50', 'four-r9700']) {
      const r = of(id);
      expect(r.estimate).toBeNull();
      expect(r.noEstimate).toMatch(/No four-card run/);
    }
    // the Surface: 64 GB by default is too small for any 4-bit file of this model
    expect(of('surface-laptop-ultra').pick.verdict).toBe('no-fit');
    // the 64 GB gaming PC cannot hold the 4-bit copy; the 128 GB one can, with no estimate and a ceiling
    expect(of('gaming-pc-64').pick.verdict).toBe('no-fit');
    const pc = of('gaming-pc-128');
    expect(pc.estimate).toBeNull();
    expect(pc.noEstimate).toMatch(/No plain run/);
    expect(pc.ceiling).toBeGreaterThan(0);
  });

  it('shows the M5 Ultra\'s estimate, with its thin plain run kept in the detail', () => {
    const u = of('mac-studio-m5-ultra');
    expect(u.headline.kind).toBe('estimate');
    expect(u.unsettled?.write).toBe(42.3);
    expect(u.speedups.length).toBeGreaterThan(0);
  });

  it('takes the reading time only from a run that measured it', () => {
    expect(of('mac-studio-m5-max').reading?.seconds).toEqual([file.document.tokens / 2201, file.document.tokens / 2966]);
    expect(of('surface-laptop-ultra').reading).toBeNull();
    for (const r of routesTable(file, qwen27, 'Q4', 4096, 16, ctx)) expect(r.reading, r.route.id).toBeNull();
  });

  it('says why the estimate may be off on a route where a run of the same kind missed', () => {
    expect(missReason(route('mac-studio-m5-ultra'), 'moe', file.routes)).toMatch(/mxfp4-bf16/);
    expect(missReason(route('mac-studio-m5-ultra'), 'dense', file.routes)).toBeNull();
    // since 0.17 the M5 Max's gpt-oss-120b estimate lands within 15 %, and the route carries no note
    expect(missReason(route('mac-studio-m5-max'), 'moe', file.routes)).toBeNull();
    // the linked Sparks borrow the single Spark's reasons
    expect(missReason(route('two-sparks-64'), 'flash_next', file.routes)).toMatch(/vLLM/);
  });
});

describe('the file a route runs', () => {
  it('uses the usual file when it fits, and steps down within the build when it does not', () => {
    const studio = pickRouteFile(route('mac-studio-m5-max'), qwen27, 'Q4', 4096, 16);
    expect(studio.quant?.label).toBe('MLX-4bit');
    const spark = pickRouteFile(route('dgx-spark'), flash, 'Q4', 4096, 16);
    // the usual 4-bit GGUF (119 GB) does not fit the Spark's 120 GB with its cache and buffers
    expect(spark.quant!.size_gb).toBeLessThan(119);
    expect(['runs', 'tight']).toContain(spark.verdict);
  });

  it('caps an offload route by memory speed, below what the card alone would allow', () => {
    const pc = route('gaming-pc-128');
    const q = pickRouteFile(pc, oss, 'Q4', 4096, 16).quant!;
    const ceiling = offloadCeiling(pc, oss, q, 4096, 16)!;
    const cardOnly = 1000 / ((5.1 * q.bits!) / 8 / pc.speed.card!.bandwidth_gbs * 1000);
    expect(ceiling).toBeGreaterThan(10);
    expect(ceiling).toBeLessThan(cardOnly);
  });
});

describe('the share link', () => {
  it('round-trips the view and drops its defaults', () => {
    const s = { ...defaultState(), view: { kind: 'routes' as const, model: 'gpt-oss-120b', bucket: 'Q8' as const, ctx: 32768 } };
    const enc = encodeState(s);
    expect(enc).toContain('vroutes*gpt-oss-120b*Q8*32');
    expect(decodeState(enc, defaultState()).state.view).toEqual(s.view);
    expect(encodeState({ ...defaultState(), view: ROUTES_VIEW_DEFAULT })).toMatch(/~vroutes$/);
  });

  it('opens a link that names only the comparison over the default table', () => {
    const d = decodeState('1~vroutes', defaultState());
    expect(d.ok).toBe(true);
    expect(d.state.view).toEqual(ROUTES_VIEW_DEFAULT);
    expect(d.state.groups).toEqual(defaultState().groups);
  });
});
