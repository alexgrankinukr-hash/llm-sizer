/**
 * What changes between the multiplier speed model + the 0.8 memory policy and the fixed-cost model + the policy in
 * `MEMORY_POLICY` (METHODOLOGY §6.1, §4, §5). Prints every verdict or speed-band change for the default first screen
 * (work apps off and on) and for a sample of machine × model × context cells. Model records come from the live API
 * and are cached under scripts/llm-sizer/.cache/.
 *
 *   npx tsx scripts/llm-sizer/verdict-diff.ts [--base http://localhost:4321] [--policy-only] [--speed-only]
 */
import { existsSync, mkdirSync, readFileSync, writeFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { evaluateCell } from '../../src/lib/llm-sizer/engine/index';
import { MEMORY_POLICY, MEMORY_POLICY_V08 } from '../../src/lib/llm-sizer/engine/constants';
import type { CellResult, Factors, Machine, MachinesFile, ModelDetail, Settings } from '../../src/lib/llm-sizer/engine/types';

const here = dirname(fileURLToPath(import.meta.url));
const root = join(here, '..', '..');
const cache = join(here, '.cache');
const args = process.argv.slice(2);
const base = args.includes('--base') ? args[args.indexOf('--base') + 1] : 'http://localhost:4321';
const policyOnly = args.includes('--policy-only');
const speedOnly = args.includes('--speed-only');

const FIRST_SCREEN_MODELS = ['glm-5.3-flash', 'qwen3.8-27b', 'gpt-oss-20b', 'gemma-4-26b-a4b', 'qwen3.8-flash-next', 'deepseek-v4-flash-0731', 'glm-5.2', 'kimi-k3'];
const SAMPLE_MODELS = [...FIRST_SCREEN_MODELS, 'mistral-small-4-119b-2603', 'gpt-oss-120b'];
const CONTEXTS = [8192, 32768, 131072];

async function model(id: string): Promise<ModelDetail | null> {
  const f = join(cache, `${id}.json`);
  if (existsSync(f)) return JSON.parse(readFileSync(f, 'utf8')) as ModelDetail;
  const r = await fetch(`${base}/api/llm-sizer/data/models/${id}.json`);
  if (!r.ok) {
    console.error(`skip ${id}: ${r.status}`);
    return null;
  }
  const j = (await r.json()) as ModelDetail;
  mkdirSync(cache, { recursive: true });
  writeFileSync(f, JSON.stringify(j));
  return j;
}

const factors = JSON.parse(readFileSync(join(root, 'public/data/llm-sizer/factors.json'), 'utf8')) as Factors;
const machines = (JSON.parse(readFileSync(join(root, 'public/data/llm-sizer/machines.json'), 'utf8')) as MachinesFile).machines;
const multiplier: Factors = { ...factors, speed_model: factors.speed_model ? { ...factors.speed_model, kind: 'multiplier' } : undefined };
const floor: Factors = { ...factors, speed_model: factors.speed_model ? { ...factors.speed_model, kind: 'floor' } : undefined };
const before = { factors: speedOnly ? multiplier : multiplier, policy: policyOnly ? MEMORY_POLICY_V08 : MEMORY_POLICY_V08 };
const after = { factors: policyOnly ? multiplier : floor, policy: speedOnly ? MEMORY_POLICY_V08 : MEMORY_POLICY };

function band(r: CellResult): string {
  return r.speed.feelsLike?.label ?? (r.speed.tokS === null ? 'no speed' : '?');
}
function label(r: CellResult): string {
  const fix = r.fix ? ` via ${r.fix.changes.map((c) => (c.kind === 'quant' ? c.quant.label : c.kind === 'context' ? `${c.tokens / 1024}K` : c.kind === 'ssdPaged' ? 'ssd build' : c.kind)).join('+')} (${r.fix.verdict})` : '';
  return `${r.verdict}${fix}${r.quant ? ` ${r.quant.label}` : ''}`;
}
function tok(r: CellResult): string {
  return r.speed.tokS === null ? 'n/a' : r.speed.tokS.toFixed(1);
}

interface Change {
  cell: string;
  before: string;
  after: string;
  tokBefore: string;
  tokAfter: string;
  kind: 'verdict' | 'band' | 'speed';
}

async function main() {
  const models = (await Promise.all(SAMPLE_MODELS.map(model))).filter((m): m is ModelDetail => !!m);
  const changes: Change[] = [];
  let cells = 0;
  const counts = new Map<string, number>();
  const bump = (k: string) => counts.set(k, (counts.get(k) ?? 0) + 1);
  const ratios: number[] = [];
  const settingsFor = (memoryGb: number, contextTokens: number, workApps: number): Settings => ({ memoryGb, workApps, override: false, contextTokens, kvBits: 16, runtime: 'gguf', qualityFloorBits: 2 });
  const rows: { machine: Machine; gb: number }[] = [];
  for (const m of machines) for (const gb of m.memory_options_gb) rows.push({ machine: m, gb });
  const firstScreen = FIRST_SCREEN_MODELS.map((id) => models.find((m) => m.id === id)).filter((m): m is ModelDetail => !!m);

  const run = (scope: string, rowsIn: typeof rows, modelsIn: ModelDetail[], contexts: number[], workApps: number[]) => {
    for (const { machine, gb } of rowsIn) {
      for (const md of modelsIn) {
        for (const ctx of contexts) {
          for (const work of workApps) {
            cells++;
            const a = evaluateCell(md, machine, settingsFor(gb, ctx, work), before.factors, before.policy);
            const b = evaluateCell(md, machine, settingsFor(gb, ctx, work), after.factors, after.policy);
            const cell = `${scope} · ${machine.id} ${gb} GB · ${md.id} · ${ctx / 1024}K${work ? ' · work on' : ''}`;
            if (label(a) !== label(b)) {
              changes.push({ cell, before: label(a), after: label(b), tokBefore: tok(a), tokAfter: tok(b), kind: 'verdict' });
              bump(`verdict ${a.verdict} → ${b.verdict}`);
            } else if (band(a) !== band(b)) {
              changes.push({ cell, before: band(a), after: band(b), tokBefore: tok(a), tokAfter: tok(b), kind: 'band' });
              bump(`band ${band(a)} → ${band(b)}`);
            }
            if (a.speed.tokS && b.speed.tokS) ratios.push(b.speed.tokS / a.speed.tokS);
          }
        }
      }
    }
  };
  run('first screen', rows, firstScreen, [32768], [0, 16]);
  run('sample', rows.filter((_, i) => i % 2 === 0), models, CONTEXTS, [0]);

  ratios.sort((x, y) => x - y);
  const q = (p: number) => ratios[Math.min(ratios.length - 1, Math.floor(p * ratios.length))];
  console.log(`# Verdict diff: multiplier + policy 0.8 → ${after.factors.speed_model?.kind ?? 'multiplier'} + policy in force\n`);
  console.log(`cells compared: ${cells}; verdict or band changes: ${changes.length}`);
  console.log(`speed ratio after/before over ${ratios.length} cells with a number: p10 ${q(0.1).toFixed(2)} · median ${q(0.5).toFixed(2)} · p90 ${q(0.9).toFixed(2)}\n`);
  console.log('## Counts by direction\n');
  for (const [k, v] of [...counts.entries()].sort((x, y) => y[1] - x[1])) console.log(`- ${k}: ${v}`);
  console.log('\n## Every change\n');
  console.log('| cell | before | after | tok/s before | tok/s after |');
  console.log('|---|---|---|---|---|');
  for (const c of changes) console.log(`| ${c.cell} | ${c.before} | ${c.after} | ${c.tokBefore} | ${c.tokAfter} |`);
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});
