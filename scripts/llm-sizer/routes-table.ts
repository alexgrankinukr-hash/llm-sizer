/**
 * Prints the 128 GB comparison's calibration table (METHODOLOGY §6.6) from public/data/llm-sizer/routes-128gb.json:
 * every published plain run against the tool's estimate for the same build, and the fitted route profiles. Run it after
 * a refit or a data refresh and paste the output into the section.
 *
 *   npx tsx scripts/llm-sizer/routes-table.ts
 */
import { readFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { calibrationTable, routesContext, type RoutesFile } from '../../src/lib/llm-sizer/app/routes';
import type { Factors, MachinesFile } from '../../src/lib/llm-sizer/engine/types';

const root = join(dirname(fileURLToPath(import.meta.url)), '..', '..');
const read = <T>(p: string) => JSON.parse(readFileSync(join(root, p), 'utf8')) as T;
const file = read<RoutesFile>('public/data/llm-sizer/routes-128gb.json');
const factors = read<Factors>('public/data/llm-sizer/factors.json');
const machines = read<MachinesFile>('public/data/llm-sizer/machines.json').machines;
const ctx = routesContext(file, factors, machines);
const MODEL: Record<string, string> = { 'qwen3.8-flash-next': 'Flash Next', 'qwen3.8-27b': 'Qwen 3.8 27B', 'gpt-oss-120b': 'gpt-oss-120b' };
const STATUS: Record<string, string> = { within: 'within 15 %', off: 'measured shown', 'other-runtime': 'other runtime: measured shown', unsettled: 'too thin to stand', 'no-estimate': 'no estimate' };
const n = (x: number) => (x >= 100 ? x.toFixed(0) : x.toFixed(1));

const out: string[] = [];
out.push('| Route | Model | Build, engine | Measured | Estimate | Error | Shown |');
out.push('|---|---|---|---:|---:|---:|---|');
for (const r of calibrationTable(file, ctx)) {
  const measured = r.range ? `${n(r.measured)} (${n(r.range[0])}–${n(r.range[1])})` : n(r.measured);
  const err = r.err === null ? '—' : `${r.err > 0 ? '+' : '−'}${Math.round(Math.abs(r.err) * 100)} %`;
  out.push(`| ${r.route}${r.variant ? `, ${r.variant}` : ''} | ${MODEL[r.modelId] ?? r.modelId} | ${r.build}, ${r.engine} | ${measured} | ${r.estimate === null ? '—' : n(r.estimate)} | ${err} | ${STATUS[r.status]} |`);
}
out.push('');
out.push('| Profile | Effective bandwidth | Share of spec | Fixed cost | Rows | Worst row |');
out.push('|---|---:|---:|---:|---:|---:|');
for (const [k, p] of Object.entries(ctx.profiles)) out.push(`| ${file.profiles[k].name} | ${Math.round(p.bEffGbs)} GB/s | ${p.bEffRatio.toFixed(2)} | ${p.t0Ms.toFixed(1)} ms${p.throughZero ? ' (through zero)' : ''} | ${p.n} | ${p.maxErrPct.toFixed(0)} % |`);
console.log(out.join('\n'));
