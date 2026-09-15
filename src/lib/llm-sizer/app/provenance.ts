/**
 * Where every number in a cell comes from: links to the Hugging Face repos, to the machine's sources row and
 * to the section of the About page (METHODOLOGY.md rendered) that computes it. Pure; used by the sheet and the About page.
 */
import { isSpecialBuild } from '../engine/memory';
import { chipKey, resolveChipProfile, resolveEfficiency, speedModelKind } from '../engine/speed';
import type { Factors, Machine, ModelDetail, Quant, SpecialBuild, Speed } from '../engine/types';

export const ABOUT_PATH = '/tools/llm-sizer/about';

/** Anchors on the About page: github-slugger slugs of METHODOLOGY.md headings (tested against the file) plus ids the page writes itself. */
export const ABOUT_ANCHORS = {
  reading: 'how-to-read-the-table',
  legend: 'what-kind-of-number-is-this',
  data: '3-where-the-data-comes-from-and-how-fresh-it-is',
  available: '4-how-much-memory-is-actually-available-for-ai',
  needed: '5-how-much-memory-a-model-needs',
  speed: '6-speed',
  decode: '61-writing-speed-decode--the-number-that-matters-day-to-day',
  runtime: '62-runtime-gguf-llamacpp--lm-studio-vs-mlx',
  acceleration: '63-acceleration-mtp-and-draft-models-dflash--dspark',
  feelsLike: '65-feels-like--the-translation-next-to-every-speed',
  otherMachines: '7-other-machines-the-same-math-different-facts',
  limitations: '8-known-limitations-read-before-arguing-with-a-number',
  examples: '9-worked-examples',
  changelog: '10-changelog',
  correct: '11-how-to-correct-us',
  // page-owned sections
  faq: 'faq',
  methodology: 'methodology',
  sources: 'sources',
  machines: 'sources-machines',
  benchmarks: 'sources-benchmarks',
  efficiency: 'sources-efficiency',
} as const;
export type AboutAnchor = keyof typeof ABOUT_ANCHORS;
export const PAGE_OWNED_ANCHORS: ReadonlySet<AboutAnchor> = new Set(['faq', 'methodology', 'sources', 'machines', 'benchmarks', 'efficiency']);

export function aboutHref(key: AboutAnchor): string {
  return `${ABOUT_PATH}#${ABOUT_ANCHORS[key]}`;
}

export function machineAnchor(machineId: string): string {
  return `machine-${machineId}`;
}

export interface SourceLink {
  label: string;
  href: string;
  external: boolean;
}

const HF = 'https://huggingface.co/';

function fmt(v: number): string {
  return v.toFixed(2);
}

/**
 * One sentence saying where this machine's speed number comes from. Under the fixed-cost model `value` is the
 * effective bandwidth in GB/s; under the multiplier model it is the efficiency factor.
 */
export function efficiencyProvenance(machine: Pick<Machine, 'platform' | 'chip' | 'family' | 'bandwidth_gbs'>, factors: Factors): { text: string; source: Speed['efficiencySource']; value: number | null } {
  const chip = chipKey(machine.chip);
  if (speedModelKind(factors) === 'floor') {
    const p = resolveChipProfile(machine, factors);
    if (!p) return { text: 'no speed profile for this platform yet', source: 'none', value: null };
    const pct = `${Math.round(p.bEffRatio * 100)} % of ${Math.round(machine.bandwidth_gbs)} GB/s`;
    const gbs = `${Math.round(p.bEffGbs)} GB/s effective`;
    const t0 = `${p.t0Ms.toFixed(1)} ms per token`;
    if (p.source === 'measured' && p.key) {
      const n = p.fitN ?? 0;
      return { text: `measured profile: ${gbs} (${pct}) and ${t0} (${p.key}, fitted from ${n} ${n === 1 ? 'row' : 'rows'})`, source: 'measured', value: p.bEffGbs };
    }
    if (p.source === 'assumed' && p.key) {
      const note = factors.efficiency.assumed[p.key]?.note ?? factors.efficiency.by_chip[p.key]?.note;
      return { text: `assumed profile: ${gbs} (${pct}) and ${t0} (${p.key}${note ? `: ${note}` : ''})`, source: 'assumed', value: p.bEffGbs };
    }
    if (p.source === 'tier' && p.key) {
      const t = factors.efficiency.by_tier[p.key];
      const chips = t?.n_chips ? `median of ${t.n_chips} measured chips` : 'tier median';
      return { text: `${p.key}-tier estimate: ${gbs} (${pct}) and ${t0} (${chips}; no measured ${chip} row yet)`, source: 'tier', value: p.bEffGbs };
    }
    return { text: 'no speed profile for this platform yet', source: 'none', value: null };
  }
  const r = resolveEfficiency(machine, factors);
  if (r.source === 'measured' && r.key && r.value !== null) {
    const e = factors.efficiency.by_chip[r.key];
    const n = e?.n ?? 1;
    const range = e?.range && e.range[0] !== e.range[1] ? `, ${fmt(e.range[0])}–${fmt(e.range[1])}` : '';
    return { text: `measured efficiency ${fmt(r.value)} (${r.key}, ${n} ${n === 1 ? 'row' : 'rows'}${range})`, source: 'measured', value: r.value };
  }
  if (r.source === 'assumed' && r.key && r.value !== null) {
    const note = factors.efficiency.assumed[r.key]?.note ?? factors.efficiency.by_chip[r.key]?.note;
    return { text: `assumed efficiency ${fmt(r.value)} (${r.key}${note ? `: ${note}` : ''})`, source: 'assumed', value: r.value };
  }
  if (r.source === 'tier' && r.key && r.value !== null) {
    const t = factors.efficiency.by_tier[r.key];
    const chips = t?.n_chips ? `median of ${t.n_chips} measured chips` : 'tier median';
    return { text: `${r.key}-tier estimate ${fmt(r.value)} (${chips}; no measured ${chip} row yet)`, source: 'tier', value: r.value };
  }
  return { text: 'no speed factor for this platform yet', source: 'none', value: null };
}

export interface CellSources {
  model: SourceLink | null;
  weights: SourceLink | null;
  cache: SourceLink;
  available: SourceLink;
  speed: SourceLink;
  efficiency: SourceLink;
  machine: SourceLink | null;
}

function quantLink(quant: Quant | SpecialBuild | null): SourceLink | null {
  if (!quant) return null;
  if (isSpecialBuild(quant)) return quant.url ? { label: `${quant.label} (community build)`, href: quant.url, external: true } : null;
  return { label: `${quant.label} · ${quant.quantizer ?? quant.repo.split('/')[0]} on Hugging Face`, href: quant.url ?? `${HF}${quant.repo}`, external: true };
}

/** Everything a cell's numbers trace to. */
export function cellSources(a: { model: ModelDetail; quant: Quant | SpecialBuild | null; machine: Machine; factors: Factors }): CellSources {
  const { model, machine } = a;
  return {
    model: model.custom ? null : { label: `${model.name} on Hugging Face`, href: model.url ?? `${HF}${model.hf_repo}`, external: true },
    weights: model.custom ? null : quantLink(a.quant),
    cache: { label: 'how the context cache is computed', href: aboutHref('needed'), external: false },
    available: { label: 'how available memory is computed', href: aboutHref('available'), external: false },
    speed: { label: 'how speed is estimated', href: aboutHref('decode'), external: false },
    efficiency: { label: 'speed profiles and their sources', href: aboutHref('efficiency'), external: false },
    machine: machine.custom ? null : { label: `${machine.family} ${machine.chip}: bandwidth, prices, sources`, href: `${ABOUT_PATH}#${machineAnchor(machine.id)}`, external: false },
  };
}
