/**
 * Machine groups: the catalog stores one row per bandwidth bin (`mac-mini-m6-16gb` and `mac-mini-m6`,
 * `macbook-pro-m5-max-32c` and `-40c`); people think "Mac mini M6" and "MacBook Pro M5 Max". A group
 * merges rows that differ only by GPU bin, keeps every memory size mapped to its own catalog row (so
 * bandwidth and price stay exact), and carries the silhouette to draw on the rail.
 */
import { chipKey } from '../engine/speed';
import { tensorCapable } from '../engine/cluster';
import { customMachine, type CustomMachineInput } from '../engine/custom';
import type { ClusterFactors, ClusterSplit, ClusterTerm, Factors, FittedTerm, Machine, Platform, SpeedModel } from '../engine/types';

export type Silhouette = 'mini' | 'studio' | 'laptop' | 'imac' | 'macpro' | 'spark' | 'box';

export interface GroupRow {
  gb: number;
  machineId: string;
  machine: Machine;
  bandwidthGbs: number;
  priceUsd: number | null;
  /** set only when two rows in the group offer the same size (a bin label tells them apart) */
  binLabel?: string;
}

export interface MachineGroup {
  id: string;
  kind: Machine['kind'];
  family: string;
  chip: string;
  platform: Platform;
  status: 'current' | 'discontinued';
  year?: number;
  silhouette: Silhouette;
  rows: GroupRow[];
  custom?: boolean;
}

/** Linked machines: how many identical machines a group pools, and how the model is split across them (METHODOLOGY §4). */
export interface LinkedInfo {
  count: 2 | 3 | 4;
  split: ClusterSplit;
}

/** `groupId:gb`, plus `:x2` / `:x4t` when the row is a linked pool (the per-machine size stays in the key). */
export type RowKey = string;

export function rowKey(groupId: string, gb: number, linked: LinkedInfo | null = null): RowKey {
  return linked ? `${groupId}:${gb}:x${linked.count}${linked.split === 'tensor' ? 't' : ''}` : `${groupId}:${gb}`;
}

const LINKED_TAIL = /^x([234])(t?)$/;

export function parseRowKey(key: string): { groupId: string; gb: number; linked: LinkedInfo | null } | null {
  let rest = key;
  let linked: LinkedInfo | null = null;
  const tail = rest.lastIndexOf(':');
  if (tail === -1) return null;
  const m = LINKED_TAIL.exec(rest.slice(tail + 1));
  if (m) {
    linked = { count: Number(m[1]) as LinkedInfo['count'], split: m[2] ? 'tensor' : 'layer' };
    rest = rest.slice(0, tail);
  }
  const i = rest.lastIndexOf(':');
  if (i === -1) return null;
  const gb = Number(rest.slice(i + 1));
  return Number.isFinite(gb) ? { groupId: rest.slice(0, i), gb, linked } : null;
}

/** The pool's nominal size: the machine's size × the count. */
export function pooledGb(gb: number, linked: LinkedInfo | null): number {
  return linked ? gb * linked.count : gb;
}

/** The pool's list price: the machine's × the count, or null when the machine has none. */
export function pooledPrice(priceUsd: number | null, linked: LinkedInfo | null): number | null {
  return priceUsd === null ? null : linked ? priceUsd * linked.count : priceUsd;
}

export function slug(text: string): string {
  return text.toLowerCase().replace(/[^a-z0-9.]+/g, '-').replace(/^-+|-+$/g, '');
}

export function silhouetteFor(machine: Pick<Machine, 'kind' | 'family'>): Silhouette {
  const f = machine.family.toLowerCase();
  if (machine.kind !== 'mac') return f.includes('spark') ? 'spark' : 'box';
  if (f.includes('mini')) return 'mini';
  if (f.includes('studio')) return 'studio';
  if (f.includes('imac')) return 'imac';
  if (f.includes('mac pro')) return 'macpro';
  return 'laptop';
}

export function groupIdFor(machine: Pick<Machine, 'kind' | 'family' | 'chip'>): string {
  const chip = chipKey(machine.chip);
  return machine.kind === 'prebuilt' ? slug(machine.family) : slug(`${machine.family} ${chip}`);
}

/** Merge the catalog into UI groups; catalog order is preserved. */
export function machineGroups(catalog: Machine[]): MachineGroup[] {
  const groups = new Map<string, MachineGroup>();
  for (const m of catalog) {
    const id = groupIdFor(m);
    let g = groups.get(id);
    if (!g) {
      g = {
        id,
        kind: m.kind,
        family: m.family,
        chip: chipKey(m.chip),
        platform: m.platform,
        status: m.status === 'discontinued' ? 'discontinued' : 'current',
        year: m.year,
        silhouette: silhouetteFor(m),
        rows: [],
      };
      groups.set(id, g);
    } else {
      if (m.status !== 'discontinued') g.status = 'current';
      if (m.year && (!g.year || m.year > g.year)) g.year = m.year;
    }
    for (const gb of m.memory_options_gb) {
      g.rows.push({ gb, machineId: m.id, machine: m, bandwidthGbs: m.bandwidth_gbs, priceUsd: m.price_usd?.[String(gb)] ?? null });
    }
  }
  for (const g of groups.values()) {
    g.rows.sort((a, b) => a.gb - b.gb || a.bandwidthGbs - b.bandwidthGbs);
    const seen = new Map<number, number>();
    for (const r of g.rows) seen.set(r.gb, (seen.get(r.gb) ?? 0) + 1);
    for (const r of g.rows) {
      if ((seen.get(r.gb) ?? 0) > 1) {
        const bin = /\(([^)]+)\)/.exec(r.machine.chip)?.[1];
        r.binLabel = bin ?? `${r.bandwidthGbs} GB/s`;
      }
    }
  }
  return [...groups.values()];
}

export function groupById(groups: MachineGroup[], id: string): MachineGroup | undefined {
  return groups.find((g) => g.id === id);
}

export function findRow(groups: MachineGroup[], key: string): { group: MachineGroup; row: GroupRow; linked: LinkedInfo | null } | null {
  const parsed = parseRowKey(key);
  if (!parsed) return null;
  const group = groupById(groups, parsed.groupId);
  const row = group?.rows.find((r) => r.gb === parsed.gb);
  return group && row ? { group, row, linked: parsed.linked } : null;
}

/** A machine the user typed in, as a one-row group. */
export function customGroup(input: CustomMachineInput): MachineGroup {
  const machine = customMachine(input);
  return {
    id: machine.id,
    kind: machine.kind,
    family: machine.family,
    chip: machine.chip,
    platform: machine.platform,
    status: 'current',
    silhouette: machine.platform === 'apple' ? 'laptop' : 'box',
    rows: [{ gb: input.memoryGb, machineId: machine.id, machine, bandwidthGbs: input.bandwidthGbs, priceUsd: null }],
    custom: true,
  };
}

/** The label the rail and pickers show: "Mac mini · M6". */
export function groupLabel(group: Pick<MachineGroup, 'family' | 'chip' | 'kind'>): string {
  return group.kind === 'prebuilt' || group.family === group.chip ? group.family : `${group.family} · ${group.chip}`;
}

/** The rail label with the count: "2 × Mac Studio · M5 Max". */
export function linkedLabel(group: Pick<MachineGroup, 'family' | 'chip' | 'kind'>, linked: LinkedInfo | null): string {
  return linked ? `${linked.count} × ${groupLabel(group)}` : groupLabel(group);
}

/** One row in words: "Mac Studio · M5 Max 128 GB" or "2 × NVIDIA DGX Spark · 128 GB each · 256 GB pooled". */
export function rowLabel(group: Pick<MachineGroup, 'family' | 'chip' | 'kind'>, row: Pick<GroupRow, 'gb' | 'binLabel'>, linked: LinkedInfo | null): string {
  const bin = row.binLabel ? ` · ${row.binLabel}` : '';
  if (!linked) return `${groupLabel(group)}${bin} ${row.gb} GB`;
  return `${linked.count} × ${groupLabel(group)}${bin} · ${row.gb} GB each · ${pooledGb(row.gb, linked)} GB pooled`;
}

/** Whether tensor parallel is on offer for the machines of a group (a fast link: Thunderbolt 5, the Spark's ports, PCIe). */
export function groupTensorCapable(group: Pick<MachineGroup, 'rows'>): boolean {
  return group.rows.length > 0 && tensorCapable(group.rows[0].machine);
}

/** The machine fields the island reads. `notes`, `sources` and the rest stay on the server (the About page shows them). */
export const ISLAND_MACHINE_FIELDS = ['id', 'kind', 'family', 'chip', 'year', 'status', 'platform', 'memory_options_gb', 'bandwidth_gbs', 'price_usd', 'gpu_cores', 'gpu_cores_by_gb', 'gpu_upgrade_usd', 'interconnect'] as const;

export function slimMachine(m: Machine): Machine {
  const out: Record<string, unknown> = {};
  for (const k of ISLAND_MACHINE_FIELDS) if (m[k] !== undefined) out[k] = m[k];
  return out as unknown as Machine;
}

function slimTerm(t: ClusterTerm): ClusterTerm {
  return { value: t.value, ...(t.n !== undefined ? { n: t.n } : {}), ...(t.source ? { source: t.source } : {}), ...(t.range ? { range: t.range } : {}) };
}

function slimCluster(c: ClusterFactors): ClusterFactors {
  return {
    hop_ms: { mac: slimTerm(c.hop_ms.mac), spark: slimTerm(c.hop_ms.spark), gpu: slimTerm(c.hop_ms.gpu) },
    tensor_c_ms: { dense: slimTerm(c.tensor_c_ms.dense), moe: slimTerm(c.tensor_c_ms.moe), gpu: slimTerm(c.tensor_c_ms.gpu) },
    prefill: { layer_factor: slimTerm(c.prefill.layer_factor), tensor_exponent: slimTerm(c.prefill.tensor_exponent) },
    dispatch_caveat: { min_layers: c.dispatch_caveat.min_layers },
  };
}

/** The factor fields the engine reads: values, tiers, ranges the UI shows; notes, examples, sources and prefill stay on the server. */
export function slimFactors(f: Factors): Factors {
  type Entry = Factors['efficiency']['by_chip'][string];
  const entry = (e: Entry): Entry => ({
    value: e.value,
    measured: e.measured,
    tier: e.tier,
    ...(e.n !== undefined ? { n: e.n } : {}),
    ...(e.range ? { range: e.range } : {}),
    // fixed-cost profile (absent in factor files from before the floor model)
    ...(e.t0_ms !== undefined ? { t0_ms: e.t0_ms } : {}),
    ...(e.b_eff_ratio !== undefined ? { b_eff_ratio: e.b_eff_ratio } : {}),
    ...(e.b_eff_gbs !== undefined ? { b_eff_gbs: e.b_eff_gbs } : {}),
    ...(e.fit_n !== undefined ? { fit_n: e.fit_n } : {}),
  });
  const mapEntries = (rec: Record<string, Entry>) => Object.fromEntries(Object.entries(rec).map(([k, v]) => [k, entry(v)]));
  const acceleration: Factors['acceleration'] = {};
  for (const [k, v] of Object.entries(f.acceleration)) if (typeof v === 'object' && v) acceleration[k] = { min: v.min, median: v.median, max: v.max, n: v.n };
  const term = (t: FittedTerm): FittedTerm => ({ value: t.value, source: t.source, ...(t.n !== undefined ? { n: t.n } : {}), ...(t.range ? { range: t.range } : {}), ...(t.platforms ? { platforms: t.platforms } : {}) });
  const terms = (rec: Record<string, FittedTerm>) => Object.fromEntries(Object.entries(rec).map(([k, v]) => [k, term(v)]));
  const sm = f.speed_model;
  const speedModel: SpeedModel | undefined = sm
    ? {
        kind: sm.kind,
        read_factor: terms(sm.read_factor),
        architecture_cost_ms: terms(sm.architecture_cost_ms),
        attention_ms_per_32k: terms(sm.attention_ms_per_32k),
        mlx: { overhead_factor: terms(sm.mlx.overhead_factor), attention_ms_per_32k: terms(sm.mlx.attention_ms_per_32k) as SpeedModel['mlx']['attention_ms_per_32k'] },
        ...(sm.validation
          ? {
              validation: {
                n_train: sm.validation.n_train,
                n_holdout: sm.validation.n_holdout,
                ...(sm.validation.holdout_median_err !== undefined ? { holdout_median_err: sm.validation.holdout_median_err } : {}),
                ...(sm.validation.holdout_p90_err !== undefined ? { holdout_p90_err: sm.validation.holdout_p90_err } : {}),
                ...(sm.validation.train_median_err !== undefined ? { train_median_err: sm.validation.train_median_err } : {}),
                ...(sm.validation.gate ? { gate: { passed: sm.validation.gate.passed } } : {}),
              },
            }
          : {}),
      }
    : undefined;
  return {
    schema: f.schema,
    data_as_of: f.data_as_of,
    efficiency: {
      by_chip: mapEntries(f.efficiency.by_chip),
      assumed: mapEntries(f.efficiency.assumed),
      by_tier: Object.fromEntries(
        Object.entries(f.efficiency.by_tier).map(([k, v]) => [
          k,
          { value: v.value, n_chips: v.n_chips, range: v.range, ...(v.t0_ms !== undefined ? { t0_ms: v.t0_ms } : {}), ...(v.b_eff_ratio !== undefined ? { b_eff_ratio: v.b_eff_ratio } : {}) },
        ]),
      ),
      ...(f.efficiency.m5_generation_lift !== undefined ? { m5_generation_lift: f.efficiency.m5_generation_lift } : {}),
    },
    ...(speedModel ? { speed_model: speedModel } : {}),
    // prefill terms, without the examples and prose (absent in factor files from before 0.12)
    ...(f.prefill
      ? {
          prefill: {
            per_core: f.prefill.per_core,
            by_chip: f.prefill.by_chip,
            chips_with_rows: f.prefill.chips_with_rows,
            kquant: { factor: f.prefill.kquant.factor, n: f.prefill.kquant.n },
            class: Object.fromEntries(Object.entries(f.prefill.class).map(([k, v]) => [k, { factor: v.factor, ...(v.n !== undefined ? { n: v.n } : {}) }])),
            class_by_model: f.prefill.class_by_model,
            mlx: { pre_m5: { factor: f.prefill.mlx.pre_m5.factor, n: f.prefill.mlx.pre_m5.n }, m5plus: { factor: f.prefill.mlx.m5plus.factor, n: f.prefill.mlx.m5plus.n } },
            context: { d0_tokens: f.prefill.context.d0_tokens, n_points: f.prefill.context.n_points, range: f.prefill.context.range, attention_share_ref: f.prefill.context.attention_share_ref },
            waits_tokens: f.prefill.waits_tokens,
            feels_like: f.prefill.feels_like,
          },
        }
      : {}),
    // linked-machine terms, values and their evidence only (absent in factor files from before 0.13)
    ...(f.cluster ? { cluster: slimCluster(f.cluster) } : {}),
    runtime: {
      gguf: { factor: f.runtime.gguf.factor },
      mlx: {
        dense_under_14b: { factor: f.runtime.mlx.dense_under_14b.factor },
        dense_14b_and_up: { factor: f.runtime.mlx.dense_14b_and_up.factor },
        moe: { factor: f.runtime.mlx.moe.factor },
        flatten_above_context: f.runtime.mlx.flatten_above_context,
        ...(f.runtime.mlx.measured_here !== undefined ? { measured_here: f.runtime.mlx.measured_here } : {}),
      },
    },
    acceleration,
    feels_like: f.feels_like.map((b) => ({ max_tok_s: b.max_tok_s, label: b.label, ...(b.description ? { description: b.description } : {}) })),
  };
}
