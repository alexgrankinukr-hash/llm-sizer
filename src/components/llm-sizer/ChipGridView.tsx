/**
 * "How fast is every Mac?": one model on every Apple chip, generations across and tiers down, the way the video's
 * speed grids read. Writing speed or the wait before the first word; each chip at the biggest memory it was sold with
 * or the smallest that runs the model; tap a chip for every memory size it came in.
 */
import { useMemo, useState } from 'react';
import type { CellResult, Factors, ModelDetail, ModelIndex, Quant } from '../../lib/llm-sizer/engine/types';
import { contextChips } from '../../lib/llm-sizer/engine/index';
import { formatBandwidth, formatContext, formatPrice, formatTokS, formatWait } from '../../lib/llm-sizer/engine/format';
import { orderCatalog, fitSettingsKey } from '../../lib/llm-sizer/app/machine-fit';
import { cellId, chipBuckets, chipGrid, shownConfig, TIER_NAMES, waitBandIndex, writeBandIndex, type ChipCell, type ChipConfig, type ChipGrid } from '../../lib/llm-sizer/app/chip-grid';
import type { MachineGroup } from '../../lib/llm-sizer/app/machines';
import { customModelDetail, type MatrixColumn, type MatrixRow, type RecordState } from '../../lib/llm-sizer/app/matrix';
import type { Bucket } from '../../lib/llm-sizer/app/quants';
import type { AppState, View } from '../../lib/llm-sizer/app/state';
import type { Analytics } from '../../lib/llm-sizer/app/analytics';
import { useModelRecord } from './useModelRecords';
import { SlidersIcon, XIcon } from './icons';

export type ChipsView = Extract<View, { kind: 'chips' }>;

export interface ChipTarget {
  model: ModelDetail;
  pick: MatrixRow;
  bucket: Bucket;
  ctx: number;
  result: CellResult;
  quant: Quant | null;
}

export interface ChipGridViewProps {
  state: AppState;
  view: ChipsView;
  groups: MachineGroup[];
  factors: Factors;
  columns: MatrixColumn[];
  /** the records the table already holds, and the ones rendered with the page: read first, so the server and the first client render agree */
  records: Record<string, RecordState>;
  seed?: Record<string, ModelDetail>;
  index: ModelIndex | null;
  analytics: Analytics;
  onView: (view: ChipsView) => void;
  onOpenCell: (target: ChipTarget) => void;
  onOpenSettings: () => void;
}

/** Fills by band, slowest first, as in the video's grids; the ring is the same hue a step darker. */
const WRITE_FILL = ['#E7E3DC', '#F2E9CF', '#EBD896', '#DFC25E', '#D4AF37'];
const WRITE_RING = ['#CFC9BF', '#E6D7A8', '#DDC56F', '#D1B13F', '#B8912B'];
/** Wait bands run the other way: the shortest wait is the brightest. */
const READ_FILL = ['#D4AF37', '#DFC25E', '#EBD896', '#E7E3DC'];
const READ_RING = ['#B8912B', '#D1B13F', '#DDC56F', '#CFC9BF'];

function writeNumber(tokS: number): string {
  return tokS >= 10 ? String(Math.round(tokS)) : tokS.toFixed(1);
}

/** "4.7 min" → ["4.7", "min"]: the number big, the unit small. */
function waitParts(seconds: number): [string, string] {
  const [n, ...unit] = formatWait(seconds).split(' ');
  return [n, unit.join(' ')];
}

function liftedText(c: ChipConfig): string {
  const parts = [c.lifted.includes('override') ? 'limit lifted' : null, c.lifted.includes('closeApps') ? 'apps closed' : null].filter(Boolean);
  return parts.join(', ');
}

/** "an 8K-token prompt", "a 32K-token prompt" */
function promptPhrase(ctx: number, unit = '-token prompt'): string {
  const text = formatContext(ctx);
  return `${/^8/.test(text) ? 'an' : 'a'} ${text}${unit}`;
}

/** A band edge in seconds as the legend says it: "3 s", "90 s", "5 min". */
function edgeText(seconds: number): string {
  return seconds < 120 ? `${seconds} s` : `${Math.round(seconds / 60)} min`;
}

function runtimeName(runtime: AppState['runtime']): string {
  return runtime === 'mlx' ? 'MLX' : 'GGUF (llama.cpp)';
}

export function ChipGridView(props: ChipGridViewProps) {
  const { state, view, groups, factors, columns, index, analytics } = props;
  const reading = view.mode === 'read' && !!factors.prefill;

  // the model: a custom one from this browser, a record the table or the page already holds, or one fetched now
  const custom = view.model.startsWith('custom-') ? (state.customModels.map(customModelDetail).find((m) => m.id === view.model) ?? null) : null;
  const held = props.records[view.model];
  const preloaded = held?.status === 'ready' ? held.model : (props.seed?.[view.model] ?? null);
  const record = useModelRecord(custom || preloaded ? null : view.model);
  const model = custom ?? preloaded ?? (record?.status === 'ready' ? record.model : null);

  const buckets = useMemo(() => (model ? chipBuckets(model, state.runtime) : []), [model, state.runtime]);
  // a build the model has no file for (after a runtime or model switch) shows as Q4, or the first build it has
  const bucket: Bucket = buckets.includes(view.bucket) ? view.bucket : buckets.includes('Q4') ? 'Q4' : (buckets[0] ?? view.bucket);
  const chips = model ? contextChips(model) : [];
  const ctx = chips.includes(view.ctx) ? view.ctx : (chips.filter((c) => c <= view.ctx).pop() ?? chips[0] ?? view.ctx);
  const settingsKey = fitSettingsKey(state);
  const grid = useMemo(() => (model ? chipGrid(model, bucket, ctx, groups, state, factors) : null), [model, bucket, ctx, groups, factors, settingsKey]); // eslint-disable-line react-hooks/exhaustive-deps
  const [openChip, setOpenChip] = useState<string | null>(null);
  const open = openChip && grid ? (grid.cells.get(openChip) ?? null) : null;

  const set = (patch: Partial<ChipsView>, toggle?: [Parameters<Analytics['toggleChanged']>[0], string | number]) => {
    props.onView({ ...view, bucket, ctx, ...patch });
    if (toggle) analytics.toggleChanged(toggle[0], toggle[1]);
  };

  // the model menu: the table's models first, then the whole catalog by size
  const tableModels = [...new Map(columns.filter((c) => !c.column.id.startsWith('custom-') || c.model).map((c) => [c.column.id, c.model?.name ?? c.column.id])).entries()];
  const inTable = new Set(tableModels.map(([id]) => id));
  const sections = index ? orderCatalog(index.models).map((s) => ({ ...s, entries: s.entries.filter((m) => !inTable.has(m.id)) })).filter((s) => s.entries.length) : [];
  const known = inTable.has(view.model) || sections.some((s) => s.entries.some((m) => m.id === view.model));

  return (
    <div>
      <div className="flex flex-wrap items-center gap-3 mb-3">
        <span className="font-heading text-lg">Every Mac running</span>
        <select aria-label="Model" className="lls-chip !py-1 appearance-none font-medium max-w-[16rem]" value={view.model} onChange={(e) => { setOpenChip(null); set({ model: e.target.value }); }}>
          {!known && <option value={view.model}>{model?.name ?? view.model}</option>}
          {tableModels.length > 0 && (
            <optgroup label="In your table">
              {tableModels.map(([id, name]) => (
                <option key={id} value={id}>
                  {name}
                </option>
              ))}
            </optgroup>
          )}
          {sections.map((s) => (
            <optgroup key={s.key} label={s.label}>
              {s.entries.map((m) => (
                <option key={m.id} value={m.id}>
                  {m.name}
                </option>
              ))}
            </optgroup>
          ))}
        </select>
        {buckets.length > 0 && (
          <div className="lls-seg" role="group" aria-label="Build">
            {buckets.map((b) => (
              <button key={b} type="button" aria-pressed={bucket === b} title={grid?.quant && bucket === b ? grid.quant.label : undefined} onClick={() => set({ bucket: b }, ['build', b])}>
                {b}
              </button>
            ))}
          </div>
        )}
        {factors.prefill && (
          <div className="lls-seg" role="group" aria-label="Writing or reading">
            <button type="button" aria-pressed={!reading} onClick={() => set({ mode: 'write' }, ['speed_mode', 'write'])} title="tokens per second while the model writes its answer">
              Writing
            </button>
            <button type="button" aria-pressed={reading} onClick={() => set({ mode: 'read' }, ['speed_mode', 'read'])} title="the wait before the first word: how long the model takes to read your prompt">
              Reading
            </button>
          </div>
        )}
      </div>
      <div className="flex flex-wrap items-center gap-x-5 gap-y-2 mb-3 text-[13px]">
        {chips.length > 0 && (
          <div className="flex items-center gap-2">
            <span className="text-[var(--color-muted)]">{reading ? 'prompt' : 'context'}</span>
            <div className="lls-seg" role="group" aria-label={reading ? 'Prompt size' : 'Context'}>
              {chips.map((c) => (
                <button key={c} type="button" aria-pressed={ctx === c} onClick={() => set({ ctx: c }, ['context', c])} title={reading ? `${promptPhrase(c)}, pasted in one go` : `${formatContext(c)} tokens in the chat`}>
                  {formatContext(c)}
                </button>
              ))}
            </div>
          </div>
        )}
        <div className="flex items-center gap-2">
          <span className="text-[var(--color-muted)]">memory</span>
          <div className="lls-seg" role="group" aria-label="Memory size per chip">
            <button type="button" aria-pressed={view.memory === 'max'} onClick={() => set({ memory: 'max' }, ['memory_pick', 'max'])} title="each chip with the most memory it was sold with: its fastest version">
              Biggest
            </button>
            <button type="button" aria-pressed={view.memory === 'min'} onClick={() => set({ memory: 'min' }, ['memory_pick', 'min'])} title="each chip with the least memory that runs this model: the cheapest way in, sometimes a slower version of the chip">
              Smallest that fits
            </button>
          </div>
        </div>
      </div>

      {!model && <p className="text-sm text-[var(--color-muted)]">{record?.status === 'missing' ? 'This model is not in the catalog any more. Pick another one above.' : record?.status === 'error' ? 'This model could not be loaded. Pick another one, or try again in a moment.' : 'Loading this model…'}</p>}
      {model && grid && !grid.quant && <p className="text-sm text-[var(--color-muted)]">{model.name} has no file for {runtimeName(state.runtime)} in this catalog.</p>}
      {model && grid && grid.quant && (
        <>
          <p className="text-sm text-[var(--color-muted)] mb-2 flex flex-wrap items-center gap-x-2 gap-y-1">
            <span>
              {[
                reading ? 'The wait before the first word' : 'Writing speed in tokens per second',
                `${model.name} ${grid.quant.label}`,
                reading ? `${promptPhrase(ctx)} pasted in one go` : `${formatContext(ctx)} context`,
                runtimeName(state.runtime),
                view.memory === 'max' ? 'the biggest memory each chip was sold with' : 'the smallest memory that runs it, chip by chip',
              ].join(' · ')}
            </span>
            <button type="button" className="lls-chip !py-0.5" onClick={props.onOpenSettings} title="MLX or GGUF, the macOS memory limit, room for your apps">
              <SlidersIcon size={12} /> runtime, memory limit, apps
            </button>
          </p>
          <Legend reading={reading} factors={factors} />
          <Grid grid={grid} view={view} reading={reading} factors={factors} openChip={openChip} onChip={(id) => setOpenChip((cur) => (cur === id ? null : id))} />
          {open && (
            <Sizes
              cell={open}
              shown={shownConfig(open, view.memory)}
              reading={reading}
              ctx={ctx}
              onClose={() => setOpenChip(null)}
              onOpen={(c) => props.onOpenCell({ model, pick: { key: c.key, group: c.group, row: c.row, linked: null }, bucket, ctx, result: c.result, quant: grid.quant })}
            />
          )}
          <p className="mt-3 text-xs text-[var(--color-light)]">
            {reading
              ? 'Reading speed is compute: it follows the GPU cores and the chip generation, not memory. In a chat only the new message is read; this is the cost of pasting the whole prompt at once.'
              : 'Writing speed follows memory bandwidth, so a bigger memory makes a chip faster only when it comes with a faster version of the chip.'}{' '}
            Estimates from the same formulas as the table, calibrated on measured runs; a dashed cell is a chip nobody has measured yet. Tap a chip for every memory size it was sold with.
          </p>
        </>
      )}
    </div>
  );
}

function Legend({ reading, factors }: { reading: boolean; factors: Factors }) {
  const swatch = (fill: string) => <i className="lls-swatch" style={{ background: fill }} aria-hidden="true" />;
  const write = factors.feels_like;
  const wait = factors.prefill?.feels_like ?? [];
  return (
    <ul className="lls-bands mb-3" aria-label="What the colours and outlines mean">
      {reading
        ? wait.map((b, i) => (
            <li key={b.label} className="lls-band">
              {swatch(READ_FILL[i] ?? READ_FILL[READ_FILL.length - 1])}
              <b>{b.label}</b>
              {b.max_s === null ? `over ${edgeText(wait[i - 1]?.max_s ?? 0)}` : i === 0 ? `under ${edgeText(b.max_s)}` : `${(wait[i - 1].max_s ?? 0) < 120 && b.max_s < 120 ? wait[i - 1].max_s : edgeText(wait[i - 1].max_s ?? 0)} to ${edgeText(b.max_s)}`}
            </li>
          ))
        : write.map((b, i) => {
            const from = i === 0 ? 0 : (write[i - 1].max_tok_s ?? 0);
            return (
              <li key={b.label} className="lls-band">
                {swatch(WRITE_FILL[i] ?? WRITE_FILL[WRITE_FILL.length - 1])}
                <b>{b.label}</b>
                {b.max_tok_s === null ? `${from} and up` : from === 0 ? `under ${b.max_tok_s}` : `${from} to ${b.max_tok_s}`}
              </li>
            );
          })}
      <li className="lls-band">
        <i className="lls-swatch lls-swatch-ring" aria-hidden="true" /> only with the macOS memory limit lifted
      </li>
      <li className="lls-band">
        <i className="lls-swatch lls-swatch-dash" aria-hidden="true" /> no measured runs on this chip yet
      </li>
    </ul>
  );
}

function Grid({ grid, view, reading, factors, openChip, onChip }: { grid: ChipGrid; view: ChipsView; reading: boolean; factors: Factors; openChip: string | null; onChip: (id: string) => void }) {
  return (
    <div className="lls-scroll">
      <table className="lls-chips">
        <caption className="sr-only">{reading ? 'Wait before the first word' : 'Writing speed'} on every Mac chip, by generation and tier</caption>
        <thead>
          <tr>
            <td />
            {grid.gens.map((g) => (
              <th key={g} scope="col">
                M{g}
                {grid.genYears[g] && <span>{grid.genYears[g]}</span>}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {grid.tiers.map((t) => (
            <tr key={t}>
              <th scope="row">
                <b>{TIER_NAMES[t]}</b>
                <span>{grid.tierFamilies[t].join(', ')}</span>
              </th>
              {grid.gens.map((g) => {
                const id = cellId(t, g);
                const cell = grid.cells.get(id);
                return <td key={g}>{cell ? <Cell cell={cell} pick={view.memory} reading={reading} factors={factors} open={openChip === id} onClick={() => onChip(id)} /> : <span className="lls-chipcell is-none">no such chip</span>}</td>;
              })}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function Cell({ cell, pick, reading, factors, open, onClick }: { cell: ChipCell; pick: ChipsView['memory']; reading: boolean; factors: Factors; open: boolean; onClick: () => void }) {
  const c = shownConfig(cell, pick);
  if (!c) return <span className="lls-chipcell is-none">no estimate</span>;
  const common = { type: 'button' as const, onClick, 'aria-expanded': open, 'aria-controls': 'lls-chip-sizes' };
  const value = reading ? c.waitS : c.writeTokS;
  if (c.status === 'no-fit' || value === null) {
    return (
      <button {...common} className={`lls-chipcell is-nofit${open ? ' is-open' : ''}`} aria-label={`${cell.name}: does not fit, ${c.gb} GB at most. Every memory size`}>
        <span className="v">{c.status === 'no-fit' ? "doesn't fit" : 'no estimate'}</span>
        <span className="m">{c.status === 'no-fit' ? `${c.gb} GB at most` : `${c.gb} GB`}</span>
      </button>
    );
  }
  const band = reading ? waitBandIndex(value, factors) : writeBandIndex(value, factors);
  const fill = reading ? READ_FILL[band] : WRITE_FILL[band];
  const ring = reading ? READ_RING[band] : WRITE_RING[band];
  const measured = reading ? c.readMeasured : c.writeMeasured;
  const lifted = c.status === 'lifted';
  const [big, unit] = reading ? waitParts(value) : [writeNumber(value), ''];
  const label = reading ? `${formatTokS(c.readTokS)}${lifted ? ' · limit lifted' : ''}` : lifted ? liftedText(c) : (factors.feels_like[band]?.label ?? '');
  const spoken = reading ? `${cell.name}, ${c.gb} GB: ${formatWait(value)} before the first word, reading at ${formatTokS(c.readTokS)}` : `${cell.name}, ${c.gb} GB: ${formatTokS(value)}, ${factors.feels_like[band]?.label ?? ''}`;
  return (
    <button
      {...common}
      className={`lls-chipcell${lifted ? ' is-lifted' : ''}${measured ? '' : ' is-est'}${open ? ' is-open' : ''}`}
      style={{ '--fill': fill, '--ring': ring } as React.CSSProperties}
      aria-label={`${spoken}${lifted ? `, ${liftedText(c)}` : ''}${measured ? '' : ', not measured on this chip yet'}. Every memory size`}
    >
      <span className="v">
        {big}
        {unit && <small>{unit}</small>}
      </span>
      <span className="l">{label}</span>
      <span className="m">{c.gb} GB</span>
    </button>
  );
}

/** Every memory size of one chip: what changes with the size and what does not. */
function Sizes({ cell, shown, reading, ctx, onClose, onOpen }: { cell: ChipCell; shown: ChipConfig | null; reading: boolean; ctx: number; onClose: () => void; onOpen: (c: ChipConfig) => void }) {
  const priced = cell.configs.some((c) => c.priceUsd !== null);
  const versions = new Set(cell.configs.map((c) => `${c.bandwidthGbs}|${c.cores}`)).size;
  const note =
    versions === 1
      ? `Every size of the ${cell.name} here is the same version of the chip, so it writes and reads at the same speed wherever the model fits. More memory buys room (longer chats, bigger builds, other apps open), not speed.`
      : `The ${cell.name} comes in ${versions === 2 ? 'two' : versions} versions. Writing follows memory bandwidth and reading follows GPU cores, so a size is faster only when it comes with the bigger version; past that, more memory buys room, not speed.`;
  return (
    <section id="lls-chip-sizes" className="lls-chip-sizes" aria-label={`${cell.name} at every memory size`}>
      <div className="flex items-start justify-between gap-4 mb-1">
        <h3 className="font-heading text-lg leading-tight">
          {cell.name} at every memory size
        </h3>
        <button type="button" className="lls-chip" onClick={onClose} aria-label="Close">
          <XIcon size={14} />
        </button>
      </div>
      <p className="text-[13px] text-[var(--color-muted)] mb-3 max-w-[760px]">{note}</p>
      <div className="lls-scroll">
        <table className="lls-mm lls-sizes-table w-full text-[13px]">
          <thead>
            <tr className="text-left text-[11px] uppercase tracking-[0.12em] text-[var(--color-light)]">
              <th>Memory</th>
              <th>Chip version</th>
              <th>Sold as</th>
              <th>Writing</th>
              <th>Reading {promptPhrase(ctx, ' prompt')}</th>
              {priced && <th>List price</th>}
            </tr>
          </thead>
          <tbody>
            {cell.configs.map((c) => {
              const isShown = shown === c;
              const up = c.result.prefill?.upgrade ?? null;
              return (
                <tr key={c.key + c.cores} className={isShown ? 'is-shown' : undefined} onClick={() => onOpen(c)}>
                  <td>
                    <button type="button" className="lls-map-rowbtn" onClick={(e) => { e.stopPropagation(); onOpen(c); }} title="Memory, speed and sources for this configuration">
                      <span className="font-semibold">{c.gb} GB</span>
                      {isShown && <span className="lls-map-badge">in the grid</span>}
                    </button>
                  </td>
                  <td>
                    {c.cores ? `${c.cores}-core GPU` : 'GPU'} · {formatBandwidth(c.bandwidthGbs)}
                  </td>
                  <td className="text-[var(--color-muted)]">{c.families.join(', ')}</td>
                  <td className={reading ? 'text-[var(--color-muted)]' : 'font-semibold'}>
                    {c.writeTokS === null ? "doesn't fit" : formatTokS(c.writeTokS)}
                    {c.status === 'lifted' && <span className="block text-[11px] font-normal text-[var(--lls-ring)]">{liftedText(c)}</span>}
                  </td>
                  <td className={reading ? 'font-semibold' : 'text-[var(--color-muted)]'}>
                    {c.waitS === null ? (c.writeTokS === null ? "doesn't fit" : '—') : `${formatWait(c.waitS)} · ${formatTokS(c.readTokS)}`}
                    {up && c.waitS !== null && (
                      <span className="block text-[11px] font-normal text-[var(--color-light)]">
                        the {up.cores}-core GPU reads {Math.round((up.ratio - 1) * 100)} % faster{up.usd !== null ? `, ${formatPrice(up.usd)} more` : ''}
                      </span>
                    )}
                  </td>
                  {priced && <td>{c.priceUsd !== null ? `from ${formatPrice(c.priceUsd)}` : 'no list price'}</td>}
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </section>
  );
}
