/**
 * "Which machine?": the buying map. One model at one build and context across every current machine, placed by
 * memory size (rows) and estimated writing speed (x), shaded into the feels-like bands; a minimum speed, a budget,
 * a form factor and the machine you own turn the marks into a recommendation and a list under the map.
 */
import { useMemo } from 'react';
import type { CellResult, Factors, ModelDetail, Quant, SpecialBuild } from '../../lib/llm-sizer/engine/types';
import { formatContext, formatPrice, formatTokS } from '../../lib/llm-sizer/engine/format';
import { atlasKey, buyingMap, evaluateBuilds, minSpeedPresets, placeRowLabels, type BuildKey, type BuyingMark, type BuyingMap } from '../../lib/llm-sizer/app/buying';
import { verdictSentence } from '../../lib/llm-sizer/app/layout';
import type { MachineGroup } from '../../lib/llm-sizer/app/machines';
import type { MatrixColumn, MatrixRow } from '../../lib/llm-sizer/app/matrix';
import type { Bucket } from '../../lib/llm-sizer/app/quants';
import type { AppState } from '../../lib/llm-sizer/app/state';
import { textWidth } from './chart';
import { SlidersIcon } from './icons';

export interface MapTarget {
  model: ModelDetail;
  pick: MatrixRow;
  bucket: Bucket;
  ctx: number;
  result: CellResult;
  /** the exact file evaluated, so the sheet names it */
  quant: Quant | SpecialBuild | null;
}

export interface MachineMapViewProps {
  state: AppState;
  column: MatrixColumn;
  columns: MatrixColumn[];
  groups: MachineGroup[];
  factors: Factors;
  build: BuildKey;
  minTokS: number;
  toggles: { work: string; limit: string };
  machinesAsOf: string | null;
  onColumn: (index: number) => void;
  onBuild: (key: BuildKey) => void;
  onMinTokS: (tokS: number) => void;
  onOpenCell: (target: MapTarget, anchor: HTMLElement) => void;
  onAddGroup: (groupId: string) => void;
  onOpenSettings: () => void;
}

const W = 900;
const RAM_W = 64;
const PLOT_X0 = RAM_W + 36;
const PLOT_X1 = W - 16;
/** a machine that does not fit is a dash just left of the zero line: never on the speed axis */
const NOFIT_X = PLOT_X0 - 16;
const TOP = 40;
const ROW_H = 52;
const R = 6.5;
/** marks that share an x are drawn smaller, so a stack reads as several machines rather than one blob */
const R_STACK = 5;
const BAND_FILLS = ['rgba(26,26,26,0.035)', 'rgba(212,175,55,0.06)', 'rgba(212,175,55,0.10)', 'rgba(212,175,55,0.14)', 'rgba(212,175,55,0.18)'];

function speedText(m: Pick<BuyingMark, 'tokS' | 'approx'>): string {
  return `${m.approx && m.tokS !== null ? '~' : ''}${formatTokS(m.tokS)}`;
}

function priceText(m: Pick<BuyingMark, 'priceUsd'>): string {
  return m.priceUsd !== null ? formatPrice(m.priceUsd) : 'no current price';
}

function markTitle(m: BuyingMark, tag: string | null): string {
  return `${m.label}${tag ? ` (${tag})` : ''}: ${verdictSentence(m.result)} · ${m.tokS !== null ? speedText(m) : m.status === 'no-fit' ? 'not placed at a speed' : 'speed not estimated'} · ${priceText(m)}`;
}

function toRow(m: BuyingMark): MatrixRow {
  return { key: m.key, group: m.group, row: m.row, linked: m.linked };
}

/** What the map says about a mark beyond its glyph: the pick. */
/** True when no mark on this build carries a speed (an engram-on-SSD build): the legend says so once, the labels need not. */
function noSpeedsAtAll(map: BuyingMap): boolean {
  return map.marks.every((m) => m.status === 'no-fit' || m.tokS === null);
}

function tagFor(m: BuyingMark, map: BuyingMap): string | null {
  if (map.recommendation.pick?.key === m.key) return m.tokS === null ? (noSpeedsAtAll(map) ? 'cheapest that fits' : 'cheapest that fits, speed not estimated') : 'cheapest that qualifies';
  return null;
}

function Swatch({ kind }: { kind: 'yes' | 'miss' | 'ring' | 'no' | 'unknown' }) {
  return (
    <svg width={16} height={16} viewBox="0 0 16 16" aria-hidden="true" style={{ flexShrink: 0 }}>
      {kind === 'yes' && <circle cx={8} cy={8} r={5.5} fill="#D4AF37" stroke="#1A1A1A" strokeWidth={1.5} />}
      {kind === 'unknown' && <circle cx={8} cy={8} r={5} fill="rgba(212,175,55,0.35)" stroke="#B8912B" strokeWidth={1.5} strokeDasharray="2 2" />}
      {kind === 'miss' && <circle cx={8} cy={8} r={5} fill="rgba(212,175,55,0.45)" />}
      {kind === 'ring' && <circle cx={8} cy={8} r={5} fill="none" stroke="#9E7C25" strokeWidth={2.5} />}
      {kind === 'no' && <rect x={2} y={6.5} width={12} height={3} fill="#8A8780" />}
    </svg>
  );
}

export function MachineMapView(props: MachineMapViewProps) {
  const { state, column, groups, factors, toggles, machinesAsOf, minTokS } = props;
  const model = column.model;
  const key = atlasKey(state, column);
  // one engine pass per column and settings; the build switch and the criteria only re-filter it
  const atlas = useMemo(() => (model ? evaluateBuilds(model, column, groups, state, factors, machinesAsOf) : null), [model, key, groups, factors, machinesAsOf]); // eslint-disable-line react-hooks/exhaustive-deps
  const map = useMemo<BuyingMap | null>(
    () => (atlas ? buyingMap(atlas, props.build, { minTokS, budgetUsd: state.budgetUsd, formFactor: state.formFactor }, toggles) : null),
    [atlas, props.build, minTokS, state.budgetUsd, state.formFactor, toggles],
  );
  const presets = useMemo(() => minSpeedPresets(factors), [factors]);
  const criteriaParts = [state.budgetUsd !== null ? `under ${formatPrice(state.budgetUsd)}` : null, state.formFactor === 'laptop' ? 'laptops only' : state.formFactor === 'desktop' ? 'desktops only' : null].filter((p): p is string => p !== null);

  return (
    <div>
      <div className="flex flex-wrap items-center gap-3 mb-4">
        <span className="font-heading text-lg">Which machine for</span>
        <select aria-label="Model" className="lls-chip !py-1 appearance-none font-medium" value={column.index} onChange={(e) => props.onColumn(Number(e.target.value))}>
          {props.columns.map((c) => (
            <option key={c.index} value={c.index}>
              {c.model?.name ?? c.column.id} · {c.quant?.label ?? '…'} · {Math.round(c.column.ctx / 1024)}K
            </option>
          ))}
        </select>
        {atlas && (
          <div className="lls-seg" role="group" aria-label="Build">
            {atlas.options.map((o) => (
              <button key={o.key} type="button" aria-pressed={map?.build.key === o.key} disabled={!o.quant} title={o.note ?? (o.quant ? o.quant.label : undefined)} onClick={() => props.onBuild(o.isColumn ? 'column' : o.key)}>
                {o.label}
              </button>
            ))}
          </div>
        )}
        <div className="flex items-center gap-2 text-[13px]">
          <span className="text-[var(--color-muted)]">tok/s at least</span>
          <div className="lls-seg" role="group" aria-label="Minimum speed">
            {presets.map((p) => (
              <button key={p.tokS} type="button" aria-pressed={minTokS === p.tokS} title={p.opens ? `${p.opens} and up` : 'no minimum'} onClick={() => props.onMinTokS(p.tokS)}>
                {p.label}
              </button>
            ))}
            {!presets.some((p) => p.tokS === minTokS) && (
              <button type="button" aria-pressed={true} title="from the link">
                {minTokS}
              </button>
            )}
          </div>
        </div>
      </div>
      {!model && <p className="text-sm text-[var(--color-muted)]">Loading this model…</p>}
      {map && atlas && model && (
        <>
          <p className="text-sm text-[var(--color-muted)] mb-2 flex flex-wrap items-center gap-x-2 gap-y-1">
            <span>{[...map.captionParts.slice(0, 5), ...criteriaParts].join(' · ')}</span>
            <button type="button" className="lls-chip !py-0.5" onClick={props.onOpenSettings} title="Budget, laptop or desktop, and your own machines">
              <SlidersIcon size={12} /> budget, form factor, your machines
            </button>
          </p>
          <ul className="lls-bands mb-3" aria-label="What the marks mean">
            <li className="lls-band"><Swatch kind="yes" /> qualifies</li>
            <li className="lls-band"><Swatch kind="miss" /> runs, misses your target</li>
            <li className="lls-band"><Swatch kind="ring" /> runs with a change, at its fix's speed</li>
            <li className="lls-band"><Swatch kind="no" /> doesn't fit</li>
            {map.marks.some((m) => m.tokS === null && m.status !== 'no-fit') && <li className="lls-band"><Swatch kind="unknown" /> fits, speed not estimated</li>}
          </ul>
          <MapSvg map={map} factors={factors} minTokS={minTokS} onOpen={(m, anchor) => props.onOpenCell({ model, pick: toRow(m), bucket: map.build.bucket, ctx: atlas.ctx, result: m.result, quant: map.build.quant }, anchor)} />
          <Verdict map={map} />
          <p className="mt-3 text-xs text-[var(--color-light)]">
            {map.captionParts.slice(5).join(' · ')}. Speeds are output-generation estimates for the configuration that fits; a ring sits at its fix's speed; a machine that doesn't fit is never placed at a speed. Tap a mark for the details. Prices are list prices at the data date.
          </p>
        </>
      )}
    </div>
  );
}

/** One line under the map: the pick is named on the map itself; here only what the map cannot show, that nothing qualifies. */
function Verdict({ map }: { map: BuyingMap }) {
  const r = map.recommendation;
  if (r.pick || r.unpricedQualifying.length > 0) {
    if (!r.pick && r.unpricedQualifying.length > 0) return <p className="mt-3 text-[13px] text-[var(--color-muted)]">Only {r.unpricedQualifying.map((m) => m.label).join('; ')} qualifies, and it has no current price, so nothing is called cheapest.</p>;
    return null;
  }
  const c = map.criteria;
  const asked = `${map.build.label} · ${formatContext(map.atlas.ctx)}`;
  const limits = `${c.minTokS > 0 ? ` at ${c.minTokS} tok/s or more` : ''}${c.budgetUsd !== null ? ` under ${formatPrice(c.budgetUsd)}` : ''}${c.formFactor !== 'any' ? ` as a ${c.formFactor}` : ''}`;
  const near = r.nearest.map((n) => `${n.kind === 'price' ? 'over budget' : n.kind === 'form' ? 'wrong form factor' : n.kind === 'speed' ? 'too slow' : 'with a change'}: ${n.mark.label} (${n.detail})`);
  return (
    <p className="mt-3 text-[13px]">
      <span className="font-semibold">Nothing qualifies.</span>{' '}
      <span className="text-[var(--color-muted)]">
        {r.nearest.length ? `No current machine runs ${map.atlas.model.name} at ${asked}${limits}. Nearest: ${near.join(' · ')}.` : `Nothing on this map runs ${map.atlas.model.name} at ${asked}, not even with a smaller build or a shorter context.${map.build.bucket !== 'Q4' && map.atlas.options.some((o) => o.key === 'Q4' && o.quant) ? ' Try Q4.' : ''}`}
      </span>
    </p>
  );
}

function MapSvg({ map, factors, minTokS, onOpen }: { map: BuyingMap; factors: Factors; minTokS: number; onOpen: (m: BuyingMark, anchor: HTMLElement) => void }) {
  const { atlas, rows } = map;
  const H = TOP + rows.length * ROW_H + 30;
  const x = (v: number) => PLOT_X0 + (Math.min(v, atlas.axisMax) / atlas.axisMax) * (PLOT_X1 - PLOT_X0);
  const yRow = (i: number) => TOP + i * ROW_H + ROW_H / 2;
  const bands = factors.feels_like;
  // the axis is read against the bands, so its numbers sit on the band edges (5, 12, 30, 60) and at the end
  const ticks = [...new Set([0, ...bands.map((b) => b.max_tok_s).filter((v): v is number => v !== null && v < atlas.axisMax), atlas.axisMax])];
  const open = (m: BuyingMark) => (e: React.SyntheticEvent<SVGRectElement>) => onOpen(m, e.currentTarget as unknown as HTMLElement);
  const onKey = (m: BuyingMark) => (e: React.KeyboardEvent<SVGRectElement>) => {
    if (e.key === 'Enter' || e.key === ' ') {
      e.preventDefault();
      onOpen(m, e.currentTarget as unknown as HTMLElement);
    }
  };
  const hit = (m: BuyingMark, hx: number, hy: number, hw: number, hh: number) => (
    <rect key={`hit-${m.key}`} className="lls-hit" x={hx} y={hy} width={hw} height={hh} fill="transparent" tabIndex={0} role="button" aria-label={markTitle(m, tagFor(m, map))} onClick={open(m)} onKeyDown={onKey(m)}>
      <title>{markTitle(m, tagFor(m, map))}</title>
    </rect>
  );

  return (
    <div className="lls-scroll">
      <svg viewBox={`0 0 ${W} ${H}`} width="100%" style={{ minWidth: 640, fontFamily: 'Inter, sans-serif' }} role="group" aria-label={`Memory against speed for ${atlas.model.name}`}>
        {bands.map((b, i) => {
          const from = i === 0 ? 0 : (bands[i - 1].max_tok_s ?? 0);
          const to = b.max_tok_s ?? atlas.axisMax;
          if (from >= atlas.axisMax) return null;
          const x0 = x(from);
          const x1 = x(to);
          const fits = textWidth(b.label, 11) < x1 - x0 - 6;
          return (
            <g key={b.label}>
              <rect x={x0} y={TOP - 4} width={Math.max(0, x1 - x0)} height={H - 24 - (TOP - 4)} fill={BAND_FILLS[i] ?? BAND_FILLS[BAND_FILLS.length - 1]} />
              {i > 0 && <line x1={x0} x2={x0} y1={TOP - 4} y2={H - 24} stroke="rgba(26,26,26,0.10)" />}
              {fits && (
                <text x={(x0 + x1) / 2} y={12} fontSize={11} textAnchor="middle" fill="#7a7974">
                  {b.label}
                </text>
              )}
            </g>
          );
        })}
        <text x={NOFIT_X + 6} y={12} fontSize={11} textAnchor="end" fill="#7a7974">
          doesn't fit
        </text>
        {ticks.map((t, i) => {
          const last = i === ticks.length - 1;
          return (
            <g key={t}>
              <line x1={x(t)} x2={x(t)} y1={TOP - 4} y2={H - 24} stroke="rgba(26,26,26,0.12)" strokeDasharray="2 4" />
              <text x={x(t)} y={H - 8} fontSize={11} textAnchor={last ? 'end' : i === 0 ? 'start' : 'middle'} fill="#98968f">
                {last ? `${t} tok/s` : t}
              </text>
            </g>
          );
        })}
        {minTokS > 0 && (
          <g>
            <line x1={x(minTokS)} x2={x(minTokS)} y1={TOP - 4} y2={H - 24} stroke="#1A1A1A" strokeWidth={1.5} strokeDasharray="5 4" />
            <text x={x(minTokS) - 6} y={28} fontSize={11} textAnchor="end" fill="#1A1A1A">
              {`at least ${minTokS} tok/s`}
            </text>
          </g>
        )}
        {rows.map((row, i) => {
          const y = yRow(i);
          // every mark goes through the same placement: dashes at the left, unknown speeds on the zero line, the rest at their speed
          // a fit with no speed figure sits just right of the zero line, never on it
          const xOf = (m: BuyingMark) => (m.status === 'no-fit' ? NOFIT_X : m.tokS === null ? PLOT_X0 + 12 : x(m.tokS));
          const labels = placeRowLabels(
            row.marks.map((m) => {
              const tag = tagFor(m, map);
              const base = m.tokS === null && m.status !== 'no-fit' && !noSpeedsAtAll(map) ? `${m.short} (speed not estimated)` : m.short;
              return { key: m.key, x: xOf(m), label: tag ? `${base} · ${tag}` : base, bold: tag !== null };
            }),
            (t) => textWidth(t, 10),
            { left: RAM_W + 4, right: PLOT_X1 },
          );
          const byKey = new Map(row.marks.map((m) => [m.key, m]));
          const stacks = new Map<string, { n: number; dy: number }>();
          for (const p of labels) for (const k of Object.keys(p.stackDy)) stacks.set(k, { n: Object.keys(p.stackDy).length, dy: p.stackDy[k] });
          return (
            <g key={row.gb}>
              <line x1={RAM_W} x2={W} y1={TOP + (i + 1) * ROW_H} y2={TOP + (i + 1) * ROW_H} stroke="rgba(26,26,26,0.06)" />
              <text x={RAM_W - 8} y={y + 4} fontSize={12} fontWeight={700} textAnchor="end" fill="#1A1A1A">
                {row.gb} GB
              </text>
              {row.marks.map((m) => {
                const st = stacks.get(m.key) ?? { n: 1, dy: 0 };
                const cx = xOf(m);
                const cy = y + st.dy;
                const r = st.n > 1 ? R_STACK : R;
                const own = labels.find((p) => p.keys.length === 1 && p.keys[0] === m.key && p.lane === 0 && p.anchor === 'start');
                const hw = own ? Math.max(20, own.x1 - (cx - 10) + 4) : 20;
                const hh = ROW_H / st.n;
                const hy = cy - hh / 2;
                return (
                  <g key={m.key}>
                    {m.status === 'no-fit' ? (
                      <rect x={cx - 6} y={cy - 1.5} width={12} height={3} fill="#8A8780" />
                    ) : m.tokS === null ? (
                      <circle cx={cx} cy={cy} r={r - 1} fill={m.qualifies ? 'rgba(212,175,55,0.35)' : 'none'} stroke={m.qualifies ? '#B8912B' : '#8A8780'} strokeWidth={1.5} strokeDasharray="2 2" />
                    ) : m.status === 'compromise' ? (
                      <circle cx={cx} cy={cy} r={r - 0.5} fill="none" stroke="#9E7C25" strokeWidth={2.5} opacity={m.wouldQualify ? 1 : 0.5} />
                    ) : m.qualifies ? (
                      <circle cx={cx} cy={cy} r={r} fill="#D4AF37" stroke="#1A1A1A" strokeWidth={1.5} />
                    ) : (
                      <circle cx={cx} cy={cy} r={r - 0.5} fill="rgba(212,175,55,0.45)" />
                    )}
                    {hit(m, cx - 10, hy, hw, hh)}
                  </g>
                );
              })}
              {labels.map((p) => {
                if (p.hidden) return null;
                const marks = p.keys.map((k) => byKey.get(k)!);
                const any = marks.some((m) => m.qualifies);
                return (
                  <text key={p.keys.join('+')} x={p.x + p.dx} y={y + p.dy} fontSize={10} fontWeight={p.bold ? 700 : 400} textAnchor={p.anchor} fill={any || p.bold ? '#1A1A1A' : '#7a7974'} style={{ pointerEvents: 'none' }}>
                    {p.text}
                  </text>
                );
              })}
            </g>
          );
        })}
      </svg>
    </div>
  );
}
