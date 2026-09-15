/**
 * The share image: the same table (or speed / one-model view) rendered by satori (@vercel/og) on the server.
 * Satori rules: every element with more than one child is display:flex; inline styles only; no CSS variables.
 * The proportions follow the video's b-roll: rail 252 · size 112 · cells share the rest.
 */
import React from 'react';
import type { AppState } from '../../../lib/llm-sizer/app/state';
import { cellKey, type Matrix } from '../../../lib/llm-sizer/app/matrix';
import type { Marker, TableLayout } from '../../../lib/llm-sizer/app/layout';
import { buyingMap, evaluateBuilds, type BuyingMap, type BuyingMark } from '../../../lib/llm-sizer/app/buying';
import type { MachineGroup } from '../../../lib/llm-sizer/app/machines';
import { formatContext, formatPrice, formatTokS } from '../../../lib/llm-sizer/engine/format';
import type { Factors } from '../../../lib/llm-sizer/engine/types';
import { pooledPrice, rowLabel } from '../../../lib/llm-sizer/app/machines';
import { SITE } from '../../../lib/llm-sizer/adapters/site';

export type ImageSize = 'og' | 'hd' | 'table';
export type ImageView = 'table' | 'speed' | 'model';
export const WATERMARK = SITE.watermark;
export const OG_MAX_COLUMNS = 4;

const BG = '#F9F8F6';
const TEXT = '#1A1A1A';
const GOLD = '#D4AF37';
const MUTED = '#6B6B6B';
const LIGHT = '#9CA3AF';
const LINE = '#E5E2DC';
const GREY = '#C9C5BE';

// base metrics at scale 1 (the 1920-wide presentation layout)
const PAD = 64;
const TITLE_H = 128;
const HEADER_H = 128;
const ROW_H = 64;
const FOOTER_H = 72;
const RAIL_W = 252;
const SIZE_W = 112;

export interface ImageProps {
  state: AppState;
  matrix: Matrix;
  layout: TableLayout;
  factors: Factors;
  size: ImageSize;
  view: ImageView;
  modelsDate: string | null;
  /** the buying map for the model view, from `buyingImageMap`; null when the column has no record */
  buying?: BuyingMap | null;
}

/** The most rows the buying list draws: the qualifying configurations first, then the rest. */
export const BUYING_IMAGE_ROWS = 14;

/** The buying map the model view's image draws: the same helper as the island, so the highlighted row is the card's pick. */
export function buyingImageMap(state: AppState, matrix: Matrix, groups: MachineGroup[], factors: Factors, machinesAsOf: string | null): BuyingMap | null {
  const view = state.view;
  const column = (view.kind === 'model' ? matrix.columns[view.column] : undefined) ?? matrix.columns[0];
  if (!column?.model) return null;
  const atlas = evaluateBuilds(column.model, column, groups, state, factors, machinesAsOf);
  const build = view.kind === 'model' ? view.build : 'column';
  const minTokS = view.kind === 'model' ? view.minTokS : 0;
  const limitPct = state.cap !== null ? `${Math.round(state.cap * 100)} % (override)` : '67 % / 75 %';
  const toggles = { work: state.work > 0 ? `apps open (${state.work} GB reserved)` : 'nothing else running', limit: `macOS memory limit ${limitPct}` };
  return buyingMap(atlas, build, { minTokS, budgetUsd: state.budgetUsd, formFactor: state.formFactor }, toggles);
}

/** The rows the buying image draws, in the list's order: qualifying, with a change, the rest. */
export function buyingImageRows(map: BuyingMap): BuyingMark[] {
  return [...map.qualifying, ...map.withChange, ...map.rest].slice(0, BUYING_IMAGE_ROWS);
}

function contentRows(view: ImageView, layout: TableLayout, matrix: Matrix, buying?: BuyingMap | null): number {
  if (view === 'speed') return matrix.columns.length;
  if (view === 'model') return buying ? buyingImageRows(buying).length : matrix.rows.length;
  return layout.rows.length;
}

/** Pixel size of the image and the scale factor everything is drawn at. */
export function imageDimensions(size: ImageSize, view: ImageView, layout: TableLayout, matrix: Matrix, buying?: BuyingMap | null): { width: number; height: number; k: number } {
  const rows = Math.max(1, contentRows(view, layout, matrix, buying));
  const rowH = view === 'table' ? ROW_H : ROW_H + 12;
  const natural = PAD * 2 + TITLE_H + HEADER_H + rows * rowH + FOOTER_H;
  if (size === 'og') {
    const k = Math.min(0.62, 630 / natural);
    return { width: 1200, height: 630, k };
  }
  if (size === 'hd') {
    const k = Math.min(1, 1080 / natural);
    return { width: 1920, height: 1080, k };
  }
  const k = 0.75;
  return { width: Math.round(1920 * k), height: Math.round(natural * k), k };
}

function MarkerGlyph({ marker, k }: { marker: Marker; k: number }) {
  const d = Math.round(26 * k);
  if (marker === 'run') return <div style={{ width: d, height: d, borderRadius: d, background: GOLD, display: 'flex' }} />;
  if (marker === 'ring')
    return (
      <div style={{ width: d, height: d, borderRadius: d, border: `${Math.max(2, Math.round(3 * k))}px solid ${GOLD}`, display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
        <div style={{ width: Math.max(3, Math.round(6 * k)), height: Math.max(3, Math.round(6 * k)), borderRadius: 6, background: GOLD, display: 'flex' }} />
      </div>
    );
  if (marker === 'no') return <div style={{ width: d, height: Math.max(3, Math.round(4 * k)), background: GREY, display: 'flex' }} />;
  return <div style={{ fontSize: Math.round(14 * k), color: LIGHT, display: 'flex' }}>{marker === 'missing' ? 'n/a' : '…'}</div>;
}

function settingsLine(state: AppState, layout: TableLayout, buying?: BuyingMap | null): string {
  const parts = [`${layout.toggles.work}`, `${layout.toggles.limit}`, state.runtime === 'mlx' ? 'MLX' : 'GGUF'];
  if (buying) {
    const c = buying.criteria;
    parts.unshift(`${buying.build.label} · ${formatContext(buying.atlas.ctx)}${c.minTokS > 0 ? ` · at least ${c.minTokS} tok/s` : ''}${c.budgetUsd !== null ? ` · under ${formatPrice(c.budgetUsd)}` : ''}${c.formFactor !== 'any' ? ` · ${c.formFactor}s only` : ''}`);
  }
  return parts.join(' · ');
}

export function TableImage({ state, matrix, layout, factors, size, view, modelsDate, buying }: ImageProps) {
  const { width, height, k } = imageDimensions(size, view, layout, matrix, buying);
  const px = (n: number) => Math.round(n * k);
  const title = view === 'speed' ? speedTitle(state, matrix) : view === 'model' ? modelTitle(state, matrix) : 'What runs on which machine';
  return (
    <div style={{ width, height, background: BG, color: TEXT, fontFamily: 'Inter', display: 'flex', flexDirection: 'column', padding: `${px(PAD)}px`, borderTop: `${px(10)}px solid ${GOLD}` }}>
      <div style={{ display: 'flex', flexDirection: 'column', height: px(TITLE_H) }}>
        <div style={{ fontFamily: 'Playfair', fontSize: px(52), lineHeight: 1.1, display: 'flex' }}>{title}</div>
        <div style={{ fontSize: px(20), color: MUTED, marginTop: px(12), display: 'flex' }}>{settingsLine(state, layout, view === 'model' ? buying : null)}</div>
      </div>
      {view === 'table' && <TableBody layout={layout} k={k} />}
      {view === 'speed' && <SpeedBody state={state} matrix={matrix} factors={factors} k={k} />}
      {view === 'model' && (buying ? <BuyingBody map={buying} k={k} /> : <ModelBody state={state} matrix={matrix} k={k} />)}
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-end', marginTop: 'auto', height: px(FOOTER_H), fontSize: px(18), color: LIGHT }}>
        <div style={{ display: 'flex' }}>{modelsDate ? `Models as of ${modelsDate} · estimates, formula in the open` : 'Estimates, formula in the open'}</div>
        <div style={{ display: 'flex' }}>{WATERMARK}</div>
      </div>
    </div>
  );
}

function TableBody({ layout, k }: { layout: TableLayout; k: number }) {
  const px = (n: number) => Math.round(n * k);
  return (
    <div style={{ display: 'flex', flexDirection: 'column', marginTop: px(16) }}>
      <div style={{ display: 'flex', height: px(HEADER_H), borderBottom: `${px(2)}px solid ${TEXT}` }}>
        <div style={{ width: px(RAIL_W + SIZE_W), display: 'flex' }} />
        {layout.headers.map((h) => (
          <div key={h.index} style={{ flex: 1, display: 'flex', flexDirection: 'column', justifyContent: 'flex-end', paddingBottom: px(12), paddingRight: px(12) }}>
            <div style={{ fontSize: px(14), letterSpacing: '0.15em', textTransform: 'uppercase', color: LIGHT, display: 'flex' }}>{h.provider}</div>
            <div style={{ fontSize: px(24), fontWeight: 500, marginTop: px(4), display: 'flex' }}>{h.name}</div>
            <div style={{ fontSize: px(16), color: MUTED, marginTop: px(4), display: 'flex' }}>{[h.params, h.quantLabel, h.contextLabel].filter(Boolean).join(' · ')}</div>
          </div>
        ))}
      </div>
      {layout.rails.map((rail) => {
        const rows = layout.rows.filter((r) => rail.rowKeys.includes(r.key));
        return (
          <div key={rail.groupId} style={{ display: 'flex', borderBottom: `1px solid ${LINE}` }}>
            <div style={{ width: px(RAIL_W), display: 'flex', flexDirection: 'column', justifyContent: 'center', paddingRight: px(16) }}>
              <div style={{ fontSize: px(22), fontWeight: 500, display: 'flex' }}>{rail.label}</div>
              {rail.price && <div style={{ fontSize: px(16), color: MUTED, marginTop: px(4), display: 'flex' }}>{rail.price}</div>}
            </div>
            <div style={{ display: 'flex', flexDirection: 'column', flex: 1 }}>
              {rows.map((row) => (
                <div key={row.key} style={{ display: 'flex', height: px(ROW_H), alignItems: 'center' }}>
                  <div style={{ width: px(SIZE_W), fontSize: px(row.linked ? 15 : 20), display: 'flex' }}>{`${row.sizeLabel}${row.binLabel ? ` · ${row.binLabel}` : ''}`}</div>
                  {row.cells.map((c) => (
                    <div key={c.key} style={{ flex: 1, display: 'flex', alignItems: 'center' }}>
                      <MarkerGlyph marker={c.marker} k={k} />
                    </div>
                  ))}
                </div>
              ))}
            </div>
          </div>
        );
      })}
      {layout.footnotes.length > 0 && (
        <div style={{ display: 'flex', flexDirection: 'column', marginTop: px(16) }}>
          {layout.footnotes.slice(0, 3).map((f, i) => (
            <div key={i} style={{ fontSize: px(14), color: MUTED, display: 'flex' }}>{`${i + 1}. ${f}`}</div>
          ))}
        </div>
      )}
    </div>
  );
}

function speedTitle(state: AppState, matrix: Matrix): string {
  const view = state.view;
  const row = view.kind === 'speed' ? matrix.rows.find((r) => r.key === view.row) : matrix.rows[0];
  return row ? `Writing speed on ${rowLabel(row.group, row.row, row.linked)}` : 'Writing speed';
}

function SpeedBody({ state, matrix, factors, k }: { state: AppState; matrix: Matrix; factors: Factors; k: number }) {
  const px = (n: number) => Math.round(n * k);
  const view = state.view;
  const row = (view.kind === 'speed' ? matrix.rows.find((r) => r.key === view.row) : undefined) ?? matrix.rows[0];
  if (!row) return <div style={{ display: 'flex', color: MUTED }}>No machine selected.</div>;
  const items = matrix.columns.map((col) => {
    const cell = matrix.cells.get(cellKey(row.key, col.index));
    const result = cell?.result ?? null;
    const tok = result && result.verdict !== 'no-fit' ? result.speed.tokS : null;
    return { name: col.model?.name ?? col.column.id, tok, feels: result?.speed.feelsLike?.label ?? null, ctx: result?.contextTokens ?? col.column.ctx, fits: !!result && result.verdict !== 'no-fit', unmeasured: result?.speed.efficiencySource !== 'measured' };
  });
  const max = Math.max(80, ...items.map((i) => i.tok ?? 0));
  const barW = 1920 - PAD * 2 - RAIL_W - 200;
  const bands = (factors.feels_like ?? []).filter((b) => b.max_tok_s !== null && (b.max_tok_s as number) < max);
  return (
    <div style={{ display: 'flex', flexDirection: 'column', marginTop: px(24) }}>
      {items.map((it, i) => (
        <div key={i} style={{ display: 'flex', alignItems: 'center', height: px(ROW_H + 12) }}>
          <div style={{ width: px(RAIL_W), fontSize: px(22), fontWeight: 500, display: 'flex' }}>{it.name}</div>
          <div style={{ width: px(barW), height: px(28), background: '#EFECE6', display: 'flex', position: 'relative' }}>
            {it.tok !== null && <div style={{ width: Math.max(px(6), Math.round(px(barW) * Math.min(1, it.tok / max))), height: px(28), background: GOLD, display: 'flex' }} />}
            {bands.map((b, j) => (
              <div key={j} style={{ position: 'absolute', left: Math.round(px(barW) * ((b.max_tok_s as number) / max)), top: 0, width: 1, height: px(28), background: LINE, display: 'flex' }} />
            ))}
          </div>
          <div style={{ width: px(200), paddingLeft: px(16), fontSize: px(20), display: 'flex', flexDirection: 'column' }}>
            <div style={{ display: 'flex' }}>{it.tok !== null ? `${formatTokS(it.tok)}${it.unmeasured ? ' ~' : ''}` : "doesn't fit"}</div>
            <div style={{ fontSize: px(14), color: MUTED, display: 'flex' }}>{it.tok !== null ? `${it.feels ?? ''} · ${formatContext(it.ctx)}` : ''}</div>
          </div>
        </div>
      ))}
      <div style={{ fontSize: px(14), color: LIGHT, marginTop: px(8), display: 'flex' }}>{`Lines mark the feels-like bands (${(factors.feels_like ?? []).map((b) => b.label).join(' · ')}); ~ = unmeasured chip`}</div>
    </div>
  );
}

function modelTitle(state: AppState, matrix: Matrix): string {
  const col = state.view.kind === 'model' ? matrix.columns[state.view.column] : matrix.columns[0];
  return col?.model ? `Which machine do I need for ${col.model.name}?` : 'Which machine do I need?';
}

/** The buying map as a list: qualifying configurations first (the pick with the gold rule), then those that would qualify with a change, then the rest. */
function BuyingBody({ map, k }: { map: BuyingMap; k: number }) {
  const px = (n: number) => Math.round(n * k);
  const rows = buyingImageRows(map);
  const pick = map.recommendation.pick;
  const nothing = !map.recommendation.pick && map.recommendation.unpricedQualifying.length === 0;
  return (
    <div style={{ display: 'flex', flexDirection: 'column', marginTop: px(24) }}>
      {nothing && <div style={{ fontSize: px(20), color: MUTED, marginBottom: px(8), display: 'flex' }}>Nothing qualifies at these settings; the nearest misses first.</div>}
      {rows.map((m) => {
        const marker: Marker = m.status === 'no-fit' ? 'no' : m.status === 'compromise' ? 'ring' : 'run';
        const dim = m.status === 'fits' && !m.qualifies;
        const highlight = pick !== null && m.key === pick.key;
        return (
          <div key={m.key} style={{ display: 'flex', alignItems: 'center', height: px(ROW_H + 12), borderLeft: `${px(4)}px solid ${highlight ? GOLD : 'transparent'}`, paddingLeft: px(16), borderBottom: `1px solid ${LINE}`, opacity: dim ? 0.55 : 1 }}>
            <div style={{ width: px(60), display: 'flex' }}>
              <MarkerGlyph marker={marker} k={k} />
            </div>
            <div style={{ flex: 1, fontSize: px(22), fontWeight: 500, display: 'flex' }}>{m.label}</div>
            <div style={{ width: px(220), fontSize: px(20), display: 'flex' }}>{m.tokS !== null ? `${formatTokS(m.tokS)}${m.approx ? ' ~' : ''}${m.status === 'compromise' ? ' · with a change' : ''}` : m.status === 'no-fit' ? "doesn't fit" : 'no estimate'}</div>
            <div style={{ width: px(200), fontSize: px(20), color: MUTED, display: 'flex', justifyContent: 'flex-end' }}>{m.priceUsd !== null ? formatPrice(m.priceUsd) : 'no current price'}</div>
          </div>
        );
      })}
      <div style={{ fontSize: px(14), color: LIGHT, marginTop: px(8), display: 'flex' }}>{`Cheapest first; a gold rule marks the pick; ~ = unmeasured chip; list prices${map.atlas.machinesAsOf ? ` as of ${map.atlas.machinesAsOf}` : ''}`}</div>
    </div>
  );
}

function ModelBody({ state, matrix, k }: { state: AppState; matrix: Matrix; k: number }) {
  const px = (n: number) => Math.round(n * k);
  const col = (state.view.kind === 'model' ? matrix.columns[state.view.column] : undefined) ?? matrix.columns[0];
  if (!col) return <div style={{ display: 'flex', color: MUTED }}>No model selected.</div>;
  const rows = matrix.rows.map((r) => {
    const result = matrix.cells.get(cellKey(r.key, col.index))?.result ?? null;
    const marker: Marker = !result ? 'loading' : result.verdict === 'no-fit' ? 'no' : result.verdict === 'runs' ? 'run' : 'ring';
    return { key: r.key, label: rowLabel(r.group, r.row, r.linked), price: pooledPrice(r.row.priceUsd, r.linked), marker, tok: result && result.verdict !== 'no-fit' ? result.speed.tokS : null };
  });
  const cheapest = rows.filter((r) => r.marker === 'run' && r.price !== null).sort((a, b) => (a.price as number) - (b.price as number))[0];
  return (
    <div style={{ display: 'flex', flexDirection: 'column', marginTop: px(24) }}>
      {rows.map((r) => (
        <div key={r.key} style={{ display: 'flex', alignItems: 'center', height: px(ROW_H + 12), borderLeft: `${px(4)}px solid ${cheapest && r.key === cheapest.key ? GOLD : 'transparent'}`, paddingLeft: px(16), borderBottom: `1px solid ${LINE}` }}>
          <div style={{ width: px(60), display: 'flex' }}>
            <MarkerGlyph marker={r.marker} k={k} />
          </div>
          <div style={{ flex: 1, fontSize: px(22), fontWeight: 500, display: 'flex' }}>{r.label}</div>
          <div style={{ width: px(200), fontSize: px(20), display: 'flex' }}>{r.tok !== null ? formatTokS(r.tok) : ''}</div>
          <div style={{ width: px(160), fontSize: px(20), color: MUTED, display: 'flex', justifyContent: 'flex-end' }}>{r.price !== null ? formatPrice(r.price) : '—'}</div>
        </div>
      ))}
    </div>
  );
}
