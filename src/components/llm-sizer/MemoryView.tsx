/**
 * "How much memory?" — the memory one configuration needs, model by model: weights, context cache and
 * buffers on an absolute scale. Independent of any machine, so the same model at two builds or two context
 * lengths can be compared side by side; a dashed line marks what one machine of yours could give it.
 */
import { useState } from 'react';
import { formatContext, formatGb } from '../../lib/llm-sizer/engine/format';
import { memoryNeeds } from '../../lib/llm-sizer/app/needs';
import { groupLabel } from '../../lib/llm-sizer/app/machines';
import { cellKey, type Matrix, type MatrixRow } from '../../lib/llm-sizer/app/matrix';
import type { AppState } from '../../lib/llm-sizer/app/state';
import { axisTicks, clip } from './chart';
import { RowSelect } from './RowSelect';
import { InfoIcon } from './icons';

export interface MemoryViewProps {
  state: AppState;
  matrix: Matrix;
  /** the machine whose limit is drawn as a reference line */
  row: MatrixRow | null;
  present?: boolean;
  onRow?: (rowKey: string) => void;
  onOpenCell?: (rowKey: string, column: number, anchor: HTMLElement) => void;
}

const W = 900;
const LABEL_W = 250;
const PARTS = [
  {
    key: 'weightsGb',
    label: 'weights',
    fill: '#D4AF37',
    about: 'The model file itself, at this build. The same file however long your prompts are.',
  },
  {
    key: 'cacheGb',
    label: 'context cache',
    fill: 'rgba(212,175,55,0.55)',
    about: 'What the model keeps about the tokens in the window. It grows with the context length, and how fast depends on the attention design.',
  },
  {
    key: 'buffersGb',
    label: 'buffers',
    fill: 'rgba(212,175,55,0.26)',
    about: 'Working room the runtime needs beside the weights and the cache: 1.5 GB plus 1 % of the weights.',
  },
  {
    key: 'scratchGb',
    label: 'prompt scratch (MLX)',
    fill: 'rgba(212,175,55,0.14)',
    about: 'What MLX sets aside to process a long prompt. Only MLX files above 8K of context carry it; GGUF files run the prompt in the buffers.',
  },
] as const;

export function MemoryView({ state, matrix, row, present, onRow, onOpenCell }: MemoryViewProps) {
  // the chart is about the models; a machine to measure them against is something you ask for
  const [compare, setCompare] = useState(false);
  const needs = memoryNeeds(state, matrix, compare && row?.linked ? { n: row.linked.count, split: row.linked.split } : null);
  const available = compare && row ? (matrix.cells.get(cellKey(row.key, matrix.columns[0]?.index ?? 0))?.result?.availability.availableGb ?? null) : null;
  const biggest = Math.max(1, ...needs.map((n) => n.need?.totalGb ?? 0), available ?? 0);
  const axisMax = biggest * 1.12;
  const ticks = axisTicks(axisMax);
  const rowH = present ? 64 : 52;
  const H = 34 + needs.length * rowH + 28;
  const x = (gb: number) => LABEL_W + (Math.min(gb, axisMax) / axisMax) * (W - LABEL_W - 20);
  const barH = present ? 26 : 18;
  // the scratch part is only worth a legend entry when a bar carries it (MLX files above 8K of context)
  const parts = PARTS.filter((p) => p.key !== 'scratchGb' || needs.some((n) => (n.need?.scratchGb ?? 0) > 0.02));
  const anySsd = needs.some((n) => (n.need?.ssdGb ?? 0) > 0.02);

  return (
    <div>
      {!present && (
        <div className="flex flex-wrap items-center gap-3 mb-4">
          <label className="lls-chip cursor-pointer" title="Draw a line at what one of your machines can give a model, and mark the bars that pass it.">
            <input type="checkbox" className="accent-[#d4af37]" checked={compare} onChange={(e) => setCompare(e.target.checked)} disabled={matrix.rows.length === 0} /> compare with a machine
          </label>
          {compare && matrix.rows.length > 0 && <RowSelect rows={matrix.rows} value={row?.key ?? matrix.rows[0].key} onChange={(k) => onRow?.(k)} label="Compare against" />}
          {compare && available !== null && <span className="text-[13px] text-[var(--color-muted)]">can give a model {formatGb(available)}</span>}
        </div>
      )}
      <p className={`${present ? 'text-[22px] font-extrabold' : 'text-sm text-[var(--color-muted)]'} mb-2`}>Memory needed, per model and build</p>
      <ul className="lls-bands mb-3" aria-label="What each part of a bar is">
        {parts.map((p) => (
          <li key={p.key} className="lls-band lls-band-help" title={p.about}>
            <span className="lls-swatch" style={{ background: p.fill }} aria-hidden="true" />
            {p.label}
            <InfoIcon size={12} />
          </li>
        ))}
        {anySsd && (
          <li className="lls-band lls-band-help" title="What an engram-on-SSD build streams from the SSD: its n-gram table, read once per token. Not memory, so it sits beside the bar.">
            <span className="lls-swatch" style={{ background: 'repeating-linear-gradient(135deg, #B8912B 0 2px, #F3E3A4 2px 4px)' }} aria-hidden="true" />
            on the SSD
            <InfoIcon size={12} />
          </li>
        )}
      </ul>
      {needs.length === 0 && <p className="text-sm text-[var(--color-muted)]">Add a model in Settings.</p>}
      {needs.length > 0 && (
        <div className="lls-scroll">
          <svg viewBox={`0 0 ${W} ${H}`} width="100%" style={{ minWidth: present ? 1600 : 640, fontFamily: 'Inter, sans-serif' }} role="img" aria-label="Memory needed per model">
          <defs>
            <pattern id="lls-ssd-hatch" width={6} height={6} patternUnits="userSpaceOnUse" patternTransform="rotate(45)">
              <rect width={3} height={6} fill="#B8912B" />
              <rect x={3} width={3} height={6} fill="#F3E3A4" />
            </pattern>
          </defs>
            {ticks.map((t, i) => {
              const last = i === ticks.length - 1;
              const room = i === 0 || x(t) - x(ticks[i - 1]) > 44;
              return (
                <g key={t}>
                  <line x1={x(t)} x2={x(t)} y1={8} y2={H - 24} stroke="rgba(26,26,26,0.12)" strokeDasharray="2 4" />
                  {room && (
                    <text x={x(t)} y={H - 8} fontSize={11} textAnchor={last ? 'end' : 'middle'} fill="#98968f">
                      {last ? `${t} GB` : t}
                    </text>
                  )}
                </g>
              );
            })}
            {available !== null && row && (
              <g>
                <line x1={x(available)} x2={x(available)} y1={22} y2={H - 24} stroke="#1A1A1A" strokeWidth={1.5} strokeDasharray="5 4" />
                <text x={x(available) - 6} y={18} fontSize={11} textAnchor="end" fill="#1A1A1A">
                  {clip(`${row.linked ? `${row.linked.count} × ${groupLabel(row.group)} ${row.row.gb} GB` : `${groupLabel(row.group)} ${row.row.gb} GB`} gives ${formatGb(available)}`, 52)}
                </text>
              </g>
            )}
            {needs.map((n, i) => {
              const y = 34 + i * rowH;
              const sub = clip(`${n.buildLabel} · ${formatContext(n.contextTokens)}${n.clamped ? ' · capped' : ''}`, present ? 34 : 42);
              const over = n.need !== null && available !== null && n.need.totalGb > available;
              let offset = LABEL_W;
              return (
                <g key={n.index}>
                  <text x={0} y={y + barH / 2 - 2} fontSize={present ? 18 : 13} fontWeight={700} fill="#1A1A1A">
                    {n.name}
                  </text>
                  <text x={0} y={y + barH / 2 + (present ? 16 : 12)} fontSize={present ? 13 : 11} fill="#7a7974">
                    {sub}
                  </text>
                  {n.need ? (
                    <>
                      {parts.map((p) => {
                        const gb = n.need![p.key];
                        const w = Math.max(0, x(gb) - LABEL_W);
                        const rect = (
                          <rect key={p.key} x={offset} y={y} width={w} height={barH} fill={p.fill}>
                            <title>{`${p.label}: ${formatGb(gb)}`}</title>
                          </rect>
                        );
                        offset += w;
                        return gb > 0.02 ? rect : null;
                      })}
                      {n.need.ssdGb > 0.02 && (
                        <rect x={x(n.need.totalGb) + 2} y={y + 3} width={Math.max(0, x(n.need.totalGb + n.need.ssdGb) - x(n.need.totalGb) - 2)} height={barH - 6} fill="url(#lls-ssd-hatch)" opacity={0.9}>
                          <title>{`on the SSD: ${formatGb(n.need.ssdGb)}, streamed, not memory`}</title>
                        </rect>
                      )}
                      <text x={Math.min(x(n.need.totalGb + n.need.ssdGb) + 8, W - 4)} y={y + barH / 2 + 4} fontSize={present ? 16 : 12} fill={over ? '#C0392B' : '#1A1A1A'}>
                        {formatGb(n.need.totalGb)}
                        {n.need.ssdGb > 0.02 ? ` + ${formatGb(n.need.ssdGb)} on the SSD` : ''}
                        {over ? ' · over' : ''}
                      </text>
                    </>
                  ) : (
                    <text x={LABEL_W + 4} y={y + barH / 2 + 4} fontSize={12} fill="#98968f">
                      {n.status === 'loading' ? 'loading…' : n.status === 'missing' ? 'not in the catalog' : 'no downloadable build known'}
                    </text>
                  )}
                  {!present && n.need && (
                    <rect
                      x={LABEL_W}
                      y={y - 6}
                      width={Math.max(0, x(n.need.totalGb) - LABEL_W)}
                      height={barH + 12}
                      fill="transparent"
                      className={row && onOpenCell ? 'lls-hit' : undefined}
                      onClick={row && onOpenCell ? (e) => onOpenCell(row.key, n.index, e.currentTarget as unknown as HTMLElement) : undefined}
                    >
                      <title>
                        {`${n.name} · ${n.buildLabel} · ${formatContext(n.contextTokens)}\n${formatGb(n.need.totalGb)} in total: ${formatGb(n.need.weightsGb)} weights, ${formatGb(n.need.cacheGb)} context cache, ${formatGb(n.need.buffersGb)} buffers${n.need.scratchGb > 0.02 ? `, ${formatGb(n.need.scratchGb)} prompt scratch` : ''}${n.need.ssdGb > 0.02 ? `; plus ${formatGb(n.need.ssdGb)} streamed from the SSD` : ''}`}
                      </title>
                    </rect>
                  )}
                </g>
              );
            })}
          </svg>
        </div>
      )}
      <p className="mt-2 text-xs text-[var(--color-light)]">
        Weights are the file on disk; the context cache grows with the context length and the model's attention design; buffers are the runtime's working room. Add the same model twice in Settings, at another build or context, to compare them here.
      </p>
    </div>
  );
}
