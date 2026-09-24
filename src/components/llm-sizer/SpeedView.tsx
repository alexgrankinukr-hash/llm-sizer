/** Speed view: writing speed for every model column on one machine row, over the feels-like bands; and, on the Reading tab,
 * the wait before the first word for a prompt size you pick (the prefill estimate, METHODOLOGY 6.4). */
import { useState } from 'react';
import type { Factors, Quant, SpecialBuild } from '../../lib/llm-sizer/engine/types';
import { isSpecialBuild } from '../../lib/llm-sizer/engine/memory';
import { formatContext, formatPrice, formatTokS, formatWait } from '../../lib/llm-sizer/engine/format';
import { firstWordWaitS, gpuUpgradeOf, prefillCoresOf } from '../../lib/llm-sizer/engine/prefill';
import { runningConfig } from '../../lib/llm-sizer/app/layout';
import { rowLabel } from '../../lib/llm-sizer/app/machines';
import type { Matrix, MatrixRow } from '../../lib/llm-sizer/app/matrix';
import type { AppState } from '../../lib/llm-sizer/app/state';
import { RowSelect } from './RowSelect';
import { InfoIcon } from './icons';
import { axisTicks, clip, textWidth } from './chart';

export interface SpeedViewProps {
  state: AppState;
  matrix: Matrix;
  row: MatrixRow;
  factors: Factors;
  present?: boolean;
  onRow?: (rowKey: string) => void;
}

const W = 900;
const LABEL_W = 250;

export type SpeedMode = 'writing' | 'reading';
/** the prompt sizes the Reading tab offers; 'column' = each column's own context, the prompt that fills it */
export type PromptPick = 'column' | 2048 | 8192 | 32768 | 131072;
const PROMPT_PICKS: PromptPick[] = [2048, 8192, 32768, 131072, 'column'];
/** "an 8K-token prompt", "a 32K-token prompt" */
const promptPhrase = (n: number) => `${/^8/.test(formatContext(n)) ? 'an' : 'a'} ${formatContext(n)}-token prompt`;
/** the reading axis is a log scale of seconds from 1 s to 1 h: waits span three orders of magnitude */
const READ_MIN_S = 1;
const READ_MAX_S = 3600;
const READ_TICKS: [number, string][] = [[1, '1 s'], [3, '3 s'], [10, '10 s'], [30, '30 s'], [60, '1 min'], [300, '5 min'], [900, '15 min'], [3600, '1 h']];




/** What a bar says when there is no number: an SSD build quotes its publisher, anything else the engine's own note. */
function noSpeedText(quant: Quant | SpecialBuild | null, notes: string[]): string {
  if (quant && isSpecialBuild(quant)) return `speed not estimated${quant.measured_tok_s ? ` · publisher: ${quant.measured_tok_s} tok/s on ${quant.measured_chip ? `an ${quant.measured_chip}` : quant.measured_on ?? 'their machine'}` : ''}`;
  return notes.find((n) => n.startsWith('no speed figure')) ?? 'no estimate for this platform';
}

/** "under 5" · "5–12" · "60 and up": the band's range, said once in the legend. */
function bandRange(from: number, to: number | null): string {
  if (to === null) return `${from} and up`;
  if (from === 0) return `under ${to}`;
  return `${from}–${to}`;
}

export function SpeedView(props: SpeedViewProps) {
  const { state, matrix, row, factors, present } = props;
  const [showAccel, setShowAccel] = useState(true);
  const [mode, setMode] = useState<SpeedMode>('writing');
  const [prompt, setPrompt] = useState<PromptPick>(8192);
  const reading = mode === 'reading' && !present && !!factors.prefill;
  const cells = matrix.columns.flatMap((c) => {
    const cell = matrix.cells.get(`${row.key}|${c.index}`);
    return cell ? [{ column: c, cell }] : [];
  });
  const speeds = cells.map(({ cell }) => cell.result?.speed.tokS ?? 0);
  const accelMax = cells.map(({ cell }) => {
    const s = cell.result?.speed;
    if (!s?.tokS || !s.accelerations.length || !showAccel) return 0;
    return s.tokS * Math.max(...s.accelerations.map((a) => a.max));
  });
  const axisMax = Math.max(80, Math.ceil((Math.max(...speeds, ...accelMax, 1) * 1.15) / 20) * 20);
  const rowH = present ? 64 : 48;
  const H = 40 + cells.length * rowH + 28;
  const x = (v: number) => LABEL_W + (Math.min(v, axisMax) / axisMax) * (W - LABEL_W - 20);
  const bands = factors.feels_like;
  const ticks = axisTicks(axisMax);

  return (
    <div>
      {!present && (
        <div className="flex flex-wrap items-center gap-3 mb-4">
          <RowSelect rows={matrix.rows} value={row.key} onChange={(k) => props.onRow?.(k)} />
          {factors.prefill && (
            <div className="lls-seg" role="group" aria-label="Writing or reading">
              <button type="button" aria-pressed={mode === 'writing'} onClick={() => setMode('writing')} title="tokens per second while the model writes its answer">
                Writing
              </button>
              <button type="button" aria-pressed={mode === 'reading'} onClick={() => setMode('reading')} title="the wait before the first word: how long the model takes to read your prompt">
                Reading
              </button>
            </div>
          )}
          {reading && (
            <div className="flex items-center gap-2 text-sm">
              <span className="text-[var(--color-muted)]">prompt</span>
              <div className="lls-seg" role="group" aria-label="Prompt size">
                {PROMPT_PICKS.map((pk) => (
                  <button key={String(pk)} type="button" aria-pressed={prompt === pk} onClick={() => setPrompt(pk)} title={pk === 'column' ? "a prompt that fills each column's whole context, pasted in one go" : promptPhrase(pk)}>
                    {pk === 'column' ? 'the whole context' : formatContext(pk)}
                  </button>
                ))}
              </div>
            </div>
          )}
          {!reading && <label
            className="lls-chip cursor-pointer"
            title="Some models can write several tokens per step (MTP) or lean on a small draft model. Where that exists, the lighter part of a bar shows how fast it can get."
          >
            <input type="checkbox" className="accent-[#d4af37]" checked={showAccel} onChange={(e) => setShowAccel(e.target.checked)} /> acceleration ranges
            <InfoIcon size={13} />
          </label>}
        </div>
      )}
      {reading && <ReadingBars cells={cells} row={row} state={state} factors={factors} prompt={prompt} />}
      {!reading && (
      <>
      <p className={`${present ? 'text-[22px] font-extrabold' : 'text-sm text-[var(--color-muted)]'} mb-2`}>
        Writing speed on {rowLabel(row.group, row.row, row.linked)} · {row.row.bandwidthGbs >= 1000 ? `${(row.row.bandwidthGbs / 1000).toFixed(1)} TB/s` : `${row.row.bandwidthGbs} GB/s`}{row.linked ? ' each' : ''} · {state.runtime === 'mlx' ? 'MLX' : 'GGUF'}{row.linked ? ` · ${row.linked.split === 'tensor' ? 'tensor parallel' : 'layer split'}` : ''}
      </p>
      <ul className="lls-bands mb-3" aria-label="How the speeds feel, in tokens per second">
        {bands.map((b, i) => (
          <li key={b.label} className="lls-band">
            <b>{b.label}</b>
            {bandRange(i === 0 ? 0 : bands[i - 1].max_tok_s ?? 0, b.max_tok_s)}
          </li>
        ))}
      </ul>
      <div className="lls-scroll">
        <svg viewBox={`0 0 ${W} ${H}`} width="100%" style={{ minWidth: present ? 1600 : 640, fontFamily: 'Inter, sans-serif' }} role="img" aria-label="Speed bars">
          {ticks.map((t, i) => {
            const last = i === ticks.length - 1;
            const room = i === 0 || x(t) - x(ticks[i - 1]) > 44;
            return (
              <g key={t}>
                <line x1={x(t)} x2={x(t)} y1={8} y2={H - 24} stroke="rgba(26,26,26,0.12)" strokeDasharray="2 4" />
                {room && (
                  <text x={x(t)} y={H - 8} fontSize={11} textAnchor={last ? 'end' : 'middle'} fill="#98968f">
                    {last ? `${t} tok/s` : t}
                  </text>
                )}
              </g>
            );
          })}
          {cells.map(({ column, cell }, i) => {
            const y = 30 + i * rowH;
            const s = cell.result?.speed;
            const tokS = s?.tokS ?? null;
            const name = column.model?.name ?? column.column.id;
            const running = runningConfig(cell.result, { quantLabel: column.quant?.label ?? null, ctx: column.column.ctx });
            const asked = `${column.quant?.label ?? 'default build'} · ${formatContext(column.column.ctx)}`;
            // the label column is 250 wide: keep this line inside it, or it runs across the bars
            const sub = clip(
              `${running.quantLabel} · ${formatContext(running.ctx)}${cell.result?.flags.includes('no-mlx-file') ? ' · GGUF file' : ''}`,
              present ? 34 : 42,
            );
            const barH = present ? 26 : 18;
            return (
              <g key={column.index}>
                <text x={0} y={y + barH / 2 - 2} fontSize={present ? 18 : 13} fontWeight={700} fill="#1A1A1A">
                  {name}
                </text>
                <text x={0} y={y + barH / 2 + (present ? 16 : 12)} fontSize={present ? 13 : 11} fill={running.atTheFix ? '#9E7C25' : '#7a7974'}>
                  {running.atTheFix ? `at the fix: ${sub}` : sub}
                </text>
                {running.atTheFix && (
                  <text x={0} y={y + barH / 2 + (present ? 32 : 24)} fontSize={present ? 12 : 10} fill="#7a7974">
                    {clip(`${asked} did not fit`, present ? 34 : 44)}
                  </text>
                )}
                {tokS !== null && cell.result?.verdict !== 'no-fit' ? (
                  (() => {
                    const accelTop = showAccel && s && s.accelerations.length > 0 ? tokS * Math.max(...s.accelerations.map((a) => a.max)) : null;
                    const tilde = s && s.efficiencySource !== 'measured' ? ' ~' : '';
                    const fontSize = present ? 16 : 12;
                    const baseLabel = `${formatTokS(tokS)}${tilde}${running.atTheFix ? ' at the fix' : ''}`;
                    // the plain speed sits after the solid bar; the ceiling after the lighter one, when the two do not collide
                    // Inter at these sizes runs about 0.42 em per character; the pad keeps the two labels apart
                    const width = textWidth;
                    const room = accelTop !== null && x(accelTop) - x(tokS) > width(baseLabel, fontSize) + 10;
                    return (
                      <>
                        {accelTop !== null && <rect x={x(tokS)} y={y} width={Math.max(0, x(accelTop) - x(tokS))} height={barH} fill="rgba(212,175,55,0.3)" rx={3} />}
                        {running.atTheFix ? (
                          <rect x={LABEL_W + 1} y={y + 1} width={Math.max(2, x(tokS) - LABEL_W - 2)} height={barH - 2} fill="rgba(212,175,55,0.18)" stroke="#9E7C25" strokeWidth={1.5} strokeDasharray="5 3" rx={3} />
                        ) : (
                          <rect x={LABEL_W} y={y} width={Math.max(2, x(tokS) - LABEL_W)} height={barH} fill="#D4AF37" rx={3} />
                        )}
                        {(accelTop === null || room) && (
                          <text x={x(tokS) + 8} y={y + barH / 2 + 4} fontSize={fontSize} fill="#1A1A1A">
                            {baseLabel}
                          </text>
                        )}
                        {accelTop !== null &&
                          (() => {
                            const text = room ? `up to ${formatTokS(accelTop)}` : `${baseLabel} · up to ${formatTokS(accelTop)}`;
                            // a label that would run off the right edge turns inward instead
                            const outside = x(accelTop) + 8 + width(text, fontSize) * 1.15 < W;
                            return (
                              <text
                                x={outside ? x(accelTop) + 8 : x(accelTop) - 8}
                                y={y + barH / 2 + 4}
                                textAnchor={outside ? 'start' : 'end'}
                                fontSize={fontSize}
                                fill={room && outside ? '#7a7974' : '#1A1A1A'}
                              >
                                {text}
                              </text>
                            );
                          })()}
                      </>
                    );
                  })()
                ) : (
                  <text x={LABEL_W + 4} y={y + barH / 2 + 4} fontSize={12} fill="#98968f">
                    {tokS === null ? noSpeedText(cell.result?.quant ?? null, s?.notes ?? []) : 'does not fit'}
                  </text>
                )}
              </g>
            );
          })}
        </svg>
      </div>
      </>
      )}
      {!reading && (
      <p className="mt-2 text-xs text-[var(--color-light)]">
        Estimates from bandwidth ÷ bytes read per token, calibrated on measured machines; "~" marks chips without measured rows. The lighter part of a bar is the ceiling with MTP or a draft model, and carries its own number. A dashed bar is "at the fix": the setup you asked for did not fit, so the row runs a smaller file or a shorter context. A smaller file reads faster, which is why a smaller machine can show a higher number than a bigger one; compare those rows with what they run, not with the others.
      </p>
      )}
    </div>
  );
}

/** The Reading tab: one bar per model, the wait before the first word for the picked prompt, on a log axis of seconds. */
function ReadingBars({ cells, row, state, factors, prompt }: { cells: { column: SpeedViewProps['matrix']['columns'][number]; cell: NonNullable<ReturnType<SpeedViewProps['matrix']['cells']['get']>> }[]; row: MatrixRow; state: AppState; factors: Factors; prompt: PromptPick }) {
  const pf = factors.prefill!;
  const rowH = 48;
  const barH = 18;
  const H = 40 + cells.length * rowH + 28;
  const lx = (sec: number) => LABEL_W + (Math.log(Math.min(Math.max(sec, READ_MIN_S), READ_MAX_S) / READ_MIN_S) / Math.log(READ_MAX_S / READ_MIN_S)) * (W - LABEL_W - 20);
  // the bin the list price buys at this size; the chip's larger bin, if any, is named beside it
  const cores = prefillCoresOf(row.row.machine, row.row.gb);
  const bigger = gpuUpgradeOf(row.row.machine, row.row.gb);
  const bands = pf.feels_like;
  // band edges on the axis, so the shading says what a wait feels like
  const edges = bands.map((b) => b.max_s ?? READ_MAX_S);
  return (
    <>
      <p className="text-sm text-[var(--color-muted)] mb-2">
        Reading speed on {rowLabel(row.group, row.row, row.linked)}{cores ? ` · ${cores} GPU cores${row.linked ? ' each' : ''}` : ''}{bigger ? ` (the ${bigger.cores}-core GPU reads ${Math.round((bigger.ratio - 1) * 100)} % faster${bigger.usd !== null ? `, ${formatPrice(bigger.usd)} more${row.linked ? ' each' : ''}` : ''})` : ''} · {state.runtime === 'mlx' ? 'MLX' : 'GGUF'}{row.linked ? ` · ${row.linked.split === 'tensor' ? 'tensor parallel' : 'layer split'}` : ''}: the wait before the first word, {prompt === 'column' ? "for a prompt that fills each column's whole context, pasted in one go" : `for ${promptPhrase(prompt)}`}
      </p>
      <ul className="lls-bands mb-3" aria-label="How the waits feel, in seconds">
        {bands.map((b, i) => (
          <li key={b.label} className="lls-band">
            <b>{b.label}</b>
            {b.max_s === null ? `over ${formatWait(bands[i - 1]?.max_s ?? 0)}` : i === 0 ? `under ${formatWait(b.max_s)}` : `${formatWait(bands[i - 1].max_s ?? 0)} to ${formatWait(b.max_s)}`}
          </li>
        ))}
      </ul>
      <div className="lls-scroll">
        <svg viewBox={`0 0 ${W} ${H}`} width="100%" style={{ minWidth: 640, fontFamily: 'Inter, sans-serif' }} role="img" aria-label="Wait before the first word">
          {edges.map((e, i) => {
            const from = i === 0 ? READ_MIN_S : edges[i - 1];
            const fills = ['rgba(212,175,55,0.16)', 'rgba(212,175,55,0.1)', 'rgba(212,175,55,0.05)', 'rgba(26,26,26,0.035)'];
            return <rect key={bands[i].label} x={lx(from)} y={8} width={Math.max(0, lx(e) - lx(from))} height={H - 32} fill={fills[i] ?? fills[3]} />;
          })}
          {READ_TICKS.map(([t, label], i) => (
            <g key={t}>
              <line x1={lx(t)} x2={lx(t)} y1={8} y2={H - 24} stroke="rgba(26,26,26,0.12)" strokeDasharray="2 4" />
              <text x={lx(t)} y={H - 8} fontSize={11} textAnchor={i === READ_TICKS.length - 1 ? 'end' : 'middle'} fill="#98968f">
                {label}
              </text>
            </g>
          ))}
          {cells.map(({ column, cell }, i) => {
            const y = 30 + i * rowH;
            const name = column.model?.name ?? column.column.id;
            const p = cell.result?.prefill ?? null;
            const running = runningConfig(cell.result, { quantLabel: column.quant?.label ?? null, ctx: column.column.ctx });
            const sub = clip(`${running.quantLabel} · ${formatContext(running.ctx)}${running.atTheFix ? ' · at the fix' : ''}`, 42);
            const tokens = prompt === 'column' ? (cell.result?.contextTokens ?? column.column.ctx) : prompt;
            const wait = p && p.tokS !== null && p.d0 !== null ? firstWordWaitS(p.tokS, p.d0, tokens) : null;
            const tilde = p && p.source !== 'measured' ? ' ~' : '';
            const label = wait === null ? null : `${formatWait(wait)}${tilde} · ${formatTokS(p!.tokS)} to start`;
            const fits = cell.result && cell.result.verdict !== 'no-fit';
            return (
              <g key={column.index}>
                <text x={0} y={y + barH / 2 - 2} fontSize={13} fontWeight={700} fill="#1A1A1A">
                  {name}
                </text>
                <text x={0} y={y + barH / 2 + 12} fontSize={11} fill={running.atTheFix ? '#9E7C25' : '#7a7974'}>
                  {sub}
                </text>
                {fits && wait !== null ? (
                  <>
                    <rect x={LABEL_W} y={y} width={Math.max(2, lx(wait) - LABEL_W)} height={barH} fill="#D4AF37" rx={3} />
                    <text x={lx(wait) + 8 + textWidth(label!, 12) * 1.05 < W ? lx(wait) + 8 : lx(wait) - 8} y={y + barH / 2 + 4} fontSize={12} textAnchor={lx(wait) + 8 + textWidth(label!, 12) * 1.05 < W ? 'start' : 'end'} fill="#1A1A1A">
                      {label}
                    </text>
                  </>
                ) : (
                  <text x={LABEL_W + 4} y={y + barH / 2 + 4} fontSize={12} fill="#98968f">
                    {!fits ? 'does not fit' : (p?.notes[0] ?? 'reading speed not estimated')}
                  </text>
                )}
              </g>
            );
          })}
        </svg>
      </div>
      <p className="mt-2 text-xs text-[var(--color-light)]">
        The wait is the prompt read at the chip's prefill rate, which is compute: it scales with GPU cores and the generation, not with memory or the file's bits, and it falls as the context fills (halved once {formatContext(pf.context.d0_tokens)} tokens are in, for the model it was fitted on). What the model has already read stays in memory, so in a chat each turn waits only for its new text; the whole-context figure is the cost of pasting that much in one go. "~" marks chips without measured prefill rows. Shorter is better. Details and sources in the sheet.
      </p>
    </>
  );
}
