/** The explanation sheet: why a cell is what it is, the fix, alternatives, and the memory bar. One native dialog. */
import { useEffect, useLayoutEffect, useMemo, useRef } from 'react';
import type { Change, Factors, Fix, Quant, SpecialBuild, Speed as CellResultSpeed } from '../../lib/llm-sizer/engine/types';
import { formatContext, formatDate, formatGb, formatPrice, formatTokS, formatWait } from '../../lib/llm-sizer/engine/format';
import { FLAG_TEXT, verdictSentence } from '../../lib/llm-sizer/app/layout';
import { pooledPrice, rowLabel, type MachineGroup } from '../../lib/llm-sizer/app/machines';
import type { MatrixCell, MatrixColumn, MatrixRow } from '../../lib/llm-sizer/app/matrix';
import type { AppState } from '../../lib/llm-sizer/app/state';
import { ABOUT_PATH, cellSources, efficiencyProvenance, type SourceLink } from '../../lib/llm-sizer/app/provenance';
import { InfoIcon, XIcon } from './icons';
import { MemoryBar } from './MemoryBar';
import { cheapestSingleForColumn, type EvalMark } from '../../lib/llm-sizer/app/buying';
import { prefillCoresOf } from '../../lib/llm-sizer/engine/prefill';


/** How a pool's speed was made from one machine's, for the sheet's "how this number is made" line. */
function clusterClause(c: NonNullable<CellResultSpeed['cluster']>): string {
  const basis = c.source === 'fitted' ? `fitted on ${c.n} published cluster measurements` : 'assumed';
  if (c.split === 'layer') return `one machine's ${c.singleMs.toFixed(1)} ms per token for the whole model plus ${c.hopMs} ms per extra machine (a layer split; ${basis})`;
  return `one machine's ${c.singleMs.toFixed(1)} ms per token ÷ ${c.machines} plus ${c.cMs} ms × log2(${c.machines}) (tensor parallel; ${basis})`;
}

/** The comparison a linked pool's sheet carries: the cheapest single machine that runs the same model as asked. */
function comparisonText(single: EvalMark | null, poolPrice: number | null, poolTokS: number | null): string {
  if (!single || single.priceUsd === null) return 'No single machine in the catalog runs this as asked: linking is the way to run it.';
  const diff = poolPrice === null ? null : poolPrice - single.priceUsd;
  const price = diff === null ? '' : diff >= 1 ? `, ${formatPrice(diff)} cheaper` : diff <= -1 ? `, ${formatPrice(-diff)} more` : ', at the same price';
  let pace = '';
  if (single.tokS !== null && poolTokS !== null && poolTokS > 0) {
    const r = single.tokS / poolTokS;
    pace = r >= 2.6 ? ' and about three times as fast' : r >= 1.8 ? ' and about twice as fast' : r >= 1.1 ? ` and about ${r.toFixed(1)}× as fast` : r >= 0.9 ? ' and about as fast' : ` and about ${(1 / r).toFixed(1)}× slower`;
  }
  const speed = single.tokS !== null ? ` (${single.approx ? '~' : ''}${formatTokS(single.tokS)}, an estimate)` : '';
  return `One ${single.label} (${formatPrice(single.priceUsd)}) runs this as asked${price}${pace}${speed}. Link only when nothing single fits.`;
}

export interface CellSheetProps {
  /** the machine catalog, so a linked pool's sheet can name the cheapest single machine that runs the model */
  groups?: MachineGroup[];
  open: boolean;
  cell: MatrixCell | null;
  row: MatrixRow | null;
  column: MatrixColumn | null;
  anchor: HTMLElement | null;
  state: AppState;
  factors: Factors;
  onClose: () => void;
  onApplyFix: (changes: Change[]) => void;
  /** absent when there is nowhere to go (the machine view with a full table) */
  onWhichMachine?: () => void;
  onReport: (text: string) => void;
  /** in the machine view only the global fixes (close apps, lift the limit) can be applied; a build or context is a cell of the grid */
  scope?: 'table' | 'machine';
  /** the machine catalog's data date: the list price is "as of" it */
  machinesAsOf?: string | null;
}

function Section({ title, aside, children }: { title: string; aside?: React.ReactNode; children: React.ReactNode }) {
  return (
    <section className="lls-sheet-section">
      <p className="lls-sheet-eyebrow">
        <span>{title}</span>
        {aside && <span className="lls-sheet-aside">{aside}</span>}
      </p>
      {children}
    </section>
  );
}


function FixRow({ fix, onApply, primary, canApply = true }: { fix: Fix; onApply: (c: Change[]) => void; primary?: boolean; canApply?: boolean }) {
  return (
    <div className={`flex items-start justify-between gap-3 ${primary ? '' : 'text-[var(--color-muted)]'}`}>
      <span className="text-[13px] leading-snug">{primary ? fix.reason : `or: ${fix.reason}`}</span>
      {canApply && (
        <button type="button" className="lls-chip shrink-0" onClick={() => onApply(fix.changes)}>
          apply
        </button>
      )}
    </div>
  );
}

function Src({ link, children, className }: { link: SourceLink | null; children: React.ReactNode; className?: string }) {
  if (!link) return <>{children}</>;
  return (
    <a className={`underline decoration-dotted underline-offset-2 hover:text-[var(--color-text)] ${className ?? ''}`} href={link.href} title={link.label} {...(link.external ? { target: '_blank', rel: 'noopener' } : {})}>
      {children}
    </a>
  );
}

export function CellSheet(props: CellSheetProps) {
  const ref = useRef<HTMLDialogElement>(null);
  const { open, cell, row, column } = props;
  const kvBits = column?.column.kvBits ?? props.state.kvBits;
  // a linked pool's sheet names the cheapest single machine that runs the same model as asked, so two boxes never quietly beat one that fits
  const single = useMemo(() => {
    const model = column?.model;
    if (!row?.linked || !model || !column || !props.groups) return null;
    return cheapestSingleForColumn(model, column, props.groups, props.state, props.factors);
  }, [row?.key, row?.linked, column, props.groups, props.state, props.factors]); // eslint-disable-line react-hooks/exhaustive-deps

  useEffect(() => {
    const d = ref.current;
    if (!d) return;
    if (open && !d.open) {
      d.showModal();
      d.querySelector<HTMLElement>('#lls-sheet-title')?.focus();
    }
    if (!open && d.open) d.close();
  }, [open]);

  // desktop: anchor the card near the cell; mobile: the stylesheet pins it to the bottom
  useLayoutEffect(() => {
    const d = ref.current;
    if (!d || !open) return;
    if (window.innerWidth < 768 || !props.anchor) {
      d.style.top = d.style.left = '';
      return;
    }
    const r = props.anchor.getBoundingClientRect();
    const width = Math.min(540, window.innerWidth * 0.92);
    const height = Math.min(d.offsetHeight || 420, window.innerHeight * 0.88);
    let left = r.right + 12;
    if (left + width > window.innerWidth - 8) left = Math.max(8, r.left - width - 12);
    let top = r.top - 24;
    if (top + height > window.innerHeight - 8) top = Math.max(8, window.innerHeight - height - 8);
    d.style.position = 'fixed';
    d.style.margin = '0';
    d.style.left = `${left}px`;
    d.style.top = `${top}px`;
  }, [open, props.anchor, cell]);

  if (!cell?.result || !row || !column) return <dialog ref={ref} className="lls-sheet" onClose={props.onClose} />;
  const result = cell.result;
  const model = column.model;
  const title = `${model?.name ?? column.column.id} on ${rowLabel(row.group, row.row, row.linked)}`;
  const poolPrice = pooledPrice(row.row.priceUsd, row.linked);
  const comparison = row.linked ? comparisonText(single, poolPrice, result.speed.tokS) : null;
  const speed = result.speed;
  const flags = result.flags.filter((f) => f !== 'unmeasured-chip');
  // the numbers in the sheet belong to the build that fits (a fix may have swapped the file), so the links do too
  const fittingQuant = result.fix ? result.fix.changes.reduce<Quant | SpecialBuild | null>((q, c) => (c.kind === 'quant' ? c.quant : c.kind === 'ssdPaged' ? c.build : q), result.quant) : result.quant;
  const src = model ? cellSources({ model, quant: fittingQuant, machine: row.row.machine, factors: props.factors }) : null;
  const eff = efficiencyProvenance(row.row.machine, props.factors);
  // a row that merges two GPU bins: the price and the reading speed both describe the bin the price buys at this size
  const pricedCores = (row.row.machine.gpu_cores?.length ?? 0) > 1 ? prefillCoresOf(row.row.machine, row.row.gb) : null;
  const upgrade = result.prefill?.upgrade ?? null;
  const upgradeWait = upgrade?.waits.find((w) => w.tokens === 32768) ?? upgrade?.waits[upgrade.waits.length - 1] ?? null;
  const canApply = (fix: Fix) => (props.scope ?? 'table') === 'table' || fix.changes.every((c) => c.kind === 'closeApps' || c.kind === 'override');

  return (
    <dialog
      ref={ref}
      className="lls-sheet"
      aria-labelledby="lls-sheet-title"
      onClose={props.onClose}
      onClick={(e) => {
        if (e.target === ref.current) props.onClose();
      }}
    >
      <div className="p-5">
        <div className="flex items-start justify-between gap-3">
          <div>
            <p className="text-[11px] uppercase tracking-[0.2em] text-[var(--color-light)]">
              <Src link={src?.model ?? null}>{model?.name ?? column.column.id}</Src> on <Src link={src?.machine ?? null}>{rowLabel(row.group, row.row, row.linked)}</Src>
            </p>
            <p id="lls-sheet-title" tabIndex={-1} className="font-heading text-xl mt-1 outline-none">
              {verdictSentence(result)}
            </p>
          </div>
          <button type="button" className="lls-chip !px-1.5 shrink-0" onClick={props.onClose} aria-label="Close">
            <XIcon size={14} />
          </button>
        </div>

        {result.verdict === 'compromise' && result.fix && (
          <div className="mt-3 space-y-2">
            <FixRow fix={result.fix} onApply={props.onApplyFix} primary canApply={canApply(result.fix)} />
            {result.alternatives.map((f, i) => (
              <FixRow key={i} fix={f} onApply={props.onApplyFix} canApply={canApply(f)} />
            ))}
          </div>
        )}
        {result.verdict === 'no-fit' && (
          <div className="mt-3 space-y-2 text-[13px]">
            {result.nearestMiss && <p className="text-[var(--color-muted)]">{result.nearestMiss}.</p>}
            {result.belowFloor && (
              <p className="text-[var(--color-muted)]">
                Below the quality floor: {result.belowFloor.reason}, quality unknown. Lower the floor in Advanced to allow it.
              </p>
            )}
          </div>
        )}

        {!row.row.machine.custom && (
          <div className="lls-sheet-price">
            {row.row.priceUsd !== null && poolPrice !== null ? (
              <>
                <span className="lls-sheet-price-big">{formatPrice(poolPrice)}</span>
                <span className="text-[12px] text-[var(--color-muted)]">
                  {row.linked ? `${row.linked.count} × ${formatPrice(row.row.priceUsd)} · ` : ''}list price{pricedCores ? ` for the ${pricedCores}-core GPU` : ''} ({row.row.machine.kind === 'mac' ? 'Apple' : row.row.machine.family.split(' ')[0]}{props.machinesAsOf ? `, as of ${formatDate(props.machinesAsOf)}` : ''})
                  {src?.machine && (
                    <>
                      {' · '}
                      <Src link={src.machine}>bandwidth, prices, sources</Src>
                    </>
                  )}
                </span>
              </>
            ) : (
              <span className="text-[13px] text-[var(--color-muted)]">
                no current price for this size{src?.machine ? <> · <Src link={src.machine}>bandwidth, prices, sources</Src></> : null}
              </span>
            )}
          </div>
        )}

        <Section title={row.linked ? `Memory across these ${row.linked.count} × ${row.row.gb} GB machines` : `Memory on this ${row.row.gb} GB machine`} aside={`${formatGb(result.breakdown.ramGb)}${row.linked ? ' pooled' : ''} in file-size units · needs ${formatGb(result.need.totalGb)} of ${formatGb(result.availability.availableGb)} available${result.fix ? ' after the fix' : ''}${result.breakdown.ssdGb > 0.05 ? ` · plus ${formatGb(result.breakdown.ssdGb)} on the SSD` : ''}`}>
          <MemoryBar breakdown={result.breakdown} showLegend />
          {result.need.cacheGb > 0.05 && (
            <p className="mt-1.5 text-[12px] text-[var(--color-light)]">
              {kvBits === 16
                ? `The context cache (${formatGb(result.need.cacheGb)}) is counted at FP16, what llama.cpp, LM Studio, Ollama and MLX store unless told otherwise. An 8-bit cache halves it, a 4-bit one quarters it: Advanced, Cache precision.`
                : `The context cache (${formatGb(result.need.cacheGb)}) is counted at ${kvBits}-bit, set under Advanced, Cache precision; at FP16, the runtimes' default, it would be ${formatGb((result.need.cacheGb * 16) / kvBits)}.`}
            </p>
          )}
        </Section>

        <Section title="Writing speed">
          {speed.tokS !== null ? (
            <div className="text-[13px]">
              <div className="flex flex-wrap items-baseline gap-x-2 gap-y-0.5">
                <span className="font-heading text-[22px] leading-none">{formatTokS(speed.tokS)}</span>
                {speed.feelsLike && <span className="text-[var(--color-muted)]">{speed.feelsLike.description}</span>}
                <span className="lls-sheet-estimate" title={`${eff.text}${speed.modelKind === 'floor' && speed.readMs !== undefined && speed.bEffGbs !== undefined && speed.t0Ms !== undefined ? ` · made of ${speed.readMs.toFixed(1)} ms to read ${speed.gbPerToken.toFixed(1)} GB per token at ${Math.round(speed.bEffGbs)} GB/s effective, ${speed.t0Ms.toFixed(1)} ms of chip fixed cost${speed.archCostMs ? `, ${speed.archCostMs.toFixed(1)} ms for the architecture` : ''}${speed.runtimeFactor !== 1 ? ` (×${speed.runtimeFactor} on MLX)` : ''}${speed.attnMs ? `, ${speed.attnMs.toFixed(1)} ms for the context` : ''}` : ''}`}>
                  <Src link={src?.speed ?? null}>estimate</Src>
                  <InfoIcon size={12} />
                </span>
              </div>
              {speed.accelerations.length > 0 && (
                <p className="mt-1 text-[var(--color-muted)]">
                  up to {formatTokS(speed.tokS * Math.max(...speed.accelerations.map((a) => a.max)))} with {speed.accelerations.map((a) => (a.kind === 'mtp' ? 'MTP' : 'a draft model')).join(' or ')}
                </p>
              )}
              {speed.notes[0] && result.verdict === 'compromise' && <p className="mt-1 text-[12px] text-[var(--color-light)]">{speed.notes[0]}</p>}
              <details className="mt-1.5 text-[12px] text-[var(--color-light)]">
                <summary className="cursor-pointer select-none">how this number is made</summary>
                <p className="mt-1">
                  <Src link={src?.efficiency ?? null}>{eff.text}</Src>
                  {speed.modelKind === 'floor' && speed.readMs !== undefined && speed.bEffGbs !== undefined && speed.t0Ms !== undefined ? (
                    <>
                      {' '}· made of {speed.readMs.toFixed(1)} ms to read {speed.gbPerToken.toFixed(1)} GB per token at {Math.round(speed.bEffGbs)} GB/s effective, {speed.t0Ms.toFixed(1)} ms of chip fixed cost{speed.archCostMs ? `, ${speed.archCostMs.toFixed(1)} ms for the architecture` : ''}{speed.runtimeFactor !== 1 ? ` (×${speed.runtimeFactor} on MLX)` : ''}{speed.attnMs ? `, ${speed.attnMs.toFixed(1)} ms for the context` : ''}
                    </>
                  ) : null}
                  {speed.cluster ? ` · then, across ${speed.cluster.machines} linked machines, ${clusterClause(speed.cluster)}` : ''}
                </p>
              </details>
              {comparison && <p className="mt-1.5 text-[12px] text-[var(--color-muted)]">{comparison}</p>}
            </div>
          ) : (
            <p className="text-[13px] text-[var(--color-muted)]">{speed.notes[0] ?? 'no estimate for this platform yet'}</p>
          )}
        </Section>

        {result.prefill && (
          <Section title="Reading speed">
            {result.prefill.tokS !== null ? (
              <div className="text-[13px]">
                <div className="flex flex-wrap items-baseline gap-x-2 gap-y-0.5">
                  <span className="font-heading text-[22px] leading-none">{formatTokS(result.prefill.tokS)}</span>
                  <span className="text-[var(--color-muted)]">reading a short prompt{result.prefill.parts?.cores ? ` on the ${result.prefill.parts.cores}-core GPU${row.linked ? 's' : ''}` : ''}{result.prefill.source !== 'measured' ? ' ~' : ''}</span>
                </div>
                <p className="mt-1 text-[var(--color-muted)]">
                  wait before the first word: {result.prefill.waits.map((w) => `${formatContext(w.tokens)} prompt ${formatWait(w.seconds)}`).join(' · ')}
                  {result.prefill.atContext && !result.prefill.waits.some((w) => w.tokens === result.prefill!.atContext!.tokens) ? ` · ${formatContext(result.prefill.atContext.tokens)} (this column, pasted in one go) ${formatWait(result.prefill.atContext.seconds)}` : ''}
                  {'. In a chat each turn waits only for its new text; what the model has already read stays in memory.'}
                </p>
                {upgrade && (
                  <p className="mt-1 text-[var(--color-muted)]">
                    The {upgrade.cores}-core GPU{upgrade.usd !== null ? ` (${formatPrice(upgrade.usd)} more${row.linked ? ' per machine' : ''} at Apple)` : ''} reads {Math.round((upgrade.ratio - 1) * 100)} % faster: {formatTokS(upgrade.tokS)}{upgradeWait ? `, a ${formatContext(upgradeWait.tokens)} prompt waits ${formatWait(upgradeWait.seconds)}` : ''}. Writing speed is the same on both.
                  </p>
                )}
                <details className="mt-1.5 text-[12px] text-[var(--color-light)]">
                  <summary className="cursor-pointer select-none">how this number is made</summary>
                  <p className="mt-1">
                    {result.prefill.parts && (
                      <>
                        {result.prefill.parts.cores !== null && result.prefill.parts.perCore !== null
                          ? `${result.prefill.parts.cores} GPU cores × ${result.prefill.parts.perCore} tokens·B/s per core (the ${result.prefill.parts.generation} generation, ${result.prefill.source === 'measured' ? 'fitted on this chip\'s measured rows' : result.prefill.source === 'generation' ? 'fitted on its generation, no rows for this chip' : 'assumed from the newest measured generation'})`
                          : `a per-machine constant of ${Math.round(result.prefill.parts.k)} tokens·B/s`}
                        {result.prefill.parts.kquant !== 1 ? ` × ${result.prefill.parts.kquant} for a K-quant file` : ''}
                        {result.prefill.parts.classFactor !== 1 ? ` × ${result.prefill.parts.classFactor} for a ${result.prefill.parts.cls.replace('_', ' ')} model` : ''}
                        {result.prefill.parts.mlx !== 1 ? ` × ${result.prefill.parts.mlx} on MLX` : ''}
                        {` ÷ ${result.prefill.parts.activeB.toFixed(1)}B active parameters. The rate halves after ${formatContext(Math.round(result.prefill.d0 ?? 0))} tokens of context, so a long prompt waits more than its length.`}
                      </>
                    )}
                    {result.prefill.notes.length > 0 && ` ${result.prefill.notes.join('. ')}.`}
                    {' '}
                    <a href={`${ABOUT_PATH}#64-reading-speed-prefill-and-the-wait-before-the-first-word`} className="underline underline-offset-2">how reading speed is estimated</a>
                  </p>
                </details>
              </div>
            ) : (
              <p className="text-[13px] text-[var(--color-muted)]">{result.prefill.notes[0] ?? 'no reading estimate'}</p>
            )}
          </Section>
        )}
        {flags.length > 0 && (
          <Section title="Notes">
            <ul className="text-[12px] text-[var(--color-muted)] space-y-0.5">
              {flags.map((f) => (
                <li key={f}>· {FLAG_TEXT[f]}</li>
              ))}
            </ul>
          </Section>
        )}

        <p className="mt-4 text-[12px] text-[var(--color-light)]">
          Sources:{' '}
          {[src?.model, src?.weights, src?.cache, src?.available, src?.speed, src?.efficiency, src?.machine]
            .filter((l): l is SourceLink => !!l)
            .map((l, i, arr) => (
              <span key={l.href + l.label}>
                <Src link={l}>{l.label}</Src>
                {i < arr.length - 1 ? ' · ' : ''}
              </span>
            ))}
        </p>

        <div className="mt-5 flex flex-wrap gap-2">
          {props.onWhichMachine && (
            <button type="button" className="lls-chip" onClick={props.onWhichMachine}>
              Which machine do I need?
            </button>
          )}
          <button type="button" className="lls-chip" onClick={() => props.onReport(`Wrong number? ${title}: ${verdictSentence(result)}`)}>
            Report a wrong number
          </button>
          <a className="lls-chip" href={ABOUT_PATH} target="_blank" rel="noopener">
            How this is computed
          </a>
        </div>
      </div>
    </dialog>
  );
}
