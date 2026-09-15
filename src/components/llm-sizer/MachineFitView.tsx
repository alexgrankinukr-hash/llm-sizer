/** "What can it run?": one machine at one memory size against the whole catalog, best build first, expandable to every build and context. */
import { memo, useDeferredValue, useMemo } from 'react';
import type { CellResult, Factors, ModelDetail, ModelIndex, ModelIndexEntry } from '../../lib/llm-sizer/engine/types';
import { formatContext, formatGb, formatTokS } from '../../lib/llm-sizer/engine/format';
import { rowLabel } from '../../lib/llm-sizer/app/machines';
import type { MatrixRow, RecordState } from '../../lib/llm-sizer/app/matrix';
import type { AppState } from '../../lib/llm-sizer/app/state';
import type { Bucket } from '../../lib/llm-sizer/app/quants';
import { fitSettingsKey, modelGrid, modelSummary, orderCatalog, orderedIds, type ModelSummary, type PickSection } from '../../lib/llm-sizer/app/machine-fit';
import { RowSelect } from './RowSelect';
import { useQueuedRecords } from './useModelRecords';

export interface FitTarget {
  model: ModelDetail;
  pick: MatrixRow;
  bucket: Bucket;
  ctx: number;
  result: CellResult;
}

export interface MachineFitViewProps {
  state: AppState;
  factors: Factors;
  pick: MatrixRow;
  sections: PickSection[];
  index: ModelIndex | null;
  indexStatus: 'loading' | 'ready' | 'error';
  expanded: boolean;
  toggles: { work: string; limit: string };
  onRow: (rowKey: string) => void;
  onExpanded: (expanded: boolean) => void;
  onOpenCell: (target: FitTarget, anchor: HTMLElement) => void;
  onBenchmarks: () => void;
  onRetry: (id: string) => void;
}

function Marker({ marker, label, onClick, disabled }: { marker: string; label: string; onClick?: (e: React.MouseEvent<HTMLButtonElement>) => void; disabled?: boolean }) {
  return (
    <button type="button" className="lls-marker" aria-label={label} title={label} onClick={onClick} disabled={disabled}>
      <span className={`lls-${marker}`} />
    </button>
  );
}

function ColumnHeads() {
  return (
    <div className="lls-fit-cols" aria-hidden="true">
      <span />
      <span>Model</span>
      <span>Parameters</span>
      <span>Memory needed</span>
      <span>Best build at 32K</span>
      <span>Context at Q4</span>
    </div>
  );
}

interface FitRowProps {
  entry: ModelIndexEntry;
  record: RecordState | undefined;
  pick: MatrixRow;
  fitKey: string;
  state: AppState;
  factors: Factors;
  expanded: boolean;
  onOpenCell: (target: FitTarget, anchor: HTMLElement) => void;
  onRetry: (id: string) => void;
}

const FitRow = memo(function FitRow({ entry, record, pick, fitKey, state, factors, expanded, onOpenCell, onRetry }: FitRowProps) {
  const model = record?.status === 'ready' ? record.model : null;
  // the summary and the grid depend on the model, the machine row and the settings the key names; nothing else
  const summary = useMemo<ModelSummary | null>(() => (model ? modelSummary(model, pick, state, factors) : null), [model, pick, fitKey]); // eslint-disable-line react-hooks/exhaustive-deps
  const grid = useMemo(() => (model && expanded ? modelGrid(model, pick, state, factors) : null), [model, pick, fitKey, expanded]); // eslint-disable-line react-hooks/exhaustive-deps
  const machine = rowLabel(pick.group, pick.row, pick.linked);

  if (!model || !summary) {
    const failed = record?.status === 'error' || record?.status === 'missing';
    return (
      <div className="lls-fit-card">
        <div className="lls-fit-summary">
          <Marker marker={failed ? 'missing' : 'loading'} label={`${entry.name} on ${machine}: ${failed ? 'could not be loaded' : 'loading'}`} disabled />
          <div className="min-w-0">
            <div className="font-semibold text-[14px] leading-tight">{entry.name}</div>
            <div className="text-[11px] text-[var(--color-light)]">{entry.provider}</div>
          </div>
          <div className="lls-fit-col text-[13px] text-[var(--color-muted)]">{entry.params_total_b ? `${Math.round(entry.params_total_b)}B` : ''}</div>
          <div className="lls-fit-col" />
          <div className="lls-fit-col text-[13px] text-[var(--color-muted)]">
            {failed ? (
              <>
                could not load this model{' '}
                <button type="button" className="lls-chip" onClick={() => onRetry(entry.id)}>
                  retry
                </button>
              </>
            ) : (
              'loading…'
            )}
          </div>
          <div className="lls-fit-col" />
        </div>
      </div>
    );
  }

  const openAsked = (anchor: HTMLElement) => {
    if (summary.asked) onOpenCell({ model, pick, bucket: 'Q4', ctx: 32768, result: summary.asked }, anchor);
  };
  return (
    <div className={`lls-fit-card${grid ? ' is-open' : ''}`}>
      <div
        className="lls-fit-summary"
        data-clickable={summary.asked ? 'true' : 'false'}
        title={summary.asked ? 'Open the details' : undefined}
        onClick={(e) => openAsked(e.currentTarget)}
      >
        <Marker
          marker={summary.marker}
          label={summary.ariaLabel}
          disabled={!summary.asked}
          onClick={(e) => {
            e.stopPropagation();
            openAsked(e.currentTarget);
          }}
        />
        <div className="min-w-0">
          <div className="font-semibold text-[14px] leading-tight">{model.name}</div>
          <div className="text-[11px] text-[var(--color-light)]">{model.provider}</div>
        </div>
        <div className="lls-fit-col text-[13px]" data-label="Parameters">
          {summary.params}
        </div>
        <div className="lls-fit-col text-[13px]" data-label="Memory needed" title={summary.status === 'no-fit' ? 'what the Q4 build at 32K would need' : 'what the build in the next column needs on this machine'}>
          {summary.memoryText}
        </div>
        <div className="lls-fit-col min-w-0 text-[13px]" data-label="Best build" title={summary.detail ?? summary.headline}>
          <span className="lls-clamp">{summary.headline}</span>
        </div>
        <div className="lls-fit-col text-[12.5px] text-[var(--color-muted)]" data-label="Context" title={summary.detail ?? undefined}>
          {summary.contextLine}
        </div>
      </div>
      {grid && (
        <div className="lls-scroll">
          <table className="lls-fit-grid text-[12px]">
            <caption className="sr-only">
              {model.name} on {machine}: every build (rows) at every context window (columns)
            </caption>
            <colgroup>
              <col style={{ width: 120 }} />
              {grid.chips.map((c) => (
                <col key={c} />
              ))}
            </colgroup>
            <thead>
              <tr>
                <th scope="col" rowSpan={2}>
                  Build
                </th>
                <th scope="colgroup" colSpan={grid.chips.length}>
                  Context window
                </th>
              </tr>
              <tr>
                {grid.chips.map((c) => (
                  <th key={c} scope="col">
                    {formatContext(c)}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {grid.rows.map((r) => (
                <tr key={r.bucket}>
                  <th scope="row">
                    {r.bucket}
                    {r.quant && <span className="block text-[10px] font-normal text-[var(--color-light)] truncate">{r.quant.label}</span>}
                  </th>
                  {r.cells[0]?.kind === 'gap' ? (
                    <td colSpan={grid.chips.length} className="!text-left text-[var(--color-light)]">
                      no {r.bucket} build
                    </td>
                  ) : (
                    r.cells.map((c) =>
                      c.kind === 'cell' ? (
                        <td key={c.ctx}>
                          <Marker marker={c.marker} label={c.ariaLabel} onClick={(e) => onOpenCell({ model, pick, bucket: c.bucket, ctx: c.ctx, result: c.result }, e.currentTarget)} />
                          <span className="lls-fit-tok">{c.tokS !== null ? `${formatTokS(c.tokS)}${c.result.speed.efficiencySource !== 'measured' ? ' ~' : ''}` : ''}</span>
                          <span className="lls-fit-tok">{formatGb((c.result.fix ?? c.result).need.totalGb)}</span>
                        </td>
                      ) : null,
                    )
                  )}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
});

export function MachineFitView(props: MachineFitViewProps) {
  const { state, factors, pick, index, indexStatus, toggles } = props;
  const sections = useMemo(() => orderCatalog(index?.models ?? []), [index]);
  const ids = useMemo(() => orderedIds(sections), [sections]);
  const records = useQueuedRecords(ids);
  const fitKey = fitSettingsKey(state);
  // the grids are hundreds of evaluations: let the toggle paint first
  const expanded = useDeferredValue(props.expanded);
  const machine = rowLabel(pick.group, pick.row, pick.linked);

  return (
    <div>
      <div className="flex flex-wrap items-center gap-3 mb-4">
        <RowSelect sections={props.sections} value={pick.key} onChange={props.onRow} />
        <label className="lls-chip cursor-pointer" title="Every quantization bucket the model has files for, at every context length it allows.">
          <input type="checkbox" className="accent-[#d4af37]" checked={props.expanded} onChange={(e) => props.onExpanded(e.target.checked)} /> show every build and context
        </label>
        <button type="button" className="lls-chip lls-chip-soon" onClick={props.onBenchmarks} title="Measured speeds next to the estimates, when they exist">
          Show benchmarks · coming soon
        </button>
      </div>
      <p className="text-sm text-[var(--color-muted)] mb-3">
        Every model in the catalog on {machine} · {state.runtime === 'mlx' ? 'MLX' : 'GGUF'} · {toggles.work} · {toggles.limit} · grouped by the size of the Q4 file
      </p>
      {props.expanded && (
        <p className="lls-fit-legend">
          In each table a row is a build (Q2 to Q8, smallest file first) and a column is a context window. A sphere runs, a ring runs with a compromise, a dash does not fit; under each mark: the writing speed and the memory needed. Tap any row or mark for the details.
        </p>
      )}
      {indexStatus === 'error' && <p className="text-sm text-[var(--color-muted)] border border-dashed border-[var(--color-border)] rounded-xl px-5 py-8 text-center">The model catalog could not be loaded. Retry above.</p>}
      {indexStatus === 'loading' && (
        <div aria-busy="true">
          {[0, 1, 2, 3, 4, 5].map((i) => (
            <div key={i} className="lls-fit-card">
              <div className="lls-fit-summary">
                <span className="lls-marker" aria-hidden="true">
                  <span className="lls-loading" />
                </span>
                <div className="h-4 w-40 rounded bg-[var(--color-border)]" />
                <div className="lls-fit-col h-4 w-12 rounded bg-[var(--color-border)]" />
                <div className="lls-fit-col h-4 w-14 rounded bg-[var(--color-border)]" />
                <div className="lls-fit-col h-4 w-56 rounded bg-[var(--color-border)]" />
                <div className="lls-fit-col" />
              </div>
            </div>
          ))}
        </div>
      )}
      {sections.length > 0 && <ColumnHeads />}
      {sections.map((section) => (
        <section key={section.key} className="mt-5 first-of-type:mt-0" aria-labelledby={`lls-fit-${section.key}`}>
          <h3 id={`lls-fit-${section.key}`} className="lls-fit-band">
            {section.label} <span>· {section.entries.length === 1 ? '1 model' : `${section.entries.length} models`}</span>
          </h3>
          {section.entries.map((entry) => (
            <FitRow key={entry.id} entry={entry} record={records[entry.id]} pick={pick} fitKey={fitKey} state={state} factors={factors} expanded={expanded} onOpenCell={props.onOpenCell} onRetry={props.onRetry} />
          ))}
        </section>
      ))}
      {sections.length > 0 && (
        <p className="mt-4 text-xs text-[var(--color-light)]">
          Best build = the highest-quality build that runs as asked at 32K; longest context = at Q4; memory needed = what that configuration takes on this machine. Every speed is an estimate; a ring's speed is its fix's. Tap a row for the reasons, the memory bar and the sources.
        </p>
      )}
    </div>
  );
}
