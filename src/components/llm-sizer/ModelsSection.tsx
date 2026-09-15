/** Settings › Models: the columns (model at a quant and a context), their order and removal, and the add flow. */
import { useEffect, useRef, useState } from 'react';
import type { Dispatch } from 'react';
import type { KvBits, ModelIndex, ModelIndexEntry } from '../../lib/llm-sizer/engine/types';
import { contextChips } from '../../lib/llm-sizer/engine/index';
import { formatContext } from '../../lib/llm-sizer/engine/format';
import type { Analytics } from '../../lib/llm-sizer/app/analytics';
import { ctxOptions, parseQuantValue, quantOptions, quantValue } from '../../lib/llm-sizer/app/column-options';
import type { Matrix } from '../../lib/llm-sizer/app/matrix';
import { availableBuckets, SIMPLE_BUCKETS, type Bucket } from '../../lib/llm-sizer/app/quants';
import { MAX_COLUMNS, type Action, type AppState } from '../../lib/llm-sizer/app/state';
import { GripIcon, PlusIcon, SearchIcon, XIcon } from './icons';
import { ModelList } from './Lists';
import { useListReorder } from './useListReorder';
import { prefetchRecord, useModelRecord } from './useModelRecords';

export interface ModelsSectionProps {
  state: AppState;
  matrix: Matrix;
  dispatch: Dispatch<Action>;
  analytics: Analytics;
  index: ModelIndex | null;
  indexStatus: 'loading' | 'ready' | 'error';
  onCustom: () => void;
  onRetry: (id: string) => void;
}

type Step = 'closed' | 'search' | { entry: ModelIndexEntry; bucket: Bucket; ctx: number };

export function ModelsSection({ state, matrix, dispatch, analytics, index, indexStatus, onCustom, onRetry }: ModelsSectionProps) {
  const columns = matrix.columns;
  const [step, setStep] = useState<Step>('closed');
  const [q, setQ] = useState('');
  const [live, setLive] = useState('');
  const addBtn = useRef<HTMLButtonElement>(null);
  const search = useRef<HTMLInputElement>(null);
  useEffect(() => {
    if (step === 'search') search.current?.focus();
  }, [step]);
  const pending = typeof step === 'object' ? step : null;
  const pendingRecord = useModelRecord(pending?.entry.id ?? null);
  const pendingModel = pendingRecord?.status === 'ready' ? pendingRecord.model : null;
  const pendingBuckets = pendingModel ? availableBuckets(pendingModel, state.runtime) : null;
  const pendingChips = pending ? contextChips({ context_max: pending.entry.context_max }) : [];
  const reorder = useListReorder(columns.length, (from, to) => dispatch({ type: 'MOVE_COLUMN', index: from, to }), setLive);
  const full = state.columns.length >= MAX_COLUMNS;

  function pick(entry: ModelIndexEntry) {
    prefetchRecord(entry.id);
    const chips = contextChips({ context_max: entry.context_max });
    setStep({ entry, bucket: 'Q4', ctx: chips.includes(32768) ? 32768 : chips[chips.length - 1] });
  }
  function add() {
    if (!pending) return;
    dispatch({ type: 'ADD_COLUMN', id: pending.entry.id, quant: pending.bucket === 'Q4' ? 'auto' : { bucket: pending.bucket }, ctx: pending.ctx });
    analytics.modelAdded(pending.entry.id, pending.entry.featured);
    setStep('closed');
    setQ('');
    addBtn.current?.focus();
  }

  return (
    <section aria-labelledby="lls-set-models">
      <div className="flex items-center justify-between gap-3 mb-2">
        <h3 id="lls-set-models" className="lls-eyebrow">Models</h3>
        <button ref={addBtn} type="button" className="lls-chip !py-1 !px-2.5 font-medium" aria-expanded={step !== 'closed'} disabled={full && step === 'closed'} title={full ? `Up to ${MAX_COLUMNS} models` : undefined} onClick={() => setStep((s) => (s === 'closed' ? 'search' : 'closed'))}>
          <PlusIcon size={13} /> Add model
        </button>
      </div>
      {step === 'search' && (
        <div className="mb-3">
          <label className="flex items-center gap-2 px-2 py-1.5 text-sm border border-[var(--color-border)] rounded-lg bg-white">
            <SearchIcon size={14} />
            <input ref={search} className="flex-1 bg-transparent outline-none" placeholder="Search models…" value={q} onChange={(e) => setQ(e.target.value)} aria-label="Search models" />
          </label>
          <ModelList index={index} indexStatus={indexStatus} chosen={state.columns.map((c) => c.id)} query={q} onPick={pick} onCustom={() => { setStep('closed'); onCustom(); }} />
        </div>
      )}
      {pending && (
        <div className="mb-3 border border-[var(--color-border)] rounded-lg p-3 bg-white space-y-2">
          <p className="text-[13.5px] font-semibold">{pending.entry.name}</p>
          <div className="flex flex-wrap items-center gap-2">
            <label className="flex items-center gap-1.5 text-[11px] uppercase tracking-[0.15em] text-[var(--color-light)]">
              Quant
              <select
                aria-label="Quantization"
                className="lls-chip !py-0.5 !px-2 text-[12px] normal-case tracking-normal text-[var(--color-text)] appearance-none pr-1"
                value={pending.bucket}
                onChange={(e) => setStep((cur) => (typeof cur === 'object' ? { ...cur, bucket: e.target.value as Bucket } : cur))}
              >
                {SIMPLE_BUCKETS.map((b) => (
                  <option key={b} value={b} disabled={pendingBuckets !== null && !pendingBuckets.includes(b)}>
                    {b === 'Q4' ? 'Q4 (default)' : b}
                  </option>
                ))}
              </select>
            </label>
            <label className="flex items-center gap-1.5 text-[11px] uppercase tracking-[0.15em] text-[var(--color-light)]">
              Context
              <select
                aria-label="Context length"
                className="lls-chip !py-0.5 !px-2 text-[12px] normal-case tracking-normal text-[var(--color-text)] appearance-none pr-1"
                value={pending.ctx}
                onChange={(e) => setStep((cur) => (typeof cur === 'object' ? { ...cur, ctx: Number(e.target.value) } : cur))}
              >
                {pendingChips.map((c) => (
                  <option key={c} value={c}>
                    {formatContext(c)}
                  </option>
                ))}
              </select>
            </label>
          </div>
          <div className="flex gap-2 pt-1">
            <button type="button" className="lls-chip !py-1 !px-3 font-medium" onClick={add}>
              Add to the table
            </button>
            <button type="button" className="lls-chip !py-1 !px-3" onClick={() => setStep('search')}>
              Back
            </button>
          </div>
        </div>
      )}
      {columns.length === 0 && <p className="text-sm text-[var(--color-muted)]">No model yet.</p>}
      <ul ref={reorder.listRef} className="space-y-2">
        {columns.map((col, i) => {
          const name = col.model?.name ?? col.column.id;
          const status = col.status === 'loading' ? 'loading…' : col.status === 'missing' ? 'not in the catalog' : col.status === 'error' ? 'could not load' : null;
          return (
            <li key={`${col.column.id}-${i}`} {...reorder.itemProps(i)} className={`lls-item ${reorder.itemProps(i).className}`}>
              <div className="flex items-center gap-2">
                <button type="button" className="lls-chip lls-handle !px-1.5" {...reorder.handleProps(i, name)}>
                  <GripIcon size={13} />
                </button>
                <span className="flex-1 min-w-0 leading-tight">
                  <span className="font-semibold text-[13.5px] block truncate">{name}</span>
                  <span className="text-[11px] text-[var(--color-light)]">{col.model?.provider ?? status ?? ''}</span>
                </span>
                {col.status === 'error' && (
                  <button type="button" className="lls-chip !py-0.5 !px-2 text-[11.5px]" onClick={() => onRetry(col.column.id)}>
                    retry
                  </button>
                )}
                <button type="button" className="lls-chip !px-1.5" aria-label={`Remove ${name}`} onClick={() => dispatch({ type: 'REMOVE_COLUMN', index: i })}>
                  <XIcon size={12} />
                </button>
              </div>
              <div className="mt-1.5 flex flex-wrap items-center gap-1 pl-9">
                <select aria-label={`${name}: quantization`} className="lls-chip !py-0.5 !px-2 text-[12px] appearance-none pr-1" value={quantValue(col, state)} onChange={(e) => dispatch({ type: 'SET_COLUMN_QUANT', index: i, quant: parseQuantValue(e.target.value) })} disabled={!col.model}>
                  {quantOptions(col, state).map((o) => (
                    <option key={o.value} value={o.value}>
                      {o.label}
                    </option>
                  ))}
                </select>
                <select aria-label={`${name}: context length`} className="lls-chip !py-0.5 !px-2 text-[12px] appearance-none pr-1" value={col.column.ctx} onChange={(e) => dispatch({ type: 'SET_COLUMN_CTX', index: i, ctx: Number(e.target.value) })}>
                  {ctxOptions(col.model, col.column.ctx).map((c) => (
                    <option key={c} value={c}>
                      {formatContext(c)}
                    </option>
                  ))}
                </select>
                {!state.simpleQuants && (
                  <select aria-label={`${name}: cache precision`} className="lls-chip !py-0.5 !px-2 text-[12px] appearance-none pr-1" value={col.column.kvBits ?? 'global'} onChange={(e) => dispatch({ type: 'SET_COLUMN_KV', index: i, kvBits: e.target.value === 'global' ? undefined : (Number(e.target.value) as KvBits) })}>
                    <option value="global">cache: default ({state.kvBits === 16 ? 'FP16' : `${state.kvBits}-bit`})</option>
                    <option value="16">cache FP16</option>
                    <option value="8">cache 8-bit</option>
                    <option value="4">cache 4-bit</option>
                  </select>
                )}
              </div>
            </li>
          );
        })}
      </ul>
      <span className="sr-only" aria-live="polite">{live}</span>
    </section>
  );
}
