/** LLM Sizer — the root island: state, data loading, URL sync, the chart switch and the settings panel. */
import { useCallback, useEffect, useMemo, useReducer, useRef, useState } from 'react';
import './llm-sizer.css';
import type { Factors, Machine, ModelDetail } from '../../lib/llm-sizer/engine/types';
import { analytics as liveAnalytics, silentAnalytics } from '../../lib/llm-sizer/app/analytics';
import { tableLayout } from '../../lib/llm-sizer/app/layout';
import { machineGroups, pooledGb } from '../../lib/llm-sizer/app/machines';
import { evaluateMatrix } from '../../lib/llm-sizer/app/matrix';
import { modelView, reducer, type Action, type AppState, type CustomModel, type View } from '../../lib/llm-sizer/app/state';
import { encodeState } from '../../lib/llm-sizer/app/url';
import { SITE } from '../../lib/llm-sizer/adapters/site';
import { pickSections, resolvePick, syntheticSheetTarget } from '../../lib/llm-sizer/app/machine-fit';
import { MAX_COLUMNS } from '../../lib/llm-sizer/app/state';
import type { Bucket } from '../../lib/llm-sizer/app/quants';
import type { MatrixCell, MatrixColumn, MatrixRow } from '../../lib/llm-sizer/app/matrix';
import { aboutHref } from '../../lib/llm-sizer/app/provenance';
import { FitTable } from './FitTable';
import { CellSheet } from './CellSheet';
import { SpeedView } from './SpeedView';
import { MemoryView } from './MemoryView';
import { MachineFitView, type FitTarget } from './MachineFitView';
import { MachineMapView, type MapTarget } from './MachineMapView';
import { ShareExport } from './ShareExport';
import { SettingsPanel } from './SettingsPanel';
import { isHiddenView, ViewTiles, type ViewKind } from './ViewTiles';
import { FeedbackDialog } from './FeedbackDialog';
import { NotifyPopup, type NotifyRequest } from './NotifyPopup';
import { MessageIcon, SlidersIcon } from './icons';
import { retryIndex, retryRecord, useModelIndex, useModelRecords } from './useModelRecords';

export interface LlmSizerProps {
  initial: AppState;
  machines: Machine[];
  factors: Factors;
  notices?: string[];
  /** model records rendered on the server for the initial columns, so the first paint is complete */
  initialRecords?: Record<string, ModelDetail>;
  present?: boolean;
  reveal?: boolean;
  /** offer the views listed in HIDDEN_VIEWS too (`?tiles=all`) */
  showHidden?: boolean;
  /** the machine catalog's data date: the buying map's list prices are "as of" it */
  machinesAsOf?: string;
}

const LOCAL_KEY = 'llm-sizer.custom';
const ADVANCED_KEY = 'llm-sizer.advanced';
const SETTINGS_KEY = 'llm-sizer.settings';
/** actions that count as "the visitor is using the tool" (the notify pop-up waits for a few of them, and for some time on the page) */
const INTERACTIONS = new Set<Action['type']>(['SET_WORK', 'SET_CAP', 'ADD_COLUMN', 'ADD_GROUP', 'TOGGLE_SIZE', 'SET_LINKED', 'SET_COLUMN_QUANT', 'SET_COLUMN_CTX', 'SET_VIEW', 'SET_RUNTIME', 'MOVE_COLUMN', 'MOVE_GROUP', 'SET_BUDGET', 'SET_FORM_FACTOR']);

function readLocal<T>(key: string): T | null {
  try {
    const raw = localStorage.getItem(key);
    return raw ? (JSON.parse(raw) as T) : null;
  } catch {
    return null;
  }
}

function writeLocal(key: string, value: unknown) {
  try {
    localStorage.setItem(key, JSON.stringify(value));
  } catch {
    /* private mode or a full store: fine */
  }
}

function Empty({ children }: { children: React.ReactNode }) {
  return <p className="text-sm text-[var(--color-muted)] border border-dashed border-[var(--color-border)] rounded-xl px-5 py-8 text-center">{children}</p>;
}

type OpenCell =
  | { kind: 'matrix'; rowKey: string; column: number; anchor: HTMLElement }
  /** a synthesised cell: from the machine view's grid, or from the buying map (which already is the "which machine" answer) */
  | { kind: 'fit'; origin: 'machine' | 'buying'; target: { cell: MatrixCell; row: MatrixRow; column: MatrixColumn }; ref: { modelId: string; bucket: Bucket; ctx: number }; anchor: HTMLElement };

export default function LlmSizer({ initial, machines, factors, notices = [], initialRecords, present = false, reveal = false, showHidden = false, machinesAsOf }: LlmSizerProps) {
  const analytics = present ? silentAnalytics : liveAnalytics;
  const [state, rawDispatch] = useReducer(reducer, initial);
  const [interactions, setInteractions] = useState(0);
  const dispatch = useCallback((action: Action) => {
    rawDispatch(action);
    if (INTERACTIONS.has(action.type)) setInteractions((n) => n + 1);
  }, []);
  const groups = useMemo(() => machineGroups(machines), [machines]);
  const index = useModelIndex();
  const records = useModelRecords(state.columns.map((c) => c.id), initialRecords);
  const matrix = useMemo(() => evaluateMatrix(state, groups, records, factors), [state, groups, records, factors]);
  const layout = useMemo(() => tableLayout(state, matrix), [state, matrix]);

  const [advancedOpen, setAdvancedOpen] = useState(false);
  const [settingsOpen, setSettingsOpen] = useState(true);
  const [openCell, setOpenCell] = useState<OpenCell | null>(null);
  const [notifyRequest, setNotifyRequest] = useState<NotifyRequest | null>(null);
  const [customForm, setCustomForm] = useState<'model' | 'machine' | null>(null);
  const [feedback, setFeedback] = useState<{ open: boolean; prefill: string | null }>({ open: false, prefill: null });
  const [noticeList, setNoticeList] = useState<string[]>(notices);
  const chartRef = useRef<HTMLDivElement>(null);
  const lastRow = useRef<string | null>('row' in initial.view ? initial.view.row : null);

  // the local mirror of custom entries and the panel flags (the URL wins on conflicts)
  useEffect(() => {
    const local = readLocal<{ customModels?: CustomModel[]; customMachines?: AppState['customMachines'] }>(LOCAL_KEY);
    if (local) rawDispatch({ type: 'HYDRATE_LOCAL', customModels: local.customModels, customMachines: local.customMachines });
    if (readLocal<boolean>(ADVANCED_KEY)) setAdvancedOpen(true);
    const savedSettings = readLocal<boolean>(SETTINGS_KEY);
    if (savedSettings === false) setSettingsOpen(false);
    const onHash = () => {
      if (window.location.hash === '#lls-feedback') setFeedback({ open: true, prefill: null });
    };
    onHash();
    window.addEventListener('hashchange', onHash);
    analytics.view();
    return () => window.removeEventListener('hashchange', onHash);
  }, []);
  useEffect(() => {
    writeLocal(LOCAL_KEY, { customModels: state.customModels, customMachines: state.customMachines });
  }, [state.customModels, state.customMachines]);
  useEffect(() => {
    writeLocal(ADVANCED_KEY, advancedOpen);
  }, [advancedOpen]);
  useEffect(() => {
    writeLocal(SETTINGS_KEY, settingsOpen);
  }, [settingsOpen]);
  useEffect(() => {
    if ('row' in state.view) lastRow.current = state.view.row;
  }, [state.view]);

  // URL sync: throttled, only when the encoded state changed, other params preserved
  const lastEncoded = useRef<string>(encodeState(initial));
  useEffect(() => {
    if (present) return;
    const encoded = encodeState(state);
    if (encoded === lastEncoded.current) return;
    const t = window.setTimeout(() => {
      lastEncoded.current = encoded;
      const url = new URL(window.location.href);
      url.searchParams.set('s', encoded);
      window.history.replaceState(null, '', url.toString());
    }, 250);
    return () => window.clearTimeout(t);
  }, [state, present]);

  const encoded = useMemo(() => encodeState(state), [state]);
  const [origin, setOrigin] = useState('');
  useEffect(() => setOrigin(window.location.origin + window.location.pathname), []);
  const shareUrl = origin ? `${origin}?s=${encodeURIComponent(encoded)}` : '';

  const openCellSheet = useCallback(
    (rowKey: string, column: number, anchor: HTMLElement) => {
      setOpenCell({ kind: 'matrix', rowKey, column, anchor });
      setInteractions((n) => n + 1);
      const cell = matrix.cells.get(`${rowKey}|${column}`);
      analytics.cellOpened(cell?.result?.verdict ?? 'unknown');
    },
    [matrix],
  );
  // the machine view's cells are not table columns: the sheet gets a synthesised target
  const openFitSheet = useCallback((t: FitTarget, anchor: HTMLElement) => {
    const pickRow = t.pick;
    setOpenCell({ kind: 'fit', origin: 'machine', target: syntheticSheetTarget(t.model, pickRow, t.bucket, t.ctx, t.result, state), ref: { modelId: t.model.id, bucket: t.bucket, ctx: t.ctx }, anchor });
    setInteractions((n) => n + 1);
    analytics.cellOpened(t.result.verdict);
  }, [state]);
  // the buying map's marks: the same sheet, named after the exact file the mark was evaluated at
  const openMapSheet = useCallback((t: MapTarget, anchor: HTMLElement) => {
    setOpenCell({ kind: 'fit', origin: 'buying', target: syntheticSheetTarget(t.model, t.pick, t.bucket, t.ctx, t.result, state, t.quant), ref: { modelId: t.model.id, bucket: t.bucket, ctx: t.ctx }, anchor });
    setInteractions((n) => n + 1);
    analytics.cellOpened(t.result.verdict);
  }, [state]);

  // a shared link may still carry a view we no longer offer: render the table at once, then tidy the link
  const hiddenNow = isHiddenView(state.view.kind) && !showHidden;
  const view: View = hiddenNow ? { kind: 'table' } : state.view;
  useEffect(() => {
    if (hiddenNow) dispatch({ type: 'SET_VIEW', view: { kind: 'table' } });
  }, [hiddenNow]);
  const rowFor = (key: string) => matrix.rows.find((r) => r.key === key) ?? matrix.rows[0] ?? null;
  const chartRow = view.kind === 'speed' || view.kind === 'memory' ? rowFor(view.row) : null;
  const modelColumn = view.kind === 'model' ? (matrix.columns[view.column] ?? matrix.columns[0] ?? null) : null;
  // the machine view may name any catalog or custom machine; a stale key falls back to the biggest table row and the link is tidied
  const pick = view.kind === 'machine' ? resolvePick(view.row, groups, state, matrix.rows) : null;
  useEffect(() => {
    if (view.kind === 'machine' && pick && pick.key !== view.row) dispatch({ type: 'SET_VIEW', view: { ...view, row: pick.key } });
  }, [view, pick]);
  const machineSections = useMemo(() => pickSections(groups, state), [groups, state.customMachines, state.groups, state.linked]); // eslint-disable-line react-hooks/exhaustive-deps

  function selectView(kind: ViewKind) {
    // the biggest machine, not the first: on the smallest one every model reads "does not fit" and the chart is empty
    const biggest = matrix.rows.reduce<(typeof matrix.rows)[number] | null>((best, r) => (best === null || pooledGb(r.row.gb, r.linked) > pooledGb(best.row.gb, best.linked) ? r : best), null);
    const row = lastRow.current && matrix.rows.some((r) => r.key === lastRow.current) ? lastRow.current : (biggest?.key ?? '');
    const column = view.kind === 'model' ? view.column : 0;
    const next: View =
      kind === 'table'
        ? { kind: 'table' }
        : kind === 'speed'
          ? { kind: 'speed', row }
          : kind === 'memory'
            ? { kind: 'memory', row }
            : kind === 'machine'
              ? { kind: 'machine', row: resolvePick(lastRow.current, groups, state, matrix.rows)?.key ?? row, expanded: false }
              : modelView(column, view);
    dispatch({ type: 'SET_VIEW', view: next });
    analytics.viewSelected(kind);
  }

  const cellForSheet = openCell ? (openCell.kind === 'fit' ? openCell.target.cell : (matrix.cells.get(`${openCell.rowKey}|${openCell.column}`) ?? null)) : null;
  const rowForSheet = openCell ? (openCell.kind === 'fit' ? openCell.target.row : (matrix.rows.find((r) => r.key === openCell.rowKey) ?? null)) : null;
  const columnForSheet = openCell ? (openCell.kind === 'fit' ? openCell.target.column : (matrix.columns[openCell.column] ?? null)) : null;
  // "Which machine do I need?" from the machine view: jump to the model's column, or add one at that build and context
  const whichMachineFromFit = (() => {
    if (openCell?.kind !== 'fit') return undefined;
    const { modelId, bucket, ctx } = openCell.ref;
    const existing = state.columns.findIndex((c) => c.id === modelId);
    if (existing === -1 && state.columns.length >= MAX_COLUMNS) return undefined;
    return () => {
      const column = existing !== -1 ? existing : state.columns.length;
      if (existing === -1) dispatch({ type: 'ADD_COLUMN', id: modelId, ctx, quant: bucket === 'Q4' ? 'auto' : { bucket } });
      dispatch({ type: 'SET_VIEW', view: modelView(column) });
      setOpenCell(null);
    };
  })();

  if (present) {
    return (
      <div className={`lls-present ${reveal ? 'lls-reveal' : ''}`}>
        <div className="flex items-end justify-between mb-6">
          <div>
            <div className="font-heading text-[40px] font-bold leading-tight">
              What runs on <em className="not-italic text-[#B8912B]">which machine</em>
            </div>
            <div className="text-[15.5px] text-[#8a8985] font-medium mt-1">
              {layout.toggles.work} · {layout.toggles.limit} · quant and context per column
            </div>
          </div>
          <Legend />
        </div>
        {view.kind === 'speed' && chartRow ? (
          <SpeedView state={state} matrix={matrix} row={chartRow} factors={factors} present />
        ) : view.kind === 'memory' ? (
          <MemoryView state={state} matrix={matrix} row={chartRow} present />
        ) : (
          <FitTable state={state} matrix={matrix} layout={layout} present reveal={reveal} />
        )}
        <div className="lls-watermark">{SITE.watermark}</div>
      </div>
    );
  }

  const chart =
    view.kind === 'speed' ? (
      chartRow ? (
        <SpeedView state={state} matrix={matrix} row={chartRow} factors={factors} onRow={(row) => dispatch({ type: 'SET_VIEW', view: { kind: 'speed', row } })} />
      ) : (
        <Empty>Add a machine in Settings to see speeds.</Empty>
      )
    ) : view.kind === 'memory' ? (
      matrix.columns.length > 0 ? (
        <MemoryView state={state} matrix={matrix} row={chartRow} onRow={(row) => dispatch({ type: 'SET_VIEW', view: { kind: 'memory', row } })} onOpenCell={openCellSheet} />
      ) : (
        <Empty>Add a model in Settings to see what it needs.</Empty>
      )
    ) : view.kind === 'machine' ? (
      pick ? (
        <MachineFitView
          state={state}
          factors={factors}
          pick={pick}
          sections={machineSections}
          index={index.status === 'ready' ? index.index : null}
          indexStatus={index.status}
          expanded={view.expanded}
          toggles={layout.toggles}
          onRow={(row) => dispatch({ type: 'SET_VIEW', view: { kind: 'machine', row, expanded: view.expanded } })}
          onExpanded={(expanded) => dispatch({ type: 'SET_VIEW', view: { kind: 'machine', row: pick.key, expanded } })}
          onOpenCell={openFitSheet}
          onBenchmarks={() => {
            analytics.benchmarksInterest(pick.key);
            setNotifyRequest({ topic: 'benchmarks', id: Date.now() });
          }}
          onRetry={retryRecord}
        />
      ) : (
        <Empty>Add a machine in Settings to see what it can run.</Empty>
      )
    ) : view.kind === 'model' ? (
      modelColumn ? (
        <MachineMapView
          state={state}
          column={modelColumn}
          columns={matrix.columns}
          groups={groups}
          factors={factors}
          build={view.build}
          minTokS={view.minTokS}
          toggles={layout.toggles}
          machinesAsOf={machinesAsOf ?? null}
          onColumn={(column) => dispatch({ type: 'SET_VIEW', view: modelView(column, view) })}
          onBuild={(build) => {
            dispatch({ type: 'SET_VIEW', view: { ...view, column: modelColumn.index, build } });
            analytics.toggleChanged('build', build);
          }}
          onMinTokS={(minTokS) => {
            dispatch({ type: 'SET_VIEW', view: { ...view, column: modelColumn.index, minTokS } });
            analytics.toggleChanged('min_tok_s', minTokS);
          }}
          onOpenCell={openMapSheet}
          onAddGroup={(id) => {
            dispatch({ type: 'ADD_GROUP', id });
            analytics.machineAdded(id);
          }}
          onOpenSettings={() => {
            setSettingsOpen(true);
            analytics.settingsOpened();
          }}
        />
      ) : (
        <Empty>Add a model in Settings to compare machines for it.</Empty>
      )
    ) : (
      <FitTable state={state} matrix={matrix} layout={layout} onOpenCell={openCellSheet} onColumnMenu={(column) => dispatch({ type: 'SET_VIEW', view: modelView(column, view) })} onRetryColumn={retryRecord} />
    );

  return (
    <div className="relative lls-root">
      {noticeList.length > 0 && (
        <div className="mb-4 text-sm text-[var(--color-muted)] border border-[var(--color-border)] rounded-lg px-4 py-2 flex items-start justify-between gap-4" role="status">
          <ul className="space-y-0.5">
            {noticeList.map((n) => (
              <li key={n}>{n}</li>
            ))}
          </ul>
          <button type="button" className="text-[var(--color-light)] hover:text-[var(--color-text)] shrink-0" onClick={() => setNoticeList([])} aria-label="Dismiss">
            ×
          </button>
        </div>
      )}
      {index.status === 'error' && (
        <div className="mb-4 text-sm border border-[var(--color-error)] rounded-lg px-4 py-2 flex items-center justify-between gap-4" role="alert">
          <span>The model catalog could not be loaded. The table cannot fill until it does.</span>
          <button type="button" className="lls-chip shrink-0" onClick={retryIndex}>
            Retry
          </button>
        </div>
      )}
      <div className="lls-layout" data-panel={settingsOpen ? 'open' : 'closed'}>
        <ViewTiles className="lls-area-tiles" value={view.kind} onSelect={selectView} showHidden={showHidden} />
        <div className="lls-area-actions flex flex-wrap items-center gap-2">
          <button
            type="button"
            className="lls-chip !py-1.5 !px-3 font-medium"
            aria-expanded={settingsOpen}
            aria-controls="lls-settings"
            aria-pressed={settingsOpen}
            onClick={() => {
              setSettingsOpen((v) => {
                if (!v) analytics.settingsOpened();
                return !v;
              });
            }}
          >
            <SlidersIcon size={14} /> Settings
          </button>
          <div className="ml-auto flex flex-wrap items-center gap-2">
            <button type="button" className="lls-chip !py-1.5 !px-3" onClick={() => setFeedback({ open: true, prefill: null })}>
              <MessageIcon size={14} /> Feedback
            </button>
            <ShareExport encoded={encoded} shareUrl={shareUrl} view={view.kind} />
          </div>
        </div>
        {settingsOpen && (
        <SettingsPanel
          id="lls-settings"
          className="lls-area-panel"
          state={state}
          dispatch={dispatch}
          groups={groups}
          matrix={matrix}
          index={index.status === 'ready' ? index.index : null}
          indexStatus={index.status}
          factors={factors}
          analytics={analytics}
          advancedOpen={advancedOpen}
          onAdvancedToggle={setAdvancedOpen}
          customForm={customForm}
          onCustomForm={setCustomForm}
          onReport={(text) => setFeedback({ open: true, prefill: text })}
          onRetry={retryRecord}
          onClose={() => {
            setSettingsOpen(false);
            const top = chartRef.current?.getBoundingClientRect().top ?? 0;
            if (top < 0 || top > window.innerHeight * 0.6) chartRef.current?.scrollIntoView({ block: 'start', behavior: 'smooth' });
          }}
        />
        )}
        <div ref={chartRef} id="lls-chart" className="lls-area-chart scroll-mt-20" role="tabpanel" aria-labelledby={`lls-tile-${view.kind}`}>
          {chart}
        </div>
        <div className="lls-area-legend">
          <div className="flex flex-wrap items-center justify-between gap-3 text-xs text-[var(--color-light)]">
            <Legend />
            <span>
              Models as of {index.status === 'ready' ? index.index.generated_at.slice(0, 10) : '…'} · every speed is an estimate ·{' '}
              <a className="underline underline-offset-2 hover:text-[var(--color-text)]" href={aboutHref('faq')}>
                how the numbers are made
              </a>{' '}
              ·{' '}
              <a className="underline underline-offset-2 hover:text-[var(--color-text)]" href={SITE.repos.code}>
                open source
              </a>{' '}
              ·{' '}
              <a className="underline underline-offset-2 hover:text-[var(--color-text)]" href={SITE.repos.data}>
                open data
              </a>
            </span>
          </div>
          {layout.footnotes.length > 0 && (
            <ol id="lls-footnotes" className="mt-3 text-xs text-[var(--color-muted)] list-decimal pl-5 space-y-0.5">
              {layout.footnotes.map((f, i) => (
                <li key={i} id={`lls-fn-${i + 1}`}>
                  {f}
                </li>
              ))}
            </ol>
          )}
        </div>
      </div>

      <FeedbackDialog open={feedback.open} prefill={feedback.prefill} onPrefillUsed={() => setFeedback((f) => ({ ...f, prefill: null }))} onClose={() => setFeedback({ open: false, prefill: null })} shareUrl={shareUrl} encoded={encoded} view={view.kind} />
      {SITE.features.notify && <NotifyPopup encoded={encoded} interactions={interactions} request={notifyRequest} />}
      <CellSheet
        groups={groups}
        open={!!openCell && !!cellForSheet?.result}
        cell={cellForSheet}
        row={rowForSheet}
        column={columnForSheet}
        anchor={openCell?.anchor ?? null}
        state={state}
        factors={factors}
        onClose={() => setOpenCell(null)}
        scope={openCell?.kind === 'fit' ? 'machine' : 'table'}
        machinesAsOf={machinesAsOf ?? null}
        onApplyFix={(changes) => {
          if (!openCell) return;
          for (const c of changes) {
            if (c.kind === 'closeApps') dispatch({ type: 'SET_WORK', gb: 0 });
            else if (c.kind === 'override') dispatch({ type: 'SET_CAP', cap: c.cap });
            else if (openCell.kind !== 'matrix') continue; // a build or context is a cell of the machine view's grid, not a setting
            else if (c.kind === 'context') dispatch({ type: 'SET_COLUMN_CTX', index: openCell.column, ctx: c.tokens });
            else if (c.kind === 'quant') dispatch({ type: 'SET_COLUMN_QUANT', index: openCell.column, quant: { label: c.quant.label, repo: c.quant.quantizer } });
            else if (c.kind === 'ssdPaged') {
              const i = matrix.columns[openCell.column]?.model?.special_builds?.findIndex((b) => b.label === c.build.label) ?? -1;
              if (i >= 0) dispatch({ type: 'SET_COLUMN_QUANT', index: openCell.column, quant: { special: i } });
            }
          }
          setOpenCell(null);
        }}
        onWhichMachine={
          openCell?.kind === 'fit'
            ? openCell.origin === 'buying'
              ? undefined
              : whichMachineFromFit
            : () => {
                if (openCell?.kind !== 'matrix') return;
                dispatch({ type: 'SET_VIEW', view: modelView(openCell.column, view) });
                setOpenCell(null);
              }
        }
        onReport={(text) => {
          setOpenCell(null);
          setFeedback({ open: true, prefill: text });
        }}
      />
    </div>
  );
}

export function Legend() {
  return (
    <div className="lls-legend flex items-center gap-5 text-xs text-[var(--color-muted)]">
      <span className="flex items-center gap-1.5">
        <span className="lls-run" /> runs
      </span>
      <span className="flex items-center gap-1.5">
        <span className="lls-ring" /> runs with a compromise · tap to see which
      </span>
      <span className="flex items-center gap-1.5">
        <span className="lls-no" /> doesn't fit
      </span>
    </div>
  );
}
