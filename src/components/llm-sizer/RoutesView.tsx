/**
 * "Every way to 128 GB": one model at one build on every route to about 128 GB of memory for local AI, side by side, with
 * the price, the memory speed, the memory the AI can use, whether it fits, the writing speed and the time for a 1,000-word
 * answer, and the reading time for a 50-page document where a run measured it. A published plain run wins over the
 * estimate; a route with no calibrated rule says so instead of showing a number (METHODOLOGY §6.6). Tap a row for the
 * file it runs, the memory it needs, how the number is made and the sources.
 */
import { Fragment, useEffect, useMemo, useState } from 'react';
import type { Factors, Machine, ModelDetail, ModelIndex } from '../../lib/llm-sizer/engine/types';
import { formatContext, formatGb, formatPrice } from '../../lib/llm-sizer/engine/format';
import { orderCatalog } from '../../lib/llm-sizer/app/machine-fit';
import { customModelDetail, type MatrixColumn, type RecordState } from '../../lib/llm-sizer/app/matrix';
import { bucketOf, quantsInBucket, SIMPLE_BUCKETS, type Bucket } from '../../lib/llm-sizer/app/quants';
import { ROUTE_GROUPS, routesContext, routesTable, secondsText, type Measured, type RouteResult, type RoutesFile } from '../../lib/llm-sizer/app/routes';
import { aboutHref } from '../../lib/llm-sizer/app/provenance';
import type { AppState, View } from '../../lib/llm-sizer/app/state';
import type { Analytics } from '../../lib/llm-sizer/app/analytics';
import { SITE } from '../../lib/llm-sizer/adapters/site';
import { useModelRecord } from './useModelRecords';
import { ChevronDownIcon } from './icons';

export type RoutesViewState = Extract<View, { kind: 'routes' }>;

const ROUTES_URL = '/data/llm-sizer/routes-128gb.json';
let routesCache: RoutesFile | null = null;
let routesPromise: Promise<RoutesFile | null> | null = null;

/** The routes file: the copy rendered with the page when the link opened on this view, else fetched once. */
function useRoutesFile(initial: RoutesFile | undefined): { file: RoutesFile | null; failed: boolean } {
  if (initial && !routesCache) routesCache = initial;
  const [file, setFile] = useState<RoutesFile | null>(routesCache);
  const [failed, setFailed] = useState(false);
  useEffect(() => {
    if (file) return;
    routesPromise ??= fetch(ROUTES_URL)
      .then(async (r) => (r.ok ? ((await r.json()) as RoutesFile) : null))
      .catch(() => null);
    let alive = true;
    routesPromise.then((f) => {
      if (!alive) return;
      if (f) {
        routesCache = f;
        setFile(f);
      } else {
        routesPromise = null;
        setFailed(true);
      }
    });
    return () => {
      alive = false;
    };
  }, [file]);
  return { file, failed };
}

export interface RoutesViewProps {
  state: AppState;
  view: RoutesViewState;
  factors: Factors;
  /** the full catalog (the Macs and the Spark the routes name) */
  machines: Machine[];
  columns: MatrixColumn[];
  records: Record<string, RecordState>;
  seed?: Record<string, ModelDetail>;
  initialRoutes?: RoutesFile;
  index: ModelIndex | null;
  analytics: Analytics;
  onView: (view: RoutesViewState) => void;
}

/** The builds a model can be shown at here: the simple buckets with a GGUF or an MLX file. */
function routeBuckets(model: ModelDetail): Bucket[] {
  return SIMPLE_BUCKETS.filter((b) => quantsInBucket(model, b, 'gguf').some((q) => q.format !== 'safetensors' && bucketOf(q.label) === b) || quantsInBucket(model, b, 'mlx').some((q) => q.format === 'mlx' && bucketOf(q.label) === b));
}

function num(tokS: number): string {
  return tokS >= 10 ? String(Math.round(tokS)) : tokS.toFixed(1);
}

function rangeText(r: [number, number]): string {
  return `${num(r[0])}–${num(r[1])}`;
}

function priceText(p: RouteResult['route']['price']): string {
  if (p.usd === null) return '—';
  if (p.usd_high && p.usd_high !== p.usd) return `${formatPrice(p.usd)}–${formatPrice(p.usd_high).slice(1)}`;
  return formatPrice(p.usd);
}

const CONDITION: Record<RouteResult['route']['condition'], string | null> = { new: null, used: 'used', preorder: 'pre-order' };

function FitCell({ r }: { r: RouteResult }) {
  const { pick } = r;
  const usable = r.route.usable.gb;
  if (pick.verdict === 'no-file') return <span className="text-[var(--color-muted)]">no file at this build</span>;
  if (pick.verdict === 'no-fit') {
    return (
      <span className="lls-routes-fit">
        <span className="lls-no" aria-hidden />
        <span>
          doesn't fit
          {pick.nearest && <span className="lls-routes-sub">needs {formatGb(pick.nearest.totalGb, 0)}</span>}
          {r.route.usable.alt && (r.altVerdict === 'runs' || r.altVerdict === 'tight') ? <span className="lls-routes-sub">fits {r.route.usable.alt.when}</span> : null}
        </span>
      </span>
    );
  }
  const need = pick.need?.totalGb ?? 0;
  const label = pick.verdict === 'runs' ? 'runs' : pick.verdict === 'tight' ? 'tight' : 'spills into RAM';
  const alt = r.route.usable.alt && r.altVerdict === 'no-fit' ? `not ${r.route.usable.alt.when}` : null;
  return (
    <span className="lls-routes-fit">
      <span className={pick.verdict === 'runs' ? 'lls-run' : 'lls-ring'} aria-hidden />
      <span>
        {label}
        <span className="lls-routes-sub">
          {pick.verdict === 'spills' ? `${formatGb(need, 0)} total` : `${formatGb(need, 0)} of ${formatGb(usable, 0)}`}
        </span>
        {alt && <span className="lls-routes-sub">{alt}</span>}
      </span>
    </span>
  );
}

function WriteCell({ r }: { r: RouteResult }) {
  const h = r.headline;
  const fits = r.pick.verdict === 'runs' || r.pick.verdict === 'tight' || r.pick.verdict === 'spills';
  const speedup = r.speedups[0];
  return (
    <span className="block">
      {h.kind === 'measured' && h.tokS !== null && (
        <>
          <span className="lls-routes-num">{num(h.tokS)}</span> <span className="lls-routes-tag" data-kind="measured">measured</span>
          {h.range && <span className="lls-routes-sub">{rangeText(h.range)}{r.measured?.n ? `, ${r.measured.n} runs` : ''}</span>}
        </>
      )}
      {h.kind === 'estimate' && h.tokS !== null && (
        <>
          <span className="lls-routes-num">≈ {num(h.tokS)}</span> <span className="lls-routes-tag" data-kind="estimate">estimate</span>
          {r.caution && <span className="lls-routes-sub lls-routes-warn">less sure here: tap the row</span>}
        </>
      )}
      {h.kind === 'none' && (
        <>
          <span className="lls-routes-num text-[var(--color-light)]">—</span>{' '}
          {fits && <span className="lls-routes-tag" data-kind="none">no estimate</span>}
          {r.ceiling !== null && <span className="lls-routes-sub">at most {num(r.ceiling)} by memory speed</span>}
        </>
      )}
      {speedup && (
        <span className="lls-routes-sub">
          speed-up: {speedup.write_range ? rangeText(speedup.write_range) : num(speedup.write)} ({speedup.label ?? speedup.engine.split(',')[0].split(' (')[0]})
        </span>
      )}
    </span>
  );
}

function readingText(r: RouteResult): { main: string; sub: string } | null {
  const rd = r.reading;
  if (!rd) return null;
  const s = rd.seconds;
  const main = Array.isArray(s) ? `${secondsText(s[1])}–${secondsText(s[0])}`.replace(/ s–/, '–') : secondsText(s);
  return { main, sub: rd.engine };
}

function sourceLabel(url: string): string {
  try {
    const u = new URL(url);
    const host = u.hostname.replace(/^www\./, '');
    if (host === 'omlx.ai') return `oMLX ${u.pathname.split('/').pop()}`;
    if (host === 'github.com') return u.pathname.split('/').slice(1, 3).join('/') + (u.pathname.includes('discussions') ? ` #${u.pathname.split('/').pop()}` : '');
    return host;
  } catch {
    return url;
  }
}

function Links({ urls }: { urls: string[] }) {
  const unique = [...new Set(urls)];
  return (
    <>
      {unique.map((u, i) => (
        <Fragment key={u}>
          {i > 0 && ', '}
          <a className="underline underline-offset-2 hover:text-[var(--color-text)]" href={u} target="_blank" rel="noreferrer">
            {sourceLabel(u)}
          </a>
        </Fragment>
      ))}
    </>
  );
}

/** A note as a sentence: its own full stop kept, one added when it has none. */
function sentence(text: string | undefined): string {
  if (!text) return '';
  return /[.!?]$/.test(text.trim()) ? `${text.trim()} ` : `${text.trim()}. `;
}

function runLine(m: Measured): string {
  const parts = [`${m.write_range ? rangeText(m.write_range) : num(m.write)} tok/s`, m.n ? `${m.n} runs` : null, m.engine, m.build, m.variant ?? null, formatContext(m.context) + ' context', m.date].filter(Boolean);
  return parts.join(' · ');
}

function Detail({ r, file }: { r: RouteResult; file: RoutesFile }) {
  const { pick, estimate, check } = r;
  const q = pick.quant;
  return (
    <div className="lls-routes-detail text-[13px] leading-relaxed text-[var(--color-muted)] space-y-2">
      <p>{r.route.detail}.</p>
      {r.route.boxes && (
        <p>
          {r.route.boxes.map((b, i) => (
            <Fragment key={b.name}>
              {i > 0 && ' · '}
              {b.source ? (
                <a className="underline underline-offset-2 hover:text-[var(--color-text)]" href={b.source} target="_blank" rel="noreferrer">
                  {b.name}
                </a>
              ) : (
                b.name
              )}{' '}
              {formatPrice(b.usd)}
              {b.stock ? ` (${b.stock})` : ''}
            </Fragment>
          ))}
        </p>
      )}
      <p>
        <span className="text-[var(--color-text)] font-medium">Price:</span> {priceText(r.route.price)}, {r.route.price.what}, as of {r.route.price.as_of} ({r.route.price.grade}). <Links urls={r.route.price.sources} />
      </p>
      <p>
        <span className="text-[var(--color-text)] font-medium">Memory:</span> {r.route.memory_speed.text}
        {r.route.memory_speed.note ? ` (${r.route.memory_speed.note})` : ''}. The AI can use {r.route.usable.text}
        {r.route.usable.note ? `: ${r.route.usable.note}` : ''}.{' '}
        {r.route.usable.source && <Links urls={[r.route.usable.source]} />}
      </p>
      {q && pick.need && (
        <p>
          <span className="text-[var(--color-text)] font-medium">The file:</span> {q.label} from {q.repo} ({formatGb(q.size_gb)}).{' '}
          {pick.tableOutGb > 0 && `Its n-gram table (${formatGb(pick.tableOutGb, 0)}) stays on the SSD, as the engines on this route allow. `}
          It needs {formatGb(pick.need.totalGb)}: {formatGb(pick.need.weightsGb)} of weights, {formatGb(pick.need.cacheGb)} of context cache, {formatGb(pick.need.buffersGb)} of working room
          {pick.need.scratchGb > 0 ? `, ${formatGb(pick.need.scratchGb)} of prompt scratch` : ''}.
        </p>
      )}
      {pick.verdict === 'no-fit' && pick.nearest && <p>The smallest file of this build needs {formatGb(pick.nearest.totalGb)}; this route gives {formatGb(r.route.usable.gb)}.</p>}
      {estimate && (
        <p>
          <span className="text-[var(--color-text)] font-medium">The estimate:</span> {estimate.how}: {estimate.parts.readMs.toFixed(1)} ms reading the weights and cache, {estimate.parts.fixedMs.toFixed(1)} ms of fixed cost
          {estimate.parts.attnMs > 0.05 ? `, ${estimate.parts.attnMs.toFixed(1)} ms for the context` : ''}
          {estimate.parts.linkMs > 0 ? `, ${estimate.parts.linkMs.toFixed(1)} ms for the link` : ''} per token, so about {num(estimate.tokS)} tok/s.{' '}
          <a className="underline underline-offset-2 hover:text-[var(--color-text)]" href={aboutHref('routes')}>
            How it is calibrated
          </a>
        </p>
      )}
      {!estimate && r.noEstimate && (
        <p>
          <span className="text-[var(--color-text)] font-medium">No estimate:</span> {r.noEstimate}
          {r.ceiling !== null && ` Memory speed alone caps it at about ${num(r.ceiling)} tok/s: the weights on the card at the card's speed, the rest at the PC's. Real runs land well below that.`}
        </p>
      )}
      {r.measured && (
        <p>
          <span className="text-[var(--color-text)] font-medium">Published, plain:</span> {runLine(r.measured)}. {sentence(r.measured.note)}
          <Links urls={r.measured.sources} />
        </p>
      )}
      {check && (
        <p>
          {check.otherRuntime
            ? `The tool's estimate for that build on the route's own runtime is ${num(check.estimate)} tok/s; the published run used another one (${r.measured?.engine}), so it is shown instead.`
            : check.ok
              ? `The tool's estimate for that build, ${num(check.estimate)} tok/s, is within ${Math.round(Math.abs(check.err) * 100)} % of it.`
              : `The tool's estimate for that build, ${num(check.estimate)} tok/s, reads ${Math.round(Math.abs(check.err) * 100)} % ${check.err > 0 ? 'high' : 'low'}, so the published run is shown.`}{' '}
          {!check.ok && r.caution}
        </p>
      )}
      {!check && r.caution && <p className="lls-routes-warn">{r.caution}</p>}
      {r.unsettled && (
        <p>
          <span className="text-[var(--color-text)] font-medium">Published, plain but unsettled:</span> {runLine(r.unsettled)}. {sentence(r.unsettled.note)}<Links urls={r.unsettled.sources} />
        </p>
      )}
      {r.speedups.map((m) => (
        <p key={m.engine + m.build + m.write}>
          <span className="text-[var(--color-text)] font-medium">With a speed-up (not comparable with plain):</span> {runLine(m)}. {sentence(m.note)}
          <Links urls={m.sources} />
        </p>
      ))}
      {r.reading && (
        <p>
          <span className="text-[var(--color-text)] font-medium">Reading:</span>{' '}
          {Array.isArray(r.reading.tokS) ? rangeText(r.reading.tokS) : num(r.reading.tokS)} tok/s at {formatContext(r.reading.context)} ({r.reading.engine}
          {r.reading.note ? `, ${r.reading.note}` : ''}), so {file.document.pages} pages ({file.document.tokens.toLocaleString('en-US')} tokens) take {readingText(r)?.main}.
        </p>
      )}
      {r.route.measured_note && <p>{sentence(r.route.measured_note)}</p>}
    </div>
  );
}

export function RoutesView(props: RoutesViewProps) {
  const { state, view, factors, machines, columns, index, analytics } = props;
  const { file, failed } = useRoutesFile(props.initialRoutes);

  const custom = view.model.startsWith('custom-') ? (state.customModels.map(customModelDetail).find((m) => m.id === view.model) ?? null) : null;
  const held = props.records[view.model];
  const preloaded = held?.status === 'ready' ? held.model : (props.seed?.[view.model] ?? null);
  const record = useModelRecord(custom || preloaded ? null : view.model);
  const model = custom ?? preloaded ?? (record?.status === 'ready' ? record.model : null);

  const buckets = useMemo(() => (model ? routeBuckets(model) : []), [model]);
  const bucket: Bucket = buckets.includes(view.bucket) ? view.bucket : buckets.includes('Q4') ? 'Q4' : (buckets[0] ?? view.bucket);
  const contexts = file ? file.contexts.filter((c) => !model?.context_max || c <= model.context_max) : [];
  const ctx = contexts.includes(view.ctx) ? view.ctx : (contexts[0] ?? view.ctx);
  const routesCtx = useMemo(() => (file ? routesContext(file, factors, machines) : null), [file, factors, machines]);
  const rows = useMemo(() => (file && model && routesCtx ? routesTable(file, model, bucket, ctx, state.kvBits, routesCtx) : []), [file, model, bucket, ctx, state.kvBits, routesCtx]);
  const [open, setOpen] = useState<string | null>(null);

  const set = (patch: Partial<RoutesViewState>, toggle?: [Parameters<Analytics['toggleChanged']>[0], string | number]) => {
    props.onView({ ...view, bucket, ctx, ...patch });
    if (toggle) analytics.toggleChanged(toggle[0], toggle[1]);
  };

  // the model menu: the comparison's three, the table's models, then the whole catalog by size
  const featured = file?.featured_models ?? [];
  const tableModels = [...new Map(columns.filter((c) => !c.column.id.startsWith('custom-') || c.model).map((c) => [c.column.id, c.model?.name ?? c.column.id])).entries()].filter(([id]) => !featured.includes(id));
  const listed = new Set([...featured, ...tableModels.map(([id]) => id)]);
  const sections = index ? orderCatalog(index.models).map((s) => ({ ...s, entries: s.entries.filter((m) => !listed.has(m.id)) })).filter((s) => s.entries.length) : [];
  const nameOf = (id: string) => index?.models.find((m) => m.id === id)?.name ?? (id === model?.id ? model.name : id);
  const known = listed.has(view.model) || sections.some((s) => s.entries.some((m) => m.id === view.model));

  if (failed) return <p className="text-sm text-[var(--color-muted)]">The route list could not be loaded. Try again in a moment.</p>;
  if (!file) return <p className="text-sm text-[var(--color-muted)]">Loading the routes…</p>;

  return (
    <div className="lls-routes">
      <div className="flex flex-wrap items-center gap-3 mb-3">
        <span className="font-heading text-lg">Every way to get 128 GB, running</span>
        <select aria-label="Model" className="lls-chip !py-1 appearance-none font-medium max-w-[16rem]" value={view.model} onChange={(e) => { setOpen(null); set({ model: e.target.value }, ['model', e.target.value]); }}>
          {!known && <option value={view.model}>{model?.name ?? view.model}</option>}
          <optgroup label="Quick picks">
            {featured.map((id) => (
              <option key={id} value={id}>
                {nameOf(id)}
              </option>
            ))}
          </optgroup>
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
              <button key={b} type="button" aria-pressed={bucket === b} onClick={() => set({ bucket: b }, ['build', b])}>
                {b}
              </button>
            ))}
          </div>
        )}
        {contexts.length > 1 && (
          <div className="lls-seg" role="group" aria-label="Context">
            {contexts.map((c) => (
              <button key={c} type="button" aria-pressed={ctx === c} onClick={() => set({ ctx: c }, ['context', c])} title={`${formatContext(c)} tokens in the chat`}>
                {formatContext(c)}
              </button>
            ))}
          </div>
        )}
      </div>
      <div className="flex flex-wrap items-center gap-2 mb-3 text-[13px]">
        <span className="text-[var(--color-muted)]">Quick picks</span>
        {featured.map((id) => (
          <button key={id} type="button" className="lls-chip !py-1" aria-pressed={view.model === id} onClick={() => { setOpen(null); set({ model: id }, ['model', id]); }}>
            {nameOf(id)}
          </button>
        ))}
      </div>

      {!model && <p className="text-sm text-[var(--color-muted)]">{record?.status === 'missing' ? 'This model is not in the catalog any more. Pick another one above.' : record?.status === 'error' ? 'This model could not be loaded. Pick another one, or try again in a moment.' : 'Loading this model…'}</p>}
      {model && (
        <>
          <p className="text-sm text-[var(--color-muted)] mb-3">
            {[`${model.name} at ${bucket}`, `${formatContext(ctx)} of context`, 'writing speed plain, with no speed-up', `a ${file.answer.words.toLocaleString('en-US')}-word answer is about ${file.answer.tokens.toLocaleString('en-US')} tokens`, `prices checked ${file.prices_checked}`].join(' · ')}
          </p>
          <div className="lls-routes-wrap">
            <table className="lls-routes-table">
              <thead>
                <tr>
                  <th scope="col">Route</th>
                  <th scope="col">Price</th>
                  <th scope="col">Memory speed</th>
                  <th scope="col">AI can use</th>
                  <th scope="col">Fits?</th>
                  <th scope="col">Writing, tok/s</th>
                  <th scope="col">{file.answer.words.toLocaleString('en-US')}-word answer</th>
                  <th scope="col">{file.document.pages} pages</th>
                </tr>
              </thead>
              <tbody>
                {ROUTE_GROUPS.map((g) => {
                  const inGroup = rows.filter((r) => r.route.group === g.key);
                  if (!inGroup.length) return null;
                  return (
                    <Fragment key={g.key}>
                      <tr className="lls-routes-group">
                        <th scope="rowgroup" colSpan={8}>
                          {g.label}
                        </th>
                      </tr>
                      {inGroup.map((r) => {
                        const isOpen = open === r.route.id;
                        const read = readingText(r);
                        const cond = CONDITION[r.route.condition];
                        return (
                          <Fragment key={r.route.id}>
                            <tr className="lls-routes-row" data-open={isOpen || undefined}>
                              <th scope="row" data-label="Route">
                                <button type="button" className="lls-routes-name" aria-expanded={isOpen} aria-controls={`lls-route-${r.route.id}`} onClick={() => setOpen(isOpen ? null : r.route.id)}>
                                  <span>
                                    {r.route.name}
                                    {cond && <span className="lls-routes-cond">{cond}</span>}
                                  </span>
                                  <ChevronDownIcon size={14} className="lls-routes-chev" />
                                </button>
                              </th>
                              <td data-label="Price">
                                <span className="tabular-nums">{priceText(r.route.price)}</span>
                                {r.route.group === 'pc' && <span className="lls-routes-sub">the RAM</span>}
                              </td>
                              <td data-label="Memory speed">
                                <span className="tabular-nums">{r.route.memory_speed.text.split(' · ')[0]}</span>
                                {r.route.memory_speed.text.includes(' · ') && <span className="lls-routes-sub">{r.route.memory_speed.text.split(' · ')[1]}</span>}
                              </td>
                              <td data-label="AI can use">
                                <span className="tabular-nums">{formatGb(r.route.usable.gb, 0)}</span>
                                {r.route.usable.alt && <span className="lls-routes-sub">{r.route.usable.alt.text}</span>}
                              </td>
                              <td data-label="Fits?">
                                <FitCell r={r} />
                              </td>
                              <td data-label="Writing, tok/s">
                                <WriteCell r={r} />
                              </td>
                              <td data-label={`${file.answer.words.toLocaleString('en-US')}-word answer`}>
                                <span className="tabular-nums">{r.headline.answerS !== null ? secondsText(r.headline.answerS) : '—'}</span>
                              </td>
                              <td data-label={`${file.document.pages} pages`}>
                                {read ? (
                                  <>
                                    <span className="tabular-nums">{read.main}</span>
                                    <span className="lls-routes-sub">{read.sub}</span>
                                  </>
                                ) : (
                                  <span className="text-[var(--color-light)]" title="no published reading speed for this model here">not measured</span>
                                )}
                              </td>
                            </tr>
                            {isOpen && (
                              <tr className="lls-routes-detail-row" id={`lls-route-${r.route.id}`}>
                                <td colSpan={8}>
                                  <Detail r={r} file={file} />
                                </td>
                              </tr>
                            )}
                          </Fragment>
                        );
                      })}
                      {g.key === 'mac' && SITE.macGuide && (
                        <tr className="lls-routes-note">
                          <td colSpan={8}>
                            {SITE.macGuide.lead}{' '}
                            <a className="underline underline-offset-2 hover:text-[var(--color-text)]" href={SITE.macGuide.href} onClick={() => analytics.promoClicked('routes-mac-guide')}>
                              {SITE.macGuide.label}
                            </a>
                          </td>
                        </tr>
                      )}
                    </Fragment>
                  );
                })}
              </tbody>
            </table>
          </div>
          <ul className="mt-3 text-xs text-[var(--color-muted)] space-y-1">
            <li>
              <span className="lls-routes-tag" data-kind="measured">measured</span> a published plain run of this model at this build, shown over the estimate.{' '}
              <span className="lls-routes-tag" data-kind="estimate">estimate</span> the tool's speed model, checked against those runs.{' '}
              <span className="lls-routes-tag" data-kind="none">no estimate</span> no calibrated rule for this route; the row says why.
            </li>
            <li>
              Reading times are shown only where a run measured them: reading depends on the chip's compute, not its memory speed, and on the DGX Spark and the RTX PRO 6000 it is most of what the extra money buys.{' '}
              Every price, speed and source is in{' '}
              <a className="underline underline-offset-2 hover:text-[var(--color-text)]" href={ROUTES_URL}>
                the data file
              </a>{' '}
              (as of {file.data_as_of}) ·{' '}
              <a className="underline underline-offset-2 hover:text-[var(--color-text)]" href={aboutHref('routes')}>
                the calibration table
              </a>
            </li>
          </ul>
        </>
      )}
    </div>
  );
}
