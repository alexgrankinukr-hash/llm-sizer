/** The fit table: machines as stacked row groups, models as columns, the three markers as cells. Nothing here configures; the settings panel does. */
import { useEffect, useMemo, useRef, useState } from 'react';
import type { TableLayout } from '../../lib/llm-sizer/app/layout';
import type { Matrix } from '../../lib/llm-sizer/app/matrix';
import type { AppState } from '../../lib/llm-sizer/app/state';
import { MachineSilhouette } from './silhouettes';
import { PlatformIcon } from './icons';

export interface FitTableProps {
  state: AppState;
  matrix: Matrix;
  layout: TableLayout;
  present?: boolean;
  reveal?: boolean;
  onOpenCell?: (rowKey: string, column: number, anchor: HTMLElement) => void;
  /** the model name opens "which machine do I need" for that column (navigation, not configuration) */
  onColumnMenu?: (column: number, anchor: HTMLElement) => void;
  onRetryColumn?: (id: string) => void;
}

function Marker({ marker }: { marker: string }) {
  const cls = marker === 'run' ? 'lls-run' : marker === 'ring' ? 'lls-ring' : marker === 'no' ? 'lls-no' : marker === 'loading' ? 'lls-loading' : 'lls-missing';
  return <span className={cls} />;
}

export function FitTable(props: FitTableProps) {
  const { matrix, layout, present, reveal } = props;
  const columns = matrix.columns;
  const railRows = useMemo(() => new Map(layout.rails.map((r) => [r.groupId, r.rowKeys.length])), [layout.rails]);
  const scroller = useRef<HTMLDivElement>(null);
  // macOS hides its scrollbars, so a table wider than its box read as "cut off" rather than "scrolls".
  // The fade on the right edge says there is more, and goes away once you reach the end.
  const [more, setMore] = useState(false);
  useEffect(() => {
    const el = scroller.current;
    if (!el) return;
    const update = () => setMore(el.scrollWidth - el.clientWidth - el.scrollLeft > 2);
    update();
    el.addEventListener('scroll', update, { passive: true });
    const ro = new ResizeObserver(update);
    ro.observe(el);
    return () => {
      el.removeEventListener('scroll', update);
      ro.disconnect();
    };
  }, [matrix, layout]);
  let cellIndex = 0;

  return (
    <div className="lls-scrollwrap" data-more={!present && more ? '1' : '0'}>
    <div className={`lls-scroll ${reveal ? 'lls-reveal' : ''}`} ref={scroller}>
      <table className="lls-table text-sm">
        <thead>
          <tr>
            <th className="lls-rail text-left align-bottom pb-3 pr-3" scope="col">
              <span className="text-[11px] uppercase tracking-[0.2em] text-[var(--color-light)]">{present ? 'Machine' : 'Your machines'}</span>
            </th>
            <th className="lls-mem text-right align-bottom pb-3 pr-3 whitespace-nowrap" scope="col">
              <span className="text-[11px] uppercase tracking-[0.2em] text-[var(--color-light)]">Memory</span>
            </th>
            {columns.map((col, i) => {
              const header = layout.headers[i];
              const status = col.status === 'loading' ? 'loading…' : col.status === 'missing' ? 'not in the catalog' : col.status === 'error' ? 'could not load' : '';
              return (
                <th
                  key={`${col.column.id}-${i}`}
                  className="lls-col-sep align-bottom px-3 pb-3 min-w-[132px]"
                  scope="col"
                  aria-describedby={header.footnotes.length > 0 ? header.footnotes.map((n) => `lls-fn-${n}`).join(' ') : undefined}
                >
                  <div className="flex flex-col items-center text-center gap-1">
                    <span className="text-[10.5px] font-semibold uppercase tracking-[0.2em] text-[var(--color-light)]">{header.provider || ' '}</span>
                    {present || !props.onColumnMenu ? (
                      <span className={`font-extrabold ${present ? 'text-[20px]' : 'text-[15px]'} leading-tight`}>{header.name}</span>
                    ) : (
                      <button
                        type="button"
                        className="font-extrabold text-[15px] leading-tight hover:text-[var(--lls-gold-text)] underline-offset-4 hover:underline"
                        onClick={(e) => props.onColumnMenu?.(i, e.currentTarget)}
                        title="Which machine do I need for this model?"
                      >
                        {header.name}
                      </button>
                    )}
                    <span className="text-xs text-[var(--color-muted)]">{header.params || status}</span>
                    <span className={present ? 'text-xs font-semibold text-[var(--lls-gold-text)]' : 'text-[11.5px] text-[var(--color-light)]'}>
                      {header.quantLabel} · {header.contextLabel}
                      {header.footnotes.length > 0 && <sup aria-hidden="true" className="lls-note">{header.footnotes.join(',')}</sup>}
                    </span>
                    {col.status === 'error' && !present && (
                      <button type="button" className="lls-chip !py-0.5 !px-2 text-[11.5px]" onClick={() => props.onRetryColumn?.(col.column.id)}>
                        retry
                      </button>
                    )}
                  </div>
                </th>
              );
            })}
            {!present && columns.length === 0 && <th className="px-6 pb-3 text-left text-[var(--color-muted)] font-normal">Add a model in Settings</th>}
          </tr>
        </thead>
        <tbody>
          {layout.rails.map((rail, railIndex) => {
            const rows = layout.rows.filter((r) => r.groupId === rail.groupId);
            return rows.map((row, ri) => (
              <tr key={row.key} className={ri === 0 && railIndex > 0 ? 'lls-group-sep' : ''}>
                {ri === 0 && (
                  <th className="lls-rail text-left align-top pt-3 pr-3 font-normal" scope="rowgroup" rowSpan={railRows.get(rail.groupId) ?? rows.length}>
                    <div className="flex flex-col gap-1">
                      <MachineSilhouette kind={rail.silhouette as never} width={present ? 96 : 64} className="text-[#6e6d69]" />
                      <span className={`flex items-center gap-1.5 font-extrabold ${present ? 'text-[22px]' : 'text-[15px]'} leading-tight`}>
                        <PlatformIcon platform={rail.platform} size={present ? 20 : 14} />
                        {rail.label}
                      </span>
                      {rail.price && <span className="text-[12.5px] text-[var(--lls-gold-text)] font-semibold">{rail.price}</span>}
                    </div>
                  </th>
                )}
                <th className={`lls-mem text-right align-middle pr-3 py-1 whitespace-nowrap font-normal ${ri === 0 ? 'pt-3' : ''}`} scope="row">
                  <span className={`font-extrabold ${present ? 'text-[24px]' : 'text-[17px]'}`}>
                    {row.gb}
                    <small className="text-[11px] font-semibold text-[var(--color-light)] ml-0.5">GB</small>
                  </span>
                  {row.binLabel && <span className="block text-[11px] text-[var(--color-light)]">{row.binLabel}</span>}
                  {row.linked && <span className="block text-[11px] text-[var(--color-light)]">each · {row.gb * row.linked.count} GB pooled</span>}
                  {!present && row.price && <span className="block text-[11px] text-[var(--color-light)]">{row.price}</span>}
                </th>
                {row.cells.map((cell) => {
                  const i = cellIndex++;
                  return (
                    <td key={cell.key} className={`lls-col-sep text-center align-middle py-1 ${ri === 0 ? 'pt-3' : ''}`}>
                      <button
                        type="button"
                        className="lls-marker"
                        style={{ '--i': i } as never}
                        aria-label={cell.ariaLabel}
                        aria-describedby={cell.footnotes.length > 0 ? cell.footnotes.map((n) => `lls-fn-${n}`).join(' ') : undefined}
                        title={cell.ariaLabel}
                        disabled={present || cell.marker === 'loading' || cell.marker === 'missing'}
                        onClick={(e) => props.onOpenCell?.(row.key, cell.column, e.currentTarget)}
                      >
                        <Marker marker={cell.marker} />
                      </button>
                      {cell.footnotes.length > 0 && !present && <sup aria-hidden="true" className="text-[10px] text-[var(--color-light)] -ml-1 align-top">{cell.footnotes.join(',')}</sup>}
                    </td>
                  );
                })}
              </tr>
            ));
          })}
          {layout.rails.length === 0 && (
            <tr>
              <td colSpan={2 + columns.length} className="py-8 text-[var(--color-muted)]">
                {present ? '' : 'Add a machine in Settings'}
              </td>
            </tr>
          )}
        </tbody>
      </table>
    </div>
    </div>
  );
}
