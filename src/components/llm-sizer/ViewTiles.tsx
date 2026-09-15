/** The chart types as a tab-like row of tiles. */
import type { View } from '../../lib/llm-sizer/app/state';
import { GaugeIcon, GridIcon, LayersIcon, ListChecksIcon, TargetIcon } from './icons';

export type ViewKind = View['kind'];

/** Views built but not offered yet (`?tiles=all` shows them for verification). */
export const HIDDEN_VIEWS: ViewKind[] = [];
export const isHiddenView = (kind: ViewKind): boolean => HIDDEN_VIEWS.includes(kind);

const TILES: { kind: ViewKind; Icon: (p: { size?: number }) => React.JSX.Element; title: string; text: string }[] = [
  { kind: 'table', Icon: GridIcon, title: 'Will it fit?', text: 'Every model on every machine, at a glance.' },
  { kind: 'speed', Icon: GaugeIcon, title: 'How fast?', text: 'Writing speed on one machine.' },
  { kind: 'model', Icon: TargetIcon, title: 'Which machine?', text: 'One model across every Mac: memory, speed, price.' },
  { kind: 'memory', Icon: LayersIcon, title: 'How much memory?', text: 'What each model, build and context needs.' },
  { kind: 'machine', Icon: ListChecksIcon, title: 'What can it run?', text: 'Every model on one machine, best build first.' },
];

export function ViewTiles({ value, onSelect, className, showHidden = false }: { value: ViewKind; onSelect: (kind: ViewKind) => void; className?: string; showHidden?: boolean }) {
  const tiles = TILES.filter((t) => showHidden || !isHiddenView(t.kind));
  const order = tiles.map((t) => t.kind);
  function onKey(e: React.KeyboardEvent<HTMLButtonElement>, kind: ViewKind) {
    const i = order.indexOf(kind);
    let next: number | null = null;
    if (e.key === 'ArrowRight') next = (i + 1) % order.length;
    else if (e.key === 'ArrowLeft') next = (i - 1 + order.length) % order.length;
    else if (e.key === 'Home') next = 0;
    else if (e.key === 'End') next = order.length - 1;
    if (next === null) return;
    e.preventDefault();
    onSelect(order[next]);
    (e.currentTarget.parentElement?.querySelectorAll<HTMLButtonElement>('[role=tab]')[next] ?? null)?.focus();
  }
  return (
    <div role="tablist" aria-label="Chart type" className={`lls-tiles ${className ?? ''}`}>
      {tiles.map(({ kind, Icon, title, text }) => {
        const selected = kind === value;
        return (
          <button
            key={kind}
            type="button"
            role="tab"
            id={`lls-tile-${kind}`}
            aria-selected={selected}
            aria-controls="lls-chart"
            tabIndex={selected ? 0 : -1}
            className="lls-tile"
            onClick={() => onSelect(kind)}
            onKeyDown={(e) => onKey(e, kind)}
          >
            <span className="lls-tile-icon">
              <Icon size={18} />
            </span>
            <span className="block font-semibold text-[14px] leading-tight">{title}</span>
            <span className="block text-[12px] text-[var(--color-muted)] mt-0.5">{text}</span>
          </button>
        );
      })}
    </div>
  );
}
