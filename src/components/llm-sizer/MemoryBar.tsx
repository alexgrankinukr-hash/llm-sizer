/** The stacked memory bar: macOS · your apps · memory-limit reserve · weights · context cache · buffers · prompt scratch (MLX) · free · over. */
import type { Breakdown } from '../../lib/llm-sizer/engine/types';
import { formatGb } from '../../lib/llm-sizer/engine/format';

export const MEMORY_SEGMENTS: [keyof Breakdown & string, string][] = [
  ['os', 'macOS'],
  ['apps', 'your apps'],
  ['capReserve', 'memory-limit reserve'],
  ['cluster', 'link buffers'],
  ['weights', 'weights'],
  ['cache', 'context cache'],
  ['buffers', 'buffers'],
  ['scratch', 'prompt scratch (MLX)'],
  ['free', 'free'],
  ['over', 'over'],
];

const CLASS: Record<string, string> = { os: 'os', apps: 'apps', capReserve: 'cap', cluster: 'cluster', weights: 'weights', cache: 'cache', buffers: 'buffers', scratch: 'scratch', free: 'free', over: 'over' };
/** Segments a linked pool holds once per machine: their labels say so. */
const PER_MACHINE = new Set(['os', 'capReserve', 'cluster', 'buffers', 'scratch']);

export interface MemoryBarProps {
  breakdown: Breakdown;
  /** GB the full width stands for; default = this machine's memory plus any overflow */
  scaleGb?: number;
  showLegend?: boolean;
  height?: number;
}

export function memorySegments(b: Breakdown): { key: string; cls: string; gb: number; label: string }[] {
  const n = b.machines ?? 1;
  return MEMORY_SEGMENTS.map(([key, label]) => ({ key, cls: CLASS[key], gb: b[key] as number, label: n > 1 && PER_MACHINE.has(key) ? `${label} × ${n}` : label })).filter((s) => s.gb > 0.05);
}

export function MemoryBar({ breakdown: b, scaleGb, showLegend = false, height }: MemoryBarProps) {
  const total = b.ramGb + b.over;
  const scale = scaleGb && scaleGb > 0 ? scaleGb : total;
  const segments = memorySegments(b);
  return (
    <div>
      <div className="relative" style={{ width: `${Math.min(100, (total / scale) * 100)}%` }}>
        <div className="lls-bar" style={height ? { height } : undefined} role="img" aria-label={segments.map((s) => `${s.label} ${formatGb(s.gb)}`).join(', ')}>
          {segments.map((s) => (
            <span key={s.key} className={s.cls} style={{ width: `${(s.gb / total) * 100}%` }} title={`${s.label}: ${formatGb(s.gb)}`} />
          ))}
        </div>
        {scaleGb && b.over > 0.05 && <span className="absolute top-0 bottom-0 w-px bg-[var(--color-text)]" style={{ left: `${(b.ramGb / total) * 100}%` }} aria-hidden="true" />}
      </div>
      {b.ssdGb > 0.05 && (
        // what a special build streams from the SSD: drawn to the same scale, beside the bar, never inside it
        <div className="mt-1.5 flex items-center gap-2 text-[12px] text-[var(--color-muted)]">
          <div className="lls-bar shrink-0" style={{ height: 8, width: `${Math.min(60, (b.ssdGb / scale) * 100)}%` }} role="img" aria-label={`on the SSD ${formatGb(b.ssdGb)}`}>
            <span className="ssd" style={{ width: '100%' }} title={`on the SSD: ${formatGb(b.ssdGb)}`} />
          </div>
          <span>on the SSD · {formatGb(b.ssdGb)} · streamed, not memory</span>
        </div>
      )}
      {showLegend && (
        <ul className="mt-2 grid grid-cols-2 gap-x-4 gap-y-0.5 text-[12px] text-[var(--color-muted)]">
          {segments.map((s) => (
            <li key={s.key} className="flex items-center gap-1.5">
              <span className="inline-block lls-bar" style={{ height: 10, width: 10 }}>
                <span className={s.cls} style={{ width: '100%' }} />
              </span>
              {s.label} · {formatGb(s.gb)}
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}
