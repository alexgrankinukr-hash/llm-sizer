/** The machine a one-machine chart is showing: its picture and platform mark, with a native menu to switch. */
import { linkedLabel, rowLabel } from '../../lib/llm-sizer/app/machines';
import type { MatrixRow } from '../../lib/llm-sizer/app/matrix';
import { ChevronDownIcon, PlatformIcon } from './icons';
import { MachineSilhouette } from './silhouettes';

export interface RowSelectProps {
  /** a flat list (the table's rows) … */
  rows?: MatrixRow[];
  /** … or grouped sections (the whole catalog), rendered as option groups */
  sections?: { label: string; rows: MatrixRow[] }[];
  value: string;
  onChange: (rowKey: string) => void;
  label?: string;
}

function optionText(r: MatrixRow): string {
  return rowLabel(r.group, r.row, r.linked);
}

export function RowSelect({ rows, sections, value, onChange, label = 'Machine' }: RowSelectProps) {
  const all = sections ? sections.flatMap((s) => s.rows) : (rows ?? []);
  const current = all.find((r) => r.key === value) ?? all[0];
  if (!current) return null;
  const option = (r: MatrixRow) => (
    <option key={r.key} value={r.key}>
      {optionText(r)}
    </option>
  );
  return (
    <div className="lls-machine-pick">
      <MachineSilhouette kind={current.group.silhouette as never} width={38} className="text-[#6e6d69] shrink-0" />
      <PlatformIcon platform={current.group.platform} size={14} />
      <span className="lls-machine-name">
        {linkedLabel(current.group, current.linked)}
        <span className="lls-machine-size">{current.row.gb} GB{current.linked ? ' each' : ''}</span>
      </span>
      <ChevronDownIcon size={14} />
      <select aria-label={label} value={current.key} onChange={(e) => onChange(e.target.value)}>
        {sections
          ? sections.map((s) => (
              <optgroup key={s.label} label={s.label}>
                {s.rows.map(option)}
              </optgroup>
            ))
          : all.map(option)}
      </select>
    </div>
  );
}
