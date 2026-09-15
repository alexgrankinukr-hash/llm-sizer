/** The searchable lists used by the settings panel: machines from the catalog, models from the index. */
import { useMemo, useState } from 'react';
import type { ModelIndex, ModelIndexEntry } from '../../lib/llm-sizer/engine/types';
import { formatGb } from '../../lib/llm-sizer/engine/format';
import { groupLabel, type LinkedInfo, type MachineGroup } from '../../lib/llm-sizer/app/machines';
import { PlatformIcon } from './icons';

export interface MachineListProps {
  groups: MachineGroup[];
  chosen: string[];
  /** per chosen machine, how many are linked as one pool; absent = one */
  linked?: Record<string, LinkedInfo>;
  query: string;
  onPick: (groupId: string) => void;
  onCustom?: () => void;
}

export function MachineList(props: MachineListProps) {
  const [showOld, setShowOld] = useState(false);
  const needle = props.query.trim().toLowerCase();
  const matches = (g: MachineGroup) => !needle || groupLabel(g).toLowerCase().includes(needle) || g.chip.toLowerCase().includes(needle);
  const current = props.groups.filter((g) => g.status === 'current' && matches(g));
  const older = props.groups.filter((g) => g.status !== 'current' && matches(g));
  const families = (list: MachineGroup[]) => {
    const out = new Map<string, MachineGroup[]>();
    for (const g of list) out.set(g.family, [...(out.get(g.family) ?? []), g]);
    return [...out.entries()];
  };
  const LINK_TAG = ['add', 'added · link a second', '×2 linked · add a third', '×3 linked · add a fourth', '×4 linked · the most'];
  const row = (g: MachineGroup) => {
    const added = props.chosen.includes(g.id);
    const count = added ? (props.linked?.[g.id]?.count ?? 1) : 0;
    const full = count >= 4;
    return (
      <button key={g.id} type="button" className="item" disabled={full} title={added && !full ? 'adds another of the same machine, linked with it as one pool' : undefined} onClick={() => props.onPick(g.id)}>
        <span className="flex items-center gap-2.5 min-w-0">
          <PlatformIcon platform={g.platform} size={15} />
          <span className="min-w-0">
            {g.chip}
            <span className="block text-[11px] text-[var(--color-light)]">{g.rows.map((r) => `${r.gb}`).join(' · ')} GB{g.year ? ` · ${g.year}` : ''}</span>
          </span>
        </span>
        <span className={`text-[11px] ${full ? 'text-[var(--color-light)]' : 'text-[var(--lls-gold-text)]'}`}>{LINK_TAG[count]}</span>
      </button>
    );
  };
  return (
    <div className="lls-list">
      {families(current).map(([family, list]) => (
        <div key={family}>
          <div className="head">{family}</div>
          {list.map(row)}
        </div>
      ))}
      {older.length > 0 && (
        <div>
          <button type="button" className="item text-[var(--color-muted)]" onClick={() => setShowOld((v) => !v)}>
            {showOld ? 'Hide' : 'Show'} older machines ({older.length})
          </button>
          {showOld &&
            families(older).map(([family, list]) => (
              <div key={family}>
                <div className="head">{family} · discontinued</div>
                {list.map(row)}
              </div>
            ))}
        </div>
      )}
      {current.length + older.length === 0 && <div className="head">no machine matches</div>}
      {props.onCustom && (
        <button type="button" className="item text-[var(--lls-gold-text)]" onClick={props.onCustom}>
          Describe a machine that isn't listed…
        </button>
      )}
    </div>
  );
}

export interface ModelListProps {
  index: ModelIndex | null;
  indexStatus: 'loading' | 'ready' | 'error';
  /** ids already in the table (shown, never disabled: another quant or context is a legitimate second column) */
  chosen: string[];
  query: string;
  onPick: (entry: ModelIndexEntry) => void;
  onCustom?: () => void;
}

const TIER_LABEL: Record<string, string> = { laptop: 'Laptop / small Mac (16–32 GB)', studio: 'Studio-class (64–256 GB)', frontier: 'Frontier-class (512 GB+)' };

export function ModelList(props: ModelListProps) {
  const models = props.index?.models ?? [];
  const needle = props.query.trim().toLowerCase();
  const filtered = useMemo(() => {
    const match = (m: ModelIndexEntry) => !needle || m.name.toLowerCase().includes(needle) || m.id.includes(needle) || m.provider.toLowerCase().includes(needle);
    return models.filter((m) => m.status !== 'hidden' && match(m));
  }, [models, needle]);
  const featured = filtered.filter((m) => m.featured).sort((a, b) => (a.rank ?? 99) - (b.rank ?? 99));
  const rest = filtered.filter((m) => !m.featured).sort((a, b) => (b.released ?? '').localeCompare(a.released ?? ''));
  const tiers = ['laptop', 'studio', 'frontier'];
  const row = (m: ModelIndexEntry) => (
    <button key={m.id} type="button" className="item" onClick={() => props.onPick(m)}>
      <span>
        {m.name}
        <span className="block text-[11px] text-[var(--color-light)]">
          {m.params_total_b !== null ? `${m.params_total_b >= 1000 ? `${(m.params_total_b / 1000).toFixed(1)}T` : `${Math.round(m.params_total_b)}B`}` : ''}
          {m.sizes.q4_gb ? ` · Q4 ${formatGb(m.sizes.q4_gb)}` : ''}
          {m.assumed ? ' · architecture assumed' : ''}
        </span>
      </span>
      <span className={`text-[11px] ${props.chosen.includes(m.id) ? 'text-[var(--color-light)]' : 'text-[var(--lls-gold-text)]'}`}>{props.chosen.includes(m.id) ? 'in the table · add again' : 'add'}</span>
    </button>
  );
  return (
    <div className="lls-list">
      {props.indexStatus === 'loading' && <div className="head">loading the catalog…</div>}
      {props.indexStatus === 'error' && <div className="head text-[var(--color-error)]">the model catalog could not be loaded</div>}
      {tiers.map((tier) => {
        const list = featured.filter((m) => m.tier === tier);
        if (!list.length) return null;
        return (
          <div key={tier}>
            <div className="head">{TIER_LABEL[tier] ?? tier}</div>
            {list.map(row)}
          </div>
        );
      })}
      {rest.length > 0 && (
        <div>
          <div className="head">Everything else · auto-imported</div>
          {rest.slice(0, needle ? 50 : 12).map(row)}
          {!needle && rest.length > 12 && <div className="head">type to search {rest.length} more…</div>}
        </div>
      )}
      {props.indexStatus === 'ready' && filtered.length === 0 && <div className="head">no model matches</div>}
      {props.onCustom && (
        <button type="button" className="item text-[var(--lls-gold-text)]" onClick={props.onCustom}>
          Describe a model that isn't listed…
        </button>
      )}
    </div>
  );
}
