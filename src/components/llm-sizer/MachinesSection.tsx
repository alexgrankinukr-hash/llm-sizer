/** Settings › Machines: the machines in the table (sizes, order, removal) and the add flow. */
import { useEffect, useRef, useState } from 'react';
import type { Dispatch } from 'react';
import type { Analytics } from '../../lib/llm-sizer/app/analytics';
import { tensorAllowed } from '../../lib/llm-sizer/engine/cluster';
import { groupLabel, groupTensorCapable, linkedLabel, type MachineGroup } from '../../lib/llm-sizer/app/machines';
import { stateGroups } from '../../lib/llm-sizer/app/matrix';
import { MAX_GROUPS, MAX_LINKED, type Action, type AppState } from '../../lib/llm-sizer/app/state';
import { GripIcon, PlatformIcon, PlusIcon, SearchIcon, XIcon } from './icons';
import { MachineList } from './Lists';
import { MachineSilhouette } from './silhouettes';
import { useListReorder } from './useListReorder';

export interface MachinesSectionProps {
  state: AppState;
  groups: MachineGroup[];
  dispatch: Dispatch<Action>;
  analytics: Analytics;
  onCustom: () => void;
}

export function MachinesSection({ state, groups, dispatch, analytics, onCustom }: MachinesSectionProps) {
  const inTable = stateGroups(state, groups);
  const [adding, setAdding] = useState(false);
  const [q, setQ] = useState('');
  const [live, setLive] = useState('');
  const addBtn = useRef<HTMLButtonElement>(null);
  const search = useRef<HTMLInputElement>(null);
  useEffect(() => {
    if (adding) search.current?.focus();
  }, [adding]);
  const reorder = useListReorder(inTable.length, (from, to) => dispatch({ type: 'MOVE_GROUP', id: inTable[from].id, to }), setLive);
  const full = state.groups.length >= MAX_GROUPS;

  function pick(id: string) {
    dispatch({ type: 'ADD_GROUP', id });
    analytics.machineAdded(id);
    setAdding(false);
    setQ('');
    addBtn.current?.focus();
  }

  return (
    <section aria-labelledby="lls-set-machines">
      <div className="flex items-center justify-between gap-3 mb-2">
        <h3 id="lls-set-machines" className="lls-eyebrow">Machines</h3>
        <button ref={addBtn} type="button" className="lls-chip !py-1 !px-2.5 font-medium" aria-expanded={adding} disabled={full && !adding} title={full ? `Up to ${MAX_GROUPS} machine groups; pick one again to link another of it` : undefined} onClick={() => setAdding((v) => !v)}>
          <PlusIcon size={13} /> Add machine
        </button>
      </div>
      {adding && (
        <div className="mb-3">
          <label className="flex items-center gap-2 px-2 py-1.5 text-sm border border-[var(--color-border)] rounded-lg bg-white">
            <SearchIcon size={14} />
            <input ref={search} className="flex-1 bg-transparent outline-none" placeholder="Search machines…" value={q} onChange={(e) => setQ(e.target.value)} aria-label="Search machines" />
          </label>
          <MachineList groups={groups} chosen={state.groups} linked={state.linked} query={q} onPick={pick} onCustom={() => { setAdding(false); onCustom(); }} />
        </div>
      )}
      {inTable.length === 0 && <p className="text-sm text-[var(--color-muted)]">No machine yet.</p>}
      <ul ref={reorder.listRef} className="space-y-2">
        {inTable.map((g, i) => {
          const hidden = state.hiddenSizes[g.id] ?? [];
          const sizes = [...new Set(g.rows.map((r) => r.gb))];
          const visibleCount = sizes.filter((gb) => !hidden.includes(gb)).length;
          const linked = state.linked[g.id] ?? null;
          const label = linkedLabel(g, linked);
          const tensorOk = groupTensorCapable(g);
          const tensorTitle = !tensorOk
            ? 'tensor parallel needs a fast link: Thunderbolt 5 on a Mac (M3 Ultra, M4 Pro and Max, the M5 family), the Spark\'s 200 Gb/s ports, or PCIe in one box'
            : linked && !tensorAllowed(linked.count)
              ? 'tensor splits need 2 or 4 machines'
              : 'MLX over Thunderbolt 5 RDMA, vLLM over the Spark\'s ports, PCIe in a GPU box: every machine reads its shard at once, so the pool writes faster than one machine';
          return (
            <li key={g.id} {...reorder.itemProps(i)} className={`lls-item ${reorder.itemProps(i).className}`}>
              <div className="flex items-center gap-2">
                <button type="button" className="lls-chip lls-handle !px-1.5" {...reorder.handleProps(i, label)}>
                  <GripIcon size={13} />
                </button>
                <MachineSilhouette kind={g.silhouette as never} width={36} className="text-[#6e6d69] shrink-0" />
                <PlatformIcon platform={g.platform} size={14} />
                <span className="font-semibold text-[13.5px] leading-tight flex-1 min-w-0">{label}</span>
                <button type="button" className="lls-chip !px-1.5" aria-label={`Remove ${label}`} onClick={() => dispatch({ type: 'REMOVE_GROUP', id: g.id })}>
                  <XIcon size={12} />
                </button>
              </div>
              {sizes.length > 1 && (
                <div className="mt-1.5 flex flex-wrap gap-1 pl-9" role="group" aria-label={`${label}: memory sizes shown`}>
                  {sizes.map((gb) => {
                    const shown = !hidden.includes(gb);
                    return (
                      <button
                        key={gb}
                        type="button"
                        className="lls-chip lls-size !py-0.5 !px-2 text-[11.5px]"
                        aria-pressed={shown}
                        disabled={shown && visibleCount === 1}
                        onClick={() => {
                          dispatch({ type: 'TOGGLE_SIZE', groupId: g.id, gb, sizes });
                          analytics.toggleChanged('size', `${g.id}:${gb}:${shown ? 'hide' : 'show'}`);
                        }}
                      >
                        {gb} GB{linked ? ' each' : ''}
                      </button>
                    );
                  })}
                </div>
              )}
              <div className="mt-1.5 flex flex-wrap items-center gap-1 pl-9" role="group" aria-label={`${groupLabel(g)}: machines linked as one pool`}>
                <span className="text-[11px] text-[var(--color-light)] mr-0.5">linked</span>
                {Array.from({ length: MAX_LINKED }, (_, i) => i + 1).map((n) => (
                  <button
                    key={n}
                    type="button"
                    className="lls-chip lls-size !py-0.5 !px-2 text-[11.5px]"
                    aria-pressed={(linked?.count ?? 1) === n}
                    title={n === 1 ? 'one machine' : `${n} identical machines pooled as one: memory adds up, the speed follows the split`}
                    onClick={() => {
                      dispatch({ type: 'SET_LINKED', groupId: g.id, count: n });
                      analytics.toggleChanged('linked', `${g.id}:${n}`);
                    }}
                  >
                    ×{n}
                  </button>
                ))}
                {linked && (
                  <div className="lls-seg ml-1" role="group" aria-label={`${groupLabel(g)}: how the model is split across the pool`}>
                    <button
                      type="button"
                      aria-pressed={linked.split === 'layer'}
                      title="llama.cpp RPC on any link, or MLX pipeline: memory pools, the pool writes at one machine's speed minus a hop per extra machine"
                      onClick={() => {
                        dispatch({ type: 'SET_LINKED', groupId: g.id, count: linked.count, split: 'layer' });
                        analytics.toggleChanged('split', `${g.id}:layer`);
                      }}
                    >
                      layer split
                    </button>
                    <button
                      type="button"
                      aria-pressed={linked.split === 'tensor'}
                      disabled={!tensorOk || !tensorAllowed(linked.count)}
                      title={tensorTitle}
                      onClick={() => {
                        dispatch({ type: 'SET_LINKED', groupId: g.id, count: linked.count, split: 'tensor' });
                        analytics.toggleChanged('split', `${g.id}:tensor`);
                      }}
                    >
                      tensor parallel
                    </button>
                  </div>
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
