/** Advanced: the finer knobs, the factors in the open, and custom models and machines. */
import { useState } from 'react';
import type { Dispatch } from 'react';
import type { Factors, KvBits, Platform } from '../../lib/llm-sizer/engine/types';
import { customMachine, customModelFull, customModelQuick } from '../../lib/llm-sizer/engine/custom';
import { resolveChipProfile } from '../../lib/llm-sizer/engine/speed';
import type { MachineGroup } from '../../lib/llm-sizer/app/machines';
import { stateGroups, type Matrix } from '../../lib/llm-sizer/app/matrix';
import { aboutHref } from '../../lib/llm-sizer/app/provenance';
import { SIMPLE_BUCKETS, type Bucket } from '../../lib/llm-sizer/app/quants';
import { CONTEXT_CHIPS } from '../../lib/llm-sizer/engine/constants';
import { formatContext } from '../../lib/llm-sizer/engine/format';
import type { Action, AppState } from '../../lib/llm-sizer/app/state';
import { Switch } from './Switch';

export interface AdvancedPanelProps {
  state: AppState;
  dispatch: Dispatch<Action>;
  groups: MachineGroup[];
  matrix: Matrix;
  factors: Factors;
  customForm: 'model' | 'machine' | null;
  onCustomForm: (form: 'model' | 'machine' | null) => void;
  onReport: (text: string) => void;
}

const field = 'w-full bg-transparent border-b border-[var(--color-border)] focus:border-[var(--color-text)] outline-none py-1 text-sm';

function CustomModelForm({ onAdd, onCancel }: { onAdd: (model: AppState['customModels'][number], id: string) => void; onCancel: () => void }) {
  const [mode, setMode] = useState<'quick' | 'full'>('quick');
  const [f, setF] = useState({ name: '', total: '', active: '', weights: '', ctx: '', attention: 'gqa', layers: '', kvLayers: '', kvHeads: '', headDim: '', kvLoraRank: '', ropeDim: '', slidingLayers: '', window: '' });
  const set = (k: keyof typeof f) => (e: React.ChangeEvent<HTMLInputElement | HTMLSelectElement>) => setF({ ...f, [k]: e.target.type === 'checkbox' ? (e.target as HTMLInputElement).checked : e.target.value });
  const n = (v: string) => (v.trim() === '' ? undefined : Number(v));
  const valid = f.name.trim() && Number(f.total) > 0 && Number(f.weights) > 0 && (mode === 'quick' || Number(f.layers) > 0);
  return (
    <form
      className="grid gap-3"
      onSubmit={(e) => {
        e.preventDefault();
        if (!valid) return;
        const base = { name: f.name.trim(), paramsTotalB: Number(f.total), paramsActiveB: n(f.active), weightsGb: Number(f.weights), contextMax: n(f.ctx) };
        if (mode === 'quick') {
          const m = { mode: 'quick' as const, ...base };
          onAdd(m, customModelQuick(m).id);
        } else {
          const m = {
            mode: 'full' as const,
            ...base,
            attention: f.attention as 'gqa' | 'mla' | 'latent',
            layers: Number(f.layers),
            kvLayers: n(f.kvLayers),
            kvHeads: n(f.kvHeads),
            headDim: n(f.headDim),
            kvLoraRank: n(f.kvLoraRank),
            ropeDim: n(f.ropeDim),
            slidingLayers: n(f.slidingLayers) ?? 0,
            window: n(f.window),
          };
          onAdd(m, customModelFull(m).id);
        }
      }}
    >
      <div className="md:col-span-2 flex items-center gap-3">
        <span className="text-sm font-semibold">Custom model</span>
        <div className="lls-seg">
          <button type="button" aria-pressed={mode === 'quick'} onClick={() => setMode('quick')}>quick</button>
          <button type="button" aria-pressed={mode === 'full'} onClick={() => setMode('full')}>full architecture</button>
        </div>
      </div>
      <input className={field} placeholder="Name" value={f.name} onChange={set('name')} required />
      <input className={field} placeholder="Total parameters (billions)" type="number" step="any" min="0" value={f.total} onChange={set('total')} required />
      <input className={field} placeholder="Active parameters (billions, mixture-of-experts only)" type="number" step="any" min="0" value={f.active} onChange={set('active')} />
      <input className={field} placeholder="Weights file size (GB)" type="number" step="any" min="0" value={f.weights} onChange={set('weights')} required />
      <input className={field} placeholder="Max context (tokens, optional)" type="number" min="0" value={f.ctx} onChange={set('ctx')} />
      {mode === 'full' && (
        <>
          <select className={field} value={f.attention} onChange={set('attention')} aria-label="Attention design">
            <option value="gqa">grouped-query attention (classic)</option>
            <option value="mla">compressed / MLA</option>
            <option value="latent">latent single-head (DeepSeek V4)</option>
          </select>
          <input className={field} placeholder="Layers" type="number" min="1" value={f.layers} onChange={set('layers')} required />
          <input className={field} placeholder="Layers with a growing cache (default: all)" type="number" min="0" value={f.kvLayers} onChange={set('kvLayers')} />
          <input className={field} placeholder="KV heads (default 8)" type="number" min="1" value={f.kvHeads} onChange={set('kvHeads')} />
          <input className={field} placeholder="Head size (default 128)" type="number" min="1" value={f.headDim} onChange={set('headDim')} />
          <input className={field} placeholder="KV latent rank (MLA)" type="number" min="0" value={f.kvLoraRank} onChange={set('kvLoraRank')} />
          <input className={field} placeholder="Rope dims (MLA / latent)" type="number" min="0" value={f.ropeDim} onChange={set('ropeDim')} />
          <input className={field} placeholder="Sliding-window layers" type="number" min="0" value={f.slidingLayers} onChange={set('slidingLayers')} />
          <input className={field} placeholder="Window size (tokens)" type="number" min="0" value={f.window} onChange={set('window')} />
        </>
      )}
      <div className="md:col-span-2 flex gap-2">
        <button type="submit" className="lls-chip !py-1.5 !px-3 font-medium" disabled={!valid}>Add to the table</button>
        <button type="button" className="lls-chip !py-1.5 !px-3" onClick={onCancel}>Cancel</button>
      </div>
    </form>
  );
}

function CustomMachineForm({ onAdd, onCancel }: { onAdd: (m: AppState['customMachines'][number], id: string) => void; onCancel: () => void }) {
  const [f, setF] = useState({ name: '', memory: '', bandwidth: '', platform: 'apple' as Platform, chip: '' });
  const valid = f.name.trim() && Number(f.memory) > 0 && Number(f.bandwidth) > 0;
  return (
    <form
      className="grid gap-3"
      onSubmit={(e) => {
        e.preventDefault();
        if (!valid) return;
        const m = { name: f.name.trim(), memoryGb: Number(f.memory), bandwidthGbs: Number(f.bandwidth), platform: f.platform, ...(f.chip.trim() ? { chip: f.chip.trim() } : {}) };
        onAdd(m, customMachine(m).id);
      }}
    >
      <span className="md:col-span-2 text-sm font-semibold">Custom machine</span>
      <input className={field} placeholder="Name" value={f.name} onChange={(e) => setF({ ...f, name: e.target.value })} required />
      <input className={field} placeholder="Memory (GB)" type="number" min="1" value={f.memory} onChange={(e) => setF({ ...f, memory: e.target.value })} required />
      <input className={field} placeholder="Memory bandwidth (GB/s)" type="number" min="1" value={f.bandwidth} onChange={(e) => setF({ ...f, bandwidth: e.target.value })} required />
      <select className={field} value={f.platform} onChange={(e) => setF({ ...f, platform: e.target.value as Platform })} aria-label="Platform">
        <option value="apple">Apple silicon (macOS rules)</option>
        <option value="cuda">NVIDIA unified box (DGX Spark-like)</option>
        <option value="rocm">AMD unified box (no speed factor yet)</option>
      </select>
      <input className={field} placeholder="Chip (optional, e.g. M4 Pro, picks the measured factor)" value={f.chip} onChange={(e) => setF({ ...f, chip: e.target.value })} />
      <div className="md:col-span-2 flex gap-2">
        <button type="submit" className="lls-chip !py-1.5 !px-3 font-medium" disabled={!valid}>Add to the table</button>
        <button type="button" className="lls-chip !py-1.5 !px-3" onClick={onCancel}>Cancel</button>
      </div>
    </form>
  );
}

export function AdvancedPanel(props: AdvancedPanelProps) {
  const { state, dispatch, factors } = props;
  const inTable = stateGroups(state, props.groups);
  const capPct = state.cap === null ? null : Math.round(state.cap * 100);
  return (
    <div className="mt-3 grid gap-5">
      <div className="space-y-4">
        <div className="flex flex-wrap items-center gap-2 text-sm">
          <span>Set every column</span>
          <select aria-label="Every column: quantization" className="lls-chip !py-0.5 appearance-none" defaultValue="" onChange={(e) => { if (e.target.value) { const b = e.target.value as Bucket; dispatch({ type: 'APPLY_ALL_COLUMNS', quant: b === 'Q4' ? 'auto' : { bucket: b } }); } e.target.value = ''; }}>
            <option value="">quant…</option>
            {SIMPLE_BUCKETS.map((b) => (
              <option key={b} value={b}>{b}</option>
            ))}
          </select>
          <select aria-label="Every column: context" className="lls-chip !py-0.5 appearance-none" defaultValue="" onChange={(e) => { if (e.target.value) dispatch({ type: 'APPLY_ALL_COLUMNS', ctx: Number(e.target.value) }); e.target.value = ''; }}>
            <option value="">context…</option>
            {CONTEXT_CHIPS.map((c) => (
              <option key={c} value={c}>{formatContext(c)}</option>
            ))}
          </select>
        </div>
        <label className="block text-sm">
          <span className="flex justify-between"><span>App budget when the work toggle is on</span><span className="text-[var(--color-muted)]">{state.work} GB</span></span>
          <input type="range" className="lls-range" min={0} max={32} step={1} value={state.work} onChange={(e) => dispatch({ type: 'SET_WORK', gb: Number(e.target.value) })} list="lls-apps" />
          <datalist id="lls-apps"><option value="8" /><option value="16" /><option value="24" /></datalist>
          <span className="text-[11px] text-[var(--color-light)]">8 light · 16 typical · 24 heavy (a browser with many tabs alone is 5–7 GB)</span>
        </label>
        <label className="block text-sm">
          <span className="flex justify-between"><span>GPU memory limit (Macs)</span><span className="text-[var(--color-muted)]">{capPct === null ? 'macOS default (67 % / 75 %)' : `${capPct} %${capPct === 100 ? ' · override' : ''}`}</span></span>
          <input type="range" className="lls-range" min={50} max={100} step={1} value={capPct ?? 75} onChange={(e) => dispatch({ type: 'SET_CAP', cap: Number(e.target.value) / 100 })} />
          <span className="flex gap-2 mt-1">
            <button type="button" className="lls-chip" aria-pressed={capPct === null} onClick={() => dispatch({ type: 'SET_CAP', cap: null })}>macOS default</button>
            <button type="button" className="lls-chip" aria-pressed={capPct === 100} onClick={() => dispatch({ type: 'SET_CAP', cap: 1 })}>override (100 %)</button>
          </span>
        </label>
        <div className="flex flex-wrap gap-4 text-sm">
          <label className="flex items-center gap-2">
            Cache precision
            <select className="lls-chip !py-0.5 appearance-none" value={state.kvBits} onChange={(e) => dispatch({ type: 'SET_KV_BITS', kvBits: Number(e.target.value) as KvBits })}>
              <option value={16}>FP16 (default)</option>
              <option value={8}>8-bit</option>
              <option value={4}>4-bit</option>
            </select>
          </label>
          <label className="flex items-center gap-2">
            Quality floor for automatic fixes
            <select className="lls-chip !py-0.5 appearance-none" value={state.floorBits} onChange={(e) => dispatch({ type: 'SET_FLOOR', bits: Number(e.target.value) })}>
              <option value={1}>1-bit builds allowed</option>
              <option value={2}>2-bit and up (default)</option>
              <option value={3}>3-bit and up</option>
              <option value={4}>4-bit and up</option>
            </select>
          </label>
          <Switch checked={!state.simpleQuants} onChange={(v) => dispatch({ type: 'SET_SIMPLE_QUANTS', simple: !v })} label="Exact quant labels" />
        </div>
        <div className="text-sm">
          <p className="text-[11px] uppercase tracking-[0.18em] text-[var(--color-light)] mb-1">Speed factors in use</p>
          <ul className="space-y-0.5 text-[13px] text-[var(--color-muted)]">
            {inTable.map((g) => {
              const p = resolveChipProfile(g.rows[0].machine, factors);
              return (
                <li key={g.id}>
                  {g.family} · {g.chip}: {p ? `${Math.round(p.bEffGbs)} GB/s effective of ${g.rows[0].bandwidthGbs} (${p.source}), ${p.t0Ms.toFixed(1)} ms fixed cost per token` : `no speed profile · ${g.rows[0].bandwidthGbs} GB/s`}
                  {g.rows.length > 1 && g.rows.some((r) => r.bandwidthGbs !== g.rows[0].bandwidthGbs) ? ` (some sizes ${g.rows[g.rows.length - 1].bandwidthGbs} GB/s)` : ''}
                </li>
              );
            })}
            {factors.cluster && (
              <li>
                Linked machines: on a layer split each extra machine adds {factors.cluster.hop_ms.mac.value} ms per token on Macs and {factors.cluster.hop_ms.spark.value} ms on Sparks ({(factors.cluster.hop_ms.mac.n ?? 0) > 0 ? `fitted on ${factors.cluster.hop_ms.mac.n} published series` : 'assumed'}); in tensor parallel one machine's time ÷ N plus {factors.cluster.tensor_c_ms.dense.value} ms × log2(N) on dense models and {factors.cluster.tensor_c_ms.moe.value} ms on mixtures of experts (fitted), {factors.cluster.tensor_c_ms.gpu.value} ms on GPU boxes (assumed); reading speed × N^{factors.cluster.prefill.tensor_exponent.value} under tensor parallel (assumed), one machine's on a layer split
              </li>
            )}
            {factors.speed_model && (
              <li>
                MLX on the fixed cost: dense ×{factors.speed_model.mlx.overhead_factor.dense.value} · small MoE on M4 and newer ×{factors.speed_model.mlx.overhead_factor.small_active_moe_m4plus.value} · on M1 to M3 ×{factors.speed_model.mlx.overhead_factor.small_active_moe_pre_m4.value} · other MoE ×{factors.speed_model.mlx.overhead_factor.moe_other.value}; plus {factors.speed_model.mlx.attention_ms_per_32k.pre_m5.value} ms per token per 32K of context on M1 to M4 chips and {factors.speed_model.mlx.attention_ms_per_32k.m5plus.value} ms on M5 and newer
              </li>
            )}
          </ul>
        </div>
      </div>
      <div className="space-y-4">
        {props.customForm === 'model' ? (
          <CustomModelForm onAdd={(model, id) => { dispatch({ type: 'ADD_CUSTOM_MODEL', model, columnId: id }); props.onCustomForm(null); }} onCancel={() => props.onCustomForm(null)} />
        ) : props.customForm === 'machine' ? (
          <CustomMachineForm onAdd={(machine, id) => { dispatch({ type: 'ADD_CUSTOM_MACHINE', machine, groupId: id }); props.onCustomForm(null); }} onCancel={() => props.onCustomForm(null)} />
        ) : (
          <div className="flex flex-wrap gap-2">
            <button type="button" className="lls-chip !py-1.5 !px-3" onClick={() => props.onCustomForm('model')}>Add a custom model</button>
            <button type="button" className="lls-chip !py-1.5 !px-3" onClick={() => props.onCustomForm('machine')}>Add a custom machine</button>
            <button type="button" className="lls-chip !py-1.5 !px-3" onClick={() => props.onReport('Missing model: ')}>Report a missing model</button>
          </div>
        )}
        {(state.customModels.length > 0 || state.customMachines.length > 0) && (
          <ul className="text-[13px] text-[var(--color-muted)] space-y-1">
            {state.customModels.map((m) => (
              <li key={m.name} className="flex justify-between gap-3">
                <span>custom model · {m.name} · {m.paramsTotalB}B · {m.weightsGb} GB</span>
                <button type="button" className="underline" onClick={() => dispatch({ type: 'REMOVE_CUSTOM', kind: 'model', name: m.name })}>forget</button>
              </li>
            ))}
            {state.customMachines.map((m) => (
              <li key={m.name} className="flex justify-between gap-3">
                <span>custom machine · {m.name} · {m.memoryGb} GB · {m.bandwidthGbs} GB/s</span>
                <button type="button" className="underline" onClick={() => dispatch({ type: 'REMOVE_CUSTOM', kind: 'machine', name: m.name })}>forget</button>
              </li>
            ))}
          </ul>
        )}
        <p className="text-[12px] text-[var(--color-light)]">Custom entries travel in the share link and stay in this browser. Every number in the tool is explained in <a className="underline" href={aboutHref('speed')}>how the numbers are made</a>.</p>
      </div>
    </div>
  );
}
