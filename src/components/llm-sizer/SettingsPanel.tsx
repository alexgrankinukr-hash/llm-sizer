/** The settings panel: everything that configures the chart lives here, never in the chart. */
import { lazy, Suspense } from 'react';
import type { Dispatch } from 'react';
import type { Factors, ModelIndex } from '../../lib/llm-sizer/engine/types';
import type { Analytics } from '../../lib/llm-sizer/app/analytics';
import type { MachineGroup } from '../../lib/llm-sizer/app/machines';
import type { Matrix } from '../../lib/llm-sizer/app/matrix';
import { BUDGET_MAX_USD, BUDGET_MIN_USD, type Action, type AppState, type FormFactor } from '../../lib/llm-sizer/app/state';
import { MachinesSection } from './MachinesSection';
import { ModelsSection } from './ModelsSection';
import { Switch } from './Switch';
import { XIcon } from './icons';

// Collapsed by default, so it is never part of the server render: loaded on demand.
const AdvancedPanel = lazy(() => import('./AdvancedPanel').then((m) => ({ default: m.AdvancedPanel })));

export interface SettingsPanelProps {
  id: string;
  className?: string;
  state: AppState;
  dispatch: Dispatch<Action>;
  groups: MachineGroup[];
  matrix: Matrix;
  index: ModelIndex | null;
  indexStatus: 'loading' | 'ready' | 'error';
  factors: Factors;
  analytics: Analytics;
  advancedOpen: boolean;
  onAdvancedToggle: (open: boolean) => void;
  customForm: 'model' | 'machine' | null;
  onCustomForm: (form: 'model' | 'machine' | null) => void;
  onReport: (text: string) => void;
  onRetry: (id: string) => void;
  onClose?: () => void;
}

const FORM_FACTORS: { value: FormFactor; label: string }[] = [
  { value: 'any', label: 'any' },
  { value: 'laptop', label: 'laptop' },
  { value: 'desktop', label: 'desktop' },
];

export function SettingsPanel(props: SettingsPanelProps) {
  const { state, dispatch, analytics } = props;
  const limitOn = state.cap === null;
  return (
    <aside id={props.id} className={`lls-panel ${props.className ?? ''}`} aria-label="Settings">
      <div className="flex items-center justify-between gap-3 mb-4">
        <h2 className="lls-eyebrow">Settings</h2>
        {props.onClose && (
          <button type="button" className="lls-chip !px-1.5" onClick={props.onClose} aria-label="Close settings">
            <XIcon size={14} />
          </button>
        )}
      </div>
      <div className="lls-panel-grid">
        <MachinesSection
          state={state}
          groups={props.groups}
          dispatch={dispatch}
          analytics={analytics}
          onCustom={() => {
            props.onAdvancedToggle(true);
            props.onCustomForm('machine');
          }}
        />
        <ModelsSection
          state={state}
          matrix={props.matrix}
          dispatch={dispatch}
          analytics={analytics}
          index={props.index}
          indexStatus={props.indexStatus}
          onRetry={props.onRetry}
          onCustom={() => {
            props.onAdvancedToggle(true);
            props.onCustomForm('model');
          }}
        />
        <section aria-labelledby="lls-set-assumptions" className="lls-panel-wide lls-panel-assumptions">
          <h3 id="lls-set-assumptions" className="lls-eyebrow mb-2">Assumptions</h3>
          <div className="flex flex-col gap-2.5">
            <Switch
              checked={state.work > 0}
              onChange={(on) => {
                dispatch({ type: 'SET_WORK', gb: on ? 16 : 0 });
                analytics.toggleChanged('work', on);
              }}
              label="I'll also use it for work"
              hint={state.work > 0 ? `(${state.work} GB for apps)` : undefined}
            />
            <Switch
              checked={limitOn}
              onChange={(on) => {
                dispatch({ type: 'SET_CAP', cap: on ? null : 1 });
                analytics.toggleChanged('limit', on);
              }}
              label="macOS memory limit"
              hint={limitOn ? '(67 % / 75 %)' : `(override: ${Math.round((state.cap ?? 1) * 100)} %)`}
            />
            <div className="flex items-center gap-3 text-sm">
              <span>Runtime</span>
              <div className="lls-seg" role="group" aria-label="Runtime">
                <button type="button" aria-pressed={state.runtime === 'gguf'} onClick={() => { dispatch({ type: 'SET_RUNTIME', runtime: 'gguf' }); analytics.toggleChanged('runtime', 'gguf'); }}>
                  GGUF
                </button>
                <button type="button" aria-pressed={state.runtime === 'mlx'} onClick={() => { dispatch({ type: 'SET_RUNTIME', runtime: 'mlx' }); analytics.toggleChanged('runtime', 'mlx'); }}>
                  MLX
                </button>
              </div>
            </div>
            {/* the map's filters: a budget and a form factor narrow "Which machine?" */}
              <div className="flex flex-wrap items-center gap-3 text-sm">
                <label htmlFor="lls-budget">Budget</label>
                <input
                  id="lls-budget"
                  className="lls-budget"
                  type="number"
                  inputMode="numeric"
                  min={BUDGET_MIN_USD}
                  max={BUDGET_MAX_USD}
                  step={100}
                  placeholder="USD, optional"
                  aria-label="Budget in US dollars"
                  value={state.budgetUsd ?? ''}
                  onChange={(e) => {
                    const usd = e.target.value === '' ? null : Number(e.target.value);
                    dispatch({ type: 'SET_BUDGET', usd });
                    analytics.toggleChanged('budget', usd ?? 'none');
                  }}
                />
                {state.budgetUsd !== null && (
                  <button type="button" className="lls-chip !py-0.5" onClick={() => { dispatch({ type: 'SET_BUDGET', usd: null }); analytics.toggleChanged('budget', 'none'); }}>
                    clear
                  </button>
                )}
              </div>
              <div className="flex items-center gap-3 text-sm">
                <span>Form factor</span>
                <div className="lls-seg" role="group" aria-label="Form factor">
                  {FORM_FACTORS.map((f) => (
                    <button key={f.value} type="button" aria-pressed={state.formFactor === f.value} onClick={() => { dispatch({ type: 'SET_FORM_FACTOR', formFactor: f.value }); analytics.toggleChanged('form_factor', f.value); }}>
                      {f.label}
                    </button>
                  ))}
                </div>
              </div>
          </div>
        </section>
        <details className="lls-details lls-panel-wide" open={props.advancedOpen} onToggle={(e) => props.onAdvancedToggle(e.currentTarget.open)}>
          <summary className="lls-eyebrow cursor-pointer select-none">Advanced</summary>
          {props.advancedOpen && (
            <Suspense fallback={<div className="h-32" aria-busy="true" />}>
              <AdvancedPanel state={state} dispatch={dispatch} groups={props.groups} matrix={props.matrix} factors={props.factors} customForm={props.customForm} onCustomForm={props.onCustomForm} onReport={props.onReport} />
            </Suspense>
          )}
        </details>
      </div>
    </aside>
  );
}
