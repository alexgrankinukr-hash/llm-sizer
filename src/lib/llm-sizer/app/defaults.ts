/** The first screen: the chart from the video, live. */
import type { AppState } from './state';
import { STATE_DEFAULTS } from './state';

export const DEFAULT_GROUPS = ['mac-mini-m6', 'mac-studio-m5-ultra', 'nvidia-dgx-spark'];
export const DEFAULT_MODELS = ['qwen3.8-27b', 'qwen3.8-flash-next', 'glm-5.3-flash', 'kimi-k3'];

export function defaultState(): AppState {
  return {
    ...STATE_DEFAULTS,
    groups: [...DEFAULT_GROUPS],
    columns: DEFAULT_MODELS.map((id) => ({ id, quant: 'auto', ctx: 32768 })),
    view: { kind: 'table' },
    hiddenSizes: {},
    linked: {},
    customModels: [],
    customMachines: [],
  };
}
