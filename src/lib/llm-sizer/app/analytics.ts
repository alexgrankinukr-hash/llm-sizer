/** The tool's analytics events with their parameter sets, over the site's GA4 wrapper. */
import { track } from '../../track';
import { EVENTS } from '../events';

export const analytics = {
  view: () => track(EVENTS.view),
  machineAdded: (group: string) => track(EVENTS.machineAdded, { group }),
  modelAdded: (modelId: string, featured: boolean) => track(EVENTS.modelAdded, { model_id: modelId, featured }),
  toggleChanged: (toggle: 'work' | 'limit' | 'runtime' | 'advanced' | 'all_columns' | 'size' | 'budget' | 'form_factor' | 'build' | 'min_tok_s' | 'linked' | 'split', value: string | number | boolean) =>
    track(EVENTS.toggleChanged, { toggle, value }),
  cellOpened: (verdict: string) => track(EVENTS.cellOpened, { verdict }),
  shareLink: (method: 'copy' | 'share') => track(EVENTS.shareLink, { method }),
  imageExported: (view: string, size: string) => track(EVENTS.imageExported, { view, size }),
  feedbackSent: (hasEmail: boolean) => track(EVENTS.feedbackSent, { has_email: hasEmail }),
  notifySignup: () => track(EVENTS.notifySignup),
  aboutView: (section: string) => track(EVENTS.aboutView, { section }),
  viewSelected: (view: string) => track(EVENTS.viewSelected, { view }),
  settingsOpened: () => track(EVENTS.settingsOpened),
  benchmarksInterest: (machine: string) => track(EVENTS.benchmarksInterest, { machine }),
};

/** The same surface with no effect: used in presentation mode so filming never counts as usage. */
export const silentAnalytics = Object.fromEntries(Object.keys(analytics).map((k) => [k, () => undefined])) as unknown as typeof analytics;

export type Analytics = typeof analytics;
