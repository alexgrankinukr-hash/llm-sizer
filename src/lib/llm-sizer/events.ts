/** Analytics event names for LLM Sizer, all prefixed so they never collide with other site features in GA4. */
export const EVENTS = {
  view: 'llm_sizer_view',
  machineAdded: 'llm_sizer_machine_added',
  modelAdded: 'llm_sizer_model_added',
  toggleChanged: 'llm_sizer_toggle_changed',
  cellOpened: 'llm_sizer_cell_opened',
  shareLink: 'llm_sizer_share_link',
  imageExported: 'llm_sizer_image_exported',
  feedbackSent: 'llm_sizer_feedback_sent',
  notifySignup: 'llm_sizer_notify_signup',
  aboutView: 'llm_sizer_about_view',
  viewSelected: 'llm_sizer_view_selected',
  settingsOpened: 'llm_sizer_settings_opened',
  benchmarksInterest: 'llm_sizer_benchmarks_interest',
} as const;

export type EventName = (typeof EVENTS)[keyof typeof EVENTS];
