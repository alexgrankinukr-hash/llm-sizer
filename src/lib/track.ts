/** Analytics stub. The upstream site sends these events to GA4; this copy sends them nowhere. Replace with your own. */
export type TrackParams = Record<string, string | number | boolean>;

export function track(_name: string, _params: TrackParams = {}): void {}
