/** "New models weekly": a small closable card that waits until the visitor has had some value from the tool: three minutes of
 * active time on the page (the tab visible) and at least two real interactions. Dismissed, it stays away for a week; subscribed, for good. */
import { useEffect, useState } from 'react';
import { NotifyForm } from './Forms';
import { XIcon } from './icons';

const KEY = 'llm-sizer.notify';
const DISMISS_DAYS = 7;
/** active seconds on the page before the automatic card may open */
const MIN_ACTIVE_SECONDS = 180;
/** real interactions (settings changed, a cell opened, a view switched) before it may open */
const MIN_INTERACTIONS = 2;

type Stored = { dismissedAt?: string; done?: boolean };

function read(): Stored {
  try {
    const raw = localStorage.getItem(KEY);
    return raw ? (JSON.parse(raw) as Stored) : {};
  } catch {
    return {};
  }
}
function write(v: Stored) {
  try {
    localStorage.setItem(KEY, JSON.stringify(v));
  } catch {
    /* fine */
  }
}

export type NotifyTopic = 'updates' | 'benchmarks';
export interface NotifyRequest {
  topic: NotifyTopic;
  /** a fresh id per click, so the same topic can be asked for twice */
  id: number;
}

const COPY: Record<NotifyTopic, { eyebrow: string; title: string; text: string }> = {
  updates: { eyebrow: 'Stay current', title: 'New models are added daily.', text: 'Get a short weekly note on what now fits your Mac.' },
  benchmarks: { eyebrow: 'Benchmarks', title: 'Measured speeds are coming.', text: 'Every number here is an estimate today. Leave your email and we will tell you when real benchmarks sit next to them.' },
};

export function NotifyPopup({ encoded, interactions, disabled = false, request = null }: { encoded: string; interactions: number; disabled?: boolean; request?: NotifyRequest | null }) {
  const [mounted, setMounted] = useState(false);
  const [open, setOpen] = useState(false);
  const [suppressed, setSuppressed] = useState(true);
  const [topic, setTopic] = useState<NotifyTopic>('updates');
  const [activeSeconds, setActiveSeconds] = useState(0);
  // an explicit ask (the benchmarks chip) always opens the card, whatever the automatic card's dismissal state
  useEffect(() => {
    if (!request) return;
    setTopic(request.topic);
    setOpen(true);
  }, [request?.id]);
  useEffect(() => {
    setMounted(true);
    const params = new URLSearchParams(window.location.search);
    const landed = params.get('notify') === 'confirmed' || params.has('link');
    const s = read();
    const recent = s.dismissedAt ? Date.now() - new Date(s.dismissedAt).getTime() < DISMISS_DAYS * 86400e3 : false;
    const allowed = !disabled && !s.done && (landed || !recent);
    setSuppressed(!allowed);
    if (allowed && landed) setOpen(true);
  }, [disabled]);
  // active time: one tick per second while the tab is visible, so a page left open in the background does not count
  useEffect(() => {
    if (!mounted || suppressed) return;
    const t = window.setInterval(() => {
      if (document.visibilityState === 'visible') setActiveSeconds((n) => n + 1);
    }, 1000);
    return () => window.clearInterval(t);
  }, [mounted, suppressed]);
  useEffect(() => {
    if (!mounted || suppressed || open) return;
    if (activeSeconds >= MIN_ACTIVE_SECONDS && interactions >= MIN_INTERACTIONS) setOpen(true);
  }, [mounted, suppressed, open, activeSeconds, interactions]);
  useEffect(() => {
    if (!open) return;
    function onKey(e: KeyboardEvent) {
      if (e.key === 'Escape' && !document.querySelector('dialog[open]')) dismiss();
    }
    document.addEventListener('keydown', onKey);
    return () => document.removeEventListener('keydown', onKey);
  });
  function dismiss() {
    write({ dismissedAt: new Date().toISOString() });
    setOpen(false);
    setSuppressed(true);
  }
  if (!open) return null;
  const copy = COPY[topic];
  return (
    <div role="dialog" aria-label={topic === 'benchmarks' ? 'Benchmarks are coming' : 'New models weekly'} aria-modal="false" className="lls-notify">
      <div className="flex items-start justify-between gap-3">
        <p className="text-[11px] uppercase tracking-[0.2em] text-[var(--color-light)]">{copy.eyebrow}</p>
        <button type="button" className="lls-chip !px-1.5 shrink-0" onClick={dismiss} aria-label="Close">
          <XIcon size={14} />
        </button>
      </div>
      <NotifyForm
        bare
        encoded={encoded}
        title={copy.title}
        text={copy.text}
        onSuccess={() => {
          write({ done: true });
          window.setTimeout(() => setOpen(false), 6000);
        }}
      />
    </div>
  );
}
