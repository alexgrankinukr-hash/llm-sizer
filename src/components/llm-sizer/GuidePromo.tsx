/** The site's own product, in a small closable card at the bottom right: it waits a few seconds of active time so it never
 * greets the visitor on load, sits beside the tool rather than over it, and stays away for two weeks once closed or followed. */
import { useEffect, useState } from 'react';
import type { SitePromo } from '../../lib/llm-sizer/adapters/site';
import type { Analytics } from '../../lib/llm-sizer/app/analytics';
import { XIcon } from './icons';

const KEY = 'llm-sizer.promo';
const AWAY_DAYS = 14;
/** active seconds on the page (the tab visible) before the card slides in */
const MIN_ACTIVE_SECONDS = 15;

type Stored = { id?: string; closedAt?: string };

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

export function GuidePromo({ promo, analytics }: { promo: SitePromo; analytics: Analytics }) {
  const [allowed, setAllowed] = useState(false);
  const [open, setOpen] = useState(false);
  const [activeSeconds, setActiveSeconds] = useState(0);
  useEffect(() => {
    const s = read();
    // a new promo (another id) shows even if the last one was closed
    const away = s.id === promo.id && s.closedAt ? Date.now() - new Date(s.closedAt).getTime() < AWAY_DAYS * 86400e3 : false;
    setAllowed(!away);
  }, [promo.id]);
  useEffect(() => {
    if (!allowed || open) return;
    const t = window.setInterval(() => {
      if (document.visibilityState === 'visible') setActiveSeconds((n) => n + 1);
    }, 1000);
    return () => window.clearInterval(t);
  }, [allowed, open]);
  useEffect(() => {
    if (allowed && !open && activeSeconds >= MIN_ACTIVE_SECONDS) {
      setOpen(true);
      analytics.promoShown(promo.id);
    }
  }, [allowed, open, activeSeconds]);
  useEffect(() => {
    if (!open) return;
    function onKey(e: KeyboardEvent) {
      if (e.key === 'Escape' && !document.querySelector('dialog[open]') && !document.querySelector('.lls-notify')) close('dismiss');
    }
    document.addEventListener('keydown', onKey);
    return () => document.removeEventListener('keydown', onKey);
  });
  function close(how: 'dismiss' | 'click') {
    write({ id: promo.id, closedAt: new Date().toISOString() });
    if (how === 'dismiss') {
      analytics.promoDismissed(promo.id);
      setOpen(false);
      setAllowed(false);
    } else {
      analytics.promoClicked(promo.id);
      // the link opens in a new tab; the card leaves this one a moment later
      window.setTimeout(() => {
        setOpen(false);
        setAllowed(false);
      }, 400);
    }
  }
  if (!open) return null;
  return (
    <aside aria-label={promo.title} className="lls-promo">
      <div className="flex items-start justify-between gap-3">
        <p className="lls-promo-eyebrow">{promo.eyebrow}</p>
        <button type="button" className="lls-chip !px-1.5 shrink-0" onClick={() => close('dismiss')} aria-label="Close">
          <XIcon size={14} />
        </button>
      </div>
      <p className="lls-promo-title">{promo.title}</p>
      <p className="lls-promo-text">{promo.text}</p>
      <a className="lls-promo-cta" href={promo.href} target="_blank" rel="noopener" onClick={() => close('click')}>
        {promo.cta} <span aria-hidden="true">→</span>
      </a>
    </aside>
  );
}
