/** The two forms under the table: "notify me" (email capture) and feedback. */
import { useEffect, useState } from 'react';
import { analytics } from '../../lib/llm-sizer/app/analytics';
import { aboutHref } from '../../lib/llm-sizer/app/provenance';
import { issueUrl, SITE } from '../../lib/llm-sizer/adapters/site';

const field = 'w-full bg-transparent border-b-2 border-[var(--color-text)] focus:border-[var(--color-accent)] outline-none py-1.5 text-base';
const honeypot: React.CSSProperties = { position: 'absolute', left: -9999, width: 1, height: 1, overflow: 'hidden' };

async function post(url: string, body: Record<string, unknown>): Promise<{ ok: boolean; error?: string; message?: string; status?: string }> {
  try {
    const res = await fetch(url, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) });
    const data = (await res.json().catch(() => ({}))) as { ok?: boolean; error?: string; message?: string; status?: string };
    return { ok: res.ok && data.ok !== false, error: data.error, message: data.message, status: data.status };
  } catch {
    return { ok: false, error: 'NETWORK' };
  }
}

export function NotifyForm({ encoded, bare = false, onSuccess, title = 'New models are added daily.', text = 'Get a short weekly note on what now fits your Mac.' }: { encoded: string; bare?: boolean; onSuccess?: () => void; title?: string; text?: string }) {
  const [email, setEmail] = useState('');
  const [hp, setHp] = useState('');
  const [state, setState] = useState<'idle' | 'sending' | 'sent' | 'already' | 'error'>('idle');
  const [message, setMessage] = useState('');
  const params = typeof window !== 'undefined' ? new URLSearchParams(window.location.search) : null;
  const confirmed = params?.get('notify') === 'confirmed';
  const linkProblem = params?.get('link'); // invalid | expired | unavailable, from the confirm route
  return (
    <form
      className={bare ? '' : 'border border-[var(--color-border)] rounded-xl p-5'}
      onSubmit={async (e) => {
        e.preventDefault();
        setState('sending');
        const r = await post('/api/llm-sizer/notify', { email, state: encoded, website: hp });
        if (r.ok) {
          setState(r.status === 'already' ? 'already' : 'sent');
          analytics.notifySignup();
          onSuccess?.();
        } else {
          setState('error');
          setMessage(r.message ?? 'Something went wrong. Try again in a minute.');
        }
      }}
    >
      <p className="font-heading text-lg">{title}</p>
      <p className="text-sm text-[var(--color-muted)] mt-1">{text}</p>
      {linkProblem && state === 'idle' && (
        <p className="mt-3 text-sm text-[var(--color-error)]">{linkProblem === 'expired' ? 'That confirmation link has expired. Enter your email again for a fresh one.' : 'That confirmation link did not work. Enter your email again for a fresh one.'}</p>
      )}
      {confirmed ? (
        <p className="mt-3 text-sm text-[var(--color-success)]">Confirmed. The first note comes with the next batch of models.</p>
      ) : state === 'sent' ? (
        <p className="mt-3 text-sm">Check your inbox: one click confirms it.</p>
      ) : state === 'already' ? (
        <p className="mt-3 text-sm">You're already on the list. The LLM Sizer note is now added.</p>
      ) : (
        <>
          <div className="mt-3 flex flex-col sm:flex-row gap-3">
            <input className={field} type="email" required placeholder="you@example.com" value={email} onChange={(e) => setEmail(e.target.value)} aria-label="Email" />
            <input style={honeypot} tabIndex={-1} autoComplete="off" aria-hidden="true" value={hp} onChange={(e) => setHp(e.target.value)} />
            <button type="submit" className="lls-chip !py-2 !px-4 font-medium whitespace-nowrap" disabled={state === 'sending'}>
              {state === 'sending' ? 'Sending…' : 'Notify me'}
            </button>
          </div>
          {state === 'error' && <p className="mt-2 text-sm text-[var(--color-error)]">{message}</p>}
          <p className="mt-2 text-[11px] text-[var(--color-light)]">One email a week at most. Confirming also enrols you in the site's free resources list; unsubscribe with one click.</p>
        </>
      )}
    </form>
  );
}

export function FeedbackForm({ shareUrl, encoded, view, prefill, onPrefillUsed, bare = false }: { shareUrl: string; encoded: string; view: string; prefill: string | null; onPrefillUsed: () => void; bare?: boolean }) {
  const [message, setMessage] = useState('');
  const [email, setEmail] = useState('');
  const [hp, setHp] = useState('');
  const [state, setState] = useState<'idle' | 'sending' | 'sent' | 'error'>('idle');
  const [error, setError] = useState('');
  useEffect(() => {
    if (prefill) {
      setMessage((m) => (m ? `${m}\n${prefill}` : prefill));
      setState('idle');
      onPrefillUsed();
      document.getElementById('lls-feedback-message')?.focus();
    }
  }, [prefill, onPrefillUsed]);
  return (
    <form
      className={bare ? '' : 'border border-[var(--color-border)] rounded-xl p-5 scroll-mt-24'}
      onSubmit={async (e) => {
        e.preventDefault();
        if (SITE.features.feedback === 'issues') {
          // no inbox on this copy: the message becomes a prefilled issue on the code repository, the table link included
          window.open(issueUrl(message.split('\n')[0].slice(0, 80) || 'Wrong number', `${message}\n\nTable: ${shareUrl}\nView: ${view}`), '_blank', 'noopener');
          setState('sent');
          analytics.feedbackSent(false);
          return;
        }
        setState('sending');
        const r = await post('/api/llm-sizer/feedback', { message, email: email || undefined, state: encoded, stateUrl: shareUrl, view, website: hp });
        if (r.ok) {
          setState('sent');
          analytics.feedbackSent(!!email);
        } else {
          setState('error');
          setError(r.message ?? 'Could not send. Try again in a minute.');
        }
      }}
    >
      <p id="lls-feedback-title" className="font-heading text-lg">Wrong number? Missing model?</p>
      <p className="text-sm text-[var(--color-muted)] mt-1">Tell us. Your current table travels with the message so we can reproduce it.</p>
      {state === 'sent' ? (
        <p className="mt-3 text-sm">
          {SITE.features.feedback === 'issues' ? 'Thanks. The issue form opened in a new tab with your table attached.' : 'Thanks, received.'} Corrections show up in the <a className="underline" href={aboutHref('changelog')}>changelog</a>.
        </p>
      ) : (
        <>
          <textarea id="lls-feedback-message" className={`${field} mt-3 min-h-[84px] resize-y`} required minLength={3} maxLength={2000} placeholder="What's off, and what you measured or expected" value={message} onChange={(e) => setMessage(e.target.value)} aria-label="Message" />
          <div className="mt-2 flex flex-col sm:flex-row gap-3">
            {SITE.features.feedback === 'inbox' && <input className={field} type="email" placeholder="Email (optional, for a reply)" value={email} onChange={(e) => setEmail(e.target.value)} aria-label="Email (optional)" />}
            <input style={honeypot} tabIndex={-1} autoComplete="off" aria-hidden="true" value={hp} onChange={(e) => setHp(e.target.value)} />
            <button type="submit" className="lls-chip !py-2 !px-4 font-medium whitespace-nowrap" disabled={state === 'sending' || message.trim().length < 3}>
              {state === 'sending' ? 'Sending…' : SITE.features.feedback === 'issues' ? 'Open an issue' : 'Send'}
            </button>
          </div>
          {state === 'error' && <p className="mt-2 text-sm text-[var(--color-error)]">{error}</p>}
          <p className="mt-2 text-[11px] text-[var(--color-light)]">
            Developers: <a className="underline" href={`${SITE.repos.code}/issues`}>open an issue</a> on the code repository, or on the <a className="underline" href={`${SITE.repos.data}/issues`}>data repository</a> for a wrong row.
          </p>
        </>
      )}
    </form>
  );
}
