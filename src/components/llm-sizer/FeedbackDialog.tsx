/** The feedback form in a dialog, opened from the Feedback button, the cell sheet and Advanced. */
import { useEffect, useRef } from 'react';
import { FeedbackForm } from './Forms';
import { XIcon } from './icons';

export interface FeedbackDialogProps {
  open: boolean;
  prefill: string | null;
  onPrefillUsed: () => void;
  onClose: () => void;
  shareUrl: string;
  encoded: string;
  view: string;
}

export function FeedbackDialog(props: FeedbackDialogProps) {
  const ref = useRef<HTMLDialogElement>(null);
  useEffect(() => {
    const d = ref.current;
    if (!d) return;
    if (props.open && !d.open) d.showModal();
    if (!props.open && d.open) d.close();
  }, [props.open]);
  return (
    <dialog
      ref={ref}
      id="lls-feedback"
      className="lls-sheet"
      aria-labelledby="lls-feedback-title"
      onClose={props.onClose}
      onClick={(e) => {
        if (e.target === ref.current) props.onClose();
      }}
    >
      <div className="p-5">
        <div className="flex items-start justify-between gap-3 mb-2">
          <p className="text-[11px] uppercase tracking-[0.2em] text-[var(--color-light)]">Feedback</p>
          <button type="button" className="lls-chip !px-1.5 shrink-0" onClick={props.onClose} aria-label="Close">
            <XIcon size={14} />
          </button>
        </div>
        {props.open && <FeedbackForm bare shareUrl={props.shareUrl} encoded={props.encoded} view={props.view} prefill={props.prefill} onPrefillUsed={props.onPrefillUsed} />}
      </div>
    </dialog>
  );
}
