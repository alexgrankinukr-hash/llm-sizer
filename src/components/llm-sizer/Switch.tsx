/** An on/off switch with a label and an optional hint. */
export function Switch({ checked, onChange, label, hint }: { checked: boolean; onChange: (v: boolean) => void; label: string; hint?: string }) {
  return (
    <button type="button" role="switch" aria-checked={checked} className="lls-switch" onClick={() => onChange(!checked)}>
      <span className="track" aria-hidden="true" />
      <span className="text-[var(--color-text)]">
        {label}
        {hint && <span className="text-[var(--color-light)] ml-1">{hint}</span>}
      </span>
    </button>
  );
}
