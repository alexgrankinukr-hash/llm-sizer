/** Inline SVG icons (Lucide geometry), the site's convention for React islands. */
import type { SVGProps } from 'react';

type IconProps = SVGProps<SVGSVGElement> & { size?: number };

function base({ size = 16, ...rest }: IconProps) {
  return { width: size, height: size, viewBox: '0 0 24 24', fill: 'none', stroke: 'currentColor', strokeWidth: 2, strokeLinecap: 'round' as const, strokeLinejoin: 'round' as const, 'aria-hidden': true, ...rest };
}

export const PlusIcon = (p: IconProps) => (
  <svg {...base(p)}>
    <path d="M12 5v14M5 12h14" />
  </svg>
);
export const XIcon = (p: IconProps) => (
  <svg {...base(p)}>
    <path d="M18 6 6 18M6 6l12 12" />
  </svg>
);
export const ChevronDownIcon = (p: IconProps) => (
  <svg {...base(p)}>
    <path d="m6 9 6 6 6-6" />
  </svg>
);
export const ChevronLeftIcon = (p: IconProps) => (
  <svg {...base(p)}>
    <path d="m15 18-6-6 6-6" />
  </svg>
);
export const LinkIcon = (p: IconProps) => (
  <svg {...base(p)}>
    <path d="M10 13a5 5 0 0 0 7.54.54l3-3a5 5 0 0 0-7.07-7.07l-1.72 1.71" />
    <path d="M14 11a5 5 0 0 0-7.54-.54l-3 3a5 5 0 0 0 7.07 7.07l1.71-1.71" />
  </svg>
);
export const DownloadIcon = (p: IconProps) => (
  <svg {...base(p)}>
    <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4M7 10l5 5 5-5M12 15V3" />
  </svg>
);
export const ShareIcon = (p: IconProps) => (
  <svg {...base(p)}>
    <path d="M4 12v8a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2v-8M16 6l-4-4-4 4M12 2v13" />
  </svg>
);
export const SlidersIcon = (p: IconProps) => (
  <svg {...base(p)}>
    <path d="M4 21v-7M4 10V3M12 21v-9M12 8V3M20 21v-5M20 12V3M1 14h6M9 8h6M17 16h6" />
  </svg>
);
export const InfoIcon = (p: IconProps) => (
  <svg {...base(p)}>
    <circle cx="12" cy="12" r="10" />
    <path d="M12 16v-4M12 8h.01" />
  </svg>
);
export const GaugeIcon = (p: IconProps) => (
  <svg {...base(p)}>
    <path d="m12 14 4-4M3.34 19a10 10 0 1 1 17.32 0" />
  </svg>
);
export const SearchIcon = (p: IconProps) => (
  <svg {...base(p)}>
    <circle cx="11" cy="11" r="8" />
    <path d="m21 21-4.3-4.3" />
  </svg>
);
export const PresentIcon = (p: IconProps) => (
  <svg {...base(p)}>
    <rect x="2" y="3" width="20" height="14" rx="2" />
    <path d="M8 21h8M12 17v4" />
  </svg>
);
export const MessageIcon = (p: IconProps) => (
  <svg {...base(p)}>
    <path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z" />
  </svg>
);
export const ExternalIcon = (p: IconProps) => (
  <svg {...base(p)}>
    <path d="M15 3h6v6M10 14 21 3M21 14v5a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h5" />
  </svg>
);
export const CheckIcon = (p: IconProps) => (
  <svg {...base(p)}>
    <path d="M20 6 9 17l-5-5" />
  </svg>
);
export const TrashIcon = (p: IconProps) => (
  <svg {...base(p)}>
    <path d="M3 6h18M8 6V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2M19 6l-1 14a2 2 0 0 1-2 2H8a2 2 0 0 1-2-2L5 6" />
  </svg>
);


export function GridIcon({ size = 16 }: { size?: number }) {
  return (
    <svg {...base({ size })}>
      <rect x="3" y="3" width="7" height="7" rx="1" /><rect x="14" y="3" width="7" height="7" rx="1" /><rect x="3" y="14" width="7" height="7" rx="1" /><rect x="14" y="14" width="7" height="7" rx="1" />
    </svg>
  );
}
export function TargetIcon({ size = 16 }: { size?: number }) {
  return (
    <svg {...base({ size })}>
      <circle cx="12" cy="12" r="9" /><circle cx="12" cy="12" r="4" /><path d="M12 3v3M12 18v3M3 12h3M18 12h3" />
    </svg>
  );
}
export function ListChecksIcon({ size = 16 }: { size?: number }) {
  return (
    <svg {...base({ size })}>
      <path d="m3 17 2 2 4-4M3 7l2 2 4-4M13 6h8M13 12h8M13 18h8" />
    </svg>
  );
}
export function LayersIcon({ size = 16 }: { size?: number }) {
  return (
    <svg {...base({ size })}>
      <rect x="3" y="5" width="18" height="5" rx="1" /><rect x="3" y="14" width="11" height="5" rx="1" /><path d="M14 16.5h7" />
    </svg>
  );
}
export function GripIcon({ size = 14 }: { size?: number }) {
  return (
    <svg {...base({ size })} fill="currentColor" stroke="none">
      <circle cx="9" cy="6" r="1.6" /><circle cx="15" cy="6" r="1.6" /><circle cx="9" cy="12" r="1.6" /><circle cx="15" cy="12" r="1.6" /><circle cx="9" cy="18" r="1.6" /><circle cx="15" cy="18" r="1.6" />
    </svg>
  );
}

/** Platform marks: which family of machine a row is (used in the rail, the machines list and the picker). */
export function AppleIcon({ size = 14 }: { size?: number }) {
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" fill="currentColor" aria-hidden="true">
      <path d="M17.1 12.7c0-2.4 2-3.6 2.1-3.6-1.1-1.7-2.9-1.9-3.6-1.9-1.5-.2-3 .9-3.7.9-.8 0-2-.9-3.2-.8-1.7 0-3.2 1-4 2.5-1.7 2.9-.4 7.2 1.2 9.6.8 1.2 1.8 2.5 3.1 2.4 1.2 0 1.7-.8 3.2-.8s1.9.8 3.2.8c1.3 0 2.2-1.2 3-2.4 1-1.4 1.3-2.7 1.4-2.8-.1 0-2.7-1.1-2.7-3.9zM14.7 5.6c.7-.8 1.1-2 1-3.2-1 0-2.2.7-2.9 1.5-.6.8-1.2 2-1 3.1 1.1.1 2.2-.6 2.9-1.4z" />
    </svg>
  );
}

export function NvidiaIcon({ size = 14 }: { size?: number }) {
  // NVIDIA's eye mark, drawn in the current text colour so it sits in the tool's palette
  return (
    <svg width={size} height={size} viewBox="0 0 64 64" fill="currentColor" aria-hidden="true">
      <path d="M23.862 23.46v-3.816l1.13-.047c10.46-.33 17.313 8.998 17.313 8.998s-7.396 10.27-15.335 10.27a9.73 9.73 0 0 1-3.086-.495v-11.59c4.075.495 4.9 2.285 7.326 6.36l5.44-4.57s-3.98-5.206-10.67-5.206c-.707-.024-1.413.024-2.12.094m0-12.626v5.7l1.13-.07c14.534-.495 24.026 11.92 24.026 11.92S38.136 41.622 26.806 41.622c-.99 0-1.955-.094-2.92-.26v3.533c.8.094 1.625.165 2.426.165 10.553 0 18.185-5.394 25.58-11.754 1.225.99 6.242 3.368 7.28 4.405-7.02 5.89-23.39 10.623-32.67 10.623a23.24 23.24 0 0 1-2.591-.141v4.97H64v-42.33zm0 27.536v3.015C14.1 39.644 11.4 29.49 11.4 29.49s4.688-5.182 12.46-6.03v3.298h-.024c-4.075-.495-7.28 3.32-7.28 3.32s1.814 6.43 7.302 8.29M6.548 29.067s5.77-8.527 17.337-9.422v-3.11C11.07 17.572 0 28.408 0 28.408s6.266 18.138 23.862 19.787v-3.298c-12.908-1.602-17.313-15.83-17.313-15.83z" />
    </svg>
  );
}

export function ChipIcon({ size = 14 }: { size?: number }) {
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.7" strokeLinejoin="round" aria-hidden="true">
      <rect x="7" y="7" width="10" height="10" rx="1.5" />
      <path d="M10 3v3M14 3v3M10 18v3M14 18v3M3 10h3M3 14h3M18 10h3M18 14h3" />
    </svg>
  );
}

/** The mark for a machine's platform, with an accessible name. */
export function PlatformIcon({ platform, size = 14 }: { platform: string; size?: number }) {
  const label = platform === 'apple' ? 'Apple silicon' : platform === 'cuda' ? 'NVIDIA' : 'other platform';
  const Mark = platform === 'apple' ? AppleIcon : platform === 'cuda' ? NvidiaIcon : ChipIcon;
  return (
    <span className="lls-platform" title={label} role="img" aria-label={label}>
      <Mark size={size} />
    </span>
  );
}


