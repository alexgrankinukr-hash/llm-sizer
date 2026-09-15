/** Line drawings of the machine families for the table's rail. Our own, in the brand's thin stroke. */
import type { Silhouette } from '../../lib/llm-sizer/app/machines';

interface Props {
  kind: Silhouette;
  width?: number;
  className?: string;
}

const STROKE = { fill: 'none', stroke: 'currentColor', strokeWidth: 1.5, strokeLinecap: 'round' as const, strokeLinejoin: 'round' as const };

export function MachineSilhouette({ kind, width = 64, className }: Props) {
  const common = { width, height: Math.round(width * 0.75), viewBox: '0 0 64 48', className, 'aria-hidden': true as const };
  switch (kind) {
    case 'mini':
      return (
        <svg {...common}>
          <rect x="10" y="20" width="44" height="16" rx="4" {...STROKE} />
          <path d="M14 36v2h36v-2" {...STROKE} />
          <circle cx="47" cy="28" r="1.4" fill="currentColor" stroke="none" />
        </svg>
      );
    case 'studio':
      return (
        <svg {...common}>
          <rect x="14" y="10" width="36" height="28" rx="5" {...STROKE} />
          <path d="M18 38v2h28v-2" {...STROKE} />
          <path d="M22 18h4M22 22h4" {...STROKE} />
        </svg>
      );
    case 'laptop':
      return (
        <svg {...common}>
          <rect x="12" y="10" width="40" height="24" rx="2.5" {...STROKE} />
          <path d="M6 38h52M8 34h48l2 4H6z" {...STROKE} />
        </svg>
      );
    case 'imac':
      return (
        <svg {...common}>
          <rect x="8" y="6" width="48" height="30" rx="2.5" {...STROKE} />
          <path d="M8 29h48M26 36l-2 6h16l-2-6" {...STROKE} />
        </svg>
      );
    case 'macpro':
      return (
        <svg {...common}>
          <rect x="18" y="8" width="28" height="34" rx="4" {...STROKE} />
          <circle cx="27" cy="18" r="2.2" {...STROKE} />
          <circle cx="37" cy="18" r="2.2" {...STROKE} />
          <circle cx="27" cy="28" r="2.2" {...STROKE} />
          <circle cx="37" cy="28" r="2.2" {...STROKE} />
          <path d="M22 6h20" {...STROKE} />
        </svg>
      );
    case 'spark':
      return (
        <svg {...common}>
          <rect x="16" y="12" width="32" height="26" rx="3" {...STROKE} />
          <path d="M16 20h32M20 24h6M20 28h6M20 32h6" {...STROKE} />
        </svg>
      );
    default:
      return (
        <svg {...common}>
          <rect x="12" y="10" width="40" height="28" rx="3" {...STROKE} />
          <path d="M18 18h10M18 24h10" {...STROKE} />
        </svg>
      );
  }
}
