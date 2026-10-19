/**
 * Site adapter for this copy. The upstream site (theaibridges.com) has its own values here: its name, its author
 * line, its email update card, its product card and its feedback inbox. This copy is neutral: no cards, and the
 * feedback form opens a prefilled issue on the code repository. Edit this one file to put the tool on your own site.
 */
export type FeedbackMode = 'inbox' | 'issues';

/** A small closable card at the bottom right pointing at your own product. */
export interface SitePromo {
  id: string;
  eyebrow: string;
  title: string;
  text: string;
  cta: string;
  href: string;
}

export interface SiteConfig {
  name: string;
  titleSuffix: string;
  toolUrl: string;
  author: { name: string; href: string; channelLabel: string } | null;
  watermark: string;
  footer: { copyright: string; links: { href: string; label: string }[] };
  repos: { code: string; data: string };
  ogFallback: string;
  features: { notify: boolean; feedback: FeedbackMode };
  promo?: SitePromo | null;
  /** one quiet line under the Macs in the 128 GB comparison, pointing at your own guide; null for none */
  macGuide?: { lead: string; label: string; href: string } | null;
  corsOrigins: string[];
}

export const SITE: SiteConfig = {
  name: 'LLM Sizer',
  titleSuffix: '',
  toolUrl: 'http://localhost:4321/tools/llm-sizer',
  author: null,
  watermark: 'LLM Sizer · open source (MIT)',
  footer: {
    copyright: 'LLM Sizer contributors. MIT license.',
    links: [{ href: 'https://github.com/alexgrankinukr-hash/llm-sizer/blob/main/LICENSE', label: 'License' }],
  },
  repos: { code: 'https://github.com/alexgrankinukr-hash/llm-sizer', data: 'https://github.com/alexgrankinukr-hash/llm-sizer-data' },
  ogFallback: '/og/default.png',
  features: { notify: false, feedback: 'issues' },
  promo: null,
  macGuide: null,
  corsOrigins: ['http://localhost:4321', 'http://127.0.0.1:4321'],
};

/** A prefilled "new issue" link on the code repository. */
export function issueUrl(title: string, body: string): string {
  return `${SITE.repos.code}/issues/new?title=${encodeURIComponent(title)}&body=${encodeURIComponent(body)}`;
}
