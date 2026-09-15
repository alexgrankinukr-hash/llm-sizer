/** Display helpers shared by the island, the page shell and the exports. Pure string formatting. */

export function formatGb(gb: number, digits = 1): string {
  if (!Number.isFinite(gb)) return '—';
  if (gb >= 1000) return `${(gb / 1000).toFixed(2)} TB`;
  return `${gb.toFixed(gb >= 100 ? 0 : digits)} GB`;
}

export function formatBandwidth(gbs: number): string {
  if (!Number.isFinite(gbs) || gbs <= 0) return '—';
  return gbs >= 1000 ? `${(gbs / 1000).toFixed(1)} TB/s` : `${Math.round(gbs)} GB/s`;
}

export function formatPrice(usd: number | null | undefined): string {
  if (usd === null || usd === undefined || !Number.isFinite(usd)) return '—';
  return `$${Math.round(usd).toLocaleString('en-US')}`;
}

export function formatTokS(tokS: number | null): string {
  if (tokS === null || !Number.isFinite(tokS)) return '—';
  if (tokS >= 100) return `${Math.round(tokS)} tok/s`;
  return `${tokS.toFixed(tokS >= 10 ? 0 : 1)} tok/s`;
}

/** A wait in seconds as people say it: "4.2 s", "28 s", "2.6 min", "14 min", "1.2 h". */
export function formatWait(seconds: number | null): string {
  if (seconds === null || !Number.isFinite(seconds)) return '—';
  if (seconds < 10) return `${seconds.toFixed(1)} s`;
  if (seconds < 60) return `${Math.round(seconds)} s`;
  if (seconds < 600) return `${(seconds / 60).toFixed(1)} min`;
  if (seconds < 3600) return `${Math.round(seconds / 60)} min`;
  return `${(seconds / 3600).toFixed(1)} h`;
}

/** 8192 → "8K", 131072 → "128K", 1048576 → "1M". */
export function formatContext(tokens: number): string {
  if (tokens >= 1_000_000) return `${(tokens / 1_048_576).toFixed(tokens % 1_048_576 ? 1 : 0)}M`;
  if (tokens >= 1024) return `${Math.round(tokens / 1024)}K`;
  return String(tokens);
}

export function formatPct(ratio: number): string {
  if (!Number.isFinite(ratio)) return '—';
  return `${Math.round(ratio * 100)} %`;
}

/** "27.8B" / "6B" for parameter counts given raw. */
export function formatParams(params: number | null): string {
  if (params === null || !Number.isFinite(params)) return '—';
  const b = params / 1e9;
  if (b >= 1000) return `${(b / 1000).toFixed(1)}T`;
  const text = b >= 100 ? String(Math.round(b)) : b.toFixed(b >= 10 ? 0 : 1).replace(/\.0$/, '');
  return `${text}B`;
}

export function formatDate(iso: string | null | undefined): string {
  if (!iso) return '—';
  return iso.slice(0, 10);
}
