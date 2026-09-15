/**
 * Data adapter for this copy: the model records come from a snapshot of the open-data repository bundled under
 * public/data/llm-sizer/models (refresh it with `npm run data:snapshot`). The upstream site reads the same records
 * from Postgres, filled nightly from Hugging Face; the shape returned here is identical, so the routes, the pages
 * and the island do not know the difference. A fork can point this at any store that returns `{ body, sha,
 * generatedAt }` per export name ('index' or 'model:<id>').
 */
export interface LiveExport {
  body: string;        // JSON text, served as-is
  sha: string;         // short content hash, used as the ETag
  generatedAt: string; // ISO timestamp
}

const files = import.meta.glob('/public/data/llm-sizer/models/*.json', { query: '?raw', import: 'default', eager: true }) as Record<string, string>;

function fnv(text: string): string {
  // a small stable hash for records that carry no sha of their own
  let h = 0x811c9dc5;
  for (let i = 0; i < text.length; i++) h = Math.imul(h ^ text.charCodeAt(i), 0x01000193) >>> 0;
  return h.toString(16).padStart(8, '0');
}

let cache: Map<string, LiveExport> | null = null;

function load(): Map<string, LiveExport> {
  if (cache) return cache;
  cache = new Map();
  for (const [path, body] of Object.entries(files)) {
    const file = path.slice(path.lastIndexOf('/') + 1).replace(/\.json$/, '');
    const name = file === 'index' ? 'index' : `model:${file}`;
    let sha = fnv(body);
    let generatedAt = new Date(0).toISOString();
    try {
      const parsed = JSON.parse(body) as { sha?: string; generated_at?: string };
      if (typeof parsed.sha === 'string' && parsed.sha) sha = parsed.sha;
      if (typeof parsed.generated_at === 'string') generatedAt = new Date(parsed.generated_at).toISOString();
    } catch {
      // a malformed file is served as-is with its content hash; the client reports the parse error
    }
    cache.set(name, { body, sha, generatedAt });
  }
  return cache;
}

/** The record for `name` ('index' or 'model:<id>'), or null when the snapshot has none. */
export async function getLiveExport(name: string): Promise<LiveExport | null> {
  return load().get(name) ?? null;
}

/** Several records at once (the share image needs every visible column's record). */
export async function getLiveExports(names: string[]): Promise<Map<string, LiveExport>> {
  const all = load();
  const out = new Map<string, LiveExport>();
  for (const n of names) {
    const row = all.get(n);
    if (row) out.set(n, row);
  }
  return out;
}
