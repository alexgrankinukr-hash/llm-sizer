/** Turns a live export row into an HTTP response with CDN caching and conditional-request support. */
import { getLiveExport } from './adapters/db';

const CACHE_FRESH = 'public, max-age=300, s-maxage=3600, stale-while-revalidate=86400, stale-if-error=86400';
const CACHE_IMMUTABLE = 'public, max-age=31536000, immutable';
const CACHE_MISS = 'public, max-age=60, s-maxage=300';

function baseHeaders(cacheControl: string): Record<string, string> {
  return {
    'Content-Type': 'application/json; charset=utf-8',
    'Cache-Control': cacheControl,
    'Access-Control-Allow-Origin': '*',
    'X-Content-Type-Options': 'nosniff',
  };
}

function etagMatches(header: string | null, etag: string): boolean {
  if (!header) return false;
  return header.split(',').some((v) => v.trim().replace(/^W\//, '') === etag);
}

/**
 * Serve the export named `name`. When `versionParam` equals the live sha (the client asked for
 * exactly this version by `?v=`), the response is marked immutable so the CDN never revalidates it.
 */
export async function serveExport(request: Request, name: string, opts: { versionParam?: string | null; missingStatus?: 404 | 503 } = {}): Promise<Response> {
  let row;
  try {
    row = await getLiveExport(name);
  } catch (err) {
    console.error('[llm-sizer] export lookup failed', err);
    return new Response(JSON.stringify({ error: 'unavailable' }), { status: 503, headers: baseHeaders('no-store') });
  }
  if (!row) {
    const status = opts.missingStatus ?? 404;
    return new Response(JSON.stringify({ error: status === 404 ? 'not_found' : 'not_exported_yet' }), {
      status,
      headers: baseHeaders(status === 404 ? CACHE_MISS : 'no-store'),
    });
  }
  const etag = `"${row.sha}"`;
  const cache = opts.versionParam && opts.versionParam === row.sha ? CACHE_IMMUTABLE : CACHE_FRESH;
  const headers = { ...baseHeaders(cache), ETag: etag, 'Last-Modified': new Date(row.generatedAt).toUTCString() };
  if (etagMatches(request.headers.get('if-none-match'), etag)) {
    return new Response(null, { status: 304, headers });
  }
  return new Response(row.body, { status: 200, headers });
}
