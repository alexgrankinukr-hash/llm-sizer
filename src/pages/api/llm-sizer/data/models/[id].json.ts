import type { APIRoute } from 'astro';
import { serveExport } from '../../../../../lib/llm-sizer/data-response';

const ID = /^[a-z0-9][a-z0-9.-]{0,79}$/;

/** One model's full record (quants, architecture, drafts). `?v=<detail_sha>` makes the response immutable. */
export const GET: APIRoute = ({ request, params, url }) => {
  const id = params.id ?? '';
  if (!ID.test(id)) {
    return new Response(JSON.stringify({ error: 'not_found' }), {
      status: 404,
      headers: { 'Content-Type': 'application/json; charset=utf-8', 'Cache-Control': 'public, max-age=60, s-maxage=300' },
    });
  }
  return serveExport(request, `model:${id}`, { versionParam: url.searchParams.get('v') });
};
