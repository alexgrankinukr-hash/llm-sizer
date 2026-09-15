/**
 * GET /api/llm-sizer/image.png?s=<state>&size=og|hd|table&view=table|speed|model
 * The share image and the link-preview card for a table state, rendered with @vercel/og (satori).
 */
import type { APIRoute } from 'astro';
import { ImageResponse } from '@vercel/og';
import fs from 'node:fs';
import path from 'node:path';
import { createHash } from 'node:crypto';
import { getLiveExports } from '../../../lib/llm-sizer/adapters/db';
import { SITE } from '../../../lib/llm-sizer/adapters/site';
import { defaultState } from '../../../lib/llm-sizer/app/defaults';
import { tableLayout } from '../../../lib/llm-sizer/app/layout';
import { machineGroups } from '../../../lib/llm-sizer/app/machines';
import { evaluateMatrix, type RecordState } from '../../../lib/llm-sizer/app/matrix';
import { decodeState } from '../../../lib/llm-sizer/app/url';
import { formatDate } from '../../../lib/llm-sizer/engine/format';
import type { Factors, Machine, ModelDetail } from '../../../lib/llm-sizer/engine/types';
import { OG_MAX_COLUMNS, TableImage, buyingImageMap, imageDimensions, type ImageSize, type ImageView } from '../../../components/llm-sizer/image/TableImage';
import machinesFile from '../../../../public/data/llm-sizer/machines.json';
import factorsFile from '../../../../public/data/llm-sizer/factors.json';

const FONTS_DIR = path.join(process.cwd(), 'public', 'fonts');
const fonts = [
  { name: 'Playfair', data: fs.readFileSync(path.join(FONTS_DIR, 'PlayfairDisplay-Bold.woff')), weight: 700 as const, style: 'normal' as const },
  { name: 'Inter', data: fs.readFileSync(path.join(FONTS_DIR, 'Inter-Regular.woff')), weight: 400 as const, style: 'normal' as const },
  { name: 'Inter', data: fs.readFileSync(path.join(FONTS_DIR, 'Inter-Medium.woff')), weight: 500 as const, style: 'normal' as const },
];
const machines = machinesFile.machines as unknown as Machine[];
const factors = factorsFile as unknown as Factors;
const groups = machineGroups(machines);
const knownGroups = new Set(groups.map((g) => g.id));

function bad(status: number, message: string): Response {
  return new Response(message, { status, headers: { 'Content-Type': 'text/plain; charset=utf-8', 'Cache-Control': 'no-store' } });
}

export const GET: APIRoute = async ({ request }) => {
  const url = new URL(request.url);
  const s = url.searchParams.get('s') ?? '';
  if (s.length > 2500) return bad(400, 'State too long.');
  const sizeParam = url.searchParams.get('size') ?? 'og';
  const viewParam = url.searchParams.get('view') ?? 'table';
  if (!['og', 'hd', 'table'].includes(sizeParam)) return bad(400, 'size must be og, hd or table.');
  if (!['table', 'speed', 'model'].includes(viewParam)) return bad(400, 'view must be table, speed or model.');
  const size = sizeParam as ImageSize;
  const view = viewParam as ImageView;

  const decoded = decodeState(s, defaultState(), { knownGroups });
  if (s && !decoded.ok) return bad(400, 'Could not read the table state.');
  let state = decoded.state;
  if (size === 'og' && state.columns.length > OG_MAX_COLUMNS) state = { ...state, columns: state.columns.slice(0, OG_MAX_COLUMNS) };

  const ids = state.columns.map((c) => c.id).filter((id) => !id.startsWith('custom-'));
  let exports: Map<string, { body: string; sha: string }>;
  try {
    exports = await getLiveExports(['index', ...ids.map((id) => `model:${id}`)]);
  } catch (err) {
    console.error('[llm-sizer] image: data unavailable', err);
    // link previews must never be blank: the site's default card stands in for the og size
    if (size === 'og') return new Response(null, { status: 302, headers: { Location: SITE.ogFallback, 'Cache-Control': 'no-store' } });
    return bad(503, 'Model data is unavailable right now.');
  }
  const index = exports.get('index');
  const records: Record<string, RecordState> = {};
  for (const id of ids) {
    const row = exports.get(`model:${id}`);
    records[id] = row ? { status: 'ready', model: JSON.parse(row.body) as ModelDetail } : { status: 'missing' };
  }
  let modelsDate: string | null = null;
  try {
    modelsDate = index ? formatDate((JSON.parse(index.body) as { generated_at?: string }).generated_at ?? null) : null;
  } catch {
    modelsDate = null;
  }

  const matrix = evaluateMatrix(state, groups, records, factors);
  const layout = tableLayout(state, matrix);
  // the model view's image is the buying list, computed by the island's own helper
  const buying = view === 'model' ? buyingImageMap(state, matrix, groups, factors, machinesFile.data_as_of ?? null) : null;
  const { width, height } = imageDimensions(size, view, layout, matrix, buying);
  const etag = `"${createHash('sha1').update(`${s}|${size}|${view}|${index?.sha ?? 'none'}`).digest('hex').slice(0, 16)}"`;
  if (request.headers.get('if-none-match') === etag) return new Response(null, { status: 304, headers: { ETag: etag } });

  return new ImageResponse(TableImage({ state, matrix, layout, factors, size, view, modelsDate, buying }), {
    width,
    height,
    fonts,
    headers: {
      'Cache-Control': 'public, max-age=300, s-maxage=86400, stale-while-revalidate=604800',
      ETag: etag,
    },
  });
};
