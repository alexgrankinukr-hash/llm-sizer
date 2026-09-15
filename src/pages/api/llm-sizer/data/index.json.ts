import type { APIRoute } from 'astro';
import { serveExport } from '../../../../lib/llm-sizer/data-response';

/** The model search index: one lean row per model. Rebuilt nightly by the export job. */
export const GET: APIRoute = ({ request }) => serveExport(request, 'index', { missingStatus: 503 });
