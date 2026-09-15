/**
 * Loads the model index once and model records on demand (immutable when the index gives a sha).
 * A module-level cache survives re-renders and re-mounts; the hook exposes plain state objects.
 */
import { useEffect, useState } from 'react';
import type { ModelDetail, ModelIndex } from '../../lib/llm-sizer/engine/types';
import type { RecordState } from '../../lib/llm-sizer/app/matrix';

const INDEX_URL = '/api/llm-sizer/data/index.json';
const MODEL_URL = (id: string, sha?: string | null) => `/api/llm-sizer/data/models/${encodeURIComponent(id)}.json${sha ? `?v=${sha}` : ''}`;

export type IndexState = { status: 'loading' } | { status: 'ready'; index: ModelIndex } | { status: 'error' };

let indexPromise: Promise<IndexState> | null = null;
let indexState: IndexState = { status: 'loading' };
const records = new Map<string, RecordState>();
const inflight = new Map<string, Promise<void>>();
const listeners = new Set<() => void>();

function notify() {
  for (const l of listeners) l();
}

function loadIndex(): Promise<IndexState> {
  if (!indexPromise) {
    indexPromise = fetch(INDEX_URL)
      .then(async (res) => {
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        const index = (await res.json()) as ModelIndex;
        indexState = { status: 'ready', index };
        return indexState;
      })
      .catch(() => {
        indexState = { status: 'error' };
        return indexState;
      })
      .finally(notify);
  }
  return indexPromise;
}

async function loadRecord(id: string): Promise<void> {
  if (records.has(id) || inflight.has(id)) return;
  const p = (async () => {
    records.set(id, { status: 'loading' });
    notify();
    const idx = await loadIndex();
    if (idx.status === 'error') {
      records.set(id, { status: 'error', message: 'catalog unavailable' });
      notify();
      return;
    }
    const entry = idx.status === 'ready' ? idx.index.models.find((m) => m.id === id) : undefined;
    if (idx.status === 'ready' && !entry) {
      records.set(id, { status: 'missing' });
      notify();
      return;
    }
    try {
      const res = await fetch(MODEL_URL(id, entry?.detail_sha));
      if (res.status === 404) records.set(id, { status: 'missing' });
      else if (!res.ok) records.set(id, { status: 'error', message: `HTTP ${res.status}` });
      else records.set(id, { status: 'ready', model: (await res.json()) as ModelDetail });
    } catch (err) {
      records.set(id, { status: 'error', message: err instanceof Error ? err.message : String(err) });
    }
    notify();
  })();
  inflight.set(id, p);
  p.finally(() => inflight.delete(id));
}

/** Start loading a record before it is needed (the add-model flow). */
export function prefetchRecord(id: string): void {
  void loadRecord(id);
}

// A queue for views that want the whole catalog: at most `limit` fetches in flight, in the order given.
const queue: string[] = [];
const queued = new Set<string>();
function pump(limit: number): void {
  while (inflight.size < limit && queue.length) {
    const id = queue.shift() as string;
    queued.delete(id);
    if (records.has(id) || inflight.has(id)) continue;
    void loadRecord(id);
    const p = inflight.get(id);
    if (p) p.finally(() => pump(limit));
  }
}

/** Load many records in order, at most `limit` at a time (the machine view: featured models first). */
export function queueRecords(ids: string[], limit = 6): void {
  for (const id of ids) {
    if (records.has(id) || inflight.has(id) || queued.has(id)) continue;
    queue.push(id);
    queued.add(id);
  }
  pump(limit);
}

/** One record, live (undefined until requested). */
export function useModelRecord(id: string | null): RecordState | undefined {
  const [state, setState] = useState<RecordState | undefined>(() => (id ? records.get(id) : undefined));
  useEffect(() => {
    if (!id) return;
    const update = () => setState(records.get(id));
    listeners.add(update);
    void loadRecord(id);
    update();
    return () => {
      listeners.delete(update);
    };
  }, [id]);
  return id ? state : undefined;
}

/** Retry a failed record. */
export function retryRecord(id: string): void {
  records.delete(id);
  void loadRecord(id);
}

/** Retry the index after a failure, then every record that failed with it. */
export function retryIndex(): void {
  if (indexState.status !== 'error') return;
  indexPromise = null;
  indexState = { status: 'loading' };
  const failed = [...records.entries()].filter(([, r]) => r.status === 'error' || r.status === 'missing').map(([id]) => id);
  for (const id of failed) records.delete(id);
  notify();
  void loadIndex().then(() => {
    for (const id of failed) void loadRecord(id);
  });
}

/** Seed the cache with records rendered on the server (the first paint is complete, no fetch needed). */
export function seedRecords(seed: Record<string, ModelDetail>): void {
  for (const [id, model] of Object.entries(seed)) if (!records.has(id)) records.set(id, { status: 'ready', model });
}

function snapshot(ids: string[]): Record<string, RecordState> {
  const out: Record<string, RecordState> = {};
  for (const id of ids) {
    const r = records.get(id);
    if (r) out[id] = r;
  }
  return out;
}

/** The model index (search, featured list, detail shas). */
export function useModelIndex(): IndexState {
  const [state, setState] = useState<IndexState>(indexState);
  useEffect(() => {
    const update = () => setState(indexState);
    listeners.add(update);
    void loadIndex();
    update();
    return () => {
      listeners.delete(update);
    };
  }, []);
  return state;
}

/** Records for many ids, started through the queue; empty on the first render so the server and the client agree. */
export function useQueuedRecords(ids: string[], limit = 6): Record<string, RecordState> {
  const key = ids.join('|');
  const [state, setState] = useState<Record<string, RecordState>>({});
  useEffect(() => {
    const list = key ? key.split('|') : [];
    const update = () => setState(snapshot(list));
    listeners.add(update);
    queueRecords(list, limit);
    update();
    return () => {
      listeners.delete(update);
    };
  }, [key, limit]);
  return state;
}

/** Records for the given catalog ids (custom ids are ignored: they are built locally). `seed` = records rendered on the server. */
export function useModelRecords(ids: string[], seed?: Record<string, ModelDetail>): Record<string, RecordState> {
  const wanted = [...new Set(ids.filter((id) => !id.startsWith('custom-')))];
  const key = wanted.join('|');
  const [state, setState] = useState<Record<string, RecordState>>(() => {
    // on the client the seed fills the shared cache once; on the server it only shapes this render
    if (seed && typeof window !== 'undefined') seedRecords(seed);
    const initial = snapshot(wanted);
    if (seed) for (const id of wanted) if (!initial[id] && seed[id]) initial[id] = { status: 'ready', model: seed[id] };
    return initial;
  });
  useEffect(() => {
    const list = key ? key.split('|') : [];
    const update = () => setState(snapshot(list));
    listeners.add(update);
    for (const id of list) void loadRecord(id);
    update();
    return () => {
      listeners.delete(update);
    };
  }, [key]);
  return state;
}
