// Test-only loaders for the engine suite: model records copied from the nightly export plus the live catalogs.
import { readFileSync } from 'node:fs';
import type { Factors, Machine, MachinesFile, ModelDetail, Quant } from '../types';

function readJson<T>(url: URL): T {
  return JSON.parse(readFileSync(url, 'utf8')) as T;
}

export function loadModel(id: string): ModelDetail {
  return readJson<ModelDetail>(new URL(`./${id}.json`, import.meta.url));
}

export function loadMachines(): Machine[] {
  return readJson<MachinesFile>(new URL('../../../../../public/data/llm-sizer/machines.json', import.meta.url)).machines;
}

export function loadFactors(): Factors {
  return readJson<Factors>(new URL('../../../../../public/data/llm-sizer/factors.json', import.meta.url));
}

/** The factor file as it was before the fixed-cost model (Release A): keeps the multiplier suites meaningful after the flip. */
export function loadMultiplierFactors(): Factors {
  return readJson<Factors>(new URL('./factors-multiplier.json', import.meta.url));
}

export function machineById(id: string): Machine {
  const m = loadMachines().find((x) => x.id === id);
  if (!m) throw new Error(`no machine ${id}`);
  return m;
}

export function quantByLabel(model: ModelDetail, label: string, repo?: string): Quant {
  const q = model.quants.find((x) => x.label === label && (!repo || x.repo === repo));
  if (!q) throw new Error(`no quant ${label} in ${model.id}`);
  return q;
}
