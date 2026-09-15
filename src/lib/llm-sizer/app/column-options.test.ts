import { describe, expect, it } from 'vitest';
import { loadModel } from '../engine/__fixtures__/load';
import { defaultState } from './defaults';
import { ctxOptions, parseQuantValue, quantOptions, quantValue } from './column-options';
import type { MatrixColumn } from './matrix';
import type { AppState } from './state';
import { resolveQuant } from './quants';

const model = loadModel('qwen3.8-27b');
const col = (quant: MatrixColumn['column']['quant']): MatrixColumn => ({ index: 0, column: { id: model.id, quant, ctx: 32768 }, status: 'ready', model, quant: resolveQuant(model, quant, 'gguf') });

describe('column options', () => {
  it('offers buckets in simple mode and files otherwise', () => {
    const simple = quantOptions(col('auto'), defaultState());
    expect(simple.map((o) => o.value)).toContain('bucket:Q4');
    expect(simple.find((o) => o.value === 'bucket:Q4')?.label).toBe('Q4 (default)');
    const exact = quantOptions(col('auto'), { ...defaultState(), simpleQuants: false });
    expect(exact[0].value).toBe('auto');
    expect(exact.some((o) => o.value.startsWith('label:'))).toBe(true);
    expect(quantOptions({ ...col('auto'), model: null, quant: null }, defaultState())).toEqual([{ value: 'auto', label: 'Q4' }]);
  });
  it('maps settings to option values and back', () => {
    expect(quantValue(col('auto'), defaultState())).toBe('bucket:Q4');
    expect(quantValue(col({ bucket: 'Q8' }), defaultState())).toBe('bucket:Q8');
    expect(quantValue(col({ label: 'Q8_0' }), { ...defaultState(), simpleQuants: false })).toBe('label:Q8_0');
    expect(parseQuantValue('bucket:Q4')).toBe('auto');
    expect(parseQuantValue('bucket:Q8')).toEqual({ bucket: 'Q8' });
    expect(parseQuantValue('label:Q8_0')).toEqual({ label: 'Q8_0' });
  });
  it('context options keep an off-chip value', () => {
    expect(ctxOptions(model, 32768)).toContain(32768);
    expect(ctxOptions(model, 20000)).toContain(20000);
    expect(ctxOptions(null, 32768)).toEqual([8192, 32768, 131072, 262144]);
  });
});

describe('special builds in the picker', () => {
  it('lists the engram-on-SSD builds after the buckets and after the files, and round-trips the setting', () => {
    const fn = loadModel('qwen3.8-flash-next');
    const c = (quant: MatrixColumn['column']['quant'] = 'auto', simpleQuants = true): [MatrixColumn, AppState] => {
      const state = { ...defaultState(), simpleQuants };
      return [{ index: 0, column: { id: fn.id, quant, ctx: 32768 }, status: 'ready', model: fn, quant: resolveQuant(fn, quant, state.runtime) }, state];
    };
    const simple = quantOptions(...c());
    expect(simple.slice(-3).map((o) => o.value)).toEqual(['special:0', 'special:1', 'special:2']);
    expect(simple.at(-3)?.label).toBe('Q4 · engram on SSD · 45.8 GB in memory');
    expect(simple.at(-1)?.label).toBe('Q5 · engram on SSD · 56.1 GB in memory');
    const exact = quantOptions(...c('auto', false));
    expect(exact.at(-3)?.label).toBe('IQ4_XS · engram on SSD · 45.8 GB in memory + 39.1 GB on SSD');
    expect(quantValue(...c({ special: 1 }))).toBe('special:1');
    expect(quantValue(...c({ special: 1 }, false))).toBe('special:1');
    expect(parseQuantValue('special:2')).toEqual({ special: 2 });
    expect(parseQuantValue('special:x')).toEqual({ special: 0 });
    expect(parseQuantValue('label:UD-Q4_K_XL')).toEqual({ label: 'UD-Q4_K_XL' });
    expect(quantOptions(...c()).some((o) => o.value.startsWith('special:'))).toBe(true);
    expect(quantOptions({ index: 0, column: { id: 'qwen3.8-27b', quant: 'auto', ctx: 32768 }, status: 'ready', model: loadModel('qwen3.8-27b'), quant: null }, defaultState()).some((o) => o.value.startsWith('special:'))).toBe(false);
  });
});
