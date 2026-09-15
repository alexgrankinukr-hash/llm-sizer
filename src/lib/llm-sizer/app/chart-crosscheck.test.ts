import { describe, expect, it } from 'vitest';
import { loadFactors, loadMachines, loadModel } from '../engine/__fixtures__/load';
import { AIRED, CHART_CORRECTIONS, CHART_MODELS, CHART_ROWS, airedToMarker, chartState, type AiredMarker, type ChartId } from './chart-corrections';
import { tableLayout } from './layout';
import { machineGroups, rowKey } from './machines';
import { evaluateMatrix, type RecordState } from './matrix';

const factors = loadFactors();
const groups = machineGroups(loadMachines());
const records: Record<string, RecordState> = Object.fromEntries(CHART_MODELS.map((id) => [id, { status: 'ready', model: loadModel(id) }]));

describe('the aired fit matrices, reproduced through the table path', () => {
  for (const chart of ['v2-128k', 'realistic'] as ChartId[]) {
    it(`${chart}: every cell matches the chart or a documented correction`, () => {
      const state = chartState(chart);
      const matrix = evaluateMatrix(state, groups, records, factors);
      const layout = tableLayout(state, matrix);
      const problems: string[] = [];
      CHART_MODELS.forEach((model, col) => {
        CHART_ROWS.forEach((r, i) => {
          const key = rowKey(r.groupId, r.gb);
          const row = layout.rows.find((x) => x.key === key);
          const matrixRow = matrix.rows.find((x) => x.key === key);
          if (!row || !matrixRow) {
            problems.push(`${chart} ${model}@${r.gb}: row ${key} missing from the table`);
            return;
          }
          if (matrixRow.row.machineId !== r.machineId) problems.push(`${chart} @${r.gb}: mapped to ${matrixRow.row.machineId}, expected ${r.machineId}`);
          const tool = row.cells[col].marker;
          const aired = AIRED[chart][model][i] as AiredMarker;
          const fix = CHART_CORRECTIONS.find((c) => c.chart === chart && c.model === model && c.gb === r.gb);
          if (fix && fix.aired !== aired) problems.push(`${chart} ${model}@${r.gb}: correction says aired ${fix.aired}, the chart says ${aired}`);
          if (fix && fix.tool === airedToMarker(aired)) problems.push(`${chart} ${model}@${r.gb}: stale correction (the tool agrees with the chart)`);
          const expected = fix ? fix.tool : airedToMarker(aired);
          if (tool !== expected) problems.push(`${chart} ${model}@${r.gb}: tool ${tool}, ${fix ? 'correction' : 'aired'} ${expected}`);
        });
      });
      expect(problems).toEqual([]);
    });
  }
});
