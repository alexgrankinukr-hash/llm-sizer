# LLM Sizer: code map

LLM Sizer answers "which open AI models fit this machine, and how fast will they run?". This file is the map of the code; [METHODOLOGY.md](METHODOLOGY.md) beside it is the map of the numbers. The top-level [README](../../../README.md) says how to run and contribute.

## Where things are

| Directory | What it holds |
|---|---|
| `src/lib/llm-sizer/engine/` | The engine, pure TypeScript: `memory.ts` (available memory, needed memory, the verdict, the breakdown bar), `fix.ts` (the compromise search), `speed.ts` (writing speed: the fixed-cost model, chip profiles, runtime factors, accelerations, the feels-like bands), `prefill.ts` (reading speed, the wait before the first word), `cluster.ts` (linked machines), `custom.ts` (user-typed models and machines), `format.ts`, `index.ts` (`evaluateCell`, `evaluateModelAcrossMachines`, `defaultQuant`, `contextChips`), `types.ts` mirroring the exported JSON, `constants.ts` explained in the methodology. No environment, no Node, no DOM. |
| `src/lib/llm-sizer/app/` | The app layer, pure and tested: `state.ts` (state + reducer, including machine and column reordering and linked pools), `url.ts` (the `?s=` share-link codec), `machines.ts` (catalog rows → machine groups, row keys, labels), `quants.ts` (Q2…Q8 buckets; a column's setting is `auto`, a bucket, an exact file label, or a special build by index), `matrix.ts` (`evaluateMatrix`), `machine-fit.ts` (the machine view: the summary rule, grids, the picker), `buying.ts` (the buying map: the machine set, the build options, the qualification rule, the price-first comparator, the recommendation, the label lanes), `needs.ts`, `layout.ts` (markers, labels, footnotes shared by the table and the share image), `provenance.ts` (source links for a cell, the About-page anchors), `column-options.ts`, `chart-corrections.ts` (the charts that aired in the launch video, as data the tool is held to), `defaults.ts`, `analytics.ts` (event names over `src/lib/track.ts`, a no-op here). |
| `src/lib/llm-sizer/adapters/` | The two files a host site owns: `db.ts` (where the model records come from; here a bundled snapshot of the open data, upstream a database filled nightly) and `site.ts` (name, author line, footer, repositories, the share-image fallback, and which site-only features exist: the email card and the feedback mode). |
| `src/components/llm-sizer/` | The React views: `LlmSizer.tsx` (root: reducer, records, URL sync, the chart switch, the layout grid), `ViewTiles`, `FitTable`, `SpeedView` (Writing and Reading), `MemoryView` + `MemoryBar`, `MachineMapView` (the buying map), `MachineFitView` (the machine view), `CellSheet` (the explanation behind every cell), `SettingsPanel` + `MachinesSection` + `ModelsSection` + `AdvancedPanel`, `Lists`, `RowSelect`, `ShareExport`, `FeedbackDialog` + `Forms` (feedback opens a prefilled issue here; the notify form is only rendered when a site enables it), `NotifyPopup`, `image/TableImage.tsx` (the share image, rendered by satori), `llm-sizer.css`. |
| `src/pages/tools/llm-sizer/` | `index.astro` (decodes `?s=` on the server, passes slimmed machines + factors and the initial columns' records as props, presentation mode, the link preview) and `about.astro` (renders `METHODOLOGY.md` with a table of contents, the FAQ, and the sources behind every machine, benchmark row and factor). |
| `src/pages/api/llm-sizer/` | `data/` (the model index and detail records, with ETags and conditional requests) and `image.png.ts` (the share image and link preview). |
| `scripts/llm-sizer/` | The Python data jobs: `discover.py` (Hugging Face → Postgres), `export.py` (Postgres → JSON, with guardrails), `calibrate.py` (benchmark rows → `factors.json`), `fit_speed.py`, `speed_model.py` (the Python mirror of the speed, reading and linked-machine formulas), `arch.py`, `naming.py`, `hf.py`, `validate_machines.py`, `migrate.py`, `models.overrides.json` (hand facts), `snapshot-data.sh` (refresh the bundled records), `check-boundary.sh`, `tests/` with the parity fixtures. |
| `public/data/llm-sizer/` | `machines.json`, `benchmarks.json`, `validation.json` (hand-curated), `factors.json` (derived, never edited by hand), `models/` (the snapshot). |
| `migrations/` | The `llm_sizer` Postgres schema the data jobs use. |

## Rules

1. Code in the tool's directories imports only from those directories plus a short allowlist (the layout, the header, `src/lib/track.ts`, React, satori in the image route, `astro` types). Nothing outside imports from the tool. `scripts/llm-sizer/check-boundary.sh` enforces it, and the engine's purity, in CI.
2. Every number is fitted, assumed or policy, and says which. A formula or constant changes in `METHODOLOGY.md` first, with a changelog entry; then in the code; then in `speed_model.py`, and the parity fixtures under `scripts/llm-sizer/tests/fixtures/` pin the two.
3. Anything a host site needs to say about itself goes through `adapters/site.ts`, never inline in a page or a view.
4. Local job output goes to `scripts/llm-sizer/.cache/` or `scripts/llm-sizer/out/`, both gitignored.

## How the data flows

Upstream, a nightly job runs `migrate.py` → `discover.py` (five quantizer groups on Hugging Face plus every featured model, resolved to the maker's base repo; `config.json` read for the architecture; exact file sizes per quant) → `export.py` (the search index and one record per model, staged, guardrail-checked, promoted) → a push to the data repository. This copy reads the snapshot in `public/data/llm-sizer/models/` through `adapters/db.ts`; `npm run data:snapshot` refreshes it. The routes under `/api/llm-sizer/data/` serve the records with ETags, and `useModelRecords.ts` fetches them on the client.

To run the jobs yourself: `pip install -r scripts/llm-sizer/requirements-dev.txt`, a Postgres in `DATABASE_URL` (environment or a `.env` at the root), optionally `HF_TOKEN` for higher Hugging Face rate limits and gated configs; then `python3 scripts/llm-sizer/migrate.py`, `discover.py --mode only --only Qwen3.8-27B`, `export.py --out scripts/llm-sizer/out`.

## Checks before a pull request

`npm test` · `npm run test:py` · `npm run boundary` · `npm run check` · `npm run build` · `python3 scripts/llm-sizer/calibrate.py --check` (the factors file is exactly what the rows produce).
