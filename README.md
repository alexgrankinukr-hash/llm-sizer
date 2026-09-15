# LLM Sizer

Which open AI models fit your machine, and how fast will they run. The code behind the free tool at [theaibridges.com/tools/llm-sizer](https://theaibridges.com/tools/llm-sizer): the real memory maths, a speed model fitted on measured runs, a reading-speed estimate, linked machines, a buying map, and a nightly model catalog from Hugging Face. Every number the tool shows links to where it came from, and every formula and constant is written out in [METHODOLOGY.md](src/lib/llm-sizer/METHODOLOGY.md).

The data is its own repository, mirrored nightly from the live tool: [llm-sizer-data](https://github.com/alexgrankinukr-hash/llm-sizer-data) (machines, benchmark rows, factors, the model records; data CC BY 4.0, formulas MIT).

## Run it

```bash
npm install
npm run dev          # http://localhost:4321/tools/llm-sizer
```

That is the whole setup. No database, no accounts, no keys: the model records are a bundled snapshot of the open data (`public/data/llm-sizer/models`). To refresh the snapshot from the data repository:

```bash
npm run data:snapshot
```

For a production build, `npm run build && npm start` runs a standalone Node server. Swap the adapter in `astro.config.mjs` for your host and set `SITE_URL` so share links and link previews carry your origin.

## Hosting notes

- Set `SITE_URL` to your origin so share links and link previews carry it.
- `/api/llm-sizer/image.png` renders a PNG per request (satori). On a public host put it behind a CDN and rate-limit it; the upstream site does both. The data routes are cheap and cacheable (ETags, `s-maxage`).
- The copy has no accounts, no database and no analytics; nothing it serves is personal. Keep Astro and `@vercel/og` current: `npm audit` reports advisories against the framework versions this copy pins, not against the tool's own code.

## What is here

| Path | What it is |
|---|---|
| `src/lib/llm-sizer/engine/` | The engine, pure TypeScript: available memory (§4 of the methodology), needed memory (§5), the compromise search, writing speed (§6.1), reading speed (§6.4), linked machines, custom entries. No environment, no DOM, no Node; a script enforces it. |
| `src/lib/llm-sizer/app/` | The app layer, pure and tested: state and reducer, the share-link codec, machine groups, the buying map, the machine view, table layout, provenance links. |
| `src/components/llm-sizer/` | The React views: the fit table, the speed and reading tabs, the memory chart, the buying map, the machine view, the sheet behind every cell, Settings. |
| `src/pages/tools/llm-sizer/` | The two pages: the tool (decodes the share link on the server, preloads the visible records, the link preview) and About (renders the methodology with the sources tables). |
| `src/pages/api/llm-sizer/` | The data routes (ETags, conditional requests) and the share image (satori). |
| `src/lib/llm-sizer/adapters/` | The two files a host site replaces: `db.ts` (where the model records come from; here, the bundled snapshot) and `site.ts` (name, author line, footer, repositories, which site-only features exist). |
| `scripts/llm-sizer/` | The Python data jobs: `discover.py` (Hugging Face → Postgres), `export.py` (Postgres → JSON records, with guardrails), `calibrate.py` (measured rows → `factors.json`), `validate_machines.py`, `speed_model.py` (the Python mirror of the speed formulas), the tests and the parity fixtures. These need Postgres and are the maintainer's tools; contributors rarely need them. |
| `public/data/llm-sizer/` | The hand-curated catalogs (`machines.json`, `benchmarks.json`, `validation.json`), the derived `factors.json`, and the model snapshot. |
| `src/lib/llm-sizer/METHODOLOGY.md` | Every formula, constant, assumption and limitation, with worked examples and the changelog. The About page renders it. |
| `migrations/` | The `llm_sizer` Postgres schema for the data jobs. |

## Checks

```bash
npm test                     # 251 TypeScript tests: the engine, the app layer, parity with the Python mirror
npm run test:py              # 66 Python tests: the data jobs, calibration, the same parity fixtures
npm run boundary             # the import boundary: the engine stays pure, the tool stays self-contained
npm run check                # astro check
```

`calibrate.py --check` confirms `factors.json` is exactly what the benchmark rows produce; CI runs all of it on every push.

## How the numbers are made

Memory is an addition: the file you download, plus the context cache (from the model's attention design and the context length), plus working room, against what the machine can actually give the model (macOS lets the GPU use two thirds of memory under 36 GB and three quarters above, minus what it keeps for itself, minus your apps). Speed is a time per token: a read of the active weights and the cache at the chip's effective bandwidth, plus a fixed cost per chip and per architecture, plus a context term; the chip profiles are fitted on the community llama.cpp table and judged on runs the fit never saw. Reading speed (the wait before the first word) is compute-bound and scales with GPU cores and the chip generation. Linked machines pool memory and follow the two measured regimes, a layer split or tensor parallel. All of it, with the evidence and the limitations, is in [METHODOLOGY.md](src/lib/llm-sizer/METHODOLOGY.md).

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). In short: a wrong number, a new machine or a model fact is a pull request on the [data repository](https://github.com/alexgrankinukr-hash/llm-sizer-data); a formula or a feature is a pull request here, with a changelog entry in the methodology. This repository is mirrored from the tool's source, so accepted pull requests are applied upstream and mirrored back within a few days.

## License and attribution

Code: [MIT](LICENSE). The bundled benchmark rows, validation rows and cluster series come from the community measurements listed in [DATA-SOURCES.md](DATA-SOURCES.md), each row carrying its source. The fonts (Inter, Playfair Display, JetBrains Mono) are under the SIL Open Font License, `public/fonts/OFL.txt`.

Built by Alex Grankin at The AI Bridge. The live tool, its email updates and its feedback inbox stay on [theaibridges.com](https://theaibridges.com/tools/llm-sizer).
