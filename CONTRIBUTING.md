# Contributing

Thank you. Most contributions are one of four things; each has a home.

## A number is wrong for your machine

Open the tool, reproduce the cell, press Share, and open an issue here with the link ([template](.github/ISSUE_TEMPLATE/wrong-number.md)): what the tool says, what you measured or expected, the runtime and the file, and a source if you cite one. If the cause is a row in the data (a machine's bandwidth or price, a benchmark that should be there), the fix is a pull request on the data repository: https://github.com/alexgrankinukr-hash/llm-sizer-data. If the cause is a formula, it is a pull request here.

## A machine the catalog should know

`machines.json` in the data repository, one row per family and chip bin where the bandwidth differs, with the memory options, the bandwidth, the GPU cores, the prices and the maker's pages as sources. `scripts/llm-sizer/validate_machines.py` checks the shape. Until it is merged, Advanced in the tool lets you type the machine in and share the link.

## A speed you measured

A row in `benchmarks.json` in the data repository: `chip`, `bandwidth_gbs`, `memory_gb`, `model`, `quant`, `weights_gb`, `runtime`, `accel`, `tg` (writing tokens per second), `pp` (reading, optional), `source` (a public link), `date`. Accelerated rows also carry `baseline_tg`. Do not edit `factors.json`: `calibrate.py` regenerates it from the rows, and CI checks that it did.

## A formula, a constant, a feature

A pull request here. The rules that keep the tool honest:

1. **The methodology comes first.** Any change to a formula, a constant or a data source is written into `src/lib/llm-sizer/METHODOLOGY.md` before the code (the section it belongs to, the evidence, and a changelog entry that says what moved and what did not). The About page renders that file, so a change nobody can read is not done.
2. **Fitted, assumed, or policy.** Every number is one of the three and says so. A constant fitted on rows carries the count and the range; an assumed one is labelled; a margin is called a margin.
3. **Two mirrors stay in step.** The speed and reading formulas live in `engine/speed.ts`, `engine/prefill.ts`, `engine/cluster.ts` and again in `scripts/llm-sizer/speed_model.py`; the parity fixtures under `scripts/llm-sizer/tests/fixtures/` pin the two, and both test suites run in CI.
4. **The engine stays pure.** No environment, no Node, no DOM inside `src/lib/llm-sizer/engine/`; `scripts/llm-sizer/check-boundary.sh` fails the build otherwise. Site-specific things go through `src/lib/llm-sizer/adapters/`.
5. **No em-dashes in strings the viewer reads**, and no number in a label that the sheet cannot explain.

Run `npm test`, `npm run test:py`, `npm run boundary` and `npm run build` before opening the pull request.

## How this repository relates to the live tool

The tool's source of truth is the site it ships in; this repository is assembled from it and pushed on every change. A merged pull request here is applied upstream first and comes back with the next mirror, usually within a few days, with your handle in the changelog. Expect the mirror to overwrite direct commits to `main` here; work on branches and pull requests.

## Data submissions from the tool

The live tool's feedback form and the data repository's issues both land with the maintainer, who adds accepted rows with the submitter's handle as the source. The same path is open to you here through issues.
