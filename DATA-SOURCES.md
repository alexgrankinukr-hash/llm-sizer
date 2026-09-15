# Data sources and attribution

Every row in `public/data/llm-sizer/benchmarks.json` and `validation.json` carries its own `source` link; the About page renders them next to the numbers they produce. This file names the sources the tool leans on most, with thanks. Licenses are those the sources publish; rows are facts (a chip, a model, a measured tokens-per-second figure) quoted with their origin.

## Machine catalog

`machines.json`: Apple's and NVIDIA's own specification and store pages, listed per row. Prices are US list prices at the date in the file.

## Model records

The nightly import reads the Hugging Face API for the five quantizer groups whose files people download (Unsloth, Bartowski, LM Studio community, the MLX community, llama.cpp's own), resolves each to the maker's base repository, and reads its `config.json` for the architecture. Each record carries the model's own license field. `models.overrides.json` in the data repository holds the few hand facts the API cannot express, each with its source.

## Benchmark rows (`benchmarks.json`)

- The community llama.cpp Apple-silicon benchmark table, [ggml-org/llama.cpp discussion #4167](https://github.com/ggml-org/llama.cpp/discussions/4167): the F16, Q8_0 and Q4_0 rows per chip that fit every chip profile, and the prefill rows.
- [mac-llm-bench](https://github.com/enescingoz/mac-llm-bench) (MIT): the MLX and llama.cpp prefill rows across chips.
- [llama.cpp's DGX Spark bench](https://github.com/ggml-org/llama.cpp/blob/master/benches/dgx-spark/dgx-spark.md): the Spark's profile.
- [mlx-dspark](https://github.com/ARahim3/mlx-dspark), [mlx-dflash](https://github.com/eauchs/mlx-dflash), [MTPLX](https://github.com/youssofal/MTPLX) and [a local MTP write-up](https://vinoth12940.github.io/blog/articles/genai-20260519-local-mtp-speculative-decoding/): the acceleration ranges.
- [heretik.io](https://heretik.io/qwen38-flash-next-262k-macbook/): the prefill depth sweep on an M5 Max.
- Linked-machine series: [exo](https://blog.exolabs.net/day-1/), [Jeff Geerling](https://www.jeffgeerling.com/blog/2025/15-tb-vram-on-mac-studio-rdma-over-thunderbolt-5/), the MLX discussions [#2990](https://github.com/ml-explore/mlx/discussions/2990), [#3209](https://github.com/ml-explore/mlx/discussions/3209) and [#3939](https://github.com/ml-explore/mlx/discussions/3939), [Openzeka](https://whitepapers.openzeka.com/papers/qwen3.6-27b-dgx-spark-scaling/), [llama.cpp discussion #16578](https://github.com/ggml-org/llama.cpp/discussions/16578), [classmethod](https://dev.classmethod.jp/en/articles/dgx-spark-two-node-clustering/), [NVIDIA's developer forums](https://forums.developer.nvidia.com/t/question-on-inference-performance-results-of-qwen3-235b-a22b-on-2x-dgx-spark/355053), [voipmonitor/rtx6kpro](https://github.com/voipmonitor/rtx6kpro/blob/master/benchmarks/results.md).
- [MacRumors](https://www.macrumors.com/2025/03/17/apples-m3-ultra-runs-deepseek-r1-efficiently/) and [LMSYS](https://lmsys.org/blog/2025-10-13-nvidia-dgx-spark/): single published figures, quoted as such.

## Validation rows (`validation.json`)

The 120 measured runs the speed model is fitted on and judged against, graded A to C by how reproducible the setup is, with a train/holdout split. Contributors, by row count: llama.cpp's DGX Spark bench, [Zach Rattner](https://zachrattner.com/projects/ai-mac-cluster/mlx-vs-ollama), [Ivan Fioravanti](https://x.com/ivanfioravanti/status/2023760975325700436), the MLX discussion [#3209](https://github.com/ml-explore/mlx/discussions/3209), [oMLX](https://omlx.ai/benchmarks/performance/32bf9vtv), [llama.cpp discussion #15396](https://github.com/ggml-org/llama.cpp/discussions/15396), [ai-kizai](https://ai-kizai.com/benchmarks/512gb), [mlx-lm issue #763](https://github.com/ml-explore/mlx-lm/issues/763), [hiesch.eu](https://hiesch.eu/blog/llamacpp-benchmarks-speculative-decoding/), [rentamac.io](https://rentamac.io/run-deepseek-v4-mac/), [Tonoken3](https://github.com/Tonoken3/DeepSeek-V4-Flash-MacStudio), [Ante Kapetanović](https://antekapetanovic.com/blog/qwen3.5-apple-silicon-benchmark/), [Raullen Chai](https://dev.to/raullen_chai_76e18e9705b0/gemma-4-on-apple-silicon-85-toks-with-a-pip-install-299a), [llmcheck.net](https://llmcheck.net/blog/apple-silicon-m5-max-local-ai-guide/), [Ollama's MLX post](https://ollama.com/blog/mlx).

## Model capability claims

The buying guide's descriptions of what each model is good at come from the makers' published evaluations and the [Artificial Analysis Intelligence Index](https://artificialanalysis.ai/), named with its version and date wherever a number is quoted. The tool itself makes no capability claims: it sizes, it does not grade.

## Fonts

Inter, Playfair Display and JetBrains Mono, under the SIL Open Font License (`public/fonts/OFL.txt`).
