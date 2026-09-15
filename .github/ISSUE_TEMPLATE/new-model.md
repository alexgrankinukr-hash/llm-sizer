---
name: Add or fix a model
about: A model the nightly import missed, or a fact the Hugging Face API cannot express
title: "Model: <name>"
labels: data
---

Models arrive nightly from Hugging Face through the five quantizer groups (Unsloth, Bartowski, LM Studio, the MLX community, llama.cpp's own). Two things need a human:

- **A model to feature or a base repo the import resolved wrongly.** Say the Hugging Face repo (`Org/Model`) and, if you know it, the quantizer repos that carry it.
- **A fact the API does not carry.** A published active-parameter count, a license threshold, a hand-maintained build (an SSD-streamed build, for instance). These go in `models.overrides.json` in the data repository: https://github.com/alexgrankinukr-hash/llm-sizer-data.

Include the source for every fact.
