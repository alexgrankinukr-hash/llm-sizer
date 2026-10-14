#!/usr/bin/env python3
"""Recompute the expected values of tests/fixtures/prefill-cases.json from factors.json with speed_model.py.

The inputs of each case (chip, cores, build, runtime, model) are kept; only the expected_* fields are rewritten. Run after
calibrate.py changes a prefill term: python3 scripts/llm-sizer/tests/emit_prefill_cases.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common import HERE, PUBLIC_DATA_DIR  # noqa: E402
from speed_model import first_word_wait_s, prefill_constant, prefill_d0, prefill_rate_at, prefill_tok_s  # noqa: E402

PATH = HERE / "tests" / "fixtures" / "prefill-cases.json"


def main() -> int:
    pf = json.loads((PUBLIC_DATA_DIR / "factors.json").read_text())["prefill"]
    doc = json.loads(PATH.read_text())
    for c in doc["cases"]:
        k, source, _ = prefill_constant(pf, c["chip"], c["gpu_cores"], c["platform"])
        tok = prefill_tok_s({**pf, "_mlx_group": c["mlx_group"]}, k, c["quant_label"], c["format"], c["arch_class"], c["runtime"], c["active_params_b"])
        d0 = prefill_d0(pf, c["kv_layers"], c["layers"], c["runtime"], c.get("model_id"))
        c.update({"expected_k": round(k, 6), "expected_source": source, "expected_tok_s": round(tok, 3), "expected_d0": round(d0, 3),
                  "expected_wait_8k_s": round(first_word_wait_s(tok, d0, 8192), 4), "expected_wait_ctx_s": round(first_word_wait_s(tok, d0, c["context_tokens"]), 4),
                  "expected_rate_at_ctx": round(prefill_rate_at(tok, d0, c["context_tokens"]), 4)})
    PATH.write_text(json.dumps(doc, indent=1, ensure_ascii=False) + "\n")
    print(f"wrote {len(doc['cases'])} cases to {PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
