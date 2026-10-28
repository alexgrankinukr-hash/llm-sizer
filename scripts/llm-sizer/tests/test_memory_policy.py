"""The memory policy margins (METHODOLOGY §5): the mirror of constants.ts, and the MLX prompt scratch by chip generation."""
import json
import re
from pathlib import Path

import pytest

from calibrate import MEMORY_POLICY
from common import PUBLIC_DATA_DIR
from fit_speed import memory_residuals

CONSTANTS_TS = Path(__file__).resolve().parents[3] / "src" / "lib" / "llm-sizer" / "engine" / "constants.ts"


def test_the_policy_is_the_engines():
    block = re.search(r"export const MEMORY_POLICY: MemoryPolicy = \{(.*?)\};", CONSTANTS_TS.read_text(), re.S)
    assert block, "MEMORY_POLICY not found in constants.ts"
    ts = {m.group(1): (None if m.group(2) == "null" else float(m.group(2))) for m in re.finditer(r"(\w+):\s*(null|[\d.]+)", block.group(1))}
    assert ts == {k: (None if v is None else float(v)) for k, v in MEMORY_POLICY.items()}


def test_the_m5_rows_grow_by_the_cache_alone():
    validation = json.loads((PUBLIC_DATA_DIR / "validation.json").read_text())
    rows = [r for r in memory_residuals(validation, MEMORY_POLICY) if r["id"].startswith("omlx-q6xqv10d-")]
    assert [r["context"] for r in rows] == [4096, 32768, 65536, 131072, 200000]
    # oMLX 0.7.1rc1 on an M5 Max: within 0.6 GB of the estimate from 4K to 200K, with no prompt scratch charged
    for r in rows:
        assert abs(r["delta_gb"]) <= 0.6, r


def test_the_scratch_follows_the_chip():
    row = {"id": "x", "kind": "memory", "split": "holdout", "runtime": "mlx", "context_tokens": 131072, "kv_bits": 16, "weights_gb": 16.05,
           "peak_gb": 26.6, "active_params_b": 27.8, "bits": 4.62,
           "arch": {"attention": "gqa", "experts_total": 0, "linear_layers": 48, "kv_bytes_per_token_8bit": 32768}}
    (m5,) = memory_residuals({"rows": [{**row, "chip": "M5 Max"}]}, MEMORY_POLICY)
    (m4,) = memory_residuals({"rows": [{**row, "chip": "M4 Max"}]}, MEMORY_POLICY)
    (unknown,) = memory_residuals({"rows": [row]}, MEMORY_POLICY)
    assert m4["predicted_gb"] - m5["predicted_gb"] == pytest.approx(18.0, abs=0.1)
    assert unknown["predicted_gb"] == m4["predicted_gb"]
