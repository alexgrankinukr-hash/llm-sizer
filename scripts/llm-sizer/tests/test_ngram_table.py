"""The n-gram table on the SSD (METHODOLOGY §5): the mirror of memory.ts, and the measured peaks it is checked against."""
import json
import re
from pathlib import Path

import pytest

from calibrate import MEMORY_POLICY
from common import PUBLIC_DATA_DIR
from fit_speed import memory_residuals
from speed_model import NGRAM_TABLES, ngram_table_share, table_on_ssd_gb

CONSTANTS_TS = Path(__file__).resolve().parents[3] / "src" / "lib" / "llm-sizer" / "engine" / "constants.ts"
FLASH = {"attention": "gqa", "experts_total": 512, "linear_layers": 36, "kv_bytes_per_token_8bit": 12288, "model_type": "qwen4_exp_text"}


def test_the_table_list_is_the_engines():
    block = re.search(r"export const NGRAM_TABLES[^=]*=\s*\[(.*?)\];", CONSTANTS_TS.read_text(), re.S)
    assert block, "NGRAM_TABLES not found in constants.ts"
    ts = [(m.group(1), float(m.group(2))) for m in re.finditer(r"modelTypePrefix:\s*'([^']+)',\s*share:\s*([\d.]+)", block.group(1))]
    assert ts == list(NGRAM_TABLES)


def test_an_mlx_file_of_flash_next_leaves_28_percent_on_the_ssd():
    assert ngram_table_share(FLASH) == 0.283
    assert table_on_ssd_gb(111.52, "mlx", FLASH) == pytest.approx(31.56, abs=0.01)
    # GGUF files keep the table in their shards; no MLX engine loads it elsewhere; other models have no table
    assert table_on_ssd_gb(119.15, "gguf", FLASH) == 0.0
    assert table_on_ssd_gb(111.52, "mlx", FLASH, loader="gguf") == 0.0
    assert table_on_ssd_gb(17.3, "mlx", {"model_type": "qwen3_5_text"}) == 0.0
    assert ngram_table_share(None) == 0.0


def test_the_measured_flash_next_peaks_sit_under_the_estimate():
    validation = json.loads((PUBLIC_DATA_DIR / "validation.json").read_text())
    rows = {r["id"]: r for r in memory_residuals(validation, MEMORY_POLICY) if "flashnext" in r["id"]}
    assert len(rows) == 4
    for r in rows.values():
        assert r["ssd_gb"] > 30
        # the table rule never undercounts; at 4K it lands within 4 GB of oMLX's allocator peak
        assert r["delta_gb"] > 0
    assert rows["omlx-0ggvto3x-m5max-flashnext-mem-149"]["predicted_gb"] == pytest.approx(82.4, abs=0.1)
    assert rows["omlx-0ggvto3x-m5max-flashnext-mem-149"]["delta_gb"] < 2
    assert rows["omlx-irhexf6y-m5max-flashnext-mem-150"]["delta_gb"] < 4


def test_a_run_with_the_table_in_memory_is_counted_whole():
    row = {"id": "x", "kind": "memory", "split": "holdout", "arch": FLASH, "runtime": "mlx", "context_tokens": 1024, "kv_bits": 16,
           "weights_gb": 106.3, "peak_gb": 106.3, "table_in_memory": True}
    (r,) = memory_residuals({"rows": [row]}, MEMORY_POLICY)
    assert r["ssd_gb"] == 0
    assert r["predicted_gb"] == pytest.approx(106.3 + 1.5 + 1.063 + 0.025, abs=0.1)
