"""Sparse attention on MLX (METHODOLOGY §6.1): the mirror of speed.ts, and the long-context rows it is checked against."""
import json
import re
from pathlib import Path

from common import PUBLIC_DATA_DIR
from fit_speed import assumed_profiles, fit_floor_profiles, fit_speed_model, spec_bandwidth, tier_profiles
from speed_model import SPARSE_ATTENTION, attended_tokens, sparse_budget

CONSTANTS_TS = Path(__file__).resolve().parents[3] / "src" / "lib" / "llm-sizer" / "engine" / "constants.ts"
FLASH = {"attention": "gqa", "experts_total": 512, "linear_layers": 36, "kv_bytes_per_token_8bit": 12288, "model_type": "qwen4_exp_text"}
# Qwen3-Coder-Next: the same 12-of-48 attention layout and cache bytes, without an index
CODER = {"attention": "gqa", "experts_total": 512, "linear_layers": 36, "kv_bytes_per_token_8bit": 12288}


def test_the_sparse_list_is_the_engines():
    block = re.search(r"export const SPARSE_ATTENTION[^=]*=\s*\[(.*?)\];", CONSTANTS_TS.read_text(), re.S)
    assert block, "SPARSE_ATTENTION not found in constants.ts"
    ts = [(m.group(1), int(m.group(2))) for m in re.finditer(r"modelTypePrefix:\s*'([^']+)',\s*budgetTokens:\s*(\d+)", block.group(1))]
    assert ts == list(SPARSE_ATTENTION)


def test_flash_next_reads_its_budget_on_mlx_only():
    assert sparse_budget(FLASH) == 2048
    assert sparse_budget(CODER) is None
    assert sparse_budget(None) is None
    assert attended_tokens(131072, "mlx", 2048) == 2048
    assert attended_tokens(1024, "mlx", 2048) == 1024
    assert attended_tokens(131072, "gguf", 2048) == 131072
    assert attended_tokens(131072, "mlx", None) == 131072


def _fit():
    bench = json.loads((PUBLIC_DATA_DIR / "benchmarks.json").read_text())
    validation = json.loads((PUBLIC_DATA_DIR / "validation.json").read_text())
    machines = json.loads((PUBLIC_DATA_DIR / "machines.json").read_text())["machines"]
    profiles, _ = fit_floor_profiles(bench["rows"], spec_bandwidth(machines))
    tiers = tier_profiles(profiles)
    return fit_speed_model(profiles, assumed_profiles(tiers), tiers, validation, None)


def test_the_long_context_flash_next_rows_land_within_15_percent():
    sm, report = _fit()
    rows = [x for x in report["rows"] if x["model"] == "Qwen 3.8 Flash Next" and x["chip"] == "M5 Max" and x["runtime"] == "mlx"]
    assert {x["context"] for x in rows} >= {4096, 32768, 65536, 131072}
    for x in rows:
        assert abs(x["floor"] / x["measured"] - 1) <= 0.15, x
    # the index pass is fitted on the M5 rows; the older chips take it scaled, labelled assumed
    index = sm["mlx"]["index_ms_per_32k"]
    assert index["m5plus"]["source"] == "fitted" and index["m5plus"]["n"] >= 3
    assert index["pre_m5"]["source"] == "assumed" and index["pre_m5"]["value"] >= index["m5plus"]["value"]


def test_the_other_hybrids_keep_their_long_context_fit():
    # the hybrids without an index read the whole context; their oMLX ladders stay within 15 % at every length
    _, report = _fit()
    rows = [x for x in report["rows"] if x["runtime"] == "mlx" and x["context"] > 8192 and x["chip"] in ("M5 Max", "M5 Pro")
            and x["model"] in ("Qwen 3.8 27B", "Qwen3-Coder-Next", "Nemotron 3 Super 120B-A12B")]
    assert len(rows) >= 10
    for x in rows:
        assert abs(x["floor"] / x["measured"] - 1) <= 0.15, x
