"""fit_speed.py: profile recovery on synthetic rows, the split rule, the gate, clamps and determinism."""
import json
import math

import pytest

from common import PUBLIC_DATA_DIR
from fit_speed import (
    CLAMP_MLX,
    CLAMP_READ,
    GATE_HOLDOUT_MEDIAN,
    assumed_profiles,
    catalog_bandwidth,
    check_split,
    fit_floor_profiles,
    fit_speed_model,
    fitted,
    spec_bandwidth,
    tier_profiles,
)
from speed_model import mlx_context_group, arch_class, decode_ms, gb_per_token, mlx_group, quant_family


def _row(chip, bw, gb, tg, **kw):
    r = {"chip": chip, "bandwidth_gbs": bw, "weights_gb": gb, "tg": tg, "runtime": "llama.cpp-gguf", "accel": "none", "platform": "apple"}
    r.update(kw)
    return r


def test_profile_recovery_from_synthetic_rows():
    # a chip that reads at 90 % of 546 GB/s with 4 ms fixed cost: 1000/tg = 4 + GB / 0.4914
    spec = {"M4 Max": [546.0, 410.0]}
    rows = [_row("M4 Max", 546, gb, 1000.0 / (4.0 + gb / (0.9 * 0.546))) for gb in (3.8, 7.2, 13.5)]
    profiles, checks = fit_floor_profiles(rows, spec)
    p = profiles["M4 Max"]
    assert abs(p["t0_ms"] - 4.0) < 0.01 and abs(p["b_eff_ratio"] - 0.9) < 0.002 and p["fit_n"] == 3
    assert p["fit_max_err_pct"] < 0.1 and checks == []


def test_profile_uses_the_top_bin_and_reports_the_others():
    spec = {"M4 Max": [546.0, 410.0]}
    rows = [_row("M4 Max", 546, gb, 1000.0 / (4.0 + gb / (0.9 * 0.546))) for gb in (7.2, 13.5)]
    rows += [_row("M4 Max", 410, gb, 1000.0 / (3.8 + gb / (0.88 * 0.410))) for gb in (7.2, 13.5)]
    profiles, checks = fit_floor_profiles(rows, spec)
    assert profiles["M4 Max"]["bandwidth_gbs"] == 546
    assert len(checks) == 1 and checks[0]["bandwidth_gbs"] == 410 and abs(checks[0]["own_ratio"] - 0.88) < 0.002


def test_rows_that_are_not_dense_short_context_llamacpp_are_ignored():
    spec = {"M4 Max": [546.0]}
    rows = [_row("M4 Max", 546, 7.2, 40.0), _row("M4 Max", 546, 13.5, 24.0),
            _row("M4 Max", 546, 60.0, 50.0, active_weights_gb=2.5),      # MoE
            _row("M4 Max", 546, 30.0, 10.0, context_tokens=32768),        # long context
            _row("M4 Max", 546, 7.2, 60.0, runtime="mlx")]                # MLX
    profiles, _ = fit_floor_profiles(rows, spec)
    assert profiles["M4 Max"]["fit_n"] == 2


def test_negative_intercept_is_clamped_and_noted():
    spec = {"X": [100.0]}
    rows = [_row("X", 100, 1.0, 1000.0 / (-1.0 + 1.0 / 0.1)), _row("X", 100, 5.0, 1000.0 / (-1.0 + 5.0 / 0.1))]
    profiles, _ = fit_floor_profiles(rows, spec)
    assert profiles["X"]["t0_ms"] == 0.0 and any("clamped" in n for n in profiles["X"]["notes"])


def test_catalog_bandwidth_snaps_to_the_nearest_bin_within_five_percent():
    spec = {"M3 Ultra": [819.0], "M3 Max": [400.0, 300.0]}
    assert catalog_bandwidth("M3 Ultra", 800.0, spec) == 819.0
    assert catalog_bandwidth("M3 Max", 300.0, spec) == 300.0
    assert catalog_bandwidth("M3 Max", 400.0, spec) == 400.0
    assert catalog_bandwidth("M9", 123.0, spec) == 123.0


def test_tier_and_assumed_profiles_are_medians():
    profiles = {"M4 Pro": {"t0_ms": 4.3, "b_eff_ratio": 0.92}, "M5 Pro": {"t0_ms": 2.6, "b_eff_ratio": 1.01}, "M1 Pro": {"t0_ms": 7.2, "b_eff_ratio": 0.95},
                "M5 Max": {"t0_ms": 0.5, "b_eff_ratio": 0.84}}
    tiers = tier_profiles(profiles)
    assert tiers["pro"]["t0_ms"] == 4.3 and tiers["pro"]["b_eff_ratio"] == 0.95 and tiers["pro"]["n_chips"] == 3
    assert tiers["max"]["n_chips"] == 1
    assumed = assumed_profiles(tiers)
    assert assumed["M6 Pro"]["b_eff_ratio"] == tiers["pro"]["b_eff_ratio"]
    assert assumed["M5 Ultra"]["b_eff_ratio"] == 0.85 and assumed["M5 Ultra"]["t0_ms"] == 1.5 and "note" in assumed["M5 Ultra"]


def test_split_rule_refuses_a_submission_that_straddles():
    check_split([{"submission_id": "a", "split": "train"}, {"submission_id": "a", "split": "report"}, {"submission_id": "b", "split": "holdout"}])
    with pytest.raises(SystemExit):
        check_split([{"submission_id": "a", "split": "train"}, {"submission_id": "a", "split": "holdout"}])


def test_fitted_falls_back_to_provisional_and_clamps():
    assert fitted([], 1.25, CLAMP_READ) == {"value": 1.25, "n": 0, "source": "assumed"}
    assert fitted([1.7], 1.25, CLAMP_READ)["source"] == "assumed"
    f = fitted([1.7, 1.9], 1.25, CLAMP_READ)
    assert f["value"] == CLAMP_READ[1] and f["source"] == "fitted" and f["n"] == 2
    assert fitted([0.1, 0.2, 0.3], 0.9, CLAMP_MLX)["value"] == CLAMP_MLX[0]


def test_mirror_helpers():
    assert quant_family("UD-Q4_K_XL") == "gguf_kquant" and quant_family("Q4_0") == "q4_0" and quant_family("MXFP4") == "mxfp4"
    assert quant_family("4bit", "mlx") == "mlx" and quant_family("MLX-8bit") == "mlx" and quant_family("Q8_0") == "q8_0"
    assert arch_class({"experts_total": 1}) == "dense" and arch_class(None) == "dense"
    assert arch_class({"experts_total": 128, "attention": "gqa", "linear_layers": 0}) == "moe"
    assert arch_class({"experts_total": 128, "attention": "gqa", "linear_layers": 24}) == "moe_hybrid"
    assert arch_class({"experts_total": 128, "attention": "mla"}) == "moe_latent"
    assert arch_class({"experts_total": 128, "attention": "mla"}, "deepseek_v4") == "deepseek_v4"
    assert mlx_group("dense", 27, "M4 Max") == "dense" and mlx_group("moe", 3.6, "M4 Max") == "small_active_moe_m4plus"
    assert mlx_group("moe", 3.6, "M1 Ultra") == "small_active_moe_pre_m4" and mlx_group("moe", 3.6, "DGX Spark") == "small_active_moe_pre_m4"
    assert mlx_group("moe_latent", 18, "M5 Max") == "moe_other"
    assert abs(gb_per_token(18, 4.5, 5632, 16, 8192) - (18 * 4.5 / 8 + 5632 * 2 * 8192 / 1e9)) < 1e-9


def test_decode_ms_shape():
    sm = {"read_factor": {"gguf_kquant": {"value": 1.25}}, "architecture_cost_ms": {"dense": {"value": 0.0}, "moe_latent": {"value": 10.0}},
          "attention_ms_per_32k": {"gqa": {"value": 0.0}, "mla": {"value": 5.0}}, "mlx": {"overhead_factor": {"dense": {"value": 0.8}}, "attention_ms_per_32k": {"pre_m5": {"value": 4.0}, "m5plus": {"value": 1.0}}}}
    p = {"t0_ms": 4.0, "b_eff_ratio": 0.9}
    plain = decode_ms(p, 500, 9.0, "q4_0", "dense", "gguf", "dense", "gqa", 0, sm)
    assert abs(plain - (9.0 / 450 * 1000 + 4.0)) < 1e-9
    kq = decode_ms(p, 500, 9.0, "gguf_kquant", "dense", "gguf", "dense", "gqa", 0, sm)
    assert abs(kq - (9.0 / 450 * 1000 * 1.25 + 4.0)) < 1e-9
    latent = decode_ms(p, 500, 9.0, "q4_0", "moe_latent", "gguf", "moe_other", "mla", 32768, sm)
    assert abs(latent - (9.0 / 450 * 1000 + 14.0 + 5.0)) < 1e-9
    mlx = decode_ms(p, 500, 9.0, "mlx", "dense", "mlx", "dense", "gqa", 16384, sm)
    assert abs(mlx - (9.0 / 450 * 1000 + 4.0 * 0.8 + 2.0)) < 1e-9
    # the M5 generation pays its own long-context cost; an unknown generation is charged the pre-M5 one
    m5 = decode_ms(p, 500, 9.0, "mlx", "dense", "mlx", "dense", "gqa", 16384, sm, "apple", "m5plus")
    assert abs(m5 - (9.0 / 450 * 1000 + 4.0 * 0.8 + 0.5)) < 1e-9
    assert mlx_context_group("M5 Max") == "m5plus" and mlx_context_group("M6") == "m5plus"
    assert mlx_context_group("M4 Max") == "pre_m5" and mlx_context_group("M1 Ultra") == "pre_m5" and mlx_context_group("Ryzen AI Max+") == "pre_m5"
    sm["read_factor"]["gguf_kquant"]["platforms"] = ["apple"]
    assert abs(decode_ms(p, 500, 9.0, "gguf_kquant", "dense", "gguf", "dense", "gqa", 0, sm, "cuda") - plain) < 1e-9
    assert abs(decode_ms(p, 500, 9.0, "gguf_kquant", "dense", "gguf", "dense", "gqa", 0, sm, "apple") - kq) < 1e-9


def test_live_fit_is_deterministic_and_gated():
    factors = json.loads((PUBLIC_DATA_DIR / "factors.json").read_text())
    sm = factors["speed_model"]
    v = sm["validation"]
    assert v["n_holdout"] >= 12 and v["n_train"] >= 40
    assert v["holdout_median_err"] <= GATE_HOLDOUT_MEDIAN
    assert v["gate"]["passed"] is True
    assert sm["kind"] in ("multiplier", "floor")
    for cls in ("dense", "moe", "moe_hybrid", "moe_latent", "deepseek_v4"):
        assert sm["architecture_cost_ms"][cls]["source"] in ("fitted", "assumed", "definition")
    assert CLAMP_READ[0] <= sm["read_factor"]["gguf_kquant"]["value"] <= CLAMP_READ[1]
    for g, t in sm["mlx"]["overhead_factor"].items():
        assert CLAMP_MLX[0] <= t["value"] <= CLAMP_MLX[1], g
    # re-running the fit on the shipped inputs reproduces the block (calibrate.py --check does this byte-exactly; here: the numbers)
    benchmarks = json.loads((PUBLIC_DATA_DIR / "benchmarks.json").read_text())
    machines = json.loads((PUBLIC_DATA_DIR / "machines.json").read_text())["machines"]
    validation = json.loads((PUBLIC_DATA_DIR / "validation.json").read_text())
    spec = spec_bandwidth(machines)
    profiles, _ = fit_floor_profiles(benchmarks["rows"], spec)
    tiers = tier_profiles(profiles)
    assumed = assumed_profiles(tiers)
    again, _ = fit_speed_model(profiles, assumed, tiers, validation, {"efficiency": factors["efficiency"], "runtime": factors["runtime"]})
    assert again["architecture_cost_ms"] == sm["architecture_cost_ms"]
    assert again["mlx"] == sm["mlx"]
    assert math.isclose(again["validation"]["holdout_median_err"], v["holdout_median_err"], rel_tol=1e-9)
