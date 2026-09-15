import json
import subprocess
import sys
from pathlib import Path

from common import PUBLIC_DATA_DIR, HERE
from calibrate import tier_of


def test_tier_names():
    assert tier_of("M5 Max") == "max" and tier_of("M2") == "base" and tier_of("M3 Ultra") == "ultra"
    assert tier_of("DGX Spark") == "spark" and tier_of("M6 Pro") == "pro"


def test_factors_json_is_current_and_sane():
    r = subprocess.run([sys.executable, str(HERE / "calibrate.py"), "--check"], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    f = json.loads((PUBLIC_DATA_DIR / "factors.json").read_text())
    tiers = f["efficiency"]["by_tier"]
    assert 0.76 <= tiers["base"]["value"] <= 0.82
    assert 0.70 <= tiers["pro"]["value"] <= 0.78
    assert 0.58 <= tiers["max"]["value"] <= 0.68
    assert 0.40 <= tiers["ultra"]["value"] <= 0.50
    assert 0.70 <= tiers["spark"]["value"] <= 0.80   # five llama.cpp maintainer rows (build b7941)
    assert f["efficiency"]["by_chip"]["M5 Max"]["measured"] and f["efficiency"]["assumed"]["M5 Ultra"]["measured"] is False
    acc = f["acceleration"]
    assert 1.5 <= acc["mtp"]["min"] and acc["mtp"]["max"] <= 2.7
    assert 1.9 <= acc["draft_dense"]["min"] and acc["draft_dense"]["max"] <= 3.7
    assert 1.1 <= acc["draft_small_active_moe"]["median"] <= 1.4


def test_speed_targets_from_factors():
    f = json.loads((PUBLIC_DATA_DIR / "factors.json").read_text())
    e = f["efficiency"]
    def tok(bw, eff, active_b, bits):
        return bw * eff / (active_b * bits / 8)
    ultra = tok(1200, e["assumed"]["M5 Ultra"]["value"], 18, 4.5)   # GLM-5.3 Flash, 18B active, ~4.5-bit
    spark = tok(273, e["by_chip"]["DGX Spark"]["value"], 18, 4.5)
    assert 51 <= ultra <= 69          # ~60 tok/s ± 15 %
    assert 17 <= spark <= 23          # ≈ 20 tok/s from the llama.cpp maintainer rows (the single SGLang row gave ~11)
    m4pro = tok(273, e["by_chip"]["M4 Pro"]["value"], 27.8, 4.62) * f["runtime"]["mlx"]["dense_14b_and_up"]["factor"]
    assert 11 <= m4pro <= 16          # measured MLX-4bit baseline on M4 Pro: 14.7 tok/s


def test_speed_model_block_and_profiles():
    f = json.loads((PUBLIC_DATA_DIR / "factors.json").read_text())
    sm = f["speed_model"]
    assert sm["kind"] in ("multiplier", "floor")
    assert set(sm["architecture_cost_ms"]) == {"dense", "moe", "moe_hybrid", "moe_latent", "deepseek_v4"}
    assert sm["architecture_cost_ms"]["dense"]["value"] == 0.0
    assert set(sm["mlx"]["overhead_factor"]) == {"dense", "small_active_moe_m4plus", "small_active_moe_pre_m4", "moe_other"}
    e = f["efficiency"]
    for chip in ("M1 Max", "M2 Ultra", "M3 Ultra", "M4 Max", "M5 Max", "DGX Spark"):
        p = e["by_chip"][chip]
        assert p["measured"] and p["fit_n"] >= 2, chip
        assert 0.75 <= p["b_eff_ratio"] <= 1.05, chip
        assert 0.0 <= p["t0_ms"] <= 9.0, chip
        assert abs(p["b_eff_gbs"] - p["b_eff_ratio"] * (273 if chip == "DGX Spark" else p["b_eff_gbs"] / p["b_eff_ratio"])) < 0.5
    assert e["by_chip"]["M5 Max"]["t0_ms"] < 3 and 4 <= e["by_chip"]["M1 Max"]["t0_ms"] <= 8
    for tier in ("base", "pro", "max", "ultra", "spark"):
        assert "t0_ms" in e["by_tier"][tier] and "b_eff_ratio" in e["by_tier"][tier], tier
    assert e["assumed"]["M5 Ultra"]["b_eff_ratio"] == 0.85 and e["assumed"]["M5 Ultra"]["t0_ms"] == 1.5
    assert e["assumed"]["M6"]["b_eff_ratio"] == e["by_tier"]["base"]["b_eff_ratio"]
    v = sm["validation"]
    assert v["n_holdout"] >= 12 and v["holdout_median_err"] <= 0.2232 and v["gate"]["passed"]


def test_floor_speed_targets_from_factors():
    from speed_model import decode_ms, gb_per_token
    f = json.loads((PUBLIC_DATA_DIR / "factors.json").read_text())
    sm, e = f["speed_model"], f["efficiency"]
    assert sm["kind"] == "floor"
    # GLM-5.3 Flash (18B active, latent attention) at Q4 (UD-Q4_K_XL, 4.97 bits), 8K, FP16 cache
    gb = gb_per_token(18, 4.97, 5632, 16, 8192)
    ultra = 1000 / decode_ms(e["assumed"]["M5 Ultra"], 1200, gb, "gguf_kquant", "moe_latent", "gguf", "moe_other", "mla", 8192, sm, "apple")
    spark = 1000 / decode_ms(e["by_chip"]["DGX Spark"], 273, gb, "gguf_kquant", "moe_latent", "gguf", "moe_other", "mla", 8192, sm, "cuda")
    assert 28 <= ultra <= 40           # ≈ 35 tok/s (METHODOLOGY §9 B)
    assert 14 <= spark <= 20           # ≈ 17 tok/s at Q4 8K; the 3-bit build at 128K lands ≈ 15
    # Qwen 3.8 27B dense on MLX 4-bit (≈ 4.5 bits) on an M4 Pro at 8K: ≈ 14 tok/s
    gb27 = gb_per_token(27, 4.5, 32768, 16, 8192)
    m4pro = 1000 / decode_ms(e["by_chip"]["M4 Pro"], 273, gb27, "mlx", "dense", "mlx", "dense", "gqa", 8192, sm, "apple")
    assert 11 <= m4pro <= 17


def test_cluster_block_is_fitted_and_in_range():
    f = json.loads((PUBLIC_DATA_DIR / "factors.json").read_text())
    c = f["cluster"]
    assert set(c["hop_ms"]) == {"mac", "spark", "gpu"}
    assert set(c["tensor_c_ms"]) == {"dense", "moe", "gpu"}
    # the hop of a layer split: the Macs' four measured points and the Sparks' two
    assert 3 <= c["hop_ms"]["mac"]["value"] <= 6 and c["hop_ms"]["mac"]["source"] == "fitted"
    assert 3.5 <= c["hop_ms"]["spark"]["value"] <= 4.5 and c["hop_ms"]["spark"]["source"] == "fitted"
    assert c["hop_ms"]["gpu"]["source"] == "assumed" and c["hop_ms"]["gpu"]["n"] == 0
    # tensor parallel's synchronisation constant: a mixture of experts pays roughly twice the dense one
    assert 4 <= c["tensor_c_ms"]["dense"]["value"] <= 5.5 and c["tensor_c_ms"]["dense"]["source"] == "fitted"
    assert 9 <= c["tensor_c_ms"]["moe"]["value"] <= 12.5 and c["tensor_c_ms"]["moe"]["source"] == "fitted"
    assert c["tensor_c_ms"]["gpu"]["source"] == "assumed" and c["tensor_c_ms"]["gpu"]["value"] == 1.5
    for key in ("mac", "spark"):
        t = c["hop_ms"][key]
        assert t["n"] >= 2 and t["range"][0] <= t["value"] <= t["range"][1], key
    # reading speed: flat on a layer split, an assumed exponent under tensor parallel
    assert c["prefill"]["layer_factor"]["value"] == 1.0 and c["prefill"]["layer_factor"]["source"] == "assumed"
    assert c["prefill"]["tensor_exponent"]["value"] == 0.75
    assert c["prefill"]["tensor_exponent"]["source"] == "assumed" and c["prefill"]["tensor_exponent"]["n"] == 0
    assert c["dispatch_caveat"]["min_layers"] == 60
    assert len(c["_sources"]) >= 10 and all(u.startswith("https://") for u in c["_sources"])
    assert set(c["_sources"]) <= set(f["sources"])


def test_cluster_series_rows_are_well_formed_and_out_of_the_single_machine_fits():
    bench = json.loads((PUBLIC_DATA_DIR / "benchmarks.json").read_text())
    series = bench["cluster_series"]
    assert len(series) >= 12 and len({s["id"] for s in series}) == len(series)
    assert "cluster_series" in bench["schema"]
    fitted_series = [s for s in series if not s.get("report_only")]
    for s in series:
        assert s["link_class"] in ("mac", "spark", "gpu") and s["split"] in ("layer", "tensor")
        assert s["link"] in ("thunderbolt5", "connectx7-200gbe", "pcie5")
        assert s["arch_class"] in ("dense", "moe", "moe_hybrid", "moe_latent")
        assert s["source"].startswith("https://") and s["date"]
        assert s["tg_by_nodes"] and all(k.isdigit() and v > 0 for k, v in s["tg_by_nodes"].items())
        # a series that is fitted has to have the one-machine point the formula subtracts
        assert ("1" in s["tg_by_nodes"]) == (s in fitted_series), s["id"]
    # they live outside `rows`, so no single-machine loop indexes them
    assert all("tg_by_nodes" not in r for r in bench["rows"])
    assert {"mac", "spark"} <= {s["link_class"] for s in fitted_series if s["split"] == "layer"}
