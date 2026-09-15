#!/usr/bin/env python3
"""Turn measured benchmarks into the speed factors the engine uses.

Reads  public/data/llm-sizer/benchmarks.json (hand-curated, sourced rows)
Writes public/data/llm-sizer/factors.json    (committed; the engine and METHODOLOGY.md must agree with it)

Usage: python3 scripts/llm-sizer/calibrate.py [--check] [--report]
  --check   exit 1 if factors.json is stale, or if the fixed-cost speed model is active and its holdout gate fails
  --report  print the fit report of the fixed-cost speed model (fit_speed.py)

Two speed models live side by side while the fixed-cost ("floor") model is validated: the multiplier model
(efficiency × bandwidth ÷ bytes, the legacy) and the floor model (t0 + bytes ÷ effective bandwidth + architecture cost).
SPEED_MODEL_KIND picks the one the engine uses; the factors file always carries both.
"""
from __future__ import annotations

import argparse
import json
import math
import re
import statistics
import sys
from collections import defaultdict

from common import PUBLIC_DATA_DIR, log
from fit_speed import assumed_profiles, fit_floor_profiles, fit_speed_model, fitted, render_report, spec_bandwidth, tier_profiles

TIERS = ["base", "pro", "max", "ultra", "spark"]

# Which speed model the engine runs: "multiplier" (legacy) or "floor" (fixed cost + effective bandwidth). Flipped in one
# reviewed commit together with the regenerated factors.json; --check enforces the holdout gate only for "floor".
SPEED_MODEL_KIND = "floor"

# Memory policy margins (mirrored in engine/constants.ts MEMORY_POLICY); listed here so fit_speed.py --memory can score them.
MEMORY_POLICY = {"bufferMinGb": 0.0, "bufferBaseGb": 1.5, "bufferPct": 0.01, "bufferCapGb": None, "mlxScratchGbPer1k": 0.15, "mlxScratchFromTokens": 8192, "osBaseGb": 6.0, "osReservePctOnOverride": 0.055}

# Published (not yet measured here) constants — see METHODOLOGY.md §6.2. Each carries its source.
RUNTIME_FACTORS = {
    "gguf": {"factor": 1.0, "note": "llama.cpp / LM Studio GGUF is the baseline"},
    "mlx": {
        "dense_under_14b": {"factor": 1.10, "range": [1.0, 1.2]},
        "dense_14b_and_up": {"factor": 1.15, "range": [1.1, 1.2]},
        "moe": {"factor": 1.9, "range": [1.5, 3.0]},
        "flatten_above_context": 36000,
        "note": "2026 community MLX-vs-llama.cpp reports; applied in full up to 24K context and fading linearly to 1.0 at 36K",
        "measured_here": False,
    },
}
# Below this active-parameter count a draft model adds little (the target is already fast).
SMALL_ACTIVE_B = 6.0


def tier_of(chip: str) -> str:
    c = chip.lower()
    if "spark" in c or "gb10" in c:
        return "spark"
    if "ultra" in c:
        return "ultra"
    if "max" in c:
        return "max"
    if "pro" in c:
        return "pro"
    if re.match(r"^m\d+$", c.strip()):
        return "base"
    return "base"


def generation_of(chip: str) -> str | None:
    m = re.match(r"^(M\d+)", chip.strip(), re.I)
    return m.group(1).upper() if m else None


def med(xs: list[float]) -> float:
    return round(statistics.median(xs), 3)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--report", action="store_true")
    args = ap.parse_args()

    bench = json.loads((PUBLIC_DATA_DIR / "benchmarks.json").read_text())
    rows = bench["rows"]
    machines = json.loads((PUBLIC_DATA_DIR / "machines.json").read_text())["machines"]
    validation = json.loads((PUBLIC_DATA_DIR / "validation.json").read_text())

    # ---- decode efficiency per chip: tg ÷ (bandwidth ÷ weights) on plain llama.cpp rows --------------
    # (the multiplier model; rows marked role = floor-fit are excluded so this median stays on its 0.8 row set)
    by_chip: dict[str, list[float]] = defaultdict(list)
    bandwidth: dict[str, float] = {}
    for r in rows:
        if r["runtime"] != "llama.cpp-gguf" or r["accel"] != "none" or not r.get("tg") or not r.get("weights_gb"):
            continue
        if r.get("role") == "floor-fit":
            continue
        # what the machine re-reads per token: the active weights for a mixture-of-experts row, else the whole file
        read_gb = r.get("active_weights_gb") or r["weights_gb"]
        eff = r["tg"] / (r["bandwidth_gbs"] / read_gb)
        by_chip[r["chip"]].append(eff)
        bandwidth[r["chip"]] = r["bandwidth_gbs"]
    chips = {chip: {"value": med(v), "n": len(v), "measured": True, "tier": tier_of(chip), "generation": generation_of(chip),
                    "range": [round(min(v), 3), round(max(v), 3)]} for chip, v in by_chip.items()}
    tiers: dict[str, dict] = {}
    for t in TIERS:
        vals = [c["value"] for c in chips.values() if c["tier"] == t]
        if vals:
            tiers[t] = {"value": med(vals), "n_chips": len(vals), "range": [round(min(vals), 3), round(max(vals), 3)]}

    # Unmeasured chips: inherit the tier median. The M5 generation measured ~1.2× its predecessors on the Max
    # tier (M5 Max 0.74 vs M4 Max 0.58), so M5 Ultra gets that lift as an explicit, labelled assumption.
    m5_lift = None
    if "M5 Max" in chips and "M4 Max" in chips:
        m5_lift = round(chips["M5 Max"]["value"] / chips["M4 Max"]["value"], 3)
    assumed = {}
    if "ultra" in tiers and m5_lift:
        assumed["M5 Ultra"] = {"value": round(min(tiers["ultra"]["value"] * m5_lift, 0.6), 3), "measured": False, "tier": "ultra", "generation": "M5",
                               "note": f"ultra-tier median {tiers['ultra']['value']} × M5 generation lift {m5_lift} (M5 Max vs M4 Max); replace when measured"}
    if "base" in tiers:
        assumed["M6"] = {"value": tiers["base"]["value"], "measured": False, "tier": "base", "generation": "M6", "note": "base-tier median until measured"}
    if "pro" in tiers:
        assumed["M6 Pro"] = {"value": tiers["pro"]["value"], "measured": False, "tier": "pro", "generation": "M6", "note": "pro-tier median until measured"}

    # ---- the fixed-cost model: chip profiles from dense llama.cpp rows, terms from the validation rows ----
    profiles, bin_checks = fit_floor_profiles(rows, spec_bandwidth(machines))
    tier_prof = tier_profiles(profiles)
    assumed_prof = assumed_profiles(tier_prof)
    for chip, p in profiles.items():
        entry = chips.setdefault(chip, {"value": None, "n": 0, "measured": True, "tier": tier_of(chip), "generation": generation_of(chip), "range": None})
        entry.update({"t0_ms": p["t0_ms"], "b_eff_gbs": p["b_eff_gbs"], "b_eff_ratio": p["b_eff_ratio"], "fit_n": p["fit_n"], "fit_max_err_pct": p["fit_max_err_pct"]})
        if p["notes"]:
            entry["fit_note"] = "; ".join(p["notes"])
    for t, v in tier_prof.items():
        tiers.setdefault(t, {"value": None, "n_chips": v["n_chips"], "range": None}).update({"t0_ms": v["t0_ms"], "b_eff_ratio": v["b_eff_ratio"]})
    for chip, v in assumed_prof.items():
        a = assumed.setdefault(chip, {"value": None, "measured": False, "tier": v["tier"], "generation": v.get("generation"), "note": v["note"]})
        a.update({"t0_ms": v["t0_ms"], "b_eff_ratio": v["b_eff_ratio"]})
        if chip == "M5 Ultra":
            a["note"] = f"{a['note']}; fixed-cost model: {v['note']}"
    legacy_view = {"efficiency": {"by_chip": chips, "assumed": assumed, "by_tier": tiers}, "runtime": RUNTIME_FACTORS}
    speed_model, fit_report = fit_speed_model(profiles, assumed_prof, tier_prof, validation, legacy_view)
    speed_model = {
        "kind": SPEED_MODEL_KIND,
        "_formula": ("ms per token = GB read per token ÷ (b_eff_ratio × spec bandwidth) × 1000 × read_factor[quant family] (the K-quant factor on the platforms it lists: Apple silicon) "
                     "+ (t0_ms + architecture_cost_ms[class]) × mlx overhead_factor[group] (GGUF: 1) "
                     "+ (attention_ms_per_32k[attention] + mlx attention_ms_per_32k[chip generation: pre_m5 | m5plus]) × context ÷ 32768; tok/s = 1000 ÷ ms"),
        "_classes": "dense | moe (mixture of experts) | moe_hybrid (mixture of experts with linear-attention layers) | moe_latent (mixture of experts with MLA or latent attention) | deepseek_v4 (per-model override)",
        "_groups": "MLX overhead groups: dense | small_active_moe_m4plus (under 6B active, M4 generation and newer) | small_active_moe_pre_m4 | moe_other. MLX context-cost groups: pre_m5 (M1 to M4, and any chip whose generation is not known) | m5plus (M5 and newer)",
        **speed_model,
        "bin_checks": bin_checks,
    }

    # ---- acceleration ratios from paired rows -----------------------------------------------------------
    groups: dict[str, list[float]] = defaultdict(list)
    examples: dict[str, list[str]] = defaultdict(list)
    for r in rows:
        if r["accel"] == "none" or not r.get("baseline_tg"):
            continue
        ratio = r["tg"] / r["baseline_tg"]
        active = r.get("active_weights_gb")
        small_active_moe = active is not None and active < SMALL_ACTIVE_B
        if r["accel"] == "mtp":
            key = "mtp"
        elif small_active_moe:
            key = "draft_small_active_moe"
        else:
            key = "draft_dense"
        groups[key].append(ratio)
        examples[key].append(f"{r['model']} {r['quant']} on {r['chip']}: {r['baseline_tg']} → {r['tg']} ({ratio:.2f}×, {r['accel']})")
    acceleration = {k: {"min": round(min(v), 2), "median": med(v), "max": round(max(v), 2), "n": len(v), "examples": examples[k]} for k, v in groups.items()}
    acceleration["_rule"] = ("MTP: native heads, factor in [min, max] when the runtime uses them. Draft models (DFlash/DSpark): use draft_dense "
                             f"unless the model is a mixture-of-experts with under {SMALL_ACTIVE_B} B active parameters, then draft_small_active_moe. "
                             "Always show the range, never the headline.")

    # ---- prefill (prompt processing): compute-bound, so tokens/s × active parameters is the constant --------------
    # per GPU core of a generation on plain llama.cpp files (the community table), a per-chip constant where cores mean
    # nothing (the Spark), then the terms that move it: K-quant files, the architecture class, MLX, and the context already read
    def k_of(r: dict) -> float:
        return r["pp"] * r["params_active_b"]
    PLAIN = {"Q4_0", "Q8_0", "F16", "BF16", "MXFP4"}
    pp_rows = [r for r in rows if r.get("pp") and r.get("params_active_b") and r["accel"] == "none" and r["runtime"] in ("llama.cpp-gguf", "mlx")]
    per_core_rows: dict[str, list[float]] = defaultdict(list)
    by_chip_rows: dict[str, list[float]] = defaultdict(list)
    for r in pp_rows:
        if r["runtime"] != "llama.cpp-gguf" or r.get("arch_class", "dense") != "dense" or r["quant"] not in PLAIN or r.get("role") == "prefill":
            continue
        gen = generation_of(r["chip"])
        if gen and r.get("gpu_cores"):
            per_core_rows[gen].append(k_of(r) / r["gpu_cores"])
        else:
            by_chip_rows[r["chip"]].append(k_of(r))
    per_core = {g: {"value": round(med(v)), "n": len(v), "range": [round(min(v)), round(max(v))]} for g, v in per_core_rows.items()}
    by_chip = {c: {"value": round(med(v)), "n": len(v), "range": [round(min(v)), round(max(v))]} for c, v in by_chip_rows.items()}
    def dense_gguf_reference(chip: str, cores: int | None) -> float | None:
        gen = generation_of(chip)
        if gen in per_core and cores:
            return per_core[gen]["value"] * cores
        return by_chip[chip]["value"] if chip in by_chip else None
    # K-quant GGUF files against the plain-file constant of the same chip and cores (community dense rows)
    kq: list[float] = []
    cls_ratios: dict[str, list[float]] = defaultdict(list)
    cls_examples: dict[str, list[str]] = defaultdict(list)
    mlx_ratios: dict[str, list[float]] = defaultdict(list)
    mlx_examples: dict[str, list[str]] = defaultdict(list)
    # dense reference per (chip, cores, runtime) among the community rows themselves, for the class ratios
    dense_ref: dict[tuple, list[float]] = defaultdict(list)
    for r in pp_rows:
        if r.get("role") == "prefill" and r.get("arch_class", "dense") == "dense":
            dense_ref[(r["chip"], r.get("gpu_cores"), r["runtime"])].append(k_of(r))
    for r in pp_rows:
        if r.get("role") != "prefill":
            continue
        ref_plain = dense_gguf_reference(r["chip"], r.get("gpu_cores"))
        cls = r.get("arch_class", "dense")
        if r["runtime"] == "llama.cpp-gguf" and cls == "dense" and ref_plain:
            kq.append(k_of(r) / ref_plain)
        if r["runtime"] == "mlx" and cls == "dense" and ref_plain:
            group = "m5plus" if (generation_of(r["chip"]) or "M0") >= "M5" else "pre_m5"
            mlx_ratios[group].append(k_of(r) / ref_plain)
            mlx_examples[group].append(f"{r['model']} on {r['chip']} {r.get('gpu_cores')}-core: {r['pp']} tok/s ({k_of(r) / ref_plain:.2f} of the plain-GGUF constant)")
        if cls != "dense":
            ref = dense_ref.get((r["chip"], r.get("gpu_cores"), r["runtime"]))
            if ref:
                cls_ratios[cls].append(k_of(r) / med(ref))
                cls_examples[cls].append(f"{r['model']} on {r['chip']} {r.get('gpu_cores')}-core ({r['runtime']}): {k_of(r) / med(ref):.2f} of the dense rate")
    # the Spark's mixture rows against its own dense rows (table rows, plain files)
    for r in pp_rows:
        if r["chip"] == "DGX Spark" and r.get("arch_class") not in (None, "dense") and r["runtime"] == "llama.cpp-gguf" and "DGX Spark" in by_chip:
            cls_ratios[r["arch_class"]].append(k_of(r) / by_chip["DGX Spark"]["value"])
            cls_examples[r["arch_class"]].append(f"{r['model']} on the DGX Spark: {k_of(r) / by_chip['DGX Spark']['value']:.2f} of the dense rate")
    kquant = {"factor": round(med(kq), 2), "n": len(kq), "range": [round(min(kq), 2), round(max(kq), 2)]} if kq else {"factor": 0.9, "n": 0, "note": "assumed"}
    classes = {"dense": {"factor": 1.0, "note": "by definition"}}
    for cls, v in cls_ratios.items():
        classes[cls] = {"factor": round(med(v), 2), "n": len(v), "range": [round(min(v), 2), round(max(v), 2)], "examples": cls_examples[cls]}
    # the depth sweeps: D0 such that rate(d) = rate(0) ÷ (1 + d ÷ D0); the sweep's own class factor from its depth-0 point
    sweeps = bench.get("prefill_depth_sweeps", [])
    d0s: list[float] = []
    share_ref = None
    for sw in sweeps:
        pts = [(d, v) for d, v in sw["points"] if d > 0]
        p0 = next(v for d, v in sw["points"] if d == 0)
        d0s.extend(d / (p0 / v - 1) for d, v in pts)
        share_ref = sw["attention_layers"] / sw["layers"]
        ref_plain = dense_gguf_reference(sw["chip"], sw.get("gpu_cores"))
        if ref_plain and sw.get("arch_class") and sw["arch_class"] not in classes:
            kqf = kquant["factor"] if sw["quant"] not in PLAIN else 1.0
            ratio = p0 * sw["params_active_b"] / (ref_plain * kqf)
            classes[sw["arch_class"]] = {"factor": round(ratio, 2), "n": 1, "range": [round(ratio, 2), round(ratio, 2)],
                                          "examples": [f"{sw['model']} on {sw['chip']} {sw['gpu_cores']}-core at an empty context: {p0} tok/s ({ratio:.2f} of the dense rate)"]}
    if "moe_latent" not in classes:
        classes["moe_latent"] = {"factor": classes.get("moe", {"factor": 0.75})["factor"], "n": 0, "note": "assumed: the mixture-of-experts factor, no measured prefill row for a latent-attention model yet"}
    if "deepseek_v4" not in classes:
        classes["deepseek_v4"] = {"factor": classes["moe_latent"]["factor"], "n": 0, "note": "assumed: the latent-attention factor"}
    mlx = {g: {"factor": round(med(v), 2), "n": len(v), "range": [round(min(v), 2), round(max(v), 2)], "examples": mlx_examples[g]} for g, v in mlx_ratios.items()}
    if "pre_m5" not in mlx:
        mlx["pre_m5"] = {"factor": 0.65, "n": 0, "note": "assumed"}
    if "m5plus" not in mlx:
        mlx["m5plus"] = {"factor": 0.9, "n": 0, "note": "assumed"}
    class_by_model = {sw["model_id"]: sw["arch_class"] for sw in sweeps if sw.get("model_id") and sw.get("arch_class") not in (None, "dense", "moe", "moe_hybrid", "moe_latent", "deepseek_v4")}
    chips_with_rows = sorted({r["chip"] for r in pp_rows if r["runtime"] == "llama.cpp-gguf" and r.get("role") != "prefill" and r.get("arch_class", "dense") == "dense"})
    prefill = {
        "class_by_model": class_by_model,
        "chips_with_rows": chips_with_rows,
        "_formula": "prefill tok/s at a short prompt = K ÷ active parameters (B), K = per_core[generation] × GPU cores (Apple) or by_chip (others), × kquant for K-quant and IQ GGUF files, × class[arch], × mlx[generation group] on MLX; rate at context depth d = short-prompt rate ÷ (1 + d ÷ d0_tokens × attention_share ÷ attention_share_ref); wait before the first word for an n-token prompt = (n + n² ÷ (2 × D0)) ÷ short-prompt rate",
        "per_core": dict(sorted(per_core.items())),
        "by_chip": dict(sorted(by_chip.items())),
        "kquant": kquant,
        "class": classes,
        "mlx": mlx,
        "context": {"d0_tokens": round(med(d0s)) if d0s else 20000, "n_points": len(d0s), "range": [round(min(d0s)), round(max(d0s))] if d0s else None,
                    "attention_share_ref": share_ref if share_ref is not None else 0.25,
                    "note": "fitted on the published depth sweep(s): D0 is the context depth at which the rate has halved; other models scale D0 by their share of full-attention layers relative to the sweep's model (assumed)"},
        "waits_tokens": [8192, 32768, 131072],
        "feels_like": [{"max_s": 3, "label": "instant"}, {"max_s": 15, "label": "short wait"}, {"max_s": 90, "label": "get a coffee"}, {"max_s": None, "label": "painful"}],
        "_sources": ["https://github.com/ggml-org/llama.cpp/discussions/4167", "https://github.com/enescingoz/mac-llm-bench", "https://heretik.io/qwen38-flash-next-262k-macbook/"],
    }


    # ---- linked machines: the hop of a layer split and the constant of tensor parallel --------------------
    # A layer split runs one machine's time for the whole model and pays a hop for every extra machine; tensor
    # parallel divides the work and pays a synchronisation constant that grows with log2 of the count. Series
    # without a one-machine point of their own runtime carry report_only: they are listed in the sources and
    # never fitted. The cluster series live in their own array, so no single-machine loop above sees them.
    series = bench.get("cluster_series", [])
    hop_vals: dict[str, list[float]] = defaultdict(list)
    c_vals: dict[str, list[float]] = defaultdict(list)
    pp_layer_ratios: list[float] = []
    pp_layer_series = 0
    for s in series:
        tg = {int(k): float(v) for k, v in s["tg_by_nodes"].items()}
        pp = {int(k): float(v) for k, v in (s.get("pp_by_nodes") or {}).items()}
        if s["split"] == "layer" and len(pp) >= 2 and 1 in pp and not s.get("report_only"):
            pp_layer_series += 1
            pp_layer_ratios.extend(pp[k] / pp[1] for k in sorted(pp) if k >= 2)
        if s.get("report_only") or 1 not in tg:
            continue
        one_ms = 1000.0 / tg[1]
        for k in sorted(tg):
            if k < 2:
                continue
            ms = 1000.0 / tg[k]
            if s["split"] == "layer":
                hop_vals[s["link_class"]].append((ms - one_ms) / (k - 1))
            else:
                c_vals["dense" if s.get("arch_class") == "dense" else "moe"].append((ms - one_ms / k) / math.log2(k))
    layer_factor = (fitted(pp_layer_ratios, 1.0) if pp_layer_series >= 2 else
                    {"value": 1.0, "n": pp_layer_series, "source": "assumed",
                     "note": ("assumed flat: only one layer-split series publishes prompt reading at two machine counts "
                              "(llama.cpp RPC on two DGX Sparks, pp2048 389.5 tok/s on one machine and 453.9 on two, 1.17 times), "
                              "so reading speed stays flat until a second series exists")})
    cluster = {
        "_formula": ("layer split: ms per token = one machine's ms for the whole model + hop_ms[link] × (machines − 1); "
                     "tensor parallel: ms = one machine's ms ÷ machines + tensor_c_ms[class] × log2(machines); "
                     "reading speed × prefill.layer_factor on a layer split, × machines ^ prefill.tensor_exponent under tensor parallel"),
        "_links": "hop_ms and the gpu tensor constant are keyed by link class: mac (Thunderbolt 5) | spark (the 200 Gb/s ports of a DGX Spark or GX10) | gpu (cards on PCIe inside one box)",
        "_sources": sorted({u for s in series for u in (s.get("source"), s.get("source_2")) if u}),
        "hop_ms": {
            "mac": fitted(hop_vals["mac"], 4.0, nd=1),
            "spark": fitted(hop_vals["spark"], 3.9, nd=1),
            "gpu": {"value": 1.0, "n": 0, "source": "assumed",
                    "note": "assumed: no published layer-split series across several cards in one box; PCIe 5 is faster than either measured link, so the hop is set below both"},
        },
        "tensor_c_ms": {
            "dense": fitted(c_vals["dense"], 4.5, nd=1),
            "moe": fitted(c_vals["moe"], 10.9, nd=1),
            "gpu": {"value": 1.5, "n": 0, "source": "assumed",
                    "note": "assumed: one MiniMax-M2.5 pair on RTX PRO 6000 Blackwell cards over PCIe 5 implies roughly this, and its quant changes with the card count, so it is quoted and not fitted"},
        },
        "prefill": {
            "layer_factor": layer_factor,
            "tensor_exponent": {"value": 0.75, "n": 0, "source": "assumed",
                                "note": ("assumed: tensor parallel divides the prompt work, so reading should scale with the count, but only one "
                                         "short-prompt point exists (vLLM on DGX Sparks, time to first word 233 ms on one machine and 149 ms on two, "
                                         "1.56 times); machines ^ 0.75 gives 1.68 at two machines, a little above that point")},
        },
        "dispatch_caveat": {
            "min_layers": 60,
            "note": ("a very deep mixture of experts can be dispatch-bound under tensor parallel: Kimi-K3 2.78T on four Mac Studio M3 Ultra wrote "
                     "2 tok/s where the bandwidth maths says about 36, because per-kernel dispatch overhead dominates on 92 layers with top-16 routing; "
                     "above this layer count the tool warns instead of promising the number"),
        },
    }

    factors = {
        "schema": "llm-sizer/factors v1 — derived by scripts/llm-sizer/calibrate.py from benchmarks.json; do not edit by hand",
        "data_as_of": bench.get("data_as_of"),
        "efficiency": {
            "_formula": "decode tok/s = bandwidth_gbs × efficiency ÷ gigabytes read per token (active weights at the quant's bits + context cache per token)",
            "_resolve": "exact chip if present in by_chip, else assumed[chip] if present, else the tier median (tier from the chip name: base | Pro | Max | Ultra; DGX Spark = spark)",
            "by_chip": dict(sorted(chips.items())),
            "assumed": assumed,
            "by_tier": tiers,
            "m5_generation_lift": m5_lift,
        },
        "runtime": RUNTIME_FACTORS,
        "speed_model": speed_model,
        "acceleration": acceleration,
        "prefill": prefill,
        "cluster": cluster,
        "feels_like": [{"max_tok_s": 5, "label": "painful", "description": "slower than reading"},
                       {"max_tok_s": 12, "label": "usable", "description": "usable for short answers"},
                       {"max_tok_s": 30, "label": "comfortable", "description": "comfortable chat"},
                       {"max_tok_s": 60, "label": "fast", "description": "agents and coding feel fine"},
                       {"max_tok_s": None, "label": "cloud-like", "description": "as fast as a hosted model"}],
        "sources": sorted({r["source"] for r in rows if r.get("source")} | {u for s in series for u in (s.get("source"), s.get("source_2")) if u}),
    }
    out = PUBLIC_DATA_DIR / "factors.json"
    text = json.dumps(factors, indent=1, ensure_ascii=False) + "\n"
    if args.report:
        print(render_report(speed_model, fit_report, profiles, tier_prof, bin_checks))
    if args.check:
        gate = speed_model["validation"]["gate"]
        if SPEED_MODEL_KIND == "floor" and not gate["passed"]:
            log(f"the fixed-cost speed model is active but its holdout gate fails: {gate}")
            return 1
        if out.exists() and out.read_text() == text:
            log("factors.json is current")
            return 0
        log("factors.json is stale — run calibrate.py")
        return 1
    out.write_text(text)
    log(f"wrote {out}")
    for t, v in tiers.items():
        log(f"  tier {t:5s} efficiency {v['value']}  (chips: {v['n_chips']}, range {v['range']})")
    for c, v in assumed.items():
        log(f"  assumed {c}: {v['value']}  — {v['note']}")
    for k, v in acceleration.items():
        if not k.startswith("_"):
            log(f"  {k}: {v['min']}–{v['max']}× (median {v['median']}, n={v['n']})")
    g = speed_model["validation"]
    log(f"  fixed-cost model ({SPEED_MODEL_KIND} active): holdout median err ×{round(2.718281828 ** g['holdout_median_err'], 3) if g['holdout_median_err'] is not None else '-'} on {g['n_holdout']} rows; gate {'passed' if g['gate']['passed'] else 'FAILED'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
