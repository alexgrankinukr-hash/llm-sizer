#!/usr/bin/env python3
"""Fit the fixed-cost speed model and judge it on rows it never saw.

Inputs   public/data/llm-sizer/benchmarks.json   (chip profiles: t0 and effective bandwidth, dense llama.cpp rows only)
         public/data/llm-sizer/validation.json   (architecture costs, runtime factors; train / holdout split)
         public/data/llm-sizer/machines.json     (spec bandwidth per chip, the denominator of the effective-bandwidth ratio)
Library  calibrate.py imports fit_floor_profiles / fit_speed_model and writes the result into factors.json.
CLI      python3 scripts/llm-sizer/fit_speed.py [--report] [--memory] [--emit-cases PATH]

Pure Python (the CI environment installs only requirements-dev.txt). Deterministic: the same inputs give the same bytes.
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
from speed_model import (ARCH_CLASSES, MLX_CONTEXT_GROUPS, MLX_GROUPS, arch_class, decode_ms, gb_per_token, legacy_tok_s, log_err, mlx_context_group,
                         mlx_group, platform_of, quant_family, read_factor)

# ---- gate and provisional values (METHODOLOGY §6.1) ---------------------------------------------------------
GATE_HOLDOUT_MEDIAN = math.log(1.25)      # median |ln(pred/meas)| on held-out A/B rows
GATE_NO_WORSE = math.log(1.30)            # no A/B row may get worse than the multiplier model by more than this
MIN_ROWS_TO_FIT = 2                       # a class or group with fewer rows keeps its provisional value, labelled assumed
PROVISIONAL_COST_MS = {"dense": 0.0, "moe": 3.0, "moe_hybrid": 7.0, "moe_latent": 17.0, "deepseek_v4": 30.0}
PROVISIONAL_MLX = {"dense": 1.0, "small_active_moe_m4plus": 0.5, "small_active_moe_pre_m4": 1.3, "moe_other": 0.9}
PROVISIONAL_READ_KQUANT = 1.25
# latent-attention decode has no clean long-context measurement yet (the one anecdote ran under memory pressure);
# until one exists the slope is the modest one measured on hybrid models, and the About page says so
PROVISIONAL_ATTN = {"gqa": 0.0, "mla": 5.0, "latent": 5.0}
PROVISIONAL_MLX_ATTN = 4.0                # per 32K, before any MLX long-context row is fitted; the M5 group falls back to the pre-M5 fit
CLAMP_READ = (1.0, 1.4)
CLAMP_MLX = (0.3, 1.6)
SHORT_CONTEXT = 8192
ASSUMED_PROFILES = {
    "M5 Ultra": {"b_eff_ratio": 0.85, "t0_ms": 1.5, "tier": "ultra", "generation": "M5",
                 "note": "unmeasured: 85 % of spec bandwidth (the Ultra tier's share) and the M5 generation's fixed cost; replace when measured"},
}


def chip_key(chip: str) -> str:
    return re.sub(r"\s*\(.*\)\s*$", "", chip).strip()


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
    return "base"


def generation_of(chip: str) -> str | None:
    m = re.match(r"^(M\d+)", chip.strip(), re.I)
    return m.group(1).upper() if m else None


def med(xs: list[float], nd: int = 3) -> float:
    return round(statistics.median(xs), nd)


def ols(points: list[tuple[float, float]]) -> tuple[float, float]:
    """Least squares y = a + b·x; two points give the exact line."""
    n = len(points)
    mx = sum(x for x, _ in points) / n
    my = sum(y for _, y in points) / n
    sxx = sum((x - mx) ** 2 for x, _ in points)
    sxy = sum((x - mx) * (y - my) for x, y in points)
    b = sxy / sxx
    return my - b * mx, b


# ---- chip profiles ---------------------------------------------------------------------------------------------
def spec_bandwidth(machines: list[dict]) -> dict[str, list[float]]:
    """The catalog's spec bandwidths per chip key (a chip can have several GPU bins: M4 Max 410 and 546)."""
    out: dict[str, set[float]] = defaultdict(set)
    for m in machines:
        out[chip_key(m["chip"])].add(float(m["bandwidth_gbs"]))
        if m.get("platform") == "cuda" and ("spark" in (m.get("family", "") + m["chip"]).lower()):
            out["DGX Spark"].add(float(m["bandwidth_gbs"]))
    return {k: sorted(v) for k, v in out.items()}


def catalog_bandwidth(chip: str, bw: float, spec_bw: dict[str, list[float]]) -> float:
    """The catalog figure a benchmark row's bandwidth stands for: the nearest bin within 5 % (800 → 819 for the M3 Ultra), else the row's own."""
    for cand in spec_bw.get(chip_key(chip), []):
        if abs(cand - bw) / cand <= 0.05:
            return cand
    return bw


def fit_floor_profiles(rows: list[dict], spec_bw: dict[str, list[float]]) -> tuple[dict, list[dict]]:
    """t0 (ms) and effective bandwidth per chip from dense llama.cpp rows at short context: 1000/tg against GB read.
    Returns (profiles by chip key, bin checks). The profile is the highest-bandwidth group of a chip; other GPU bins
    are fitted too and reported as a check that the ratio transfers."""
    groups: dict[tuple[str, float], list[dict]] = defaultdict(list)
    for r in rows:
        if r["runtime"] != "llama.cpp-gguf" or r["accel"] != "none" or not r.get("tg") or not r.get("weights_gb"):
            continue
        if r.get("active_weights_gb") or (r.get("context_tokens") or 0) > 4096:
            continue
        groups[(r["chip"], float(r["bandwidth_gbs"]))].append(r)
    fits: dict[tuple[str, float], dict] = {}
    for (chip, bw), rs in groups.items():
        if len({r["weights_gb"] for r in rs}) < 2:
            continue
        pts = [(float(r["weights_gb"]), 1000.0 / r["tg"]) for r in rs]
        t0, slope = ols(pts)
        notes: list[str] = []
        if t0 < 0:
            notes.append(f"intercept {t0:.2f} ms clamped to 0")
            t0 = 0.0
        b_eff = 1000.0 / slope
        spec = catalog_bandwidth(chip, bw, spec_bw)
        ratio = b_eff / spec
        if ratio > 1.05:
            notes.append(f"effective bandwidth {b_eff:.0f} GB/s exceeds the spec {spec:.0f}")
        errs = [abs(1000.0 / (t0 + slope * x) / (1000.0 / y) - 1) for x, y in pts]
        fits[(chip, bw)] = {"t0_ms": round(t0, 2), "b_eff_gbs": round(b_eff, 1), "b_eff_ratio": round(ratio, 3), "fit_n": len(rs),
                            "fit_max_err_pct": round(100 * max(errs), 1), "bandwidth_gbs": bw, "spec_bandwidth_gbs": spec, "notes": notes}
    profiles: dict[str, dict] = {}
    bin_checks: list[dict] = []
    for chip in {c for c, _ in fits}:
        bins = sorted([bw for c, bw in fits if c == chip], reverse=True)
        top = fits[(chip, bins[0])]
        profiles[chip] = top
        for bw in bins[1:]:
            own = fits[(chip, bw)]
            bin_checks.append({"chip": chip, "bandwidth_gbs": bw, "own_ratio": own["b_eff_ratio"], "profile_ratio": top["b_eff_ratio"],
                               "own_t0_ms": own["t0_ms"], "profile_t0_ms": top["t0_ms"]})
    return profiles, sorted(bin_checks, key=lambda b: (b["chip"], b["bandwidth_gbs"]))


def tier_profiles(profiles: dict) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for t in ("base", "pro", "max", "ultra", "spark"):
        rs = [p for c, p in profiles.items() if tier_of(c) == t]
        if rs:
            out[t] = {"b_eff_ratio": med([p["b_eff_ratio"] for p in rs]), "t0_ms": med([p["t0_ms"] for p in rs], 2), "n_chips": len(rs)}
    return out


def assumed_profiles(tiers: dict[str, dict]) -> dict[str, dict]:
    out = {k: dict(v) for k, v in ASSUMED_PROFILES.items()}
    if "base" in tiers:
        out["M6"] = {"b_eff_ratio": tiers["base"]["b_eff_ratio"], "t0_ms": tiers["base"]["t0_ms"], "tier": "base", "generation": "M6", "note": "base-tier medians until measured"}
    if "pro" in tiers:
        out["M6 Pro"] = {"b_eff_ratio": tiers["pro"]["b_eff_ratio"], "t0_ms": tiers["pro"]["t0_ms"], "tier": "pro", "generation": "M6", "note": "pro-tier medians until measured"}
    return out


def resolve_profile(chip: str, profiles: dict, assumed: dict, tiers: dict) -> tuple[dict | None, str]:
    k = chip_key(chip)
    if k in profiles:
        return profiles[k], "measured"
    if k in assumed:
        return assumed[k], "assumed"
    t = tiers.get(tier_of(k))
    return (t, "tier") if t else (None, "none")


# ---- the validation rows -----------------------------------------------------------------------------------------
def prepare(rows: list[dict], profiles: dict, assumed: dict, tiers: dict) -> list[dict]:
    out = []
    for r in rows:
        if r.get("kind") != "speed":
            continue
        prof, src = resolve_profile(r["chip"], profiles, assumed, tiers)
        if prof is None:
            continue
        arch = r["arch"]
        cls = arch_class(arch, arch.get("arch_cost_class"))
        rt = "mlx" if r["runtime"] in ("mlx", "ollama-mlx") else "gguf"
        fam = quant_family(r["quant_label"], "mlx" if rt == "mlx" else "gguf")
        gb = gb_per_token(r["active_params_b"], r["bits"], arch["kv_bytes_per_token_8bit"], r.get("kv_bits", 16), r["context_tokens"])
        out.append({**r, "_profile": prof, "_profile_source": src, "_cls": cls, "_rt": rt, "_fam": fam, "_gb": gb, "_platform": platform_of(r["chip"]),
                    "_group": mlx_group(cls, r["active_params_b"], r["chip"]), "_mlx_ctx": mlx_context_group(r["chip"]), "_attn": arch.get("attention") or "gqa",
                    "_is_moe": (arch.get("experts_total") or 0) > 1, "_meas_ms": 1000.0 / r["tg"]})
    return out


def check_split(rows: list[dict]) -> None:
    by_sub: dict[str, set[str]] = defaultdict(set)
    for r in rows:
        by_sub[r["submission_id"]].add(r["split"])
    bad = {s: sorted(v) for s, v in by_sub.items() if len(v - {"report"}) > 1}
    if bad:
        raise SystemExit(f"validation.json: submissions straddle the train/holdout split: {bad}")


def fitted(values: list[float], provisional: float, clamp: tuple[float, float] | None = None, nd: int = 2) -> dict:
    if len(values) >= MIN_ROWS_TO_FIT:
        v = statistics.median(values)
        if clamp:
            v = min(clamp[1], max(clamp[0], v))
        return {"value": round(v, nd), "n": len(values), "source": "fitted", "range": [round(min(values), nd), round(max(values), nd)]}
    return {"value": provisional, "n": len(values), "source": "assumed"}


def fit_speed_model(profiles: dict, assumed: dict, tiers: dict, validation: dict, legacy: dict | None) -> tuple[dict, dict]:
    """Coordinate-descent fit of the architecture costs and runtime factors on the train split; evaluation on every split.
    Returns (speed_model block for factors.json, report dict)."""
    rows = prepare(validation["rows"], profiles, assumed, tiers)
    check_split(validation["rows"])
    train = [r for r in rows if r["split"] == "train" and r["grade"] in ("A", "B")]
    sm = {
        "read_factor": {f: {"value": 1.0, "source": "definition"} for f in ("q4_0", "q8_0", "f16", "mxfp4", "mlx")},
        "architecture_cost_ms": {c: {"value": PROVISIONAL_COST_MS[c], "n": 0, "source": "assumed"} for c in ARCH_CLASSES},
        "attention_ms_per_32k": {a: {"value": PROVISIONAL_ATTN[a], "n": 0, "source": "assumed"} for a in ("gqa", "mla", "latent")},
        "mlx": {"overhead_factor": {g: {"value": PROVISIONAL_MLX[g], "n": 0, "source": "assumed"} for g in MLX_GROUPS},
                "attention_ms_per_32k": {g: {"value": PROVISIONAL_MLX_ATTN, "n": 0, "source": "assumed"} for g in MLX_CONTEXT_GROUPS}},
    }
    sm["read_factor"]["gguf_kquant"] = {"value": PROVISIONAL_READ_KQUANT, "n": 0, "source": "assumed"}
    sm["architecture_cost_ms"]["dense"] = {"value": 0.0, "n": 0, "source": "definition"}
    sm["attention_ms_per_32k"]["gqa"] = {"value": 0.0, "n": 0, "source": "definition"}
    sm["mlx"]["overhead_factor"]["dense"] = {"value": 1.0, "n": 0, "source": "definition"}

    def parts(r: dict) -> tuple[float, float, float, float]:
        p = r["_profile"]
        b_eff = p["b_eff_ratio"] * r["bandwidth_gbs"]
        read1 = r["_gb"] / b_eff * 1000  # read at factor 1.0
        return p["t0_ms"], read1, sm["architecture_cost_ms"][r["_cls"]]["value"], r["context_tokens"] / 32768

    for _ in range(3):
        # 1. K-quant read penalty: dense GGUF K-quant rows at short context, on Metal (no CUDA K-quant row exists,
        #    so the factor is scoped to Apple silicon and the Spark reads K-quants at the plain rate)
        vals = []
        for r in train:
            if r["_rt"] == "gguf" and r["_fam"] == "gguf_kquant" and r["_cls"] == "dense" and r["context_tokens"] <= SHORT_CONTEXT and r["_platform"] == "apple":
                t0, read1, _, _ = parts(r)
                vals.append((r["_meas_ms"] - t0) / read1)
        sm["read_factor"]["gguf_kquant"] = {**fitted(vals, PROVISIONAL_READ_KQUANT, CLAMP_READ), "platforms": ["apple"]}
        # 2. architecture costs: GGUF rows of each class at short context
        for cls in ARCH_CLASSES:
            if cls == "dense":
                continue
            vals = []
            for r in train:
                if r["_rt"] == "gguf" and r["_cls"] == cls and r["context_tokens"] <= SHORT_CONTEXT:
                    t0, read1, _, _ = parts(r)
                    vals.append(max(0.0, r["_meas_ms"] - read1 * read_factor(sm, r["_fam"], r["_platform"]) - t0))
            sm["architecture_cost_ms"][cls] = fitted(vals, PROVISIONAL_COST_MS[cls], nd=1)
        # 3. attention cost per 32K for latent attention: GGUF rows past the short context
        for attn in ("mla", "latent"):
            vals = []
            for r in train:
                if r["_rt"] == "gguf" and r["_attn"] == attn and r["context_tokens"] > SHORT_CONTEXT:
                    t0, read1, cost, units = parts(r)
                    vals.append(max(0.0, (r["_meas_ms"] - read1 * read_factor(sm, r["_fam"], r["_platform"]) - t0 - cost) / units))
            sm["attention_ms_per_32k"][attn] = fitted(vals, PROVISIONAL_ATTN[attn], nd=1)
        # 4. MLX overhead per group: MLX rows at short context, factor on the fixed cost
        for g in MLX_GROUPS:
            if g == "dense":
                continue
            vals = []
            for r in train:
                if r["_rt"] == "mlx" and r["_group"] == g and r["context_tokens"] <= SHORT_CONTEXT:
                    t0, read1, cost, _ = parts(r)
                    if t0 + cost > 0:
                        vals.append((r["_meas_ms"] - read1) / (t0 + cost))
            sm["mlx"]["overhead_factor"][g] = fitted(vals, PROVISIONAL_MLX[g], CLAMP_MLX)
        # the dense MLX factor is fitted too, so a dense MLX bonus can only appear where the rows show one
        vals = []
        for r in train:
            if r["_rt"] == "mlx" and r["_group"] == "dense" and r["context_tokens"] <= SHORT_CONTEXT:
                t0, read1, cost, _ = parts(r)
                if t0 + cost > 0:
                    vals.append((r["_meas_ms"] - read1) / (t0 + cost))
        sm["mlx"]["overhead_factor"]["dense"] = fitted(vals, 1.0, CLAMP_MLX)
        # 5. MLX attention cost per 32K, per chip generation: MLX rows past the short context. The M5 GPU generation
        #    pays a different long-context cost than M1 to M4 (the oMLX M5 Max ladders), so each group is fitted on
        #    its own rows; a group with too few rows takes the pre-M5 value and says so.
        for ctx_group in MLX_CONTEXT_GROUPS:
            vals = []
            for r in train:
                if r["_rt"] == "mlx" and r["context_tokens"] > SHORT_CONTEXT and r["_mlx_ctx"] == ctx_group:
                    t0, read1, cost, units = parts(r)
                    fixed = (t0 + cost) * sm["mlx"]["overhead_factor"][r["_group"]]["value"]
                    attn = sm["attention_ms_per_32k"][r["_attn"] if r["_attn"] in sm["attention_ms_per_32k"] else "gqa"]["value"] * units
                    vals.append(max(0.0, (r["_meas_ms"] - read1 - fixed - attn) / units))
            sm["mlx"]["attention_ms_per_32k"][ctx_group] = fitted(vals, PROVISIONAL_MLX_ATTN, nd=1)
        if sm["mlx"]["attention_ms_per_32k"]["m5plus"]["source"] == "assumed":
            pre = sm["mlx"]["attention_ms_per_32k"]["pre_m5"]
            sm["mlx"]["attention_ms_per_32k"]["m5plus"] = {"value": pre["value"], "n": 0, "source": "assumed", "note": "no M5-generation MLX row past 8K yet; the pre-M5 value applies"}

    # ---- evaluation ---------------------------------------------------------------------------------------------
    evaluated = []
    for r in rows:
        pred_ms = decode_ms(r["_profile"], r["bandwidth_gbs"], r["_gb"], r["_fam"], r["_cls"], r["_rt"], r["_group"], r["_attn"], r["context_tokens"], sm, r["_platform"], r["_mlx_ctx"])
        pred = 1000.0 / pred_ms
        old = None
        if legacy:
            eff = legacy_efficiency(r["chip"], legacy)
            old = legacy_tok_s(eff, r["bandwidth_gbs"], r["_gb"], r["_rt"], r["_is_moe"], r["active_params_b"], r["context_tokens"], legacy["runtime"])
        e_new, e_old = log_err(pred, r["tg"]), log_err(old, r["tg"])
        evaluated.append({"id": r["id"], "split": r["split"], "grade": r["grade"], "chip": r["chip"], "model": r["model"], "quant": r["quant_label"],
                          "runtime": r["runtime"], "context": r["context_tokens"], "class": r["_cls"], "group": r["_group"] if r["_rt"] == "mlx" else None,
                          "measured": r["tg"], "floor": round(pred, 1), "multiplier": round(old, 1) if old else None,
                          "err_floor": round(e_new, 3) if e_new is not None else None, "err_multiplier": round(e_old, 3) if e_old is not None else None,
                          "profile_source": r["_profile_source"]})
    def summary(rs: list[dict], key: str) -> dict:
        es = [x[key] for x in rs if x[key] is not None]
        if not es:
            return {"n": 0}
        es_sorted = sorted(es)
        p90 = es_sorted[min(len(es) - 1, int(math.ceil(0.9 * len(es))) - 1)]
        return {"n": len(es), "median": round(statistics.median(es), 3), "p90": round(p90, 3),
                "median_factor": round(math.exp(statistics.median(es)), 3), "p90_factor": round(math.exp(p90), 3)}
    ab = [x for x in evaluated if x["grade"] in ("A", "B")]
    holdout = [x for x in ab if x["split"] == "holdout"]
    train_e = [x for x in ab if x["split"] == "train"]
    worse = [x for x in ab if x["split"] in ("train", "holdout") and x["err_floor"] is not None and x["err_multiplier"] is not None
             and x["err_floor"] - x["err_multiplier"] > GATE_NO_WORSE]
    s_hold, s_train = summary(holdout, "err_floor"), summary(train_e, "err_floor")
    gate = {"holdout_median_max": round(GATE_HOLDOUT_MEDIAN, 3), "no_worse_than_multiplier_by": round(GATE_NO_WORSE, 3),
            "holdout_median": s_hold.get("median"), "rows_worse": [x["id"] for x in worse],
            "passed": bool(s_hold.get("n", 0) >= 12 and s_hold.get("median", 9) <= GATE_HOLDOUT_MEDIAN and not worse)}
    by_class = {c: summary([x for x in ab if x["class"] == c], "err_floor") for c in ARCH_CLASSES}
    by_runtime = {rt: summary([x for x in ab if x["runtime"] == rt], "err_floor") for rt in sorted({x["runtime"] for x in ab})}
    by_contrib = {}
    for c in sorted({r["contributor"] for r in rows}):
        by_contrib[c] = summary([x for x in ab if next(r for r in rows if r["id"] == x["id"])["contributor"] == c], "err_floor")
    validation_block = {
        "holdout_rule": validation.get("holdout_rule"), "n_train": s_train.get("n", 0), "n_holdout": s_hold.get("n", 0),
        "train_median_err": s_train.get("median"), "train_p90_err": s_train.get("p90"),
        "holdout_median_err": s_hold.get("median"), "holdout_p90_err": s_hold.get("p90"),
        "multiplier_holdout_median_err": summary(holdout, "err_multiplier").get("median"),
        "multiplier_train_median_err": summary(train_e, "err_multiplier").get("median"),
        "gate": gate, "fitted_at": validation.get("data_as_of"),
    }
    sm_out = {**sm, "validation": validation_block}
    report = {"rows": evaluated, "by_class": by_class, "by_runtime": by_runtime, "by_contributor": by_contrib,
              "train": s_train, "holdout": s_hold, "train_multiplier": summary(train_e, "err_multiplier"), "holdout_multiplier": summary(holdout, "err_multiplier"), "gate": gate}
    return sm_out, report


def legacy_efficiency(chip: str, legacy: dict) -> float | None:
    """The multiplier model's efficiency for a chip: by_chip → assumed → tier (speed.ts:resolveEfficiency)."""
    k = chip_key(chip)
    e = legacy["efficiency"]
    if k in e["by_chip"]:
        return e["by_chip"][k]["value"]
    if k in e["assumed"]:
        return e["assumed"][k]["value"]
    t = e["by_tier"].get(tier_of(k))
    return t["value"] if t else None


# ---- memory rows: informational residuals of the policy margins ---------------------------------------------------
def memory_residuals(validation: dict, policy: dict) -> list[dict]:
    out = []
    for r in validation["rows"]:
        if r.get("kind") != "memory" or r.get("split") == "report":
            continue
        a = r["arch"]
        weights = r.get("weights_gb")
        if weights is None:
            weights = r["active_params_b"] * r["bits"] / 8 if (a.get("experts_total") or 0) <= 1 else None
        if weights is None:
            continue
        cache = a["kv_bytes_per_token_8bit"] * (r.get("kv_bits", 16) / 8) * r["context_tokens"] / 1e9
        buffers = max(policy["bufferMinGb"], policy.get("bufferBaseGb", 0.0) + policy["bufferPct"] * weights)
        if policy.get("bufferCapGb"):
            buffers = min(policy["bufferCapGb"], buffers)
        scratch = policy["mlxScratchGbPer1k"] * max(0, r["context_tokens"] - policy["mlxScratchFromTokens"]) / 1024 if r["runtime"] in ("mlx", "ollama-mlx") else 0.0
        pred = weights + cache + buffers + scratch
        out.append({"id": r["id"], "split": r["split"], "context": r["context_tokens"], "measured_gb": r["peak_gb"], "predicted_gb": round(pred, 1),
                    "delta_gb": round(pred - r["peak_gb"], 1)})
    return out


# ---- report -----------------------------------------------------------------------------------------------------
def render_report(sm: dict, report: dict, profiles: dict, tiers: dict, bin_checks: list[dict]) -> str:
    L = []
    L.append("# Floor speed model: fit report\n")
    L.append("## Chip profiles (from benchmarks.json)\n")
    L.append("| chip | effective GB/s | share of spec | t0 ms | rows | max err |")
    L.append("|---|---:|---:|---:|---:|---:|")
    for c, p in sorted(profiles.items()):
        L.append(f"| {c} | {p['b_eff_gbs']} | {p['b_eff_ratio']:.2f} | {p['t0_ms']} | {p['fit_n']} | {p['fit_max_err_pct']} % |")
    L.append("\nTier medians: " + ", ".join(f"{t}: {v['b_eff_ratio']:.2f} / {v['t0_ms']} ms ({v['n_chips']} chips)" for t, v in tiers.items()))
    if bin_checks:
        L.append("\nOther GPU bins (ratio transfers across bins when own ≈ profile): " + "; ".join(f"{b['chip']} @{b['bandwidth_gbs']:.0f}: own {b['own_ratio']:.2f}/{b['own_t0_ms']} ms vs profile {b['profile_ratio']:.2f}/{b['profile_t0_ms']} ms" for b in bin_checks))
    L.append("\n## Fitted terms\n")
    L.append("| term | value | n | source | range |")
    L.append("|---|---:|---:|---|---|")
    for k, v in sm["read_factor"].items():
        L.append(f"| read factor {k} | {v['value']} | {v.get('n', '')} | {v['source']} | {v.get('range', '')} |")
    for k, v in sm["architecture_cost_ms"].items():
        L.append(f"| architecture cost {k} (ms) | {v['value']} | {v.get('n', '')} | {v['source']} | {v.get('range', '')} |")
    for k, v in sm["attention_ms_per_32k"].items():
        L.append(f"| attention cost {k} (ms per 32K) | {v['value']} | {v.get('n', '')} | {v['source']} | {v.get('range', '')} |")
    for k, v in sm["mlx"]["overhead_factor"].items():
        L.append(f"| MLX fixed-cost factor {k} | {v['value']} | {v.get('n', '')} | {v['source']} | {v.get('range', '')} |")
    for k, v in sm["mlx"]["attention_ms_per_32k"].items():
        L.append(f"| MLX attention cost {k} (ms per 32K) | {v['value']} | {v.get('n', '')} | {v['source']} | {v.get('range', '')} |")
    g = report["gate"]
    L.append("\n## Errors (|ln(predicted ÷ measured)|, shown as a factor)\n")
    for name, s, m in (("train", report["train"], report["train_multiplier"]), ("holdout", report["holdout"], report["holdout_multiplier"])):
        L.append(f"- **{name}** ({s.get('n', 0)} rows): floor median ×{s.get('median_factor', '-')} (p90 ×{s.get('p90_factor', '-')}); multiplier median ×{m.get('median_factor', '-')} (p90 ×{m.get('p90_factor', '-')})")
    L.append(f"- **gate**: holdout median ≤ ×1.25 and no A/B row worse than the multiplier by more than ×1.30 → **{'PASSED' if g['passed'] else 'FAILED'}**" + (f"; rows worse: {', '.join(g['rows_worse'])}" if g["rows_worse"] else ""))
    L.append("\n| class | n | floor median | p90 |")
    L.append("|---|---:|---:|---:|")
    for c, s in report["by_class"].items():
        L.append(f"| {c} | {s.get('n', 0)} | ×{s.get('median_factor', '-')} | ×{s.get('p90_factor', '-')} |")
    L.append("\n| runtime | n | floor median | p90 |")
    L.append("|---|---:|---:|---:|")
    for c, s in report["by_runtime"].items():
        L.append(f"| {c} | {s.get('n', 0)} | ×{s.get('median_factor', '-')} | ×{s.get('p90_factor', '-')} |")
    L.append("\n| contributor | n | floor median | p90 |")
    L.append("|---|---:|---:|---:|")
    for c, s in report["by_contributor"].items():
        if s.get("n"):
            L.append(f"| {c} | {s['n']} | ×{s['median_factor']} | ×{s['p90_factor']} |")
    L.append("\n## Every row\n")
    L.append("| id | split | chip | model · quant · runtime | ctx | measured | floor | multiplier | class |")
    L.append("|---|---|---|---|---:|---:|---:|---:|---|")
    for x in report["rows"]:
        L.append(f"| {x['id']} | {x['split']} | {x['chip']} | {x['model']} · {x['quant']} · {x['runtime']} | {x['context']} | {x['measured']} | {x['floor']} | {x['multiplier'] if x['multiplier'] is not None else '-'} | {x['class']}{(' / ' + x['group']) if x['group'] else ''} |")
    return "\n".join(L) + "\n"


def emit_cases(profiles: dict, assumed: dict, tiers: dict, sm: dict, validation: dict, path: str) -> None:
    """Parity fixtures for the TypeScript engine: inputs and the expected milliseconds to three decimals."""
    rows = prepare(validation["rows"], profiles, assumed, tiers)
    picks = ["llamacpp-15396-m4max-001", "spark-b7941-gpt-oss-120b-011", "zachrattner-m1ultra-qwen27b-q4km-032", "zachrattner-m1ultra-qwen27b-mlx4bit-036",
             "antek-m4max-qwen35-042", "fioravanti-m3ultra-qwen397-4bit-046", "mlx3209-m3ultra-qwen32b-058", "rentamac-m3ultra-kimi-gguf-097",
             "rentamac-m5max-dsv4-bench-093", "kizai-m3ultra-glm52-090", "hiesch-m3max-gemma26-071", "spark-b7941-qwen2.5-coder-7b-025",
             # the M5-generation MLX context cost: a 64K row on the M5 Max and the pre-M5 sweep row at the same depth
             "omlx-32bf9vtv-m5max-qwen27-079", "omlx-u79n8shx-m5max-nemotron-081"]
    cases = []
    for r in rows:
        if r["id"] not in picks:
            continue
        ms = decode_ms(r["_profile"], r["bandwidth_gbs"], r["_gb"], r["_fam"], r["_cls"], r["_rt"], r["_group"], r["_attn"], r["context_tokens"], sm, r["_platform"], r["_mlx_ctx"])
        cases.append({"id": r["id"], "chip": r["chip"], "platform": r["_platform"], "bandwidth_gbs": r["bandwidth_gbs"], "profile": {"t0_ms": r["_profile"]["t0_ms"], "b_eff_ratio": r["_profile"]["b_eff_ratio"]},
                      "active_params_b": r["active_params_b"], "bits": r["bits"], "kv_bytes_per_token_8bit": r["arch"]["kv_bytes_per_token_8bit"], "kv_bits": r.get("kv_bits", 16),
                      "context_tokens": r["context_tokens"], "quant_label": r["quant_label"], "runtime": r["_rt"], "arch_class": r["_cls"], "mlx_group": r["_group"], "mlx_ctx": r["_mlx_ctx"],
                      "attention": r["_attn"], "gb_per_token": round(r["_gb"], 6), "expected_ms": round(ms, 3), "expected_tok_s": round(1000 / ms, 2)})
    with open(path, "w") as f:
        json.dump({"schema": "llm-sizer/speed-cases v1 — parity fixtures emitted by fit_speed.py --emit-cases; the engine's speed-parity test must reproduce expected_ms", "cases": cases}, f, indent=1)
    log(f"wrote {len(cases)} parity cases to {path}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--memory", action="store_true")
    ap.add_argument("--emit-cases", metavar="PATH")
    args = ap.parse_args()
    bench = json.loads((PUBLIC_DATA_DIR / "benchmarks.json").read_text())
    validation = json.loads((PUBLIC_DATA_DIR / "validation.json").read_text())
    machines = json.loads((PUBLIC_DATA_DIR / "machines.json").read_text())["machines"]
    legacy = json.loads((PUBLIC_DATA_DIR / "factors.json").read_text()) if (PUBLIC_DATA_DIR / "factors.json").exists() else None
    profiles, bin_checks = fit_floor_profiles(bench["rows"], spec_bandwidth(machines))
    tiers = tier_profiles(profiles)
    assumed = assumed_profiles(tiers)
    sm, report = fit_speed_model(profiles, assumed, tiers, validation, legacy)
    if args.report or not (args.memory or args.emit_cases):
        print(render_report(sm, report, profiles, tiers, bin_checks))
    if args.memory:
        from calibrate import MEMORY_POLICY
        for row in memory_residuals(validation, MEMORY_POLICY):
            print(row)
    if args.emit_cases:
        emit_cases(profiles, assumed, tiers, sm, validation, args.emit_cases)
    return 0


if __name__ == "__main__":
    sys.exit(main())
