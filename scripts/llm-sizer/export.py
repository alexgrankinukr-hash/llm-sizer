#!/usr/bin/env python3
"""Nightly export: Postgres -> the JSON the tool page reads (`index` + one detail record per model).

Writes every record as `staged`, runs the guardrails against the current `live` set, promotes in one
transaction when they pass, and writes the same files to --out for the public data mirror.

Usage: python3 scripts/llm-sizer/export.py [--out scripts/llm-sizer/out] [--force-promote] [--no-promote] [--dry-run]
Exit codes: 0 promoted (or nothing to do) · 2 guardrails blocked promotion · 1 error
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

import psycopg2.extras

from common import HERE, OUT_DIR, connect, load_env, log

SCHEMA_VERSION = "llm-sizer/models v1"
REF_4BIT = ["Q4_K_M", "UD-Q4_K_XL", "UD-Q4_K_M", "Q4_K_S", "IQ4_XS", "MLX-4bit", "Q4_0", "MXFP4", "MXFP4_MOE", "MLX-MXFP4", "NVFP4", "MLX-NVFP4"]
REF_8BIT = ["Q8_0", "MLX-8bit", "UD-Q8_K_XL", "FP8", "MLX-MXFP8", "MXFP8"]
REF_MLX_4BIT = ["MLX-4bit", "MLX-MXFP4", "MLX-NVFP4"]
QUANTIZER_PREFERENCE = ["lmstudio-community", "unsloth", "bartowski", "ggml-org", "mlx-community", "official"]


def canonical(obj) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str)


def sha_of(obj) -> str:
    return hashlib.sha256(canonical(obj).encode()).hexdigest()[:16]


def gb(n: int | float | None) -> float | None:
    return None if n is None else round(float(n) / 1e9, 2)


def pick_reference(quants: list[dict], labels: list[str]) -> dict | None:
    for lab in labels:
        rows = [q for q in quants if q["label"] == lab]
        if rows:
            rows.sort(key=lambda q: QUANTIZER_PREFERENCE.index(q["quantizer"]) if q["quantizer"] in QUANTIZER_PREFERENCE else 99)
            return rows[0]
    return None


def load_rows(conn) -> tuple[list[dict], dict[str, dict], dict[str, list[dict]]]:
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute("SELECT * FROM llm_sizer.models WHERE status <> 'hidden' ORDER BY featured DESC, featured_rank NULLS LAST, id")
        models = [dict(r) for r in cur.fetchall()]
        cur.execute("SELECT * FROM llm_sizer.model_architecture")
        arch = {r["model_id"]: dict(r) for r in cur.fetchall()}
        cur.execute("SELECT * FROM llm_sizer.model_quants WHERE withdrawn_at IS NULL ORDER BY size_bytes")
        quants: dict[str, list[dict]] = {}
        for r in cur.fetchall():
            quants.setdefault(r["model_id"], []).append(dict(r))
    return models, arch, quants


SPECIAL_BUILD_FORMATS = {"gguf", "mlx"}


def special_builds(ov: dict, model_id: str) -> list[dict]:
    """The hand-maintained builds of a model, checked for shape: a label, an in-memory size, an on-SSD size, a known format.

    They are written by hand in models.overrides.json and the app reads them without parsing, so a missing number would
    surface as a blank cell; fail the export instead."""
    out = []
    for i, b in enumerate(ov.get("special_builds", []) or []):
        where = f"{model_id} special_builds[{i}]"
        if not isinstance(b, dict) or not isinstance(b.get("label"), str) or not b["label"].strip():
            raise ValueError(f"{where}: needs a label")
        for key in ("resident_gb", "ssd_gb"):
            v = b.get(key)
            if not isinstance(v, (int, float)) or isinstance(v, bool) or v < 0:
                raise ValueError(f"{where}: {key} must be a non-negative number")
        if b.get("format") not in SPECIAL_BUILD_FORMATS:
            raise ValueError(f"{where}: format must be one of {sorted(SPECIAL_BUILD_FORMATS)}")
        if "disk_gb" in b and (not isinstance(b["disk_gb"], (int, float)) or b["disk_gb"] + 1e-6 < b["resident_gb"]):
            raise ValueError(f"{where}: disk_gb must be a number at least resident_gb")
        out.append(dict(b))
    return out


def build_detail(m: dict, a: dict | None, qs: list[dict], ov: dict, generated_at: str) -> dict:
    arch_out = None
    if a:
        arch_out = {k: v for k, v in a.items() if k not in ("config", "model_id", "updated_at")}
        arch_out["kv_cache_gb_at_128k_8bit"] = round((a["full_bytes_per_token_8bit"] * 131072
                                                     + a["sliding_bytes_per_token_8bit"] * min(131072, a["window_size"] or 131072)
                                                     + a["linear_state_bytes"]) / 1e9, 2)
        # a hand-set class for the speed model's fixed cost (METHODOLOGY §6.1); the engine derives the class otherwise
        if ov.get("arch_cost_class"):
            arch_out["arch_cost_class"] = ov["arch_cost_class"]
    quants_out = [{
        "repo": q["repo_id"], "url": f"https://huggingface.co/{q['repo_id']}", "format": q["format"], "label": q["label"],
        "bits": q["bits_effective"], "size_gb": q["size_gb"], "size_bytes": int(q["size_bytes"]), "quantizer": q["quantizer"],
        "file_count": q["file_count"], "qat": q["qat"], "first_seen": q["first_seen"].date().isoformat(),
    } for q in qs]
    def ref(q: dict | None) -> dict | None:
        return {"repo": q["repo_id"], "label": q["label"]} if q else None
    native = next((q for q in qs if q["repo_id"] == m["hf_repo"] and q["format"] == "safetensors"), None)
    reference = {"q4": ref(pick_reference(qs, REF_4BIT)), "q8": ref(pick_reference(qs, REF_8BIT)),
                 "mlx4": ref(pick_reference(qs, REF_MLX_4BIT)), "native": ref(native)}
    return {
        "schema": SCHEMA_VERSION, "id": m["id"], "name": m["name"], "provider": m["provider"], "hf_repo": m["hf_repo"],
        "url": f"https://huggingface.co/{m['hf_repo']}", "variant": m["variant"], "released": m["released"].isoformat() if m["released"] else None,
        "params_total": m["params_total"], "params_active": m["params_active"], "params_active_source": m["params_active_source"],
        "context_max": m["context_max"], "native_bits": m["native_bits"],
        "license": {"name": m["license"], "url": m["license_url"], "commercial": m["license_commercial"], "note": m["license_note"]},
        "tags": m["tags"], "featured": m["featured"], "tier": m["featured_tier"], "rank": m["featured_rank"], "gated": m["gated"],
        "status": m["status"], "superseded_by": m["superseded_by"], "notes": m["notes"], "four_bit_native": bool(ov.get("four_bit_native")),
        "architecture": arch_out, "quants": quants_out, "reference": reference, "special_builds": special_builds(ov, m["id"]),
        "drafts": {"dflash": a.get("accel_dflash_repo") if a else None, "dspark": a.get("accel_dspark_repo") if a else None},
        "timestamps": {"first_seen": m["first_seen"].isoformat(), "last_checked": m["last_checked"].isoformat(), "last_changed": m["last_changed"].isoformat()},
        "generated_at": generated_at,
    }


def build_index_entry(m: dict, a: dict | None, qs: list[dict], ov: dict, detail_sha: str) -> dict:
    q4 = pick_reference(qs, REF_4BIT)
    q8 = pick_reference(qs, REF_8BIT)
    native = next((q for q in qs if q["repo_id"] == m["hf_repo"] and q["format"] == "safetensors"), None)
    return {
        "id": m["id"], "name": m["name"], "provider": m["provider"], "hf_repo": m["hf_repo"],
        "released": m["released"].isoformat() if m["released"] else None,
        "params_total_b": gb(m["params_total"]), "params_active_b": gb(m["params_active"]), "context_max": m["context_max"],
        "featured": m["featured"], "tier": m["featured_tier"], "rank": m["featured_rank"], "gated": m["gated"], "status": m["status"],
        "license": m["license"], "commercial": m["license_commercial"],
        "arch": a["attention"] if a else "assumed", "hybrid": bool(a and (a["linear_layers"] or a["sliding_layers"])),
        "moe": bool(a and a["experts_total"]), "assumed": bool(a is None or a["conventional_assumed"]),
        "accel": {"mtp": bool(a and a["accel_mtp"]), "dflash": bool(a and a["accel_dflash_repo"]), "dspark": bool(a and a["accel_dspark_repo"])},
        "kv_bytes_per_token_8bit": (a["full_bytes_per_token_8bit"] if a else None),
        "sizes": {"min_gb": qs[0]["size_gb"] if qs else None, "q4_gb": q4["size_gb"] if q4 else None,
                  "q8_gb": q8["size_gb"] if q8 else None, "native_gb": native["size_gb"] if native else None},
        "formats": sorted({q["format"] for q in qs}), "quant_count": len(qs), "four_bit_native": bool(ov.get("four_bit_native")),
        "has_special_build": bool(ov.get("special_builds")),
        "detail_sha": detail_sha,
    }


def guardrail_problems(live_index: dict | None, live_details: dict[str, dict], staged_index: dict, staged_details: dict[str, dict]) -> list[str]:
    """Pure comparison of a staged export against the live one. Empty list = safe to promote."""
    problems: list[str] = []
    if not live_index:
        return problems  # first export: nothing to compare against
    live_models = {m["id"]: m for m in live_index.get("models", [])}
    staged_models = {m["id"]: m for m in staged_index["models"]}
    if len(staged_models) < 0.9 * len(live_models):
        problems.append(f"index shrank from {len(live_models)} to {len(staged_models)} models")
    for mid, lm in live_models.items():
        featured = bool(lm.get("featured"))
        s = staged_models.get(mid)
        if not s:
            if featured:
                problems.append(f"featured model {mid} disappeared")
            continue
        if featured and not s["sizes"]["min_gb"]:
            problems.append(f"featured model {mid} has no quants")
        ld, sd = live_details.get(mid), staged_details.get(mid)
        if not ld or not sd:
            continue
        lq = {(q["repo"], q["label"]): q["size_bytes"] for q in ld.get("quants", [])}
        sq = {(q["repo"], q["label"]): q["size_bytes"] for q in sd.get("quants", [])}
        vanished = [k for k in lq if k not in sq]
        if len(vanished) > 2:
            problems.append(f"{mid}: {len(vanished)} quants vanished (e.g. {vanished[0][0]} {vanished[0][1]})")
        for k, old in lq.items():
            new = sq.get(k)
            if new and old and abs(new - old) / old > 0.10:
                problems.append(f"{mid}: {k[0]} {k[1]} size moved {old / 1e9:.1f} -> {new / 1e9:.1f} GB")
        la, sa = (ld.get("architecture") or {}).get("source"), (sd.get("architecture") or {}).get("source")
        if la in ("config", "override") and sa == "assumed":
            problems.append(f"{mid}: architecture regressed to assumed")
    return problems


def guardrails(conn, staged_index: dict, staged_details: dict[str, dict]) -> list[str]:
    with conn.cursor() as cur:
        cur.execute("SELECT body FROM llm_sizer.exports WHERE name='index' AND status='live'")
        row = cur.fetchone()
    if not row:
        return []
    live_index = row[0]
    live_ids = [m["id"] for m in live_index.get("models", [])]
    with conn.cursor() as cur:
        cur.execute("SELECT name, body FROM llm_sizer.exports WHERE status='live' AND name = ANY(%s)", ([f"model:{mid}" for mid in live_ids],))
        live_details = {r[0].split(":", 1)[1]: r[1] for r in cur.fetchall()}
    return guardrail_problems(live_index, live_details, staged_index, staged_details)


def write_files(out: Path, index: dict, details: dict[str, dict]) -> None:
    out.mkdir(parents=True, exist_ok=True)
    models_dir = out / "models"
    if models_dir.exists():
        shutil.rmtree(models_dir)
    models_dir.mkdir()
    (out / "index.json").write_text(json.dumps(index, indent=1, ensure_ascii=False, default=str) + "\n")
    for mid, d in details.items():
        (models_dir / f"{mid}.json").write_text(json.dumps(d, indent=1, ensure_ascii=False, default=str) + "\n")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=str(OUT_DIR))
    ap.add_argument("--force-promote", action="store_true", help="promote even if the guardrails object")
    ap.add_argument("--no-promote", action="store_true", help="stage and write files only")
    ap.add_argument("--dry-run", action="store_true", help="build and write files, touch nothing in the database")
    args = ap.parse_args()

    overrides = json.loads((HERE / "models.overrides.json").read_text()).get("models", {})
    conn = connect(load_env()["DATABASE_URL"])
    generated_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    models, arch, quants = load_rows(conn)
    if not models:
        log("no models in the database; nothing to export")
        return 0
    with conn.cursor() as cur:
        cur.execute("SELECT id FROM llm_sizer.discovery_runs ORDER BY id DESC LIMIT 1")
        row = cur.fetchone()
        run_id = row[0] if row else None

    details: dict[str, dict] = {}
    entries: list[dict] = []
    for m in models:
        a = arch.get(m["id"])
        qs = quants.get(m["id"], [])
        ov = overrides.get(m["id"], {})
        d = build_detail(m, a, qs, ov, generated_at)
        d["sha"] = sha_of({k: v for k, v in d.items() if k != "generated_at"})
        details[m["id"]] = d
        entries.append(build_index_entry(m, a, qs, ov, d["sha"]))
    index = {"schema": SCHEMA_VERSION, "generated_at": generated_at, "count": len(entries),
             "featured_count": sum(1 for e in entries if e["featured"]), "models": entries}
    index["sha"] = sha_of({k: v for k, v in index.items() if k != "generated_at"})
    log(f"built index with {len(entries)} models ({index['featured_count']} featured), {sum(len(v) for v in quants.values())} quant rows")

    out = Path(args.out)
    write_files(out, index, details)
    log(f"wrote {out / 'index.json'} and {len(details)} detail files")
    if args.dry_run:
        return 0

    # stage
    with conn, conn.cursor() as cur:
        cur.execute("DELETE FROM llm_sizer.exports WHERE status='staged'")
        rows = [("index", "staged", psycopg2.extras.Json(index), index["sha"], run_id)]
        rows += [(f"model:{mid}", "staged", psycopg2.extras.Json(d), d["sha"], run_id) for mid, d in details.items()]
        psycopg2.extras.execute_batch(cur, "INSERT INTO llm_sizer.exports (name, status, body, sha, run_id) VALUES (%s,%s,%s,%s,%s)", rows, page_size=200)
    log(f"staged {len(rows)} export rows")

    problems = guardrails(conn, index, details)
    if problems and not args.force_promote:
        for p in problems:
            log(f"GUARDRAIL: {p}")
        log("promotion blocked; staged rows kept for review (re-run with --force-promote after checking)")
        return 2
    if args.no_promote:
        log("staged only (--no-promote)")
        return 0
    with conn, conn.cursor() as cur:
        cur.execute("DELETE FROM llm_sizer.exports WHERE status='live'")
        cur.execute("UPDATE llm_sizer.exports SET status='live' WHERE status='staged'")
        promoted = cur.rowcount
    log(f"promoted {promoted} rows to live" + (" (forced)" if problems else ""))
    conn.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
