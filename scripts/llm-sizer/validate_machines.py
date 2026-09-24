#!/usr/bin/env python3
"""Validate public/data/llm-sizer/machines.json (or a file given as the first argument).

Checks: unique ids, required fields, bandwidth > 0, at least one memory option, sources present,
prices only on current rows and monotonic in memory, tier/bandwidth consistency, sensible enums, and the GPU-bin
fields (gpu_cores_by_gb, gpu_upgrade_usd) consistent with gpu_cores and the memory options.
Exit 1 with a list of problems; exit 0 when clean.
"""
from __future__ import annotations

import json
import re
import sys

from common import PUBLIC_DATA_DIR

REQUIRED = ["id", "kind", "family", "chip", "generation", "year", "memory_options_gb", "memory_kind", "bandwidth_gbs",
            "gpu_cores", "platform", "interconnect", "status", "price_usd", "notes", "sources"]
ENUMS = {"kind": {"mac", "prebuilt", "pc"}, "memory_kind": {"unified", "vram"}, "platform": {"apple", "cuda", "rocm"},
         "status": {"current", "discontinued"}, "interconnect": {"thunderbolt3", "thunderbolt4", "thunderbolt5", "connectx7-200gbe", "none"}}
# Apple-published bandwidth per chip name (GB/s); a row whose chip matches must use one of these numbers.
APPLE_BANDWIDTH = {
    "M1": {68}, "M1 Pro": {200}, "M1 Max": {400}, "M1 Ultra": {800},
    "M2": {100}, "M2 Pro": {200}, "M2 Max": {400}, "M2 Ultra": {800},
    "M3": {100}, "M3 Pro": {150}, "M3 Max": {300, 400}, "M3 Ultra": {819},
    "M4": {120}, "M4 Pro": {273}, "M4 Max": {410, 546},
    "M5": {153}, "M5 Pro": {307}, "M5 Max": {460, 614}, "M5 Ultra": {1200},
    "M6": {153, 170}, "M6 Pro": {307},
}


def chip_key(chip: str) -> str:
    return re.sub(r"\s*\(.*\)\s*$", "", chip).strip()


def validate(doc: dict) -> list[str]:
    problems: list[str] = []
    rows = doc.get("machines")
    if not isinstance(rows, list) or not rows:
        return ["no machines array"]
    ids: set[str] = set()
    for i, m in enumerate(rows):
        tag = f"row {i} ({m.get('id', '?')})"
        for k in REQUIRED:
            if k not in m:
                problems.append(f"{tag}: missing {k}")
        if not all(k in m for k in REQUIRED):
            continue
        if m["id"] in ids:
            problems.append(f"{tag}: duplicate id")
        ids.add(m["id"])
        if not re.match(r"^[a-z0-9][a-z0-9-]*$", m["id"]):
            problems.append(f"{tag}: id must be a lowercase slug")
        for k, allowed in ENUMS.items():
            if m[k] not in allowed:
                problems.append(f"{tag}: {k}={m[k]!r} not in {sorted(allowed)}")
        if not (isinstance(m["bandwidth_gbs"], (int, float)) and m["bandwidth_gbs"] > 0):
            problems.append(f"{tag}: bandwidth_gbs must be > 0")
        mem = m["memory_options_gb"]
        if not (isinstance(mem, list) and mem and all(isinstance(x, int) and x > 0 for x in mem) and mem == sorted(mem)):
            problems.append(f"{tag}: memory_options_gb must be a sorted non-empty list of positive integers")
        if not (isinstance(m["sources"], list) and m["sources"] and all(str(s).startswith("http") for s in m["sources"])):
            problems.append(f"{tag}: needs at least one http source")
        if not (isinstance(m["year"], int) and 2020 <= m["year"] <= 2030):
            problems.append(f"{tag}: year {m['year']} out of range")
        if m["platform"] == "apple":
            key = chip_key(m["chip"])
            allowed_bw = APPLE_BANDWIDTH.get(key)
            if allowed_bw is None:
                problems.append(f"{tag}: unknown Apple chip {key!r}")
            elif m["bandwidth_gbs"] not in allowed_bw:
                problems.append(f"{tag}: {key} bandwidth {m['bandwidth_gbs']} not in Apple's published {sorted(allowed_bw)}")
            if not m["generation"].startswith("M"):
                problems.append(f"{tag}: generation should be like 'M4'")
        price = m["price_usd"]
        if m["status"] == "current":
            if not isinstance(price, dict) or not price:
                problems.append(f"{tag}: current rows need a price_usd object")
            else:
                for k in price:
                    if int(k) not in mem:
                        problems.append(f"{tag}: price key {k} is not a memory option")
                seq = [price.get(str(g)) for g in mem if price.get(str(g)) is not None]
                if seq != sorted(seq):
                    problems.append(f"{tag}: prices must rise with memory ({seq})")
        elif price not in (None, {}):
            problems.append(f"{tag}: discontinued rows must have price_usd null")
        problems.extend(validate_gpu_bins(m, tag))
    return problems


def validate_gpu_bins(m: dict, tag: str) -> list[str]:
    """A row that merges two GPU bins reads on the one its price buys at each size: the smallest, unless
    gpu_cores_by_gb names the larger for a size that comes only with it. gpu_upgrade_usd is Apple's price for the
    largest bin at a size where the row reads on a smaller one (current rows only)."""
    problems: list[str] = []
    cores = m.get("gpu_cores") or []
    mem = m["memory_options_gb"]
    by_gb = m.get("gpu_cores_by_gb")
    upgrade = m.get("gpu_upgrade_usd")
    if by_gb is not None:
        if not (isinstance(by_gb, dict) and by_gb):
            problems.append(f"{tag}: gpu_cores_by_gb must be a non-empty object")
            by_gb = {}
        for k, v in by_gb.items():
            if not k.isdigit() or int(k) not in mem:
                problems.append(f"{tag}: gpu_cores_by_gb key {k} is not a memory option")
            if v not in cores:
                problems.append(f"{tag}: gpu_cores_by_gb {k}: {v} is not one of gpu_cores {cores}")
            elif cores and v == min(cores):
                problems.append(f"{tag}: gpu_cores_by_gb {k}: {v} is the smallest bin, the default; drop it")
    if upgrade is not None:
        if m["status"] != "current":
            problems.append(f"{tag}: gpu_upgrade_usd only on current rows")
        if not (isinstance(upgrade, dict) and upgrade):
            problems.append(f"{tag}: gpu_upgrade_usd must be a non-empty object")
            upgrade = {}
        for k, v in upgrade.items():
            if not k.isdigit() or int(k) not in mem:
                problems.append(f"{tag}: gpu_upgrade_usd key {k} is not a memory option")
                continue
            priced = (by_gb or {}).get(k, min(cores) if cores else None)
            if not cores or priced == max(cores):
                problems.append(f"{tag}: gpu_upgrade_usd {k}: the row already reads on the largest GPU bin at this size")
            if not (isinstance(v, int) and v > 0):
                problems.append(f"{tag}: gpu_upgrade_usd {k} must be a positive whole number of dollars")
    return problems


def main() -> int:
    path = sys.argv[1] if len(sys.argv) > 1 else str(PUBLIC_DATA_DIR / "machines.json")
    doc = json.loads(open(path).read())
    problems = validate(doc)
    rows = doc.get("machines", [])
    if problems:
        print(f"{len(problems)} problem(s) in {path}:")
        for p in problems:
            print("  -", p)
        return 1
    current = sum(1 for m in rows if m.get("status") == "current")
    print(f"OK: {len(rows)} machines ({current} current), {len({m['family'] for m in rows})} families, data_as_of {doc.get('data_as_of')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
