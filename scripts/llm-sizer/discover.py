#!/usr/bin/env python3
"""Discovery job: find open models and their quantizations on Hugging Face and write them to Postgres.

Modes
  daily     repos updated in the quantizer orgs during the last N days (default 7) + every featured model
  backfill  repos created in the quantizer orgs since 2025-09-01 + every featured model
  only      just the featured models (or the --only stems) — the development mode

Usage
  python3 scripts/llm-sizer/discover.py --mode only --only Qwen3.8-27B
  python3 scripts/llm-sizer/discover.py --mode daily
  python3 scripts/llm-sizer/discover.py --mode backfill --limit 50 --dry-run
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
import traceback
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone

import psycopg2.extras

import arch as archmod
import naming
from common import HERE, connect, load_env, log
from hf import HfClient, Listing, RepoInfo

QUANTIZERS = naming.QUANTIZER_ORGS
BACKFILL_SINCE = datetime(2025, 9, 1, tzinfo=timezone.utc)
TEXT_TAGS = {"text-generation", "image-text-to-text", "any-to-any", "image-to-text", None}
MIN_PARAMS = 1_000_000_000
MIN_DOWNLOADS = 1000
DRAFT_ORG_PREFERENCE = ["z-lab", "incoai", "deepseek-ai", "RedHatAI", "nvidia"]
ACTIVE_IN_NAME = re.compile(r"-a(\d+(?:\.\d+)?)b(?:-|$)", re.I)
PRECISION_TOKENS = {"fp8", "nvfp4", "mxfp4", "mxfp8", "fp4", "int4", "int8", "awq", "gptq", "w4a16", "w8a8", "w4a4", "block", "dynamic", "bf16", "fp16"}


def looks_like_precision_variant(name: str) -> bool:
    return bool(set(naming.tokens(name)) & PRECISION_TOKENS)
LICENSE_COMMERCIAL = {"apache-2.0": True, "mit": True, "bsd-3-clause": True, "gemma": True, "llama4": True, "llama3.3": True,
                      "openrail": True, "cc-by-4.0": True, "nvidia-open-model-license": True, "cc-by-nc-4.0": False}


def load_overrides() -> dict:
    return json.loads((HERE / "models.overrides.json").read_text())


def git_sha() -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=HERE, text=True).strip()
    except Exception:  # noqa: BLE001
        return None


@dataclass
class BaseWork:
    repo_id: str            # canonical hf repo (may carry a -BF16 suffix when no bare repo exists)
    org: str
    stem: str               # name used for quant matching (precision suffix stripped)
    featured: dict | None = None
    listing: Listing | None = None
    quant_candidates: dict[str, Listing] = field(default_factory=dict)


class Discovery:
    def __init__(self, hf: HfClient, conn, overrides: dict, dry_run: bool = False):
        self.hf = hf
        self.conn = conn
        self.ov = overrides
        self.dry = dry_run
        self.errors: list[dict] = []
        self.stats = {"models_seen": 0, "models_added": 0, "quants_added": 0, "quants_changed": 0, "repos_rejected": 0, "repos_skipped": 0}
        self.known: dict[str, dict] = {}
        self.existing_models: set[str] = set()
        self.featured_by_repo = {f["hf_repo"]: f for f in overrides.get("featured", [])}

    # ---------------------------------------------------------------- db helpers
    def load_known(self) -> None:
        with self.conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cur:
            cur.execute('SELECT id, sha, kind, model_id, reject_reason FROM llm_sizer.repos')
            self.known = {r["id"]: dict(r) for r in cur.fetchall()}
            cur.execute("SELECT id FROM llm_sizer.models")
            self.existing_models = {r[0] for r in cur.fetchall()}

    def exec(self, sql: str, params: tuple = ()) -> None:
        if self.dry:
            return
        with self.conn.cursor() as cur:
            cur.execute(sql, params)

    def record_repo(self, repo: Listing | RepoInfo, kind: str, model_id: str | None = None, reject_reason: str | None = None) -> None:
        self.exec(
            """INSERT INTO llm_sizer.repos (id, model_id, kind, org, pipeline_tag, card_base_model, gated, downloads, sha, last_modified, created_at, reject_reason)
               VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
               ON CONFLICT (id) DO UPDATE SET model_id=EXCLUDED.model_id, kind=EXCLUDED.kind, pipeline_tag=EXCLUDED.pipeline_tag,
                 card_base_model=EXCLUDED.card_base_model, gated=EXCLUDED.gated, downloads=EXCLUDED.downloads, sha=EXCLUDED.sha,
                 last_modified=EXCLUDED.last_modified, created_at=EXCLUDED.created_at, reject_reason=EXCLUDED.reject_reason, last_checked=now()""",
            (repo.id, model_id, kind, repo.org, repo.pipeline_tag, list(repo.base_models or []), _gated(repo.gated), repo.downloads,
             repo.sha, repo.last_modified, repo.created_at, reject_reason),
        )
        self.known[repo.id] = {"id": repo.id, "sha": repo.sha, "kind": kind, "model_id": model_id, "reject_reason": reject_reason}
        if kind == "rejected":
            self.stats["repos_rejected"] += 1

    # ---------------------------------------------------------------- candidates
    def quantizer_candidates(self, mode: str, days: int, limit: int | None) -> dict[str, Listing]:
        out: dict[str, Listing] = {}
        for org in QUANTIZERS:
            if mode == "backfill":
                since, by = BACKFILL_SINCE, "createdAt"
            else:
                since, by = datetime.now(timezone.utc) - timedelta(days=days), "lastModified"
            n = 0
            for row in self.hf.list_org_updates(org, since, by=by):
                out[row.id] = row
                n += 1
                if limit and n >= limit:
                    break
            log(f"  {org}: {n} candidate repos")
        return out

    def resolve_base(self, cand: Listing) -> tuple[str, str, str] | None:
        """Return (base_repo_id, org, stem) for a quant repo, or None."""
        _, name = naming.split_repo(cand.id)
        suf = naming.strip_suffix(name)
        # 1. card metadata base_model (follow one hop through quantizer mirrors)
        for target in cand.base_models[:2]:
            for _hop in range(2):
                org, tname = naming.split_repo(target)
                if not org or not tname:
                    break
                if org in QUANTIZERS:
                    try:
                        info = self.hf.repo_info(target)
                    except Exception:  # noqa: BLE001
                        break
                    if info.base_models:
                        target = info.base_models[0]
                        continue
                    break
                stem = naming.strip_precision_variant(tname)
                if naming.reject_token(stem):
                    return None
                if naming.same_model(stem, suf.stem) or naming.same_model(tname, suf.stem):
                    return self._canonical(org, tname, stem)
                # the card points somewhere else (e.g. an unquantized QAT checkpoint) — fall through to search
                break
        # 2. org prefix hint (bartowski `org_Name`, mlx-community `org-Name`)
        m = re.match(r"^([A-Za-z0-9][A-Za-z0-9.\-]*)_(.+)$", name)
        hints: list[tuple[str, str]] = []
        if m:
            hints.append((m.group(1), naming.strip_suffix(m.group(2)).stem))
        # 3. search by stem, prefer maker orgs and downloads
        try:
            rows = self.hf.search(suf.stem, limit=30)
        except Exception as exc:  # noqa: BLE001
            self.errors.append({"repo": cand.id, "stage": "resolve", "error": f"{type(exc).__name__}: {exc}"})
            rows = []
        best: Listing | None = None
        best_key: tuple | None = None
        for r in rows:
            org, rname = naming.split_repo(r.id)
            if org in QUANTIZERS or naming.reject_token(rname) or naming.looks_like_quant_name(rname):
                continue
            stem = naming.strip_precision_variant(rname)
            if naming.strip_suffix(rname).kind in ("gguf", "mlx"):
                continue
            if not naming.same_model(stem, suf.stem):
                continue
            # a bare maker repo beats any precision re-pack, whatever the download counts say
            key = (looks_like_precision_variant(rname), -(r.downloads or 0))
            if best is None or key < best_key:
                best, best_key = r, key
        if best:
            org, rname = naming.split_repo(best.id)
            return self._canonical(org, rname, naming.strip_precision_variant(rname))
        for org, stem in hints:
            try:
                info = self.hf.repo_info(f"{org}/{stem}")
                return self._canonical(org, stem, stem)
            except Exception:  # noqa: BLE001
                continue
        return None

    def _canonical(self, org: str, repo_name: str, stem: str) -> tuple[str, str, str]:
        """Prefer the bare (non-FP8) maker repo when it exists; otherwise keep the precision-suffixed one."""
        if repo_name != stem:
            try:
                self.hf.repo_info(f"{org}/{stem}")
                return f"{org}/{stem}", org, stem
            except Exception:  # noqa: BLE001
                return f"{org}/{repo_name}", org, stem
        return f"{org}/{repo_name}", org, stem

    # ---------------------------------------------------------------- per base model
    def quant_search(self, work: BaseWork) -> None:
        """Rule (i): author search per quantizer org + the maker's own org."""
        for org in list(QUANTIZERS) + [work.org]:
            try:
                for r in self.hf.search(work.stem, author=org, limit=40):
                    work.quant_candidates.setdefault(r.id, r)
            except Exception as exc:  # noqa: BLE001
                self.errors.append({"repo": work.repo_id, "stage": "quant_search", "org": org, "error": f"{type(exc).__name__}: {exc}"})

    def find_drafts(self, stem: str) -> tuple[str | None, str | None]:
        out: dict[str, str | None] = {"DFlash": None, "DSpark": None}
        for method in out:
            try:
                rows = self.hf.search(f"{stem} {method}", limit=20)
            except Exception:  # noqa: BLE001
                continue
            ranked: list[tuple[int, int, str]] = []
            for r in rows:
                org, rname = naming.split_repo(r.id)
                if method.lower() not in rname.lower() and method.lower() not in r.id.lower():
                    continue
                if not naming.same_model(re.split(r"[-_](?i:dflash|dspark)", rname)[0], stem):
                    continue
                pref = DRAFT_ORG_PREFERENCE.index(org) if org in DRAFT_ORG_PREFERENCE else len(DRAFT_ORG_PREFERENCE)
                ranked.append((pref, -(r.downloads or 0), r.id))
            if ranked:
                ranked.sort()
                out[method] = ranked[0][2]
        return out["DFlash"], out["DSpark"]

    def process_base(self, work: BaseWork, mode: str) -> None:
        self.stats["models_seen"] += 1
        info = self.hf.repo_info(work.repo_id, sha=(work.listing.sha if work.listing else None))
        base_known = self.known.get(work.repo_id)
        featured = work.featured
        # noise floor for auto-imported models
        if not featured and work.repo_id not in self.ov.get("always_import", []):
            if info.pipeline_tag not in TEXT_TAGS:
                self.record_repo(info, "rejected", reject_reason=f"pipeline_tag:{info.pipeline_tag}")
                return
            if (info.safetensors_total or 0) < MIN_PARAMS or (info.downloads or 0) < MIN_DOWNLOADS:
                self.record_repo(info, "rejected", reject_reason="below_floor")
                return
            if info.created_at and info.created_at < BACKFILL_SINCE:
                self.record_repo(info, "rejected", reject_reason="outside_window")
                return
            if naming.looks_like_quant_name(work.stem):
                self.record_repo(info, "rejected", reject_reason="quant_label_in_name")
                return
        model_id = featured["id"] if featured else naming.slugify(naming.stem_key(work.stem))
        if not featured and model_id in self.existing_models and base_known and base_known.get("model_id") not in (None, model_id):
            model_id = naming.slugify(work.org + "-" + naming.stem_key(work.stem))

        cfg, cfg_err = self.hf.config(work.repo_id, info.sha)
        if cfg is None and cfg_err == "gated" and not self.hf.token:
            cfg_err = "gated"
        ov = self.ov.get("models", {}).get(model_id, {})
        a = None
        if cfg is not None:
            if not _looks_like_text_model(cfg):
                self.record_repo(info, "rejected", reject_reason="not_a_text_model")
                return
            a = archmod.classify(cfg)
        status = "needs_token" if (cfg is None and cfg_err == "gated") else "active"
        if ov.get("superseded_by"):
            status = "superseded"

        params_total = info.safetensors_total or ov.get("params_total")
        if a is not None and params_total:
            active, active_src = archmod.active_params(a, params_total)
        else:
            active, active_src = params_total, "dense"
        m = ACTIVE_IN_NAME.search(work.stem)
        if m and a is not None and a.is_moe:
            active, active_src = int(float(m.group(1)) * 1e9), "name"
        if ov.get("params_active"):
            active, active_src = int(ov["params_active"]), "override"

        lic = next((t.split(":", 1)[1] for t in info.tags if t.startswith("license:")), None)
        lic_commercial = LICENSE_COMMERCIAL.get(lic) if lic else None
        name = featured["name"] if featured else work.stem
        notes = list(ov.get("notes", []))
        if a is not None and a.conventional_assumed:
            notes.append("architecture read from config, but the model family is unfamiliar — treat cache math as approximate")
        if cfg is None:
            notes.append("config.json unavailable (gated or missing); architecture assumed conventional")

        is_new = model_id not in self.existing_models
        self.exec(
            """INSERT INTO llm_sizer.models (id, name, provider, hf_repo, variant, released, params_total, params_active, params_active_source,
                 context_max, native_bits, license, license_commercial, license_note, tags, featured, featured_tier, featured_rank, gated, downloads,
                 status, superseded_by, notes)
               VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
               ON CONFLICT (id) DO UPDATE SET name=EXCLUDED.name, provider=EXCLUDED.provider, hf_repo=EXCLUDED.hf_repo, variant=EXCLUDED.variant,
                 released=COALESCE(llm_sizer.models.released, EXCLUDED.released), params_total=EXCLUDED.params_total, params_active=EXCLUDED.params_active,
                 params_active_source=EXCLUDED.params_active_source, context_max=EXCLUDED.context_max, native_bits=EXCLUDED.native_bits,
                 license=EXCLUDED.license, license_commercial=EXCLUDED.license_commercial, license_note=EXCLUDED.license_note, tags=EXCLUDED.tags,
                 featured=EXCLUDED.featured, featured_tier=EXCLUDED.featured_tier, featured_rank=EXCLUDED.featured_rank, gated=EXCLUDED.gated,
                 downloads=EXCLUDED.downloads, status=EXCLUDED.status, superseded_by=EXCLUDED.superseded_by, notes=EXCLUDED.notes,
                 last_checked=now(),
                 last_changed=CASE WHEN llm_sizer.models.params_total IS DISTINCT FROM EXCLUDED.params_total
                                     OR llm_sizer.models.context_max IS DISTINCT FROM EXCLUDED.context_max THEN now() ELSE llm_sizer.models.last_changed END""",
            (model_id, name, work.org, work.repo_id, naming.variant(work.stem), (info.created_at.date() if info.created_at else None),
             params_total, active, active_src, (a.context_max if a else None), (a.native_bits if a else None), lic, lic_commercial,
             ov.get("license_note"), [t for t in info.tags if not t.startswith(("license:", "region:", "base_model:", "arxiv:"))][:30],
             bool(featured), featured["tier"] if featured else None, featured["rank"] if featured else None, _gated(info.gated), info.downloads,
             status, ov.get("superseded_by"), notes),
        )
        if is_new:
            self.stats["models_added"] += 1
            self.existing_models.add(model_id)
        self.record_repo(info, "base", model_id=model_id)

        # the base repo's own weights are the "native" record (BF16 / FP8 / FP4 as shipped)
        native_label = {16.0: "BF16", 8.0: "FP8", 4.0: "FP4", 32.0: "F32"}.get(a.native_bits if a and a.native_bits else 16.0, "BF16")
        base_quants = naming.quants_from_files(info.siblings, "safetensors", native_label, (a.native_bits if a else None), params_total)
        if base_quants:
            self.upsert_quants(model_id, info.id, base_quants, "official", params_total)

        # architecture
        dflash, dspark = (None, None)
        if featured or is_new or mode == "only":
            dflash, dspark = self.find_drafts(work.stem)
        if a is not None:
            arch_ov = ov.get("architecture", {})
            self.exec(
                """INSERT INTO llm_sizer.model_architecture (model_id, model_type, attention, layers, kv_layers, sliding_layers, linear_layers, kv_heads, head_dim,
                     k_eq_v, kv_lora_rank, rope_dim, window_size, full_bytes_per_token_8bit, sliding_bytes_per_token_8bit, linear_state_bytes, hidden_size,
                     experts_total, experts_active, experts_shared, moe_intermediate, moe_layers, accel_mtp, accel_dflash_repo, accel_dspark_repo,
                     conventional_assumed, source, config, config_sha, notes)
                   VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
                   ON CONFLICT (model_id) DO UPDATE SET model_type=EXCLUDED.model_type, attention=EXCLUDED.attention, layers=EXCLUDED.layers,
                     kv_layers=EXCLUDED.kv_layers, sliding_layers=EXCLUDED.sliding_layers, linear_layers=EXCLUDED.linear_layers, kv_heads=EXCLUDED.kv_heads,
                     head_dim=EXCLUDED.head_dim, k_eq_v=EXCLUDED.k_eq_v, kv_lora_rank=EXCLUDED.kv_lora_rank, rope_dim=EXCLUDED.rope_dim,
                     window_size=EXCLUDED.window_size, full_bytes_per_token_8bit=EXCLUDED.full_bytes_per_token_8bit,
                     sliding_bytes_per_token_8bit=EXCLUDED.sliding_bytes_per_token_8bit, linear_state_bytes=EXCLUDED.linear_state_bytes,
                     hidden_size=EXCLUDED.hidden_size, experts_total=EXCLUDED.experts_total, experts_active=EXCLUDED.experts_active,
                     experts_shared=EXCLUDED.experts_shared, moe_intermediate=EXCLUDED.moe_intermediate, moe_layers=EXCLUDED.moe_layers,
                     accel_mtp=EXCLUDED.accel_mtp, accel_dflash_repo=COALESCE(EXCLUDED.accel_dflash_repo, llm_sizer.model_architecture.accel_dflash_repo),
                     accel_dspark_repo=COALESCE(EXCLUDED.accel_dspark_repo, llm_sizer.model_architecture.accel_dspark_repo),
                     conventional_assumed=EXCLUDED.conventional_assumed, source=EXCLUDED.source, config=EXCLUDED.config, config_sha=EXCLUDED.config_sha,
                     notes=EXCLUDED.notes, updated_at=now()""",
                (model_id, a.model_type, a.attention, a.layers, arch_ov.get("kv_layers", a.kv_layers), a.sliding_layers, a.linear_layers, a.kv_heads,
                 a.head_dim, a.k_eq_v, a.kv_lora_rank, a.rope_dim, a.window_size, arch_ov.get("full_bytes_per_token_8bit", a.full_bytes_per_token_8bit),
                 a.sliding_bytes_per_token_8bit, a.linear_state_bytes, a.hidden_size, a.experts_total, a.experts_active, a.experts_shared,
                 a.moe_intermediate, a.moe_layers, a.accel_mtp, dflash, dspark, a.conventional_assumed, "override" if arch_ov else a.source,
                 psycopg2.extras.Json(archmod.stripped_config(cfg)), archmod.config_sha(cfg), a.notes),
            )
        elif ov.get("architecture"):
            ao = ov["architecture"]
            self.exec(
                """INSERT INTO llm_sizer.model_architecture (model_id, attention, layers, kv_layers, kv_heads, head_dim, full_bytes_per_token_8bit, accel_mtp, source, notes)
                   VALUES (%s,%s,%s,%s,%s,%s,%s,%s,'override',%s)
                   ON CONFLICT (model_id) DO UPDATE SET attention=EXCLUDED.attention, layers=EXCLUDED.layers, kv_layers=EXCLUDED.kv_layers,
                     kv_heads=EXCLUDED.kv_heads, head_dim=EXCLUDED.head_dim, full_bytes_per_token_8bit=EXCLUDED.full_bytes_per_token_8bit,
                     accel_mtp=EXCLUDED.accel_mtp, source='override', updated_at=now()""",
                (model_id, ao.get("attention", "gqa"), ao["layers"], ao.get("kv_layers", ao["layers"]), ao.get("kv_heads"), ao.get("head_dim"),
                 ao["full_bytes_per_token_8bit"], ao.get("accel_mtp", False), ["architecture from overrides"]),
            )

        # quantizations
        for cand_id, cand in sorted(work.quant_candidates.items()):
            known = self.known.get(cand_id)
            if known and known["sha"] == cand.sha and known["kind"] in ("quant", "precision_variant", "rejected") and mode != "only":
                self.stats["repos_skipped"] += 1
                continue
            match = naming.classify_quant_repo(cand_id, work.org, work.stem)
            if not match.accepted:
                self.record_repo(cand, "rejected", model_id=None, reject_reason=match.reject_reason)
                continue
            try:
                qinfo = self.hf.repo_info(cand_id, sha=cand.sha)
            except Exception as exc:  # noqa: BLE001
                self.errors.append({"repo": cand_id, "stage": "repo_info", "error": f"{type(exc).__name__}: {exc}"})
                continue
            quants = naming.quants_from_files(qinfo.siblings, match.format, match.label, match.bits, params_total, match.qat)
            if not quants:
                self.record_repo(qinfo, "rejected", model_id=model_id, reject_reason="no_weight_files")
                continue
            kind = "precision_variant" if match.format == "safetensors" else "quant"
            self.record_repo(qinfo, kind, model_id=model_id)
            self.upsert_quants(model_id, qinfo.id, quants, match.quantizer, params_total)
        if not self.dry:
            self.conn.commit()

    def upsert_quants(self, model_id: str, repo_id: str, quants: list[naming.Quant], quantizer: str, params_total: int | None) -> None:
        existing: dict[str, int] = {}
        with self.conn.cursor() as cur:
            cur.execute("SELECT label, size_bytes FROM llm_sizer.model_quants WHERE repo_id=%s AND withdrawn_at IS NULL", (repo_id,))
            existing = {r[0]: int(r[1]) for r in cur.fetchall()}
        gone = [lab for lab in existing if lab not in {q.label for q in quants}]
        if gone:
            # the repo's current listing is the truth; labels the quantizer removed (or we mis-read before) are withdrawn, not deleted
            self.exec("UPDATE llm_sizer.model_quants SET withdrawn_at = now(), last_checked = now() WHERE repo_id=%s AND label = ANY(%s) AND withdrawn_at IS NULL", (repo_id, gone))
            self.stats["quants_changed"] += len(gone)
        for q in quants:
            prev = existing.get(q.label)
            if prev is None:
                self.stats["quants_added"] += 1
            elif prev != q.size_bytes:
                self.stats["quants_changed"] += 1
            self.exec(
                """INSERT INTO llm_sizer.model_quants (model_id, repo_id, format, label, bits_effective, size_bytes, size_gb, quantizer, file_count, qat)
                   VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
                   ON CONFLICT (repo_id, label) DO UPDATE SET model_id=EXCLUDED.model_id, format=EXCLUDED.format, bits_effective=EXCLUDED.bits_effective,
                     size_bytes=EXCLUDED.size_bytes, size_gb=EXCLUDED.size_gb, quantizer=EXCLUDED.quantizer, file_count=EXCLUDED.file_count, qat=EXCLUDED.qat,
                     withdrawn_at=NULL, last_checked=now(),
                     last_changed=CASE WHEN llm_sizer.model_quants.size_bytes <> EXCLUDED.size_bytes THEN now() ELSE llm_sizer.model_quants.last_changed END""",
                (model_id, repo_id, q.format, q.label, q.bits, q.size_bytes, q.size_gb, quantizer, q.file_count, q.qat),
            )

    # ---------------------------------------------------------------- orchestration
    def run(self, mode: str, only: list[str], days: int, limit: int | None) -> None:
        self.load_known()
        bases: dict[str, BaseWork] = {}

        def ensure(repo_id: str, org: str, stem: str, featured: dict | None = None) -> BaseWork:
            w = bases.get(repo_id)
            if w is None:
                w = bases[repo_id] = BaseWork(repo_id, org, stem, featured)
            if featured and not w.featured:
                w.featured = featured
            return w

        # featured + always-import models
        for f in self.ov.get("featured", []):
            org, rname = naming.split_repo(f["hf_repo"])
            if only and not any(naming.same_model(o, naming.strip_precision_variant(rname)) or o.lower() in f["id"] for o in only):
                continue
            ensure(f["hf_repo"], org, naming.strip_precision_variant(rname), f)
        if mode != "only" or not only:
            for repo in self.ov.get("always_import", []):
                org, rname = naming.split_repo(repo)
                if only and not any(naming.same_model(o, rname) for o in only):
                    continue
                ensure(repo, org, naming.strip_precision_variant(rname))
        for o in only:
            key = naming.stem_key(o)
            if not any(naming.same_model(o, w.stem) or key in naming.stem_key(w.stem) for w in bases.values()):
                rows = [r for r in self.hf.search(o, limit=20)
                        if r.org not in QUANTIZERS and naming.strip_suffix(r.name).kind == "none"
                        and not looks_like_precision_variant(r.name) and not naming.reject_token(r.name)
                        and not naming.looks_like_quant_name(r.name)]
                rows.sort(key=lambda r: -(r.downloads or 0))
                if rows:
                    ensure(rows[0].id, rows[0].org, naming.strip_precision_variant(rows[0].name))
                else:
                    log(f"--only {o}: no base repo found")

        # quantizer feeds
        if mode in ("daily", "backfill"):
            log(f"listing quantizer orgs ({mode}) …")
            cands = self.quantizer_candidates(mode, days, limit)
            log(f"{len(cands)} candidate repos; resolving bases …")
            for cid, cand in cands.items():
                known = self.known.get(cid)
                if known and known["sha"] == cand.sha and known["kind"] == "rejected" and not known["reject_reason"].startswith("base_unresolved"):
                    self.stats["repos_skipped"] += 1
                    continue
                if naming.reject_token(cand.name):
                    if not known:
                        self.record_repo(cand, "rejected", reject_reason=f"variant_token:{naming.reject_token(cand.name)}")
                    continue
                if naming.strip_suffix(cand.name).kind == "none":
                    if not known:
                        self.record_repo(cand, "rejected", reject_reason="no_quant_suffix")
                    continue
                if known and known["kind"] in ("quant", "precision_variant") and known["sha"] == cand.sha:
                    self.stats["repos_skipped"] += 1
                    continue
                if known and known["model_id"] and known["kind"] in ("quant", "precision_variant"):
                    # repo changed; re-read under its existing model
                    with self.conn.cursor() as cur:
                        cur.execute("SELECT hf_repo, provider FROM llm_sizer.models WHERE id=%s", (known["model_id"],))
                        row = cur.fetchone()
                    if row:
                        org, rname = naming.split_repo(row[0])
                        ensure(row[0], row[1], naming.strip_precision_variant(rname)).quant_candidates[cid] = cand
                        continue
                resolved = self.resolve_base(cand)
                if not resolved:
                    self.record_repo(cand, "pending", reject_reason="base_unresolved")
                    continue
                base_id, org, stem = resolved
                ensure(base_id, org, stem).quant_candidates[cid] = cand
            if not self.dry:
                self.conn.commit()

        log(f"{len(bases)} base models to process")
        for i, (base_id, work) in enumerate(sorted(bases.items(), key=lambda kv: (not bool(kv[1].featured), kv[0])), 1):
            t0 = time.time()
            try:
                if work.featured or mode == "only" or not work.quant_candidates:
                    self.quant_search(work)
                self.process_base(work, mode)
                log(f"[{i}/{len(bases)}] {base_id}: {len(work.quant_candidates)} quant repos, {time.time() - t0:.1f}s, {self.hf.calls} calls")
            except Exception as exc:  # noqa: BLE001
                if not self.dry:
                    self.conn.rollback()
                self.errors.append({"repo": base_id, "stage": "process", "error": f"{type(exc).__name__}: {exc}", "trace": traceback.format_exc()[-800:]})
                log(f"[{i}/{len(bases)}] {base_id}: ERROR {type(exc).__name__}: {exc}")


def _gated(v) -> str | None:
    if v is None:
        return None
    if v is False:
        return "false"
    return str(v)


def _looks_like_text_model(cfg: dict) -> bool:
    inner = archmod.unwrap(cfg)
    has_layers = bool(inner.get("num_hidden_layers") or inner.get("n_layer") or inner.get("layers_block_type") or inner.get("hybrid_override_pattern"))
    return has_layers and bool(inner.get("num_attention_heads") or inner.get("n_head") or inner.get("vocab_size"))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", choices=["daily", "backfill", "only"], default="daily")
    ap.add_argument("--only", action="append", default=[], help="model stem(s) to process (implies featured-only otherwise)")
    ap.add_argument("--days", type=int, default=7, help="daily window in days")
    ap.add_argument("--limit", type=int, default=None, help="max candidate repos per quantizer org (dev)")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    env = load_env()
    hf = HfClient(env["HF_TOKEN"])
    conn = connect(env["DATABASE_URL"])
    overrides = load_overrides()
    started = datetime.now(timezone.utc)
    run_id = None
    if not args.dry_run:
        with conn, conn.cursor() as cur:
            cur.execute("INSERT INTO llm_sizer.discovery_runs (mode, code_sha) VALUES (%s,%s) RETURNING id", (args.mode, git_sha()))
            run_id = cur.fetchone()[0]
    d = Discovery(hf, conn, overrides, dry_run=args.dry_run)
    ok = True
    try:
        d.run(args.mode, args.only, args.days, args.limit)
    except Exception as exc:  # noqa: BLE001
        ok = False
        d.errors.append({"stage": "run", "error": f"{type(exc).__name__}: {exc}", "trace": traceback.format_exc()[-1200:]})
        log(f"RUN FAILED: {exc}")
    summary = {**d.stats, "hf_calls": hf.calls, "cache_hits": hf.cache_hits, "seconds": round((datetime.now(timezone.utc) - started).total_seconds(), 1), "errors": len(d.errors)}
    log("summary: " + json.dumps(summary))
    if run_id is not None:
        with conn, conn.cursor() as cur:
            cur.execute(
                """UPDATE llm_sizer.discovery_runs SET finished_at=now(), ok=%s, hf_calls=%s, models_seen=%s, models_added=%s, quants_added=%s,
                   quants_changed=%s, errors=%s, summary=%s WHERE id=%s""",
                (ok and not d.errors, hf.calls, d.stats["models_seen"], d.stats["models_added"], d.stats["quants_added"], d.stats["quants_changed"],
                 psycopg2.extras.Json(d.errors), psycopg2.extras.Json(summary), run_id),
            )
    conn.close()
    if d.errors:
        log(f"{len(d.errors)} error(s); first: {json.dumps(d.errors[0])[:400]}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
