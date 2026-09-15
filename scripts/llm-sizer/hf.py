"""Thin Hugging Face Hub client for the discovery job: retries, call counting, per-sha caching."""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

from huggingface_hub import HfApi, hf_hub_download
from huggingface_hub.utils import GatedRepoError, RepositoryNotFoundError, EntryNotFoundError, HfHubHTTPError

from common import CACHE_DIR, log

LISTING_EXPAND = ["cardData", "sha", "lastModified", "createdAt", "downloads", "pipeline_tag", "safetensors", "gated", "tags"]


def _base_models(card: Any) -> list[str]:
    if card is None:
        return []
    try:
        val = card.get("base_model") if hasattr(card, "get") else getattr(card, "base_model", None)
    except Exception:  # noqa: BLE001
        return []
    if isinstance(val, str):
        return [val]
    if isinstance(val, list):
        return [str(v) for v in val if v]
    return []


def _dt(v: Any) -> datetime | None:
    if v is None:
        return None
    if isinstance(v, datetime):
        return v if v.tzinfo else v.replace(tzinfo=timezone.utc)
    try:
        return datetime.fromisoformat(str(v).replace("Z", "+00:00"))
    except ValueError:
        return None


@dataclass
class Listing:
    id: str
    sha: str | None
    last_modified: datetime | None
    created_at: datetime | None
    downloads: int | None
    pipeline_tag: str | None
    base_models: list[str]
    safetensors_total: int | None
    gated: Any = None
    tags: list[str] = field(default_factory=list)

    @property
    def org(self) -> str:
        return self.id.partition("/")[0]

    @property
    def name(self) -> str:
        return self.id.partition("/")[2]


@dataclass
class RepoInfo(Listing):
    siblings: list[dict] = field(default_factory=list)   # [{rfilename, size}]
    safetensors_by_dtype: dict[str, int] = field(default_factory=dict)

    def to_json(self) -> dict:
        return {
            "id": self.id, "sha": self.sha, "last_modified": self.last_modified.isoformat() if self.last_modified else None,
            "created_at": self.created_at.isoformat() if self.created_at else None, "downloads": self.downloads,
            "pipeline_tag": self.pipeline_tag, "base_models": self.base_models, "safetensors_total": self.safetensors_total,
            "gated": self.gated, "tags": self.tags, "siblings": self.siblings, "safetensors_by_dtype": self.safetensors_by_dtype,
        }

    @classmethod
    def from_json(cls, d: dict) -> "RepoInfo":
        return cls(id=d["id"], sha=d.get("sha"), last_modified=_dt(d.get("last_modified")), created_at=_dt(d.get("created_at")),
                   downloads=d.get("downloads"), pipeline_tag=d.get("pipeline_tag"), base_models=d.get("base_models") or [],
                   safetensors_total=d.get("safetensors_total"), gated=d.get("gated"), tags=d.get("tags") or [],
                   siblings=d.get("siblings") or [], safetensors_by_dtype=d.get("safetensors_by_dtype") or {})


class HfClient:
    def __init__(self, token: str | None, cache_dir: Path = CACHE_DIR, sleep: float = 0.0):
        self.api = HfApi(token=token)
        self.token = token
        self.cache_dir = cache_dir
        self.calls = 0
        self.cache_hits = 0
        self.sleep = sleep

    # ---- plumbing --------------------------------------------------------------
    def _call(self, what: str, fn, *args, **kwargs):
        delay = 2.0
        for attempt in range(1, 5):
            try:
                self.calls += 1
                out = fn(*args, **kwargs)
                if self.sleep:
                    time.sleep(self.sleep)
                return out
            except (GatedRepoError, RepositoryNotFoundError, EntryNotFoundError):
                raise
            except HfHubHTTPError as exc:
                status = getattr(getattr(exc, "response", None), "status_code", None)
                if status in (429, 500, 502, 503, 504) and attempt < 4:
                    log(f"HF {what}: HTTP {status}; retry {attempt} in {delay:.0f}s")
                    time.sleep(delay)
                    delay *= 2
                    continue
                raise
            except (ConnectionError, TimeoutError, OSError) as exc:
                if attempt < 4:
                    log(f"HF {what}: {type(exc).__name__}; retry {attempt} in {delay:.0f}s")
                    time.sleep(delay)
                    delay *= 2
                    continue
                raise
        raise RuntimeError(f"HF {what}: retries exhausted")

    def _list(self, what: str, **kwargs):
        """`list_models` with descending sort; newer huggingface_hub versions dropped the `direction` argument."""
        try:
            return self._call(what, self.api.list_models, direction=-1, **kwargs)
        except TypeError as exc:
            if "direction" not in str(exc):
                raise
            return self._call(what, self.api.list_models, **kwargs)

    def _listing(self, m: Any) -> Listing:
        st = getattr(m, "safetensors", None)
        return Listing(
            id=m.id, sha=getattr(m, "sha", None), last_modified=_dt(getattr(m, "last_modified", None)),
            created_at=_dt(getattr(m, "created_at", None)), downloads=getattr(m, "downloads", None),
            pipeline_tag=getattr(m, "pipeline_tag", None), base_models=_base_models(getattr(m, "card_data", None)),
            safetensors_total=(getattr(st, "total", None) if st else None), gated=getattr(m, "gated", None),
            tags=list(getattr(m, "tags", None) or []),
        )

    # ---- listings ----------------------------------------------------------------
    def list_org_updates(self, org: str, since: datetime | None, by: str = "lastModified", limit: int = 20000) -> Iterator[Listing]:
        """Repos of `org` newest-first by `by` (lastModified | createdAt), stopping once older than `since`."""
        it = self._list(f"list_models({org})", author=org, sort=by, limit=limit, expand=LISTING_EXPAND)
        for m in it:
            self.calls += 0  # pagination happens inside the iterator; counted once above
            row = self._listing(m)
            stamp = row.last_modified if by == "lastModified" else row.created_at
            if since and stamp and stamp < since:
                break
            yield row

    def search(self, query: str, author: str | None = None, limit: int = 30, sort: str = "downloads") -> list[Listing]:
        it = self._list(f"search({query})", search=query, author=author, sort=sort, limit=limit, expand=LISTING_EXPAND)
        return [self._listing(m) for m in it]

    # ---- per-repo ---------------------------------------------------------------------
    def _cache_path(self, repo_id: str, sha: str, kind: str) -> Path:
        org, _, name = repo_id.partition("/")
        return self.cache_dir / "hf" / org / name / f"{sha}.{kind}.json"

    def repo_info(self, repo_id: str, sha: str | None = None) -> RepoInfo:
        """`model_info(files_metadata=True)`, cached by commit sha when the sha is known up front."""
        if sha:
            p = self._cache_path(repo_id, sha, "info")
            if p.exists():
                self.cache_hits += 1
                return RepoInfo.from_json(json.loads(p.read_text()))
        info = self._call(f"model_info({repo_id})", self.api.model_info, repo_id, files_metadata=True)
        st = getattr(info, "safetensors", None)
        row = RepoInfo(
            id=info.id, sha=info.sha, last_modified=_dt(info.last_modified), created_at=_dt(getattr(info, "created_at", None)),
            downloads=info.downloads, pipeline_tag=info.pipeline_tag, base_models=_base_models(getattr(info, "card_data", None)),
            safetensors_total=(st.total if st else None), gated=info.gated, tags=list(info.tags or []),
            siblings=[{"rfilename": s.rfilename, "size": s.size} for s in (info.siblings or [])],
            safetensors_by_dtype=(dict(st.parameters) if st and getattr(st, "parameters", None) else {}),
        )
        if row.sha:
            p = self._cache_path(repo_id, row.sha, "info")
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(json.dumps(row.to_json()))
        return row

    def config(self, repo_id: str, sha: str | None) -> tuple[dict | None, str | None]:
        """Download `config.json`. Returns (config, error) where error is 'gated' | 'missing' | 'error' | None."""
        if sha:
            p = self._cache_path(repo_id, sha, "config")
            if p.exists():
                self.cache_hits += 1
                return json.loads(p.read_text()), None
        try:
            path = self._call(f"config({repo_id})", hf_hub_download, repo_id, "config.json", revision=sha, cache_dir=str(self.cache_dir / "hub"), token=self.token)
        except GatedRepoError:
            return None, "gated"
        except (RepositoryNotFoundError, EntryNotFoundError):
            return None, "missing"
        except HfHubHTTPError as exc:
            status = getattr(getattr(exc, "response", None), "status_code", None)
            return None, "gated" if status in (401, 403) else "error"
        cfg = json.loads(Path(path).read_text())
        if sha:
            p = self._cache_path(repo_id, sha, "config")
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(json.dumps(cfg))
        return cfg, None
