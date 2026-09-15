"""Shared helpers for the LLM Sizer data jobs: environment, database connection, retries.

Environment variables (from the process environment, else the repo-root `.env`):
  DATABASE_URL  Postgres connection string (the tool uses its own schema `llm_sizer`)
  HF_TOKEN      Hugging Face read token (optional; higher rate limits, gated configs)
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from typing import Callable, TypeVar
from urllib.parse import urlparse, parse_qs, urlencode, urlunparse

HERE = Path(__file__).resolve().parent          # scripts/llm-sizer
REPO_ROOT = HERE.parent.parent                   # repo root
CACHE_DIR = HERE / ".cache"
OUT_DIR = HERE / "out"
MIGRATIONS_DIR = REPO_ROOT / "migrations"
PUBLIC_DATA_DIR = REPO_ROOT / "public" / "data" / "llm-sizer"

T = TypeVar("T")


def _read_dotenv(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    if not path.exists():
        return values
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, val = line.split("=", 1)
        val = val.strip()
        if len(val) >= 2 and val[0] == val[-1] and val[0] in "\"'":
            val = val[1:-1]
        values[key.strip()] = val
    return values


def load_env() -> dict[str, str | None]:
    """Return DATABASE_URL and HF_TOKEN from the environment, falling back to the repo `.env`."""
    dotenv = _read_dotenv(REPO_ROOT / ".env")
    return {
        "DATABASE_URL": os.environ.get("DATABASE_URL") or dotenv.get("DATABASE_URL"),
        "HF_TOKEN": os.environ.get("HF_TOKEN") or dotenv.get("HF_TOKEN"),
    }


def with_ssl(url: str) -> str:
    """Force `sslmode=require` on a Postgres URL that does not set it (hosted Postgres proxies want TLS)."""
    parts = urlparse(url)
    query = parse_qs(parts.query, keep_blank_values=True)
    if "sslmode" not in query:
        query["sslmode"] = ["require"]
    flat = {k: v[0] for k, v in query.items()}
    return urlunparse(parts._replace(query=urlencode(flat)))


def connect(url: str | None = None):
    """Open a psycopg2 connection with the settings hosted Postgres proxies expect (TLS, keepalives)."""
    import psycopg2  # imported lazily so the parsers stay importable without it

    url = url or load_env()["DATABASE_URL"]
    if not url:
        raise SystemExit("DATABASE_URL is not set (environment or .env)")
    return psycopg2.connect(
        with_ssl(url),
        connect_timeout=15,
        keepalives=1,
        keepalives_idle=30,
        keepalives_interval=10,
        keepalives_count=5,
        application_name="llm-sizer",
        options="-c statement_timeout=60000 -c search_path=llm_sizer,public",
    )


def with_retry(fn: Callable[[], T], tries: int = 3, delay: float = 2.0, what: str = "operation") -> T:
    last: Exception | None = None
    for attempt in range(1, tries + 1):
        try:
            return fn()
        except Exception as exc:  # noqa: BLE001 - we re-raise after the last attempt
            last = exc
            if attempt == tries:
                break
            log(f"{what} failed ({type(exc).__name__}: {exc}); retry {attempt}/{tries - 1} in {delay:.0f}s")
            time.sleep(delay)
            delay *= 2
    assert last is not None
    raise last


def log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)
