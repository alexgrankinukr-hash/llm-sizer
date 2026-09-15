#!/usr/bin/env python3
"""Apply the LLM Sizer SQL migrations (`migrations/*llm-sizer*.sql`) in filename order, once each.

Usage: python3 scripts/llm-sizer/migrate.py [--dry-run]
"""
from __future__ import annotations

import argparse
import sys

from common import MIGRATIONS_DIR, connect, log

BOOTSTRAP = """
CREATE SCHEMA IF NOT EXISTS "llm_sizer";
CREATE TABLE IF NOT EXISTS "llm_sizer"."schema_migrations" (
  "filename" text PRIMARY KEY,
  "applied_at" timestamp with time zone DEFAULT now() NOT NULL
);
"""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    files = sorted(p for p in MIGRATIONS_DIR.glob("*.sql") if "llm-sizer" in p.name)
    if not files:
        log("no llm-sizer migrations found")
        return 1
    conn = connect()
    try:
        with conn, conn.cursor() as cur:
            cur.execute(BOOTSTRAP)
            cur.execute('SELECT "filename" FROM "llm_sizer"."schema_migrations"')
            done = {r[0] for r in cur.fetchall()}
        pending = [p for p in files if p.name not in done]
        log(f"{len(files)} migration file(s), {len(done)} applied, {len(pending)} pending")
        for p in pending:
            if args.dry_run:
                log(f"would apply {p.name}")
                continue
            with conn, conn.cursor() as cur:
                cur.execute(p.read_text())
                cur.execute('INSERT INTO "llm_sizer"."schema_migrations" ("filename") VALUES (%s) ON CONFLICT DO NOTHING', (p.name,))
            log(f"applied {p.name}")
    finally:
        conn.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
