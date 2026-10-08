from pathlib import Path

import psycopg

MIGRATIONS_DIR = Path(__file__).parent / "migrations"
_LOCK_ID = 727274  # two callers at once (python -m rag_lab.setup run twice) apply each file once


def apply_migrations(database_url: str) -> list[str]:
    """Apply any numbered .sql file not yet recorded in schema_migrations. Returns what it applied."""
    applied = []
    with psycopg.connect(database_url) as conn:
        conn.execute("SELECT pg_advisory_xact_lock(%s)", (_LOCK_ID,))
        conn.execute(
            """CREATE TABLE IF NOT EXISTS schema_migrations (
                   version text PRIMARY KEY,
                   applied_at timestamptz NOT NULL DEFAULT now())"""
        )
        done = {row[0] for row in conn.execute("SELECT version FROM schema_migrations")}
        for path in sorted(MIGRATIONS_DIR.glob("*.sql")):
            if path.stem not in done:
                conn.execute(path.read_text(encoding="utf-8"))
                conn.execute("INSERT INTO schema_migrations (version) VALUES (%s)", (path.stem,))
                applied.append(path.stem)
    return applied
