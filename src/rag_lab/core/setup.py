"""Check the databases and make our tables: python -m rag_lab setup

Connects with every URL of config/connections.yaml, applies our migrations inside `app_database.schema`,
and says what is missing. It creates no database and no role: that is scripts/provision.sql, which an
admin runs (the local stack runs it by itself, see docker-compose.local.yml). The `setup` service runs
this once when the stack starts, before the API, the UI and Dagster; it is safe to run again.
It reads all three settings files first, so a mistake in one of them stops the start with its message."""

import sys

import psycopg

from rag_lab.core.settings import ConfigError, load
from rag_lab.core.store import apply_migrations, in_schema

PROVISION = "Run scripts/provision.sql as an admin of the PostgreSQL server (its first lines say how)."


def _problem(what: str, url: str) -> str | None:
    """Why this role cannot connect, or None."""
    try:
        psycopg.connect(url, connect_timeout=10).close()
    except psycopg.OperationalError as e:
        return f"{what}: cannot connect ({str(e).strip().splitlines()[-1]})"
    return None


def main() -> None:
    try:
        connections = load().connections
    except ConfigError as e:
        sys.exit(str(e))  # the file and the key, without a traceback
    app, tables = connections.app_database, connections.tables_database
    problems = [_problem("app_database.url", app.url), _problem("tables_database.reader_url", tables.reader_url)]
    if tables.loader_url:
        problems.append(_problem("tables_database.loader_url", tables.loader_url))
    done: list[str] = []
    if problems[0] is None:
        with psycopg.connect(app.url) as conn:
            exists = conn.execute(
                "SELECT has_schema_privilege(oid, 'CREATE') FROM pg_namespace WHERE nspname = %s", (app.schema_name,)
            ).fetchone()
        if exists is None:
            problems.append(f"app_database.schema: there is no schema '{app.schema_name}'")
        elif not exists[0]:
            problems.append(f"app_database.schema: this role may not create tables in '{app.schema_name}'")
        else:
            done = [f"applied the migration {name}" for name in apply_migrations(in_schema(app.url, app.schema_name))]
    problems = [p for p in problems if p]
    if problems:
        sys.exit("config/connections.yaml:\n  " + "\n  ".join(problems) + f"\n{PROVISION}")
    print("\n".join(done) if done else "Nothing to do: the databases can be reached and our tables are in place.")


if __name__ == "__main__":
    main()
