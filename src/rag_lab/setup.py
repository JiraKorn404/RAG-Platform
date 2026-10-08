"""Get the databases ready: python -m rag_lab.setup

Creates the database of `app_database.url` (config/connections.yaml) if it is missing and applies its
migrations, then makes sure the database for imported tables exists with its two roles
(sql/bootstrap.py). The `setup` service in docker-compose.yml runs this once when the stack starts,
before the UI and Dagster; it is safe to run again, and changes nothing when everything is in place.
It reads all three settings files first, so a mistake in one of them stops the start with its message."""

import sys

import psycopg
from psycopg import sql
from psycopg.conninfo import conninfo_to_dict, make_conninfo

from rag_lab.metrics.migrate import apply_migrations
from rag_lab.settings import ConfigError, load
from rag_lab.sql.bootstrap import ensure_database


def ensure_app_database(database_url: str) -> list[str]:
    """Create the database the URL names if it does not exist. Returns what was done."""
    name = conninfo_to_dict(database_url)["dbname"]
    with psycopg.connect(make_conninfo(database_url, dbname="postgres"), autocommit=True) as admin:
        if admin.execute("SELECT 1 FROM pg_database WHERE datname = %s", (name,)).fetchone():
            return []
        admin.execute(sql.SQL("CREATE DATABASE {}").format(sql.Identifier(name)))
    return [f"created the database {name}"]


def main() -> None:
    try:
        database_url = load().connections.app_database.url
    except ConfigError as e:
        sys.exit(str(e))  # the file and the key, without a traceback
    done = ensure_app_database(database_url)
    done += [f"applied the migration {name}" for name in apply_migrations(database_url)]
    done += ensure_database()
    print("\n".join(done) if done else "Nothing to do: the databases, the tables and the roles are in place.")


if __name__ == "__main__":
    main()
