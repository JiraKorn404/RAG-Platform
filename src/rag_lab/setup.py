"""Get the databases ready: python -m rag_lab.setup

Creates the `rag_metrics` database if it is missing and applies its migrations, then makes sure the
database for imported tables exists with its two roles (sql/bootstrap.py). The `setup` service in
docker-compose.yml runs this once when the stack starts, before the UI and Dagster; it is safe to run
again, and changes nothing when everything is in place. It needs METRICS_DATABASE_URL (the admin
credentials), SQL_LOADER_URL and SQL_READER_URL."""

import os

import psycopg
from psycopg import sql
from psycopg.conninfo import conninfo_to_dict, make_conninfo

from rag_lab.metrics.migrate import apply_migrations
from rag_lab.sql.bootstrap import ensure_database


def ensure_metrics_database(database_url: str) -> bool:
    """Create the database the URL names if it does not exist. Returns whether it was created."""
    name = conninfo_to_dict(database_url)["dbname"]
    with psycopg.connect(make_conninfo(database_url, dbname="postgres"), autocommit=True) as admin:
        if admin.execute("SELECT 1 FROM pg_database WHERE datname = %s", (name,)).fetchone():
            return False
        admin.execute(sql.SQL("CREATE DATABASE {}").format(sql.Identifier(name)))
    return True


def main() -> None:
    database_url = os.environ["METRICS_DATABASE_URL"]
    done = ["created the database rag_metrics"] if ensure_metrics_database(database_url) else []
    done += [f"applied the migration {name}" for name in apply_migrations(database_url)]
    done += ensure_database()
    print("\n".join(done) if done else "Nothing to do: the databases, the tables and the roles are in place.")


if __name__ == "__main__":
    main()
