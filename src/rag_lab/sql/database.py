"""The database for imported tables (`tables_database` in config/connections.yaml): its roles, its
connections and its schemas.

Two roles, and nothing else touches it. The loader makes schemas and tables and loads data. The reader
can only select; it is what runs a question's SQL. The admin role is used only by `bootstrap.py`.
A schema is one dataset; the registry of schemas (`db_schemas`, in `rag_metrics`) is kept in step with
the schemas that exist here."""

import re

import psycopg
from psycopg import sql
from psycopg.conninfo import conninfo_to_dict

from rag_lab.config import NAME_PATTERN
from rag_lab.metrics.store import MetricsStore
from rag_lab.settings import load
from rag_lab.sql.examples import drop_index
from rag_lab.storage.qdrant import QdrantStore

def _url(which: str) -> str:
    """A role's connection URL, from `tables_database` in config/connections.yaml."""
    url = getattr(load().connections.tables_database, which)
    if not conninfo_to_dict(url).get("password"):
        raise RuntimeError(
            f"config/connections.yaml: tables_database.{which} has no password. Set SQL_LOADER_PASSWORD and "
            "SQL_READER_PASSWORD in .env (letters and digits only) and recreate the containers."
        )
    return url


def loader_url() -> str:
    return _url("loader_url")


def reader_url() -> str:
    return _url("reader_url")


def role_of(url: str) -> str:
    return conninfo_to_dict(url)["user"]


def check_schema_name(name: str) -> None:
    """Raise ValueError for a name that is not a usable schema name. Names are quoted everywhere they
    are used, so the pattern is only there to keep them plain."""
    if not re.fullmatch(NAME_PATTERN, name) or len(name.encode()) > 63:
        raise ValueError(
            f"'{name}' is not a usable schema name: lower-case letters, digits, '_' and '-', starting with "
            "a letter or digit, at most 63 bytes."
        )
    if name in ("public", "information_schema") or name.startswith("pg_"):
        raise ValueError(f"'{name}' is reserved by PostgreSQL.")


def check_table_name(name: str) -> None:
    """Raise ValueError for a table name that could not have come from the importer: no path separators,
    and at most 63 bytes (it is always quoted in SQL)."""
    if not name or len(name.encode()) > 63 or any(c in name for c in ("/", "\\", "\x00")) or name.startswith("."):
        raise ValueError(f"'{name}' is not a usable table name.")


def create_schema(metrics: MetricsStore, name: str, description: str = "") -> None:
    """Make a schema the reader can read and the loader can fill, and register it. It can be repeated
    after a failure: the schema is made only if missing, and a name already registered is refused."""
    check_schema_name(name)
    if metrics.get_db_schema(name):
        raise ValueError(f"The schema '{name}' already exists.")
    schema, reader = sql.Identifier(name), sql.Identifier(role_of(reader_url()))
    with psycopg.connect(loader_url()) as conn:
        conn.execute(sql.SQL("CREATE SCHEMA IF NOT EXISTS {}").format(schema))
        conn.execute(sql.SQL("GRANT USAGE ON SCHEMA {} TO {}").format(schema, reader))
        # The loader owns what it creates, so this makes every table it adds readable by the reader.
        conn.execute(sql.SQL("ALTER DEFAULT PRIVILEGES IN SCHEMA {} GRANT SELECT ON TABLES TO {}").format(schema, reader))
    metrics.add_db_schema(name, description)


def drop_schema(metrics: MetricsStore, name: str, qdrant: QdrantStore) -> None:
    """Drop a schema with everything in it, its good answers' collection in Qdrant, then its registry
    rows (its chats and good answers go with them). The stores first, the rows last, so a failure can
    be repeated. The CSV files in data/tables are not touched."""
    check_schema_name(name)
    drop_index(qdrant, name)
    with psycopg.connect(loader_url()) as conn:
        conn.execute(sql.SQL("DROP SCHEMA IF EXISTS {} CASCADE").format(sql.Identifier(name)))
    metrics.delete_db_schema(name)


def drop_table(metrics: MetricsStore, schema: str, table: str) -> None:
    """Drop an imported table and its registry row, the data first so it can be repeated. Its CSV file
    in data/tables is not touched."""
    check_schema_name(schema)
    with psycopg.connect(loader_url()) as conn:
        conn.execute(sql.SQL("DROP TABLE IF EXISTS {}").format(sql.Identifier(schema, table)))
    metrics.delete_db_table(schema, table)
