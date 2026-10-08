"""The database the SQL chatbot answers from (`tables_database` in config/connections.yaml): its roles,
its connections and its schemas.

Two roles, and nothing else touches it. The reader can only select; it is what runs a question's SQL.
The loader makes schemas and tables and loads the CSV files; where it is left out of the settings (a
database whose tables are already there) the import is off. Neither is an admin: the roles and their
limits are made by scripts/provision.sql.
A schema is one dataset; the registry of schemas (`db_schemas`, among our tables) is kept in step with
what the chatbot may read here. A schema is either made by the import, or `registered`: it was already in
the database and is only described in the registry (ingest/register.py)."""

import re

import psycopg
from psycopg import sql
from psycopg.conninfo import conninfo_to_dict

from rag_lab.core.config import NAME_PATTERN
from rag_lab.core.qdrant import QdrantStore
from rag_lab.core.settings import load
from rag_lab.core.store import MetricsStore


def _url(which: str) -> str:
    """A role's connection URL, from `tables_database` in config/connections.yaml."""
    url = getattr(load().connections.tables_database, which)
    if url is None:
        raise RuntimeError(
            f"config/connections.yaml: tables_database.{which} is not set, so CSV files cannot be imported here."
        )
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
    """Raise ValueError for a name the import cannot use for a schema. Names are quoted everywhere they
    are used, so the pattern is only there to keep them plain. A registered schema is not checked here:
    it has the name it was given, and may be `public`."""
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
    """Remove a schema from what the chatbot knows: its good answers' collection in Qdrant, then its
    registry rows (its chats and good answers go with them). An imported schema is dropped from the
    database too, with everything in it; a registered schema's tables are never touched. The stores
    first, the rows last, so a failure can be repeated. The CSV files in data/tables are not touched."""
    found = metrics.get_db_schema(name)
    qdrant.drop_examples(name)
    if not (found and found["registered"]):
        check_schema_name(name)
        with psycopg.connect(loader_url()) as conn:
            conn.execute(sql.SQL("DROP SCHEMA IF EXISTS {} CASCADE").format(sql.Identifier(name)))
    metrics.delete_db_schema(name)


def drop_table(metrics: MetricsStore, schema: str, table: str) -> None:
    """Drop an imported table and its registry row, the data first so it can be repeated. Its CSV file
    in data/tables is not touched. A table of a registered schema only leaves the registry."""
    found = metrics.get_db_schema(schema)
    if not (found and found["registered"]):
        check_schema_name(schema)
        with psycopg.connect(loader_url()) as conn:
            conn.execute(sql.SQL("DROP TABLE IF EXISTS {}").format(sql.Identifier(schema, table)))
    metrics.delete_db_table(schema, table)
