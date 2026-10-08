"""Describe tables that are already in the database, so the SQL chatbot can be asked about them.

The chatbot reads a description of each table from the registry (`db_schemas`, `db_tables`): its columns,
their types, example values and ranges. The CSV import writes that for the tables it loads. For a
database whose tables are already there, `register` writes it instead, for every schema named in
`tables_database.existing_schemas` (config/connections.yaml): the tables and column types come from
PostgreSQL's catalog, and each column is profiled with the queries the import uses.

It runs as the reader role, so it cannot change anything, and the reader must have been granted the schema
(scripts/grant_read.sql). Descriptions are optional and go in config/schemas/<schema>.yaml, in the format
of a folder's `_schema.yaml` without `type`. Run it again when the tables change: tables that are gone
leave the registry, and so does a schema that is no longer listed (with its chats and good answers; its
tables are never touched). Plain Python, no Dagster."""

import psycopg
from psycopg import sql

from rag_lab.core.config import ImportConfig
from rag_lab.core.database import reader_url
from rag_lab.core.qdrant import QdrantStore
from rag_lab.core.settings import config_dir, load
from rag_lab.core.store import MetricsStore
from rag_lab.ingest.folder import read_notes
from rag_lab.ingest.importer import CsvError, profile_column

# The tables, views and materialised views of a schema that this role may read, with every column's type
# as PostgreSQL writes it and the category of that type (N number, D date or time, S text, E enum).
COLUMNS = """
    SELECT c.relname, a.attname, format_type(a.atttypid, a.atttypmod), t.typcategory
    FROM pg_class c
    JOIN pg_namespace n ON n.oid = c.relnamespace
    JOIN pg_attribute a ON a.attrelid = c.oid AND a.attnum > 0 AND NOT a.attisdropped
    JOIN pg_type t ON t.oid = a.atttypid
    WHERE n.nspname = %s AND c.relkind IN ('r', 'p', 'v', 'm') AND NOT c.relispartition
      AND has_table_privilege(c.oid, 'SELECT')
    ORDER BY c.relname, a.attnum"""


def _describe(conn: psycopg.Connection, schema: str, table: str, columns: list[dict], cfg: ImportConfig) -> tuple[int, list[str]]:
    """Count the table's rows and profile its columns. A query the database refuses or stops (a type
    that cannot be compared, a table too large for the reader's time limit) leaves that part empty and is
    returned as a warning: a table with half a profile is still better described than one left out."""
    warnings = []
    rows = 0
    try:
        rows = conn.execute(sql.SQL("SELECT count(*) FROM {}").format(sql.Identifier(schema, table))).fetchone()[0]
    except psycopg.Error as e:
        warnings.append(f"{schema}.{table}: the rows could not be counted ({(e.diag.message_primary or str(e)).strip()})")
    for column in columns:
        category = column.pop("category")
        try:
            profile_column(conn, schema, table, column, cfg, ranged=category in ("N", "D"), sampled=category in ("S", "E"))
        except psycopg.Error as e:
            warnings.append(f"{schema}.{table}.{column['name']}: not profiled ({(e.diag.message_primary or str(e)).strip()})")
    return rows, warnings


def register(metrics: MetricsStore, qdrant: QdrantStore, cfg: ImportConfig) -> dict:
    """Bring the registry in step with `existing_schemas`. Returns the tables of each schema and the
    warnings. `qdrant` is for the good answers of a schema that is no longer listed."""
    wanted = load().connections.tables_database.existing_schemas
    known = {s["name"]: s for s in metrics.list_db_schemas()}
    done: dict = {"schemas": {}, "removed": [], "warnings": []}
    for name, found in known.items():
        if found["registered"] and name not in wanted:
            qdrant.drop_examples(name)
            metrics.delete_db_schema(name)
            done["removed"].append(name)
    if not wanted:
        return done

    # Autocommit: each query stands alone, so one that fails does not stop the ones after it.
    with psycopg.connect(reader_url(), autocommit=True) as conn:
        for schema in wanted:
            if schema in known and not known[schema]["registered"]:
                raise CsvError(f"The schema '{schema}' was made by the CSV import, so it cannot be registered as existing.")
            notes = read_notes(schema, config_dir() / "schemas" / f"{schema}.yaml")
            tables: dict[str, list[dict]] = {}
            for table, column, kind, category in conn.execute(COLUMNS, (schema,)):
                tables.setdefault(table, []).append({"name": column, "type": kind, "category": category})
            if not tables:
                done["warnings"].append(
                    f"{schema}: no table can be read. Does the schema exist, and was scripts/grant_read.sql run for it?"
                )
            if schema not in known:
                metrics.add_db_schema(schema, registered=True)
            metrics.set_db_schema_description(schema, notes.description or "")
            for table, columns in tables.items():
                written = notes.tables.get(table)
                unknown = sorted(set(written.columns) - {c["name"] for c in columns}) if written else []
                if unknown:
                    raise CsvError(f"config/schemas/{schema}.yaml: tables.{table}.columns.{unknown[0]}: the table has no such column.")
                for c in columns:
                    note = written.columns.get(c["name"]) if written else None
                    c["description"] = (note.description if note else None) or ""
                rows, warnings = _describe(conn, schema, table, columns, cfg)
                done["warnings"] += warnings
                metrics.upsert_db_table(schema, table, None, rows, columns)
                metrics.set_db_table_descriptions(schema, table, (written.description if written else None) or "", {})
            for gone in {t["table_name"] for t in metrics.list_db_tables(schema)} - set(tables):
                metrics.delete_db_table(schema, gone)
            done["schemas"][schema] = sorted(tables)
    return done
