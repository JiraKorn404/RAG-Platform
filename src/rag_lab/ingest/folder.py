"""The CSV files to import: data/tables/<schema>/<table>.csv.

A folder is a schema (one dataset) and a CSV file in it is a table, named after the file. A folder may
also hold `_schema.yaml`, which describes the dataset for the text-to-SQL model and can set the type of
a column the import would guess differently. Everything in it is optional:

    description: Orders of the web shop, 2023 to 2024
    tables:
      orders:
        description: One row per order
        columns:
          total: {description: "Order total in baht, with VAT", type: double precision}
          status: {description: "Shipped, Cancelled or Open"}

Columns are named as they are in the table (`python -m rag_lab tables schema --schema <name>` prints them).
Dagster's sensor (definitions.py) starts an import for every file that is new or has changed;
`import_file` is what that run does. Plain Python, no Dagster."""

import time
from dataclasses import dataclass
from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, ValidationError

from rag_lab.core.config import ImportConfig
from rag_lab.core.database import check_schema_name, create_schema
from rag_lab.core.settings import TABLES_DIR, problems
from rag_lab.core.store import MetricsStore
from rag_lab.ingest.importer import CsvError, clean_identifier, import_csv, parse_sample
from rag_lab.ingest.source import content_hash

NOTES_FILE = "_schema.yaml"
SETTLED_SECONDS = 5  # a file changed more recently than this may still be being written


class _Notes(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ColumnNotes(_Notes):
    description: str | None = None
    type: str | None = None  # one of importer.TYPES, instead of the guessed one


class TableNotes(_Notes):
    description: str | None = None
    columns: dict[str, ColumnNotes] = {}


class SchemaNotes(_Notes):
    description: str | None = None
    tables: dict[str, TableNotes] = {}


@dataclass(frozen=True)
class TableFile:
    schema: str  # the folder's name
    table: str  # the file's name, made into a table name
    path: Path

    @property
    def key(self) -> str:
        """`schema.table`: the table's partition in Dagster. Neither part can contain a dot."""
        return f"{self.schema}.{self.table}"


def table_files() -> tuple[list[TableFile], list[str]]:
    """Every CSV file in data/tables, and what cannot be imported with the reason (a folder whose name
    cannot be a schema, two files that give the same table name)."""
    files, unusable = [], []
    if not TABLES_DIR.exists():
        return files, unusable
    for folder in sorted(p for p in TABLES_DIR.iterdir() if p.is_dir()):
        try:
            check_schema_name(folder.name)
        except ValueError as e:
            unusable.append(f"data/tables/{folder.name}/: {e}")
            continue
        seen: dict[str, str] = {}
        for path in sorted(folder.iterdir()):
            if not path.is_file() or path.suffix.lower() != ".csv":
                continue
            table = clean_identifier(path.stem, fallback="table")
            if table in seen:
                unusable.append(
                    f"data/tables/{folder.name}/{path.name}: gives the same table name as {seen[table]} ('{table}')"
                )
                continue
            seen[table] = path.name
            files.append(TableFile(folder.name, table, path))
    return files, unusable


def is_settled(path: Path) -> bool:
    """Whether the file can be read and has not changed for a few seconds. A file that Windows is still
    copying into the folder cannot be opened from the container until the copy ends."""
    try:
        with path.open("rb"):
            pass
        return time.time() - path.stat().st_mtime >= SETTLED_SECONDS
    except OSError:
        return False


def version(file: TableFile) -> str:
    """Changes when the file or its folder's `_schema.yaml` changes, so either one imports it again."""
    notes = file.path.parent / NOTES_FILE
    return content_hash(file.path) + (f":{content_hash(notes)}" if notes.exists() else "")


def read_notes(schema: str, path: Path | None = None) -> SchemaNotes:
    """What is written about a schema: its `_schema.yaml`, or the file at `path` (a registered schema's)."""
    path = path or TABLES_DIR / schema / NOTES_FILE
    if not path.exists():
        return SchemaNotes()
    try:
        return SchemaNotes.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")) or {})
    except yaml.YAMLError as e:
        raise CsvError(f"{path}: not valid YAML: {e}") from e
    except ValidationError as e:
        raise CsvError(f"{path}: {problems(e)}") from e


def import_file(metrics: MetricsStore, file: TableFile, cfg: ImportConfig) -> dict:
    """Load one CSV file as its table, replacing the table if it exists, and write what `_schema.yaml`
    says about it. The schema is made if this is its first table. Returns the numbers the run shows.
    A file that cannot be loaded raises CsvError and leaves the table as it was."""
    notes = read_notes(file.schema)
    table_notes = notes.tables.get(file.table, TableNotes())
    data = file.path.read_bytes()
    sample = parse_sample(data, cfg=cfg)

    unknown = sorted(set(table_notes.columns) - set(sample.names))
    if unknown:
        raise CsvError(
            f"{NOTES_FILE}: tables.{file.table}.columns.{unknown[0]}: {file.path.name} has no such column "
            f"(its columns are {', '.join(sample.names)})."
        )
    wanted = {name: column.type for name, column in table_notes.columns.items() if column.type}
    types = [wanted.get(name, guessed) for name, guessed in zip(sample.names, sample.types)]

    if not metrics.get_db_schema(file.schema):
        create_schema(metrics, file.schema)
    rows = import_csv(
        metrics,
        file.schema,
        file.table,
        data,
        delimiter=sample.delimiter,
        headers=sample.headers,
        names=sample.names,
        types=types,
        replace=True,
        source_file=file.path.name,
        cfg=cfg,
    )

    if notes.description is not None:
        metrics.set_db_schema_description(file.schema, notes.description)
    described = {name: c.description for name, c in table_notes.columns.items() if c.description is not None}
    if table_notes.description is not None or described:
        kept = metrics.get_db_table(file.schema, file.table)["description"]
        description = table_notes.description if table_notes.description is not None else kept
        metrics.set_db_table_descriptions(file.schema, file.table, description, described)
    return {
        "schema": file.schema,
        "table": file.table,
        "source_file": file.path.name,
        "rows": rows,
        "columns": dict(zip(sample.names, types)),
        "bytes": len(data),
    }
