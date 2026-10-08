"""The schema as one block of text for the model: every table with its description, every column with its
type, description and example values or range, and the joins that look likely.

There is no retrieval and no index: the model is given the whole schema, which is what a few tables
allow. It is read from the registry (`db_tables`), where the import left each column's profile, so
rendering needs no query against the data. A schema too long for the budget is refused, not cut."""

import re
from dataclasses import dataclass

from rag_lab.core.store import MetricsStore

# A column is a likely join key when it has the same name and type in two tables and either exactly one
# side is unique (a key, and what refers to it) or it is named like a key. `status` in two tables is not
# a join, and neither is `name` when it is unique in both.
KEY_SUFFIXES = ("_id", "_key", "_code", "_no", "_number")
KEY_NAMES = {"id", "key", "code"}
VALUE_LIMIT = 40  # characters of an example value that go in the text


class SchemaTooLarge(ValueError):
    """The schema's description is longer than the model is to be given."""


@dataclass
class SchemaText:
    schema: str
    text: str
    tables: list[str]

    @property
    def chars(self) -> int:
        return len(self.text)

    def check(self, budget: int) -> None:
        """Raise `SchemaTooLarge` with the size when the text is over the budget."""
        if self.chars > budget:
            raise SchemaTooLarge(
                f"The schema '{self.schema}' is {self.chars:,} characters ({len(self.tables)} tables); the model "
                f"is given at most {budget:,}. Showing the model only the tables a question needs is not built "
                "yet, so remove tables or columns, or raise `schema_char_budget` if the model's context allows it."
            )


def quote(name: str) -> str:
    """The name as SQL needs it: plain when it is a simple lower-case identifier, else in double quotes."""
    return name if re.fullmatch(r"[a-z_][a-z0-9_]*", name) else '"' + name.replace('"', '""') + '"'


def _value(value) -> str:
    text = " ".join(str(value).split())
    if len(text) > VALUE_LIMIT:
        text = text[: VALUE_LIMIT - 1] + "…"
    return "'" + text.replace("'", "''") + "'"


def _bound(value: str) -> str:
    return value.removesuffix(" 00:00:00")  # a timestamp that is always midnight is a date


def _is_unique(column: dict, rows: int) -> bool:
    return rows > 1 and column.get("distinct") == rows - column.get("nulls", 0) and column.get("nulls", 0) < rows


def _column_line(column: dict, rows: int) -> str:
    notes = []
    if column.get("description"):
        notes.append(column["description"].strip().rstrip("."))
    distinct, samples = column.get("distinct") or 0, column.get("samples") or []
    if _is_unique(column, rows):  # a key: say so, and what it looks like (the first values are no more telling)
        notes.append("unique" + ("; e.g. " + ", ".join(_value(v) for v in samples[:3]) if samples else ""))
    elif samples:
        shown = ", ".join(_value(v) for v in samples)
        notes.append(f"values: {shown}" if len(samples) >= distinct else f"most common of {distinct} values: {shown}")
    elif distinct and column["type"] != "boolean":
        notes.append(f"{distinct} distinct values")
    if column.get("min") is not None and column.get("max") is not None:
        notes.append(f"range {_bound(column['min'])} to {_bound(column['max'])}")
    if column.get("nulls"):
        notes.append(f"{column['nulls']} NULL")
    return f"  - {quote(column['name'])} ({column['type']})" + (": " + "; ".join(notes) if notes else "")


def likely_joins(schema: str, tables: list[dict]) -> list[str]:
    """Pairs of columns that look like the same key in two tables. A guess from names and types: a CSV
    declares no foreign keys."""
    joins = []
    for i, a in enumerate(tables):
        for b in tables[i + 1 :]:
            shared = {c["name"]: c for c in a["columns"]}
            for column in b["columns"]:
                other = shared.get(column["name"])
                if not other or other["type"] != column["type"]:
                    continue
                name = column["name"]
                one_side_unique = _is_unique(other, a["row_count"]) != _is_unique(column, b["row_count"])
                if one_side_unique or name in KEY_NAMES or name.endswith(KEY_SUFFIXES):
                    left = f"{quote(schema)}.{quote(a['table_name'])}.{quote(name)}"
                    right = f"{quote(schema)}.{quote(b['table_name'])}.{quote(name)}"
                    joins.append(f"{left} = {right} ({column['type']})")
    return joins


def render(metrics: MetricsStore, schema: str) -> SchemaText:
    """The whole schema as text. Raises ValueError for a schema that does not exist."""
    found = metrics.get_db_schema(schema)
    if not found:
        raise ValueError(f"There is no schema '{schema}'.")
    tables = metrics.list_db_tables(schema)
    head = f"Schema {quote(schema)}." + (f" {found['description'].strip()}" if found["description"].strip() else "")
    if not tables:
        return SchemaText(schema, f"{head}\nIt has no tables.", [])
    blocks = [head]
    for table in tables:
        title = f"Table {quote(schema)}.{quote(table['table_name'])}: {table['row_count']:,} rows."
        if table["description"].strip():
            title += f" {table['description'].strip()}"
        lines = [_column_line(c, table["row_count"]) for c in table["columns"]]
        blocks.append("\n".join([title, *lines]))
    joins = likely_joins(schema, tables)
    if len(tables) > 1:
        blocks.append(
            "Likely joins (guessed from column names, not declared):\n" + "\n".join(f"  - {j}" for j in joins)
            if joins
            else "Likely joins: none found (no column has the same name and type in two tables)."
        )
    return SchemaText(schema, "\n\n".join(blocks), [t["table_name"] for t in tables])
