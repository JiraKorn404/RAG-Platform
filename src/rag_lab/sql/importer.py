"""Import a CSV file into a table of the database for imported tables.

Reading the start of a file gives the delimiter, clean names and a guess at each column's type; loading
is one transaction as the loader role (create the table, COPY the file in, profile the columns), so a bad
row leaves no table. The pure parts (names, types, delimiter) are what everything after rests on."""

import csv
import io
import unicodedata
from collections import Counter
from dataclasses import dataclass
from datetime import date, datetime

import psycopg
from psycopg import sql

from rag_lab.config import ImportConfig
from rag_lab.metrics.store import MetricsStore
from rag_lab.sql.database import check_schema_name, check_table_name, loader_url

TYPES = ("bigint", "double precision", "boolean", "date", "timestamp", "timestamptz", "text")
DELIMITERS = (",", ";", "\t", "|")
# PostgreSQL does not allow a column to be named like one of its system columns.
SYSTEM_COLUMNS = {"tableoid", "xmin", "cmin", "xmax", "cmax", "ctid"}
BIGINT_MAX = 2**63 - 1


class CsvError(ValueError):
    """A file that cannot be imported, with a message for the person who put it there."""


NOT_UTF8 = "The file is not UTF-8. Save it as 'CSV UTF-8' and put it in the folder again."


# --- names ---------------------------------------------------------------------------------------


def clean_identifier(text: str, fallback: str = "column") -> str:
    """A name PostgreSQL accepts and that stays readable: lower case, anything that is not a letter, a
    digit or a combining mark (Thai vowels are marks) becomes `_`, a leading digit gets a prefix, at most
    63 bytes. Letters of any language are kept. `fallback` stands in for a name with nothing left."""
    text = unicodedata.normalize("NFKC", text).strip().lower()
    kept = "".join(c if c.isalnum() or c == "_" or unicodedata.category(c).startswith("M") else "_" for c in text)
    name = "_".join(part for part in kept.split("_") if part)
    if not name:
        name = fallback
    if name[0].isdigit():
        name = f"{fallback}_{name}"
    return _cut(name, 63)


def _cut(name: str, size: int) -> str:
    """At most `size` bytes, without cutting a character in two."""
    return name.encode()[:size].decode(errors="ignore")


def clean_names(headers: list[str]) -> list[str]:
    """Clean names for a header row: unique, and not a system column. Duplicates get `_2`, `_3`."""
    names: list[str] = []
    for header in headers:
        name = clean_identifier(header)
        if name in SYSTEM_COLUMNS:
            name += "_col"
        base, n = name, 1
        while name in names:
            n += 1
            suffix = f"_{n}"
            name = _cut(base, 63 - len(suffix)) + suffix
        names.append(name)
    return names


# --- types ---------------------------------------------------------------------------------------


def _has_leading_zero(value: str) -> bool:
    """007, 0042, -007: a code or an identifier written with digits, not a number. 0 and 0.5 are numbers."""
    digits = value.lstrip("+-")
    return len(digits) > 1 and digits[0] == "0" and digits[1].isdigit()


def _is_int(value: str) -> bool:
    digits = value[1:] if value[:1] in "+-" else value
    return (
        digits.isascii()
        and digits.isdigit()
        and not _has_leading_zero(value)
        and len(digits) <= 19
        and int(digits) <= BIGINT_MAX
    )


def _is_float(value: str) -> bool:
    try:
        float(value)
    except ValueError:
        return False
    return (
        value.isascii()
        and value.lower() not in ("nan", "inf", "-inf", "+inf", "infinity", "-infinity")
        and "_" not in value
        and not value.lstrip("+-").isdigit()  # digits only and not a bigint: an id too long to be a number
    )


def _is_date(value: str) -> bool:
    try:
        return len(value) == 10 and date.fromisoformat(value) is not None
    except ValueError:
        return False


def _timestamp_kind(value: str) -> str | None:
    """'timestamp', 'timestamptz' (it has an offset) or None, for an ISO timestamp."""
    if len(value) < 16 or value[10] not in " T" or not _is_date(value[:10]):
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    return "timestamptz" if parsed.tzinfo else "timestamp"


def infer_type(values: list[str]) -> str:
    """The narrowest type every non-empty value fits: bigint, double precision, boolean, date, timestamp
    (or timestamptz when they carry an offset), otherwise text. Only ISO dates and timestamps are
    recognised; a column that is all empty is text."""
    values = [v.strip() for v in values if v.strip()]
    if not values:
        return "text"
    if all(_is_int(v) for v in values):
        return "bigint"
    if all(_is_int(v) or (_is_float(v) and not _has_leading_zero(v)) for v in values):
        return "double precision"
    if all(v.lower() in ("true", "false") for v in values):
        return "boolean"
    if all(_is_date(v) for v in values):
        return "date"
    kinds = set()
    for v in values:
        kind = "timestamp" if _is_date(v) else _timestamp_kind(v)  # a date among timestamps is one at midnight
        if kind is None:
            return "text"
        kinds.add(kind)
    return kinds.pop() if len(kinds) == 1 else "text"  # offsets on some values and not on others: text


# --- reading the start of a file -----------------------------------------------------------------


def sniff_delimiter(text: str) -> str:
    """The delimiter that splits the first lines into the same number of fields, more than one. A comma
    when none does."""
    lines = text.splitlines()[:20]
    best, best_key = ",", (0, 0)
    for delimiter in DELIMITERS:
        counts = [len(row) for row in csv.reader(lines, delimiter=delimiter) if row]
        if not counts:
            continue
        fields, lines_with = Counter(counts).most_common(1)[0]
        if fields > 1 and (lines_with, fields) > best_key:
            best, best_key = delimiter, (lines_with, fields)
    return best


@dataclass
class CsvSample:
    delimiter: str
    headers: list[str]  # as written in the file
    names: list[str]  # clean and unique: the column names the table will have
    types: list[str]  # guessed from the sample
    rows: list[list[str]]  # the sample
    truncated: bool  # the file is longer than the sample


def _decode_sample(data: bytes, cfg: ImportConfig) -> tuple[str, bool]:
    head = data[: cfg.sample_bytes]
    truncated = len(data) > len(head)
    try:
        text = head.decode("utf-8-sig")
    except UnicodeDecodeError as e:
        if truncated and e.start >= len(head) - 4:  # a character cut in two by the end of the sample
            text = head[: e.start].decode("utf-8-sig")
        else:
            raise CsvError(NOT_UTF8) from None
    if truncated and "\n" in text:
        text = text[: text.rindex("\n") + 1]  # the last line may be cut
    return text, truncated


def parse_sample(data: bytes, delimiter: str | None = None, cfg: ImportConfig | None = None) -> CsvSample:
    """Read the start of a file: its delimiter (sniffed unless given), headers, clean names, and a type
    for each column guessed from up to `sample_rows` rows."""
    cfg = cfg or ImportConfig()
    text, truncated = _decode_sample(data, cfg)
    if not text.strip():
        raise CsvError("The file is empty.")
    delimiter = delimiter or sniff_delimiter(text)
    reader = csv.reader(io.StringIO(text, newline=""), delimiter=delimiter)
    try:
        headers = next(reader)
        rows = [row for _, row in zip(range(cfg.sample_rows), reader) if row]
    except csv.Error as e:
        raise CsvError(f"The file could not be read as CSV: {e}") from None
    if not rows:
        raise CsvError("The file has a header but no rows.")
    width = len(headers)
    rows = [(row + [""] * width)[:width] for row in rows]  # a short or long row is reported when it is loaded
    types = [infer_type([row[i] for row in rows]) for i in range(width)]
    return CsvSample(delimiter, headers, clean_names(headers), types, rows, truncated)


# --- loading -------------------------------------------------------------------------------------


def _profile(conn: psycopg.Connection, schema: str, table: str, columns: list[dict], cfg: ImportConfig) -> None:
    """Fill in each column's profile from the loaded data: nulls, distinct values, example values for a text
    column with few distinct values, the range of numbers and dates. It is what lets a small model see that
    `status` holds `Shipped`, not `shipped`."""
    source = sql.Identifier(schema, table)
    for column in columns:
        name, kind = sql.Identifier(column["name"]), column["type"]
        ranged = kind in ("bigint", "double precision", "date", "timestamp", "timestamptz")
        query = sql.SQL(
            "SELECT count(*) - count({c}), count(DISTINCT {c})" + (", min({c})::text, max({c})::text" if ranged else "")
            + " FROM {t}"
        ).format(c=name, t=source)
        row = conn.execute(query).fetchone()
        column["nulls"], column["distinct"] = row[0], row[1]
        column["min"], column["max"] = (row[2], row[3]) if ranged else (None, None)
        column["samples"] = []
        if kind == "text" and 0 < row[1] <= cfg.max_distinct_for_samples:
            samples = conn.execute(
                sql.SQL(
                    "SELECT {c} FROM {t} WHERE {c} IS NOT NULL GROUP BY {c} ORDER BY count(*) DESC, {c} LIMIT %s"
                ).format(c=name, t=source),
                (cfg.sample_values,),
            ).fetchall()
            column["samples"] = [r[0] for r in samples]


def import_csv(
    metrics: MetricsStore,
    schema: str,
    table: str,
    data: bytes,
    *,
    delimiter: str,
    headers: list[str],
    names: list[str],
    types: list[str],
    replace: bool = False,
    source_file: str | None = None,
    cfg: ImportConfig | None = None,
) -> int:
    """Load a CSV file into `schema.table` as the loader role and register it. Returns the row count.

    One transaction: create the table, COPY the file in, check the size and profile the columns, so a bad
    row or a file over the limits leaves nothing behind (and, with `replace`, the old table). A table
    that exists is refused unless `replace`."""
    cfg = cfg or ImportConfig()
    check_schema_name(schema)
    check_table_name(table)
    if not metrics.get_db_schema(schema):
        raise CsvError(f"There is no schema '{schema}'.")
    if len(data) > cfg.max_bytes:
        raise CsvError(f"The file is {len(data) / 1e6:.1f} MB; the limit is {cfg.max_bytes / 1e6:.1f} MB.")
    if not (len(headers) == len(names) == len(types)) or len(set(names)) != len(names):
        raise CsvError("The columns do not match the file's header.")
    if any(t not in TYPES for t in types):
        raise CsvError(f"A column type is not one of {', '.join(TYPES)}.")

    target = sql.Identifier(schema, table)
    old = metrics.get_db_table(schema, table)
    columns = [
        {"name": n, "type": t, "header": h, "description": h if h != n else ""}
        for h, n, t in zip(headers, names, types)
    ]
    if old:  # a table imported again keeps what was written about its columns
        kept = {c["name"]: c["description"] for c in old["columns"]}
        for column in columns:
            column["description"] = kept.get(column["name"], column["description"])
    try:
        with psycopg.connect(loader_url()) as conn:
            exists = conn.execute(
                "SELECT 1 FROM information_schema.tables WHERE table_schema = %s AND table_name = %s",
                (schema, table),
            ).fetchone()
            if exists and not replace:
                raise CsvError(f"The table {schema}.{table} already exists. Choose replace to import over it.")
            if exists:
                conn.execute(sql.SQL("DROP TABLE {}").format(target))
            conn.execute(
                sql.SQL("CREATE TABLE {} ({})").format(
                    target, sql.SQL(", ").join(sql.SQL("{} {}").format(sql.Identifier(n), sql.SQL(t)) for n, t in zip(names, types))
                )
            )
            copy = sql.SQL("COPY {} ({}) FROM STDIN WITH (FORMAT csv, HEADER true, DELIMITER {})").format(
                target, sql.SQL(", ").join(map(sql.Identifier, names)), sql.Literal(delimiter)
            )
            with conn.cursor() as cur:
                with cur.copy(copy) as writer:
                    text = io.TextIOWrapper(io.BytesIO(data), encoding="utf-8-sig", newline="")
                    while chunk := text.read(1 << 16):
                        writer.write(chunk)
                rows = cur.rowcount
            if rows > cfg.max_rows:
                raise CsvError(f"The file has {rows:,} rows; the limit is {cfg.max_rows:,}.")
            _profile(conn, schema, table, columns, cfg)
    except UnicodeDecodeError:
        raise CsvError(NOT_UTF8) from None
    except psycopg.Error as e:  # a bad row: say which one
        where = (e.diag.context or "").splitlines()[-1] if e.diag.context else ""
        raise CsvError(f"{(e.diag.message_primary or str(e)).strip()}. {where}".strip()) from None

    # The table is loaded and committed by now. If this fails the table exists without a registry row,
    # and importing it again with replace puts that right.
    metrics.upsert_db_table(schema, table, source_file, rows, columns)
    return rows
