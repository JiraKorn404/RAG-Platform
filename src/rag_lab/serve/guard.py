"""Checks a generated query before it runs. It is not the only defence: the reader role can only select,
and runs in a read-only transaction with a timeout. The guard is what turns a refusal into a message the
model can repair from, and what keeps a chat to its own schema, which the role alone cannot do (the
reader may read every schema).

The SQL that runs is the one this module writes out from the parsed query (`sqlglot`, PostgreSQL
dialect), not the text the model wrote, so what was checked is what runs: nothing the parser read as a
comment or a string can reach the database as a statement."""

from collections.abc import Iterable

import sqlglot
from sqlglot import exp
from sqlglot.errors import SqlglotError


class Refused(ValueError):
    """Why a query was not allowed, worded for the model that wrote it."""


# A node of any of these anywhere in the query (a data-modifying CTE hides one inside a SELECT) is refused.
FORBIDDEN = tuple(
    getattr(exp, name)
    for name in (
        "Insert", "Update", "Delete", "Merge", "Create", "Drop", "Alter", "Command", "Set", "Copy",
        "Into", "Lock", "Transaction", "Commit", "Rollback", "TruncateTable", "Grant", "Revoke",
    )
    if hasattr(exp, name)
)
# A function sqlglot does not know by name is refused unless it is one of these: ordinary reading
# functions. This is what keeps `pg_read_file`, `pg_sleep`, `set_config`, `nextval` and `query_to_xml`
# (which runs another query, outside the schema check) out. Add a name here when a real question needs it.
ANONYMOUS_ALLOWED = {
    "age", "generate_series", "to_date", "to_timestamp", "initcap", "split_part", "regexp_replace",
    "regexp_match", "width_bucket", "percentile_cont", "percentile_disc", "mode", "corr", "string_agg",
    "array_agg", "bool_and", "bool_or", "stddev", "variance", "date_part", "justify_interval", "to_number",
    "btrim", "ltrim", "rtrim", "lpad", "rpad", "reverse", "repeat", "left", "right", "ceiling", "trunc",
}


def _name(identifier: exp.Identifier) -> str:
    """The name PostgreSQL would use: as written when quoted, lower case when not."""
    return identifier.name if identifier.quoted else identifier.name.lower()


def guard(sql_text: str, schema: str, tables: Iterable[str], max_rows: int) -> str:
    """The SQL to run for `sql_text`: one SELECT, every table in `schema` and named in `tables`, written
    out schema-qualified and with a LIMIT of at most `max_rows`. Raises `Refused` with the reason."""
    allowed = set(tables)
    try:
        statements = [s for s in sqlglot.parse(sql_text, read="postgres") if s is not None]
    except SqlglotError as e:
        raise Refused(f"The SQL could not be parsed: {str(e).strip().splitlines()[0][:200]}") from None
    if not statements:
        raise Refused("There is no SQL to run.")
    if len(statements) > 1:
        raise Refused("Write exactly one SELECT statement; this has several.")
    root = statements[0]
    if not isinstance(root, (exp.Select, exp.SetOperation)):
        raise Refused(f"Only a SELECT statement may be run, not {type(root).__name__.upper()}. Write a SELECT query.")
    forbidden = root.find(*FORBIDDEN)
    if forbidden is not None:
        raise Refused(f"This query contains {type(forbidden).__name__.upper()}, which is not allowed: it must only read.")
    for function in root.find_all(exp.Anonymous):
        if function.name.lower() not in ANONYMOUS_ALLOWED:
            raise Refused(
                f"The function {function.name}() is not allowed. Use standard SQL: count, sum, avg, min, max, "
                "lower, upper, coalesce, round, date_trunc, extract, cast, case."
            )

    ctes = {_name(cte.args["alias"].this) for cte in root.find_all(exp.CTE) if cte.args.get("alias")}
    listing = ", ".join(sorted(allowed)) or "none"
    for table in root.find_all(exp.Table):
        if not isinstance(table.this, exp.Identifier):
            continue  # a table function: its function is checked above
        if table.args.get("catalog"):
            raise Refused(f"Only tables of the schema {schema} may be used; drop the database name.")
        name = _name(table.this)
        db = table.args.get("db")
        if db is None and name in ctes:
            continue
        if db is not None and _name(db) != schema:
            raise Refused(f"Only tables of the schema {schema} may be used, not {_name(db)}.{name}. Tables: {listing}.")
        if name not in allowed:
            raise Refused(f"There is no table {name} in the schema {schema}. Tables: {listing}.")
        table.set("db", exp.to_identifier(schema))

    limit = root.args.get("limit")
    if isinstance(limit, exp.Fetch):  # FETCH FIRST n ROWS ONLY is a LIMIT; its variants mean something else
        options = limit.args.get("limit_options")
        if options is not None and (options.args.get("with_ties") or options.args.get("percent")):
            raise Refused("Use LIMIT instead of FETCH FIRST ... WITH TIES or PERCENT.")
        wanted = limit.args.get("count")
        limit = exp.Limit(expression=wanted) if wanted is not None else None
        root.set("limit", limit)
    else:
        wanted = limit.expression if limit is not None else None
    if not (isinstance(wanted, exp.Literal) and wanted.is_int and int(wanted.this) <= max_rows):
        root = root.limit(max_rows, copy=False)
    return root.sql(dialect="postgres", comments=False)  # a comment is text the model wrote; none runs
