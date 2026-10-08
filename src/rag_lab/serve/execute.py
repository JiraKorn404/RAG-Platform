"""Runs a checked query as the reader role: a read-only transaction pinned to the chat's schema, with a
statement timeout and a row cap. `explain` plans a query without running it, which catches an unknown
column or a type error before the query is executed."""

import math
import time
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from datetime import time as time_of_day
from decimal import Decimal

import psycopg
from psycopg import sql

from rag_lab.core.database import reader_url


class SqlError(Exception):
    """The database refused or could not finish a query. The message is worded for the model."""


@dataclass
class QueryResult:
    columns: list[str]
    rows: list[list]  # JSON-safe values, at most `row_limit` of them
    row_count: int
    truncated: bool  # there were more rows than the limit
    ms: float


def jsonable(value):
    """A value as plain data: numbers, strings, booleans, None, lists and dicts."""
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else str(value)
    if isinstance(value, Decimal):
        return float(value) if value.is_finite() else str(value)
    if isinstance(value, (date, datetime, time_of_day)):
        return value.isoformat()
    if isinstance(value, timedelta):
        return str(value)
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    return str(value)


def _message(error: psycopg.Error, timeout_s: float) -> str:
    if isinstance(error, psycopg.errors.QueryCanceled):
        return f"The query took longer than {timeout_s:g} seconds and was stopped. Make it cheaper."
    text = (error.diag.message_primary or str(error)).strip()
    hint = (error.diag.message_hint or "").strip()
    return f"{text}. {hint}" if hint else text


@contextmanager
def _session(schema: str, timeout_s: float):
    """A read-only transaction as the reader, pinned to `schema`, that cannot run long."""
    try:
        with psycopg.connect(reader_url(), connect_timeout=10) as conn:
            conn.read_only = True
            conn.execute(sql.SQL("SET LOCAL statement_timeout = {}").format(sql.Literal(f"{int(timeout_s * 1000)}ms")))
            conn.execute(sql.SQL("SET LOCAL search_path = {}").format(sql.Identifier(schema)))
            yield conn
    except psycopg.Error as e:
        raise SqlError(_message(e, timeout_s)) from None


def explain(schema: str, query: str, timeout_s: float) -> None:
    """Plan a checked query without running it. Raises `SqlError` for what the database would refuse."""
    with _session(schema, timeout_s) as conn:
        conn.execute("EXPLAIN " + query)


def run(schema: str, query: str, row_limit: int, timeout_s: float) -> QueryResult:
    """Run a checked query. At most `row_limit + 1` rows are read, to tell whether the result was cut."""
    start = time.perf_counter()
    with _session(schema, timeout_s) as conn:
        cursor = conn.execute(query)
        columns = [d.name for d in cursor.description]
        rows = cursor.fetchmany(row_limit + 1)
    return QueryResult(
        columns=columns,
        rows=[[jsonable(v) for v in row] for row in rows[:row_limit]],
        row_count=min(len(rows), row_limit),
        truncated=len(rows) > row_limit,
        ms=(time.perf_counter() - start) * 1000,
    )
