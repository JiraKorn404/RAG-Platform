"""Our tables in PostgreSQL (`app_database` in config/connections.yaml): one method per table, and the
migrations that make them. Plain Python, no Dagster."""

from pathlib import Path

import psycopg
from psycopg.conninfo import make_conninfo
from psycopg.types.json import Jsonb


def in_schema(database_url: str, schema: str) -> str:
    """The connection string with `schema` as its search path, so our tables are made and found there."""
    return make_conninfo(database_url, options=f"-c search_path={schema}")


class MetricsStore:
    def __init__(self, database_url: str, schema: str = "public"):
        self.database_url = in_schema(database_url, schema)

    def _execute(self, sql: str, params: tuple | list, many: bool = False):
        """Runs one statement in its own transaction; returns the first row if it has RETURNING."""
        with psycopg.connect(self.database_url) as conn:
            if many:
                with conn.cursor() as cur:
                    cur.executemany(sql, params)
                return None
            cur = conn.execute(sql, params)
            return cur.fetchone() if "RETURNING" in sql else None

    def upsert_experiment(self, config_hash: str, name: str, config: dict) -> None:
        self._execute(
            """INSERT INTO experiments (config_hash, name, config) VALUES (%s, %s, %s)
               ON CONFLICT (config_hash) DO UPDATE SET name = EXCLUDED.name, config = EXCLUDED.config""",
            (config_hash, name, Jsonb(config)),
        )

    def get_experiment(self, name: str) -> tuple[str, dict] | None:
        """(config_hash, config) recorded under this experiment name."""
        with psycopg.connect(self.database_url) as conn:
            row = conn.execute(
                "SELECT config_hash, config FROM experiments WHERE name = %s "
                "ORDER BY created_at DESC LIMIT 1",
                (name,),
            ).fetchone()
        return (row[0], row[1]) if row else None

    def add_stage_metric(
        self,
        config_hash: str,
        doc_id: str,
        stage: str,
        duration_ms: float,
        items: int | None = None,
        throughput: float | None = None,
        details: dict | None = None,
        dagster_run_id: str | None = None,
    ) -> None:
        self._execute(
            """INSERT INTO ingestion_stage_metrics
                   (config_hash, doc_id, stage, dagster_run_id, duration_ms, items, throughput, details)
               VALUES (%s, %s, %s, %s, %s, %s, %s, %s)""",
            (config_hash, doc_id, stage, dagster_run_id, duration_ms, items, throughput,
             Jsonb(details or {})),
        )

    def list_experiments(self) -> list[dict]:
        """The newest row of each experiment name, with how many documents it holds (distinct documents
        in its stage metrics)."""
        with psycopg.connect(self.database_url) as conn:
            rows = conn.execute(
                """SELECT DISTINCT ON (e.name) e.name, e.config_hash, e.config,
                          (SELECT count(DISTINCT m.doc_id) FROM ingestion_stage_metrics m
                           WHERE m.config_hash = e.config_hash) AS documents,
                          e.created_at
                   FROM experiments e ORDER BY e.name, e.created_at DESC"""
            ).fetchall()
        return [
            {
                "name": r[0],
                "config_hash": r[1],
                "config": r[2],
                "documents": r[3],
                "created_at": r[4],
            }
            for r in rows
        ]

    def delete_document_rows(self, config_hash: str, doc_id: str) -> int:
        """Remove one document's stage metric rows in one experiment. Returns how many rows went."""
        with psycopg.connect(self.database_url) as conn:
            return conn.execute(
                "DELETE FROM ingestion_stage_metrics WHERE config_hash = %s AND doc_id = %s",
                (config_hash, doc_id),
            ).rowcount

    def add_search_log(
        self,
        config_hash: str,
        query_text: str,
        top_k: int,
        embed_ms: float,
        search_ms: float,
        total_ms: float,
        top_similarity: float | None,
    ) -> None:
        self._execute(
            """INSERT INTO search_log (config_hash, query_text, top_k, embed_ms, search_ms,
                                       total_ms, top_similarity)
               VALUES (%s, %s, %s, %s, %s, %s, %s)""",
            (config_hash, query_text, top_k, embed_ms, search_ms, total_ms, top_similarity),
        )

    def add_chat_turn(
        self,
        config_hash: str | None,
        schema_name: str | None,
        session_id: str,
        question: str,
        query: str,
        model: str,
        think: bool,
        answer: str,
        thinking: str,
        hits: list[dict],
        timings: dict,
        model_states: list[dict],
        *,
        kind: str,
        events: list[dict],
        abstained: bool = False,
        cited: list[int] | None = None,
        total_ms: float | None = None,
        settings: dict | None = None,
        error: str | None = None,
    ) -> int:
        """Save one turn and return its id. Its chat is made with its first turn (titled by the question) and its
        `updated_at` moves with every turn; both happen in one transaction. A chat has one `kind` for its
        whole life: a turn of another kind is refused by the database (a composite foreign key)."""
        with psycopg.connect(self.database_url) as conn:
            conn.execute(
                """INSERT INTO chat_sessions (session_id, kind, config_hash, schema_name, title)
                   VALUES (%s, %s, %s, %s, %s)
                   ON CONFLICT (session_id) DO UPDATE SET updated_at = now()""",
                (session_id, kind, config_hash, schema_name, question[:80]),
            )
            return conn.execute(
                """INSERT INTO chat_turns (config_hash, session_id, kind, question, query, model, think, answer,
                                           thinking, hits, timings, model_states, events, abstained,
                                           cited, total_ms, settings, error)
                   VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                   RETURNING id""",
                (config_hash, session_id, kind, question, query, model, think, answer, thinking,
                 Jsonb(hits), Jsonb(timings), Jsonb(model_states),
                 Jsonb(events), abstained,
                 Jsonb(cited) if cited is not None else None, total_ms,
                 Jsonb(settings) if settings is not None else None, error),
            ).fetchone()[0]

    def get_chat_session(self, session_id: str) -> dict | None:
        with psycopg.connect(self.database_url) as conn:
            row = conn.execute(
                "SELECT config_hash, title, kind, schema_name FROM chat_sessions WHERE session_id = %s",
                (session_id,),
            ).fetchone()
        return {"config_hash": row[0], "title": row[1], "kind": row[2], "schema_name": row[3]} if row else None

    def list_chat_sessions(self, limit: int = 30) -> list[dict]:
        """The chats of both kinds, most recently used first, each with its kind, what it searches (the
        experiment's name or the schema) and how many turns it has."""
        keys = ["session_id", "title", "updated_at", "kind", "target", "turns"]
        with psycopg.connect(self.database_url) as conn:
            rows = conn.execute(
                """SELECT s.session_id, s.title, s.updated_at, s.kind,
                          CASE WHEN s.kind = 'documents'
                               THEN (SELECT e.name FROM experiments e WHERE e.config_hash = s.config_hash LIMIT 1)
                               ELSE s.schema_name END,
                          (SELECT count(*) FROM chat_turns t WHERE t.session_id = s.session_id)
                   FROM chat_sessions s ORDER BY s.updated_at DESC LIMIT %s""",
                (limit,),
            ).fetchall()
        return [dict(zip(keys, r)) for r in rows]

    def get_chat_turns(self, session_id: str | None = None, turn_id: int | None = None) -> list[dict]:
        """A chat's turns, oldest first, or the one turn with `turn_id`."""
        keys = ["id", "session_id", "kind", "question", "query", "model", "think", "answer", "thinking", "hits",
                "timings", "model_states", "events", "abstained", "cited", "total_ms", "settings", "error"]
        with psycopg.connect(self.database_url) as conn:
            rows = conn.execute(
                f"SELECT {', '.join(keys)} FROM chat_turns WHERE session_id = %s OR id = %s ORDER BY created_at, id",
                (session_id, turn_id),
            ).fetchall()
        return [dict(zip(keys, r)) for r in rows]

    def delete_chat_session(self, session_id: str) -> None:
        """Delete a chat; its turns go with it."""
        self._execute("DELETE FROM chat_sessions WHERE session_id = %s", (session_id,))

    # --- imported tables ---------------------------------------------------------------------------

    def add_db_schema(self, name: str, description: str = "", registered: bool = False) -> None:
        """`registered`: the schema was already in the database (ingest/register.py), not made by the import."""
        self._execute(
            "INSERT INTO db_schemas (schema_name, description, registered) VALUES (%s, %s, %s)",
            (name, description, registered),
        )

    def get_db_schema(self, name: str) -> dict | None:
        found = [s for s in self.list_db_schemas() if s["name"] == name]
        return found[0] if found else None

    def list_db_schemas(self) -> list[dict]:
        """The schemas, with how many tables each has, by name."""
        keys = ["name", "description", "created_at", "registered", "tables"]
        with psycopg.connect(self.database_url) as conn:
            rows = conn.execute(
                """SELECT s.schema_name, s.description, s.created_at, s.registered,
                          (SELECT count(*) FROM db_tables t WHERE t.schema_name = s.schema_name)
                   FROM db_schemas s ORDER BY s.schema_name"""
            ).fetchall()
        return [dict(zip(keys, r)) for r in rows]

    def delete_db_schema(self, name: str) -> None:
        """Remove a schema's rows; its tables' rows and its chats go with it. The data is dropped by the
        caller."""
        self._execute("DELETE FROM db_schemas WHERE schema_name = %s", (name,))

    def upsert_db_table(
        self, schema: str, table: str, source_file: str | None, row_count: int, columns: list[dict]
    ) -> None:
        """Record an imported table. A table imported again keeps its description."""
        self._execute(
            """INSERT INTO db_tables (schema_name, table_name, source_file, row_count, columns)
               VALUES (%s, %s, %s, %s, %s)
               ON CONFLICT (schema_name, table_name) DO UPDATE
               SET source_file = EXCLUDED.source_file, row_count = EXCLUDED.row_count,
                   columns = EXCLUDED.columns, imported_at = now()""",
            (schema, table, source_file, row_count, Jsonb(columns)),
        )

    def get_db_table(self, schema: str, table: str) -> dict | None:
        found = [t for t in self.list_db_tables(schema) if t["table_name"] == table]
        return found[0] if found else None

    def list_db_tables(self, schema: str) -> list[dict]:
        keys = ["table_name", "source_file", "imported_at", "row_count", "description", "columns"]
        with psycopg.connect(self.database_url) as conn:
            rows = conn.execute(
                f"SELECT {', '.join(keys)} FROM db_tables WHERE schema_name = %s ORDER BY table_name",
                (schema,),
            ).fetchall()
        return [dict(zip(keys, r)) for r in rows]

    def set_db_table_descriptions(
        self, schema: str, table: str, description: str, columns: dict[str, str]
    ) -> None:
        """Save what was written about a table and its columns (column name -> description). A column not
        named keeps its description."""
        found = self.get_db_table(schema, table)
        if not found:
            raise ValueError(f"There is no table {schema}.{table}.")
        updated = [{**c, "description": columns.get(c["name"], c["description"])} for c in found["columns"]]
        self._execute(
            "UPDATE db_tables SET description = %s, columns = %s WHERE schema_name = %s AND table_name = %s",
            (description, Jsonb(updated), schema, table),
        )

    def set_db_schema_description(self, name: str, description: str) -> None:
        self._execute("UPDATE db_schemas SET description = %s WHERE schema_name = %s", (description, name))

    def delete_db_table(self, schema: str, table: str) -> None:
        self._execute("DELETE FROM db_tables WHERE schema_name = %s AND table_name = %s", (schema, table))

    # --- good answers -------------------------------------------------------------------------------

    def add_sql_example(
        self, schema: str, question: str, standalone: str, sql: str, turn_id: int | None
    ) -> int:
        """Keep a question and its SQL as an example and return its id. The same question and SQL is kept
        once: saving it again keeps the first turn it came from."""
        with psycopg.connect(self.database_url) as conn:
            return conn.execute(
                """INSERT INTO sql_examples (schema_name, question, standalone, sql, turn_id)
                   VALUES (%s, %s, %s, %s, %s)
                   ON CONFLICT (schema_name, md5(standalone), md5(sql))
                   DO UPDATE SET question = sql_examples.question RETURNING id""",
                (schema, question, standalone, sql, turn_id),
            ).fetchone()[0]

    def list_sql_examples(self, schema: str | None = None) -> list[dict]:
        """A schema's examples (all schemas when none is given), newest first."""
        keys = ["id", "schema_name", "question", "standalone", "sql", "turn_id", "created_at"]
        with psycopg.connect(self.database_url) as conn:
            rows = conn.execute(
                f"SELECT {', '.join(keys)} FROM sql_examples WHERE %s::text IS NULL OR schema_name = %s "
                "ORDER BY created_at DESC, id DESC",
                (schema, schema),
            ).fetchall()
        return [dict(zip(keys, r)) for r in rows]

    def delete_sql_example(self, example_id: int) -> None:
        self._execute("DELETE FROM sql_examples WHERE id = %s", (example_id,))

    def delete_experiment(self, name: str) -> None:
        """Remove an experiment's rows (its stage metrics go with it). The collection and files are
        removed by the caller."""
        self._execute("DELETE FROM experiments WHERE name = %s", (name,))


# --- migrations --------------------------------------------------------------------------------------

MIGRATIONS_DIR = Path(__file__).parent / "migrations"
_LOCK_ID = 727274  # two callers at once (python -m rag_lab setup run twice) apply each file once


def apply_migrations(database_url: str) -> list[str]:
    """Apply any numbered .sql file not yet recorded in schema_migrations. Returns what it applied.
    The tables are made in the search path of `database_url` (see `in_schema`)."""
    applied = []
    with psycopg.connect(database_url) as conn:
        conn.execute("SELECT pg_advisory_xact_lock(%s)", (_LOCK_ID,))
        conn.execute(
            """CREATE TABLE IF NOT EXISTS schema_migrations (
                   version text PRIMARY KEY,
                   applied_at timestamptz NOT NULL DEFAULT now())"""
        )
        done = {row[0] for row in conn.execute("SELECT version FROM schema_migrations")}
        for path in sorted(MIGRATIONS_DIR.glob("*.sql")):
            if path.stem not in done:
                conn.execute(path.read_text(encoding="utf-8"))
                conn.execute("INSERT INTO schema_migrations (version) VALUES (%s)", (path.stem,))
                applied.append(path.stem)
    return applied
