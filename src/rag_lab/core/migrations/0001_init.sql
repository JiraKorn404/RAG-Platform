-- The tables of the rag_metrics database. A later change is a new numbered file, never an edit here.

-- An experiment is a Qdrant collection with the settings it was made with. Settings cannot change
-- under a name, so a name is one row.
CREATE TABLE experiments (
    config_hash text PRIMARY KEY,
    name        text NOT NULL UNIQUE,
    config      jsonb NOT NULL,
    created_at  timestamptz NOT NULL DEFAULT now()
);

-- One row per document and stage of an ingestion run.
CREATE TABLE ingestion_stage_metrics (
    id             bigserial PRIMARY KEY,
    config_hash    text NOT NULL REFERENCES experiments ON DELETE CASCADE,
    doc_id         text NOT NULL,
    stage          text NOT NULL CHECK (stage IN ('parse', 'chunk', 'embed', 'index')),
    dagster_run_id text,
    duration_ms    double precision NOT NULL,
    items          integer,
    throughput     double precision,
    details        jsonb NOT NULL DEFAULT '{}',
    created_at     timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE search_log (
    id             bigserial PRIMARY KEY,
    config_hash    text NOT NULL REFERENCES experiments ON DELETE CASCADE,
    query_text     text NOT NULL,
    top_k          integer NOT NULL,
    embed_ms       double precision NOT NULL,
    search_ms      double precision NOT NULL,
    total_ms       double precision NOT NULL,
    top_similarity double precision,
    created_at     timestamptz NOT NULL DEFAULT now()
);

-- What was imported into the database for tables (`rag_data`, which is a separate database: a foreign
-- key cannot cross it, so these rows are kept in step with it by rag_lab/sql/database.py). A schema is
-- one dataset; a table's columns are described here: type, description, and a profile of its values
-- (null count, distinct count, example values, range).
CREATE TABLE db_schemas (
    schema_name text PRIMARY KEY,
    description text NOT NULL DEFAULT '',
    created_at  timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE db_tables (
    schema_name text NOT NULL REFERENCES db_schemas ON DELETE CASCADE,
    table_name  text NOT NULL,
    source_file text,
    imported_at timestamptz NOT NULL DEFAULT now(),
    row_count   bigint NOT NULL DEFAULT 0,
    description text NOT NULL DEFAULT '',
    columns     jsonb NOT NULL DEFAULT '[]',
    PRIMARY KEY (schema_name, table_name)
);

-- A chat searches one thing for its whole life: the documents of an experiment, or a database schema.
-- It is made with its first turn and goes with what it searches.
CREATE TABLE chat_sessions (
    session_id  text PRIMARY KEY,
    kind        text NOT NULL CHECK (kind IN ('documents', 'database')),
    config_hash text REFERENCES experiments ON DELETE CASCADE,
    schema_name text REFERENCES db_schemas ON DELETE CASCADE,
    title       text NOT NULL,
    created_at  timestamptz NOT NULL DEFAULT now(),
    updated_at  timestamptz NOT NULL DEFAULT now(),
    CONSTRAINT chat_sessions_session_kind_key UNIQUE (session_id, kind),
    CONSTRAINT chat_sessions_one_target CHECK (
        (kind = 'documents' AND config_hash IS NOT NULL AND schema_name IS NULL)
        OR (kind = 'database' AND schema_name IS NOT NULL AND config_hash IS NULL)
    )
);

-- One row per turn: the question, the query it was made into, what was retrieved and said, how long
-- each step took, and its events in order (without the token-by-token ones), from which the page shows
-- it again. A turn repeats its chat's kind under a composite foreign key, so the database refuses a
-- turn of the other kind. Deleting a chat deletes its turns.
CREATE TABLE chat_turns (
    id           bigserial PRIMARY KEY,
    session_id   text NOT NULL,
    kind         text NOT NULL,
    config_hash  text REFERENCES experiments ON DELETE CASCADE,
    question     text NOT NULL,
    query        text NOT NULL,
    model        text NOT NULL,
    think        boolean NOT NULL,
    answer       text NOT NULL,
    thinking     text NOT NULL,
    hits         jsonb NOT NULL,
    timings      jsonb NOT NULL,
    model_states jsonb NOT NULL,
    events       jsonb NOT NULL,
    abstained    boolean NOT NULL DEFAULT false,
    cited        jsonb,
    total_ms     double precision,
    settings     jsonb,
    error        text,
    created_at   timestamptz NOT NULL DEFAULT now(),
    CONSTRAINT chat_turns_session_kind_fk
        FOREIGN KEY (session_id, kind) REFERENCES chat_sessions (session_id, kind) ON DELETE CASCADE
);

CREATE INDEX chat_turns_session ON chat_turns (session_id, created_at);

-- Database answers a user marked as good: the question (as typed and made standalone) and the SQL that
-- produced the answer. They are shown to the text-to-SQL agent when a similar question comes later, and
-- indexed in Qdrant (rag_lab/sql/examples.py). An example belongs to its schema and goes with it. It
-- keeps the turn it came from for provenance, but outlives that turn's chat.
CREATE TABLE sql_examples (
    id          bigserial PRIMARY KEY,
    schema_name text NOT NULL REFERENCES db_schemas ON DELETE CASCADE,
    question    text NOT NULL,
    standalone  text NOT NULL,
    sql         text NOT NULL,
    turn_id     bigint REFERENCES chat_turns ON DELETE SET NULL,
    created_at  timestamptz NOT NULL DEFAULT now()
);

-- The same question and SQL is kept once (hashed: a long query does not fit in an index entry).
CREATE UNIQUE INDEX sql_examples_once ON sql_examples (schema_name, md5(standalone), md5(sql));
