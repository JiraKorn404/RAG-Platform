"""Good answers: questions about a schema that were answered correctly, kept with the SQL that answered
them and shown to the text-to-SQL agent when a similar question comes later.

The examples are rows in `rag_metrics.sql_examples` (the source of truth) and points in a small Qdrant
collection per schema, `sqlexamples__<schema>`, one point for each example, embedded from its
standalone question. A change goes to the table first and then to Qdrant, so a failure between the two
is put right by `reindex`, which rebuilds the collection from the table."""

import uuid
from dataclasses import dataclass

from qdrant_client.models import Distance, PayloadSchemaType, PointIdsList, PointStruct, VectorParams

from rag_lab.config import EmbedConfig, SqlAgentConfig
from rag_lab.embedding.ollama import OllamaEmbedder
from rag_lab.metrics.store import MetricsStore
from rag_lab.storage.qdrant import SQL_EXAMPLES_PREFIX, QdrantStore

_NAMESPACE = uuid.UUID("0b8c1a52-3f0e-4d4c-8a43-9c7d1c1f0b22")


@dataclass
class Example:
    id: int
    question: str  # the standalone question: what was embedded
    sql: str
    score: float = 0.0  # the similarity to the question asked, for a retrieved example


def collection_name(schema: str) -> str:
    return SQL_EXAMPLES_PREFIX + schema


def point_id(schema: str, example_id: int) -> str:
    return str(uuid.uuid5(_NAMESPACE, f"{schema}/{example_id}"))


def embed_key(cfg: EmbedConfig) -> str:
    """What the vectors in a collection were made with. A point made with another setup is not matched,
    and the collection is rebuilt with `reindex`."""
    return f"{cfg.model}|{cfg.dimension or 'native'}"


def _check_size(store: QdrantStore, schema: str, size: int) -> None:
    if store.stats(collection_name(schema))["vector_dimension"] != size:
        raise ValueError(f"The examples of '{schema}' were indexed with another embedding setup. Run `python -m rag_lab.sql examples --reindex`.")


def _upsert(embedder: OllamaEmbedder, store: QdrantStore, cfg: SqlAgentConfig, schema: str, rows: list[dict]) -> None:
    """Embed the rows' standalone questions and write their points, creating the collection if needed."""
    if not rows:
        return
    vectors = embedder.embed([r["standalone"] for r in rows], cfg.embed, kind="document").vectors
    name, client = collection_name(schema), store.client
    if not client.collection_exists(name):
        client.create_collection(name, vectors_config=VectorParams(size=len(vectors[0]), distance=Distance.COSINE))
        client.create_payload_index(name, "embed_key", PayloadSchemaType.KEYWORD)
    else:
        _check_size(store, schema, len(vectors[0]))
    client.upsert(
        name,
        points=[
            PointStruct(
                id=point_id(schema, r["id"]),
                vector=vector,
                payload={
                    "example_id": r["id"],
                    "question": r["standalone"],
                    "sql": r["sql"],
                    "embed_key": embed_key(cfg.embed),
                },
            )
            for r, vector in zip(rows, vectors)
        ],
    )


def _delete_point(store: QdrantStore, schema: str, example_id: int) -> None:
    name = collection_name(schema)
    if store.client.collection_exists(name):
        store.client.delete(name, points_selector=PointIdsList(points=[point_id(schema, example_id)]))


def drop_index(store: QdrantStore, schema: str) -> None:
    """Remove a schema's collection of examples, if it has one."""
    name = collection_name(schema)
    if store.client.collection_exists(name):
        store.delete_collection(name)


def save(
    metrics: MetricsStore, embedder: OllamaEmbedder, store: QdrantStore, cfg: SqlAgentConfig,
    schema: str, question: str, standalone: str, sql: str, turn_id: int | None,
) -> int:
    """Keep a good answer as an example and index it. Returns its id."""
    example_id = metrics.add_sql_example(schema, question, standalone, sql, turn_id)
    _upsert(embedder, store, cfg, schema, [{"id": example_id, "standalone": standalone, "sql": sql}])
    return example_id


def remove(metrics: MetricsStore, store: QdrantStore, schema: str, example_id: int) -> None:
    metrics.delete_sql_example(example_id)
    _delete_point(store, schema, example_id)


def has_examples(store: QdrantStore, schema: str) -> bool:
    name = collection_name(schema)
    return store.client.collection_exists(name) and store.count(name) > 0


def retrieve(embedder: OllamaEmbedder, store: QdrantStore, cfg: SqlAgentConfig, schema: str, question: str) -> list[Example]:
    """The examples whose question is most like `question`, at most `examples_k` and each at least
    `examples_min_score` similar, most similar first."""
    vector = embedder.embed([question], cfg.embed, kind="query").vectors[0]
    _check_size(store, schema, len(vector))
    hits = store.query(collection_name(schema), vector, cfg.examples_k, filters={"embed_key": embed_key(cfg.embed)})
    return [
        Example(h.payload["example_id"], h.payload["question"], h.payload["sql"], h.score)
        for h in hits
        if h.score >= cfg.examples_min_score
    ]


def reindex(
    metrics: MetricsStore, embedder: OllamaEmbedder, store: QdrantStore, cfg: SqlAgentConfig, schema: str | None = None
) -> dict[str, int]:
    """Rebuild the collection of a schema (every schema when none is given) from the examples in the
    table. Returns how many examples each schema has indexed."""
    found = metrics.list_sql_examples(schema)
    schemas = [schema] if schema else sorted({e["schema_name"] for e in found} | {
        c.name.removeprefix(SQL_EXAMPLES_PREFIX)
        for c in store.client.get_collections().collections
        if c.name.startswith(SQL_EXAMPLES_PREFIX)
    })
    done = {}
    for name in schemas:
        drop_index(store, name)
        rows = [e for e in found if e["schema_name"] == name]
        _upsert(embedder, store, cfg, name, rows)
        done[name] = len(rows)
    return done
