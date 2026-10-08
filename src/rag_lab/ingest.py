"""The stage bodies the Dagster assets run. Plain Python, no Dagster: the assets wrap these and keep
only what is Dagster's (partition key, Failure, run id, MaterializeResult)."""

import json
import time
from datetime import datetime, timezone

import numpy as np
from llama_index.core.schema import NodeRelationship, RelatedNodeInfo, TextNode
from llama_index.vector_stores.qdrant import QdrantVectorStore

from rag_lab.chunking.models import read_chunks
from rag_lab.config import ExperimentConfig
from rag_lab.embedding.ollama import OllamaEmbedder
from rag_lab.embedding.sparse import document_vectors
from rag_lab.metrics.store import MetricsStore
from rag_lab.parsing.parse import ParsedDocument
from rag_lab.paths import artifacts_dir
from rag_lab.settings import ConfigError, load
from rag_lab.storage.qdrant import QdrantStore, to_point_id

# Below this many extracted characters per page, the PDF is almost certainly scanned images.
SCANNED_PDF_CHARS_PER_PAGE = 50
# What the parse stage row says about OCR, when the experiment has it on.
OCR_DETAILS = ("ocr_regions", "ocr_pages", "ocr_seconds", "ocr_output_tokens", "ocr_cut_regions")


class IngestError(Exception):
    """A stage cannot run, for example because the previous stage's files are missing."""


def configured_experiment(store: MetricsStore) -> ExperimentConfig:
    """The experiment config/pipeline.yaml and config/llm.yaml describe now, recorded under its name.
    Every stage of a run starts with it, so a file edited to other settings under the same name stops
    the run instead of mixing two sets of settings in one collection."""
    try:
        config = load().experiment
    except ConfigError as e:
        raise IngestError(str(e)) from e
    register_experiment(store, config)
    return config


def register_experiment(store: MetricsStore, config: ExperimentConfig) -> str:
    """Record the experiment's settings under its name; returns the config hash. A name that
    already has other settings is refused."""
    problem = experiment_conflict(store, config)
    if problem:
        raise IngestError(problem)
    config_hash = config.config_hash()
    store.upsert_experiment(config_hash, config.name, config.model_dump(mode="json"))
    return config_hash


def experiment_conflict(store: MetricsStore, config: ExperimentConfig) -> str | None:
    """Why documents cannot be ingested under `config.name`, or None when they can: the name is free,
    or it is this same experiment. An experiment's collection holds one set of settings, so the same
    name with other settings is refused instead of mixing them."""
    found = store.get_experiment(config.name)
    if found is None or found[0] == config.config_hash():
        return None
    return (
        f"An experiment named '{config.name}' already exists with other settings. Settings cannot change "
        "under the same name: choose a new `name` in config/pipeline.yaml (a new experiment), or put the "
        "settings back."
    )


def record_parse(
    store: MetricsStore,
    config: ExperimentConfig,
    doc_id: str,
    parsed: ParsedDocument,
    run_id: str | None = None,
) -> None:
    store.add_stage_metric(
        config.config_hash(),
        doc_id,
        "parse",
        duration_ms=parsed.parse_seconds * 1000,
        items=parsed.pages,
        throughput=parsed.pages_per_second,
        details={
            "source_file": parsed.source_file,
            "status": parsed.status,
            "tables": parsed.tables,
            "table_cells": parsed.table_cells,
            "text_items": parsed.text_items,
            "chars_per_page": parsed.chars_per_page,
            "model_load_seconds": parsed.model_load_seconds,
            **({k: getattr(parsed, k) for k in OCR_DETAILS} if config.parse.ocr else {}),
        },
        dagster_run_id=run_id,
    )


def record_chunk(
    store: MetricsStore,
    config: ExperimentConfig,
    doc_id: str,
    summary: dict,
    stats: dict,
    warnings: list[str],
    seconds: float,
    run_id: str | None = None,
) -> None:
    """`summary` is chunking.summarise(chunks); `stats` and `warnings` come from the chunk context."""
    store.add_stage_metric(
        config.config_hash(),
        doc_id,
        "chunk",
        duration_ms=seconds * 1000,
        items=summary["chunks"],
        throughput=summary["chunks"] / seconds if seconds else None,
        details={
            "strategy": config.chunk.strategy,
            **summary,
            **stats,
            "warnings": warnings,
        },
        dagster_run_id=run_id,
    )


def embed_chunks(
    store: MetricsStore,
    embedder: OllamaEmbedder,
    config: ExperimentConfig,
    doc_id: str,
    run_id: str | None = None,
) -> dict:
    """Embed every chunk of the document (document mode, no query prefix) with Ollama.
    Reads chunk/<doc_id>.chunks.jsonl and writes embed/<doc_id>.npy: one float32 row per chunk.
    Returns the numbers the asset shows as metadata."""
    chunk_path = artifacts_dir(config.name, "chunk") / f"{doc_id}.chunks.jsonl"
    if not chunk_path.exists():
        raise IngestError(f"No chunks at {chunk_path}. Materialise `chunks` first (same experiment name).")

    rows = read_chunks(chunk_path)
    texts: list[str | None] = [c.text for c in rows]
    images = None
    pictures = sum(1 for c in rows if c.image)
    if pictures and config.embed.picture_input != "caption":
        parse_dir = artifacts_dir(config.name, "parse")
        images = [(parse_dir / c.image).read_bytes() if c.image else None for c in rows]
        if config.embed.picture_input == "image":
            texts = [None if c.image else c.text for c in rows]
    result = embedder.embed(texts, config.embed, kind="document", images=images)
    vectors = np.array(result.vectors, dtype=np.float32)
    np.save(artifacts_dir(config.name, "embed") / f"{doc_id}.npy", vectors)

    seconds = result.wall_ms / 1000
    details = {
        "model": config.embed.model,
        "dimension": int(vectors.shape[1]),
        "batch_size": config.embed.batch_size,
        "batches": result.batches,
        "prompt_tokens": result.prompt_tokens,
        "tokens_per_second": round(result.prompt_tokens / seconds, 1) if seconds else None,
        "ollama_ms": round(result.ollama_ms, 1),
        "model_load_ms": round(result.load_ms, 1),
    }
    if pictures:
        details["pictures"] = pictures
        details["picture_input"] = config.embed.picture_input
    store.add_stage_metric(
        config.config_hash(),
        doc_id,
        "embed",
        duration_ms=result.wall_ms,
        items=len(texts),
        throughput=len(texts) / seconds if seconds else None,
        details=details,
        dagster_run_id=run_id,
    )
    return {
        "chunks": len(texts),
        "chunks_per_second": round(len(texts) / seconds, 2) if seconds else 0,
        "embed_seconds": round(seconds, 3),
        **details,
    }


def index_chunks(
    store: MetricsStore,
    qdrant: QdrantStore,
    config: ExperimentConfig,
    doc_id: str,
    run_id: str | None = None,
) -> dict:
    """Write the document's chunks and vectors to the experiment's Qdrant collection.
    Idempotent: the document's existing points are deleted first and point ids are deterministic.
    Returns the details the asset shows as metadata."""
    chunk_path = artifacts_dir(config.name, "chunk") / f"{doc_id}.chunks.jsonl"
    vector_path = artifacts_dir(config.name, "embed") / f"{doc_id}.npy"
    if not (chunk_path.exists() and vector_path.exists()):
        raise IngestError("Missing chunks or embeddings. Materialise `chunks` and `embeddings` first.")

    rows = read_chunks(chunk_path)
    vectors = np.load(vector_path)
    if len(rows) != len(vectors):
        raise IngestError(
            f"{len(rows)} chunks but {len(vectors)} vectors: the embeddings are stale. "
            "Materialise `embeddings` again."
        )

    # source_file comes from the parse step's metadata file
    meta_path = artifacts_dir(config.name, "parse") / f"{doc_id}.meta.json"
    source_file = (
        json.loads(meta_path.read_text(encoding="utf-8")).get("source_file") if meta_path.exists() else None
    )

    config_hash = config.config_hash()
    ingested_at = datetime.now(timezone.utc).isoformat()
    payloads = [
        {
            "chunk_id": c.chunk_id,
            "doc_id": c.doc_id,
            "text": c.text,
            "modality": c.modality,
            "strategy": c.strategy,
            "page": c.page,
            "headings": c.headings,
            "bbox": c.bbox,
            "token_count": c.token_count,
            "experiment": config.name,
            "config_hash": config_hash,
            "source_file": source_file,
            "ingested_at": ingested_at,
            **({"image": c.image} if c.image else {}),
        }
        for c in rows
    ]

    collection = config.collection
    t0 = time.perf_counter()
    qdrant.ensure_collection(collection, int(vectors.shape[1]), config.index)
    qdrant.delete_document(collection, doc_id)
    # The collection exists now, so LlamaIndex's store reuses its (unnamed) vector instead of creating
    # its own. It writes each node's metadata flat into the payload and the Document id as `doc_id`.
    vector_store = QdrantVectorStore(collection_name=collection, client=qdrant.client, batch_size=256)
    nodes = []
    for chunk, vector, payload in zip(rows, vectors, payloads):
        metadata = {k: v for k, v in payload.items() if k != "doc_id"}
        nodes.append(
            TextNode(
                id_=to_point_id(chunk.chunk_id),
                text=chunk.text,
                embedding=vector.tolist(),
                metadata=metadata,
                excluded_embed_metadata_keys=list(metadata),
                excluded_llm_metadata_keys=list(metadata),
                relationships={NodeRelationship.SOURCE: RelatedNodeInfo(node_id=chunk.doc_id)},
            )
        )
    t1 = time.perf_counter()
    written = len(vector_store.add(nodes))
    upsert_seconds = time.perf_counter() - t1
    sparse_seconds = 0.0
    if config.index.sparse:
        t2 = time.perf_counter()
        qdrant.add_sparse(
            collection, [to_point_id(c.chunk_id) for c in rows], document_vectors([c.text for c in rows])
        )
        sparse_seconds = time.perf_counter() - t2
    total = qdrant.count(collection)

    # A failed batch raises out of add, so there is no partial-failure count to report.
    details = {
        "collection": collection,
        "points_written": written,
        "batches": -(-written // 256),
        "upsert_ms": round(upsert_seconds * 1000, 1),
        "setup_and_delete_ms": round((t1 - t0) * 1000, 1),
        "collection_points": total,
    }
    if config.index.sparse:
        details["sparse_ms"] = round(sparse_seconds * 1000, 1)
    store.add_stage_metric(
        config_hash,
        doc_id,
        "index",
        duration_ms=(time.perf_counter() - t0) * 1000,
        items=written,
        throughput=written / upsert_seconds if upsert_seconds else None,
        details=details,
        dagster_run_id=run_id,
    )
    return details

