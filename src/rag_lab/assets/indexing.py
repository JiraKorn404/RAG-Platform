from dagster import AssetExecutionContext, Failure, MaterializeResult, asset

from rag_lab import clients
from rag_lab.assets.chunking import chunks
from rag_lab.assets.partitions import documents_partitions
from rag_lab.ingest import IngestError, configured_experiment, embed_chunks, index_chunks


@asset(partitions_def=documents_partitions, deps=[chunks], group_name="ingestion")
def embeddings(context: AssetExecutionContext) -> MaterializeResult:
    """Embed every chunk of the document (document mode, no query prefix) with Ollama.
    Reads chunk/<doc_id>.chunks.jsonl and writes embed/<doc_id>.npy: one float32 row per chunk."""
    store = clients.metrics()
    try:
        config = configured_experiment(store)
        metadata = embed_chunks(store, clients.embedder(), config, context.partition_key, context.run_id)
    except IngestError as e:
        raise Failure(str(e)) from e
    return MaterializeResult(metadata=metadata)


@asset(partitions_def=documents_partitions, deps=[embeddings], group_name="ingestion")
def qdrant_index(context: AssetExecutionContext) -> MaterializeResult:
    """Write the document's chunks and vectors to the experiment's Qdrant collection.
    Idempotent: the document's existing points are deleted first and point ids are deterministic."""
    store = clients.metrics()
    try:
        config = configured_experiment(store)
        details = index_chunks(store, clients.qdrant(), config, context.partition_key, context.run_id)
    except IngestError as e:
        raise Failure(str(e)) from e
    return MaterializeResult(metadata=details)
