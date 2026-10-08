import time

from dagster import AssetExecutionContext, Failure, MaterializeResult, MetadataValue, asset
from docling_core.types.doc import DoclingDocument

from rag_lab.assets.parsing import parsed_document
from rag_lab.assets.partitions import documents_partitions
from rag_lab.chunking import chunk_document, summarise
from rag_lab.chunking.models import write_chunks
from rag_lab.ingest import record_chunk, register_experiment
from rag_lab.paths import artifacts_dir
from rag_lab.resources import ExperimentResource, MetricsStoreResource, OllamaResource


@asset(partitions_def=documents_partitions, deps=[parsed_document], group_name="ingestion")
def chunks(
    context: AssetExecutionContext,
    experiment: ExperimentResource,
    metrics: MetricsStoreResource,
    ollama: OllamaResource,
) -> MaterializeResult:
    """Split the parsed document into chunks with the configured strategy.
    Reads parse/<doc_id>.json and writes chunk/<doc_id>.chunks.jsonl under the experiment folder."""
    config = experiment.config()
    doc_id = context.partition_key
    parsed_path = artifacts_dir(config.name, "parse") / f"{doc_id}.json"
    if not parsed_path.exists():
        raise Failure(
            f"No parsed document at {parsed_path}. Materialise parsed_document for this "
            f"partition with the same experiment name first."
        )

    store = metrics.store()
    register_experiment(store, config)

    doc = DoclingDocument.load_from_json(parsed_path)
    embedder = ollama.embedder() if config.chunk.strategy == "semantic" else None
    start = time.perf_counter()
    result, ctx = chunk_document(doc, doc_id, config, embedder)
    seconds = time.perf_counter() - start

    for warning in ctx.warnings:
        context.log.warning(warning)
    if not result:
        raise Failure(f"No chunks produced for {doc_id}. Is the PDF scanned, or is everything skipped?")

    write_chunks(result, artifacts_dir(config.name, "chunk") / f"{doc_id}.chunks.jsonl")
    summary = summarise(result)
    sample = result[0]

    record_chunk(store, config, doc_id, summary, ctx.stats, ctx.warnings, seconds, context.run_id)

    return MaterializeResult(
        metadata={
            "experiment": config.name,
            "strategy": config.chunk.strategy,
            "chunks": summary["chunks"],
            "by_modality": MetadataValue.json(summary["by_modality"]),
            "tokens_min": summary["tokens_min"],
            "tokens_mean": summary["tokens_mean"],
            "tokens_max": summary["tokens_max"],
            "chunk_seconds": round(seconds, 3),
            **({f"semantic_{k}": v for k, v in ctx.stats.items()} if ctx.stats else {}),
            "sample_chunk": MetadataValue.md(f"**{sample.modality}, page {sample.page}**\n\n{sample.text[:600]}"),
        }
    )
