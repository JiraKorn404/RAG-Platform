from dagster import AssetExecutionContext, Failure, MaterializeResult, MetadataValue, asset

from rag_lab import clients
from rag_lab.assets.partitions import documents_partitions
from rag_lab.documents import scan_raw
from rag_lab.ingest import (
    OCR_DETAILS,
    SCANNED_PDF_CHARS_PER_PAGE,
    IngestError,
    configured_experiment,
    record_parse,
)
from rag_lab.parsing.parse import parse_pdf
from rag_lab.paths import artifacts_dir


@asset(partitions_def=documents_partitions, group_name="ingestion")
def parsed_document(context: AssetExecutionContext) -> MaterializeResult:
    """Docling parse of one PDF. Output goes to data/artifacts/<experiment>/parse/. Ollama is only
    called when the experiment has `parse.ocr` on."""
    store = clients.metrics()
    try:
        config = configured_experiment(store)
    except IngestError as e:
        raise Failure(str(e)) from e
    doc_id = context.partition_key
    path = scan_raw().get(doc_id)
    if path is None:
        raise Failure(f"No PDF in data/raw with content hash {doc_id} (changed or removed?)")

    parsed = parse_pdf(
        path,
        doc_id,
        config.parse,
        artifacts_dir(config.name, "parse"),
        clients.ollama_url(),
        lambda done, total: context.log.info(f"OCR: page {done} of the {total} that need it"),
    )
    if parsed.chars_per_page < SCANNED_PDF_CHARS_PER_PAGE:
        context.log.warning(
            f"{path.name}: only {parsed.chars_per_page} characters per page. "
            + (
                "OCR was on and found little text."
                if config.parse.ocr
                else "This looks like a scanned PDF; set `parse.ocr` in config/pipeline.yaml to read it."
            )
        )

    record_parse(store, config, doc_id, parsed, context.run_id)

    return MaterializeResult(
        metadata={
            "experiment": config.name,
            "config_hash": config.config_hash(),
            "source_file": parsed.source_file,
            "status": parsed.status,
            "pages": parsed.pages,
            "tables": parsed.tables,
            "text_items": parsed.text_items,
            "chars_per_page": parsed.chars_per_page,
            "parse_seconds": parsed.parse_seconds,
            "pages_per_second": parsed.pages_per_second,
            "model_load_seconds": parsed.model_load_seconds,
            **({k: getattr(parsed, k) for k in OCR_DETAILS} if config.parse.ocr else {}),
            "markdown_preview": MetadataValue.md(parsed.markdown_preview),
        }
    )
