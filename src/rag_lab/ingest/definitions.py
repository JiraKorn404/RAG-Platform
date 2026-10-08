"""What Dagster loads: the assets, the jobs and the sensors. They are thin: each asset reads the settings,
calls a stage body (documents.py, folder.py) and keeps only what is Dagster's (the partition key,
Failure, the run id, MaterializeResult)."""

import time

from dagster import (
    AssetExecutionContext,
    DefaultSensorStatus,
    Definitions,
    DynamicPartitionsDefinition,
    Failure,
    MaterializeResult,
    MetadataValue,
    RunRequest,
    SensorEvaluationContext,
    SensorResult,
    SkipReason,
    asset,
    define_asset_job,
    in_process_executor,
    sensor,
)
from docling_core.types.doc import DoclingDocument

from rag_lab.core import connections
from rag_lab.core.settings import ConfigError, artifacts_dir, load
from rag_lab.ingest.chunk import summarise, write_chunks
from rag_lab.ingest.documents import (
    OCR_DETAILS,
    SCANNED_PDF_CHARS_PER_PAGE,
    IngestError,
    chunk_document,
    configured_experiment,
    embed_chunks,
    experiment_conflict,
    index_chunks,
    record_chunk,
    record_parse,
)
from rag_lab.ingest.folder import import_file, is_settled, table_files, version
from rag_lab.ingest.importer import CsvError
from rag_lab.ingest.parse import parse_pdf
from rag_lab.ingest.register import register
from rag_lab.ingest.source import doc_id_for, is_complete_pdf, raw_pdfs, scan_raw

# --- partitions --------------------------------------------------------------------------------------

# One partition per document, keyed by content hash, and one per imported table, keyed `schema.table`.
# The sensors below register them.
documents_partitions = DynamicPartitionsDefinition(name="documents")
tables_partitions = DynamicPartitionsDefinition(name="tables")


# --- the PDF assets: parse ---------------------------------------------------------------------------

@asset(partitions_def=documents_partitions, group_name="ingestion")
def parsed_document(context: AssetExecutionContext) -> MaterializeResult:
    """Docling parse of one PDF. Output goes to data/artifacts/<experiment>/parse/. Ollama is only
    called when the experiment has `parse.ocr` on."""
    store = connections.metrics()
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
        connections.ollama_url(),
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


# --- chunk -------------------------------------------------------------------------------------------

@asset(partitions_def=documents_partitions, deps=[parsed_document], group_name="ingestion")
def chunks(context: AssetExecutionContext) -> MaterializeResult:
    """Split the parsed document into chunks with the configured strategy.
    Reads parse/<doc_id>.json and writes chunk/<doc_id>.chunks.jsonl under the experiment folder."""
    store = connections.metrics()
    try:
        config = configured_experiment(store)
    except IngestError as e:
        raise Failure(str(e)) from e
    doc_id = context.partition_key
    parsed_path = artifacts_dir(config.name, "parse") / f"{doc_id}.json"
    if not parsed_path.exists():
        raise Failure(
            f"No parsed document at {parsed_path}. Materialise parsed_document for this "
            f"partition first (the experiment is '{config.name}')."
        )

    doc = DoclingDocument.load_from_json(parsed_path)
    embedder = connections.embedder() if config.chunk.strategy == "semantic" else None
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


# --- embed and index ---------------------------------------------------------------------------------

@asset(partitions_def=documents_partitions, deps=[chunks], group_name="ingestion")
def embeddings(context: AssetExecutionContext) -> MaterializeResult:
    """Embed every chunk of the document (document mode, no query prefix) with Ollama.
    Reads chunk/<doc_id>.chunks.jsonl and writes embed/<doc_id>.npy: one float32 row per chunk."""
    store = connections.metrics()
    try:
        config = configured_experiment(store)
        metadata = embed_chunks(store, connections.embedder(), config, context.partition_key, context.run_id)
    except IngestError as e:
        raise Failure(str(e)) from e
    return MaterializeResult(metadata=metadata)


@asset(partitions_def=documents_partitions, deps=[embeddings], group_name="ingestion")
def qdrant_index(context: AssetExecutionContext) -> MaterializeResult:
    """Write the document's chunks and vectors to the experiment's Qdrant collection.
    Idempotent: the document's existing points are deleted first and point ids are deterministic."""
    store = connections.metrics()
    try:
        config = configured_experiment(store)
        details = index_chunks(store, connections.qdrant(), config, context.partition_key, context.run_id)
    except IngestError as e:
        raise Failure(str(e)) from e
    return MaterializeResult(metadata=details)


# --- the CSV asset -----------------------------------------------------------------------------------

@asset(partitions_def=tables_partitions, group_name="tables")
def imported_table(context: AssetExecutionContext) -> MaterializeResult:
    """One CSV file of data/tables/<schema>/ loaded as a table of that schema, replacing the table if it
    exists. A file that cannot be loaded fails the run with the reason and leaves the table as it was."""
    files, _ = table_files()
    file = next((f for f in files if f.key == context.partition_key), None)
    if file is None:
        raise Failure(f"No CSV file in data/tables for the table {context.partition_key} (moved or removed?)")
    start = time.perf_counter()
    try:
        done = import_file(connections.metrics(), file, load().tables)
    except (CsvError, ConfigError) as e:
        raise Failure(str(e)) from e
    return MaterializeResult(
        metadata={
            **done,
            "columns": MetadataValue.json(done["columns"]),
            "import_seconds": round(time.perf_counter() - start, 3),
        }
    )


@asset(group_name="tables")
def registered_tables(context: AssetExecutionContext) -> MaterializeResult:
    """The tables that are already in the database, described in the registry for the SQL chatbot: every
    schema of `tables_database.existing_schemas` (config/connections.yaml). Nothing in the database is
    changed. Materialise it again when the tables change."""
    try:
        done = register(connections.metrics(), connections.qdrant(), load().tables)
    except (CsvError, ConfigError, RuntimeError) as e:
        raise Failure(str(e)) from e
    for warning in done["warnings"]:
        context.log.warning(warning)
    return MaterializeResult(
        metadata={
            "schemas": MetadataValue.json(done["schemas"]),
            "tables": sum(len(tables) for tables in done["schemas"].values()),
            "removed": MetadataValue.json(done["removed"]),
            "warnings": len(done["warnings"]),
        }
    )


# --- jobs --------------------------------------------------------------------------------------------

# Parse -> chunk -> embed -> index for the selected document partitions. A run reads its settings from
# config/*.yaml when it starts, so there is no run config.
# The four steps run in the run's own process. With the default executor each step starts a process
# of its own, which imports Docling again: about 45 s a step, 170 s of a 204 s run for a two-page PDF.
ingest_job = define_asset_job(
    "ingest_job",
    selection=[parsed_document, chunks, embeddings, qdrant_index],
    partitions_def=documents_partitions,
    executor_def=in_process_executor,
)

# One CSV file of data/tables into its table.
tables_job = define_asset_job(
    "tables_job",
    selection=[imported_table],
    partitions_def=tables_partitions,
    executor_def=in_process_executor,
)

# Describe the tables that are already in the database. Started by hand; no sensor watches a database.
register_job = define_asset_job("register_job", selection=[registered_tables], executor_def=in_process_executor)


# --- sensors -----------------------------------------------------------------------------------------

@sensor(job=ingest_job, minimum_interval_seconds=30, default_status=DefaultSensorStatus.RUNNING)
def new_pdf_sensor(context: SensorEvaluationContext):
    """Starts `ingest_job` for each PDF that is new in data/raw. The run reads its settings from
    config/pipeline.yaml and config/llm.yaml when it starts. New means not yet a partition: a PDF that
    was already in the folder, or the same bytes under another file name, starts nothing. A file that is
    still being written waits for a later tick. The settings are read at every tick that finds a new
    PDF; when they do not load, or name an experiment that exists with other settings, the tick fails
    with the reason and the PDF stays new, so it runs once the file is put right."""
    known = set(context.instance.get_dynamic_partitions(documents_partitions.name))
    waiting, new = [], {}
    for path in raw_pdfs():
        if not is_complete_pdf(path):
            waiting.append(path.name)
        elif (doc_id := doc_id_for(path)) not in known:
            new.setdefault(doc_id, path)
    note = ""
    if waiting:
        note = f" Waiting for {len(waiting)} file(s) that are not complete yet: {', '.join(waiting)}."
    if not new:
        return SkipReason("No new PDFs." + note)

    config = load().experiment
    problem = experiment_conflict(connections.metrics(), config)
    if problem:
        names = ", ".join(path.name for path in new.values())
        raise ValueError(f"{problem} Not started: {names}.")
    context.log.info(f"{len(new)} new PDF(s) for the experiment '{config.name}'." + note)
    return SensorResult(
        dynamic_partitions_requests=[documents_partitions.build_add_request(list(new))],
        run_requests=[
            RunRequest(
                run_key=f"{config.name}:{doc_id}",
                partition_key=doc_id,
                tags={"experiment": config.name, "source_file": path.name},
            )
            for doc_id, path in new.items()
        ],
    )


@sensor(job=tables_job, minimum_interval_seconds=30, default_status=DefaultSensorStatus.RUNNING)
def new_csv_sensor(context: SensorEvaluationContext):
    """Starts `tables_job` for each CSV file in data/tables/<schema>/ that is new or has changed: the run
    key is the file's content hash (with that of the folder's `_schema.yaml`), so the same file is never
    imported twice and an edit to either imports it again. A file that is still being written waits for
    a later tick. A run that failed is not started again until the file changes; start it by hand for
    its partition, or fix the file. Without a loader role in config/connections.yaml the import is off."""
    if load().connections.tables_database.loader_url is None:
        return SkipReason("The CSV import is off: `tables_database.loader_url` is not set in config/connections.yaml.")
    files, unusable = table_files()
    for reason in unusable:
        context.log.warning(reason)
    ready = [f for f in files if is_settled(f.path)]
    note = f" Not usable: {'; '.join(unusable)}." if unusable else ""
    if len(ready) < len(files):
        note += f" Waiting for {len(files) - len(ready)} file(s) that are not complete yet."
    if not ready:
        return SkipReason("No CSV files to import." + note)

    known = set(context.instance.get_dynamic_partitions(tables_partitions.name))
    new = [f.key for f in ready if f.key not in known]
    return SensorResult(
        dynamic_partitions_requests=[tables_partitions.build_add_request(new)] if new else [],
        run_requests=[
            RunRequest(
                run_key=f"{f.key}:{version(f)}",
                partition_key=f.key,
                tags={"schema": f.schema, "table": f.table, "source_file": f.path.name},
            )
            for f in ready
        ],
    )


# No run config and no resources: a run reads config/*.yaml when it starts (core/settings.py), and
# the assets get their clients from core/connections.py.
defs = Definitions(
    assets=[parsed_document, chunks, embeddings, qdrant_index, imported_table, registered_tables],
    jobs=[ingest_job, tables_job, register_job],
    sensors=[new_pdf_sensor, new_csv_sensor],
)
