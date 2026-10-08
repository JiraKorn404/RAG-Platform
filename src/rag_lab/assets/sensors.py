from dagster import (
    DefaultSensorStatus,
    RunRequest,
    SensorEvaluationContext,
    SensorResult,
    SkipReason,
    sensor,
)

from rag_lab import clients
from rag_lab.assets.jobs import ingest_job, tables_job
from rag_lab.assets.partitions import documents_partitions, tables_partitions
from rag_lab.documents import doc_id_for, is_complete_pdf, raw_pdfs
from rag_lab.ingest import experiment_conflict
from rag_lab.settings import load
from rag_lab.sql.folder import is_settled, table_files, version


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
    problem = experiment_conflict(clients.metrics(), config)
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
    its partition, or fix the file."""
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
