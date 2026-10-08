import time

from dagster import AssetExecutionContext, Failure, MaterializeResult, MetadataValue, asset

from rag_lab import clients
from rag_lab.assets.partitions import tables_partitions
from rag_lab.settings import ConfigError, load
from rag_lab.sql.folder import import_file, table_files
from rag_lab.sql.importer import CsvError


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
        done = import_file(clients.metrics(), file, load().tables)
    except (CsvError, ConfigError) as e:
        raise Failure(str(e)) from e
    return MaterializeResult(
        metadata={
            **done,
            "columns": MetadataValue.json(done["columns"]),
            "import_seconds": round(time.perf_counter() - start, 3),
        }
    )
