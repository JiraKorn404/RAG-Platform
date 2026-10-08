from dagster import Definitions

from rag_lab.assets.chunking import chunks
from rag_lab.assets.indexing import embeddings, qdrant_index
from rag_lab.assets.jobs import ingest_job, tables_job
from rag_lab.assets.parsing import parsed_document
from rag_lab.assets.sensors import new_csv_sensor, new_pdf_sensor
from rag_lab.assets.tables import imported_table

# No run config and no resources: a run reads config/*.yaml when it starts (rag_lab/settings.py), and
# the assets get their clients from rag_lab/clients.py.
defs = Definitions(
    assets=[parsed_document, chunks, embeddings, qdrant_index, imported_table],
    jobs=[ingest_job, tables_job],
    sensors=[new_pdf_sensor, new_csv_sensor],
)
