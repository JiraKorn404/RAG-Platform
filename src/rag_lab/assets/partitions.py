from dagster import DynamicPartitionsDefinition

# One partition per document, keyed by content hash, and one per imported table, keyed `schema.table`.
# assets/sensors.py registers them.
documents_partitions = DynamicPartitionsDefinition(name="documents")
tables_partitions = DynamicPartitionsDefinition(name="tables")
