import os
from pathlib import Path

DATA_DIR = Path(os.environ.get("DATA_DIR", "data"))
RAW_DIR = DATA_DIR / "raw"  # PDFs to ingest
TABLES_DIR = DATA_DIR / "tables"  # CSV files to import: <schema>/<table>.csv


def artifacts_dir(experiment: str, stage: str) -> Path:
    """Per-stage outputs: data/artifacts/<experiment>/<stage>/. Created on demand."""
    path = DATA_DIR / "artifacts" / experiment / stage
    path.mkdir(parents=True, exist_ok=True)
    return path
