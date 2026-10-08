"""What is in each experiment, and deleting a document or a whole experiment. Plain Python (no Streamlit,
no Dagster).

An experiment is a Qdrant collection (what search sees), its `experiments` row, its stage metric rows and
its folder data/artifacts/<name>/. Deleting always goes Qdrant first, then files, then database rows, so
that a half-finished delete can be finished by running it again. A PDF in data/raw and anything in
Dagster are never touched.
"""

import re
import shutil
from pathlib import Path

from rag_lab.core.config import NAME_PATTERN
from rag_lab.core.qdrant import SQL_EXAMPLES_PREFIX, QdrantStore
from rag_lab.core.settings import DATA_DIR
from rag_lab.core.store import MetricsStore

ARTIFACT_STAGES = ("parse", "chunk", "embed")


def _artifact_dir(name: str) -> Path | None:
    """data/artifacts/<name>, or None when the name could not be an experiment's. Names starting with
    an underscore (_cache) never match the pattern, and the folder must stay inside artifacts."""
    if not re.fullmatch(NAME_PATTERN, name):
        return None
    root = (DATA_DIR / "artifacts").resolve()
    path = (root / name).resolve()
    return path if path.parent == root else None


def list_library(metrics: MetricsStore, qdrant: QdrantStore) -> list[dict]:
    """Every experiment with its documents, newest first, then the collections that have no experiment
    row. Each entry: name, config (None without a row), has_row, has_collection, documents (name, id,
    points, ingested_at), points, created_at."""
    collections = {
        c.name for c in qdrant.client.get_collections().collections if not c.name.startswith(SQL_EXAMPLES_PREFIX)
    }
    entries = []
    for row in sorted(metrics.list_experiments(), key=lambda r: r["created_at"], reverse=True):
        present = row["name"] in collections
        docs = qdrant.documents(row["name"]) if present else []
        entries.append(
            {
                "name": row["name"],
                "config": row["config"],
                "has_row": True,
                "has_collection": present,
                "documents": docs,
                "points": sum(d["points"] for d in docs),
                "created_at": row["created_at"],
            }
        )
        collections.discard(row["name"])
    for name in sorted(collections):
        docs = qdrant.documents(name)
        entries.append(
            {
                "name": name,
                "config": None,
                "has_row": False,
                "has_collection": True,
                "documents": docs,
                "points": sum(d["points"] for d in docs),
                "created_at": None,
            }
        )
    return entries


def delete_document(name: str, doc_id: str, metrics: MetricsStore, qdrant: QdrantStore) -> dict:
    """Remove one document from one experiment: its points, its parse, chunk and embed files, and its
    stage metric rows there. Returns how many of each went."""
    if not re.fullmatch(r"[0-9a-f]{16}", doc_id):
        raise ValueError(f"'{doc_id}' is not a document id")
    points = 0
    if qdrant.client.collection_exists(name):
        points = qdrant.count_document(name, doc_id)
        qdrant.delete_document(name, doc_id)
    files = 0
    folder = _artifact_dir(name)
    if folder is not None:
        for stage in ARTIFACT_STAGES:
            for path in (folder / stage).glob(f"{doc_id}.*"):
                if path.is_dir():  # <doc_id>.pictures, the document's picture files
                    files += sum(1 for p in path.iterdir() if p.is_file())
                    shutil.rmtree(path)
                else:
                    path.unlink()
                    files += 1
    rows = 0
    found = metrics.get_experiment(name)
    if found is not None:
        rows = metrics.delete_document_rows(found[0], doc_id)
    return {"points": points, "files": files, "rows": rows}


def delete_experiment(name: str, metrics: MetricsStore, qdrant: QdrantStore) -> dict:
    """Remove an experiment: its collection, its folder under data/artifacts and its `experiments` row
    (the stage metric rows and search log go with it). A collection with no row is removed as a
    collection. Returns what was removed."""
    collection = qdrant.client.collection_exists(name)
    if collection:
        qdrant.delete_collection(name)
    folder = _artifact_dir(name)
    removed_folder = folder is not None and folder.exists()
    if folder is not None:
        shutil.rmtree(folder, ignore_errors=True)
    row = metrics.get_experiment(name) is not None
    if row:
        metrics.delete_experiment(name)
    return {"collection": collection, "folder": removed_folder, "row": row}
