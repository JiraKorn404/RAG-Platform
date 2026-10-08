"""Finding PDFs in data/raw. A document's id is a hash of its content, so renaming a file does not
make it a new document, and the same bytes under two names count once."""

import hashlib
from pathlib import Path

from rag_lab.paths import RAW_DIR

# (path, mtime_ns, size) -> hash, so a sensor does not re-hash every file on every tick
_hash_cache: dict[tuple[str, int, int], str] = {}


def content_hash(path: Path) -> str:
    """The first 16 hex characters of the SHA-256 of a file's bytes."""
    stat = path.stat()
    key = (str(path), stat.st_mtime_ns, stat.st_size)
    if key not in _hash_cache:
        _hash_cache[key] = hashlib.sha256(path.read_bytes()).hexdigest()[:16]
    return _hash_cache[key]


doc_id_for = content_hash  # a document's id is the hash of its content


def raw_pdfs() -> list[Path]:
    return [p for p in sorted(RAW_DIR.iterdir()) if p.is_file() and p.suffix.lower() == ".pdf"]


def is_complete_pdf(path: Path) -> bool:
    """Whether the file can be read and ends as a PDF does (`%%EOF` in its last kilobyte). A file that
    Windows is still copying into the folder cannot be opened from the container at all (permission
    denied until the copy ends); the end marker covers a program that writes the file bit by bit."""
    try:
        with path.open("rb") as f:
            size = f.seek(0, 2)
            f.seek(max(size - 1024, 0))
            return b"%%EOF" in f.read()
    except OSError:
        return False


def scan_raw() -> dict[str, Path]:
    """doc_id -> path for every PDF in data/raw."""
    return {doc_id_for(p): p for p in raw_pdfs()}
