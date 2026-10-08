"""What every chunking strategy is given."""

from dataclasses import dataclass, field

from docling_core.types.doc import DoclingDocument

from rag_lab.chunking.tokens import Tokens
from rag_lab.config import ExperimentConfig
from rag_lab.embedding.ollama import OllamaEmbedder


@dataclass
class ChunkContext:
    doc: DoclingDocument
    doc_id: str
    cfg: ExperimentConfig
    tokens: Tokens
    embedder: OllamaEmbedder | None = None  # only the semantic strategy needs it
    stats: dict = field(default_factory=dict)  # strategy-specific numbers, reported as metrics
    warnings: list[str] = field(default_factory=list)
