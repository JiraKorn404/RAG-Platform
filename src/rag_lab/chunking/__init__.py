import statistics
from collections import Counter

from docling_core.types.doc import DoclingDocument

from rag_lab.chunking.base import ChunkContext
from rag_lab.chunking.docling_chunkers import docling_chunks
from rag_lab.chunking.models import Chunk
from rag_lab.chunking.pictures import picture_chunks
from rag_lab.chunking.text_splitters import section_chunks
from rag_lab.chunking.tokens import load_tokenizer
from rag_lab.config import ExperimentConfig
from rag_lab.embedding.ollama import OllamaEmbedder


def chunk_document(
    doc: DoclingDocument, doc_id: str, cfg: ExperimentConfig, embedder: OllamaEmbedder | None = None
) -> tuple[list[Chunk], ChunkContext]:
    """Chunk one parsed document with the strategy in `cfg.chunk`. The context carries the
    strategy's own stats and any warnings."""
    strategy = cfg.chunk.strategy
    ctx = ChunkContext(doc, doc_id, cfg, load_tokenizer(cfg.embed.tokenizer), embedder)
    if cfg.chunk.overlap and strategy != "fixed":
        ctx.warnings.append(f"overlap is only used by 'fixed'; ignored for '{strategy}'")

    chunks = docling_chunks(ctx) if strategy in ("hybrid", "hierarchical") else section_chunks(ctx)
    if cfg.parse.pictures:
        chunks = chunks + picture_chunks(ctx)
    for i, chunk in enumerate(chunks):
        chunk.chunk_id = f"{doc_id}:{i:05d}"
        chunk.token_count = ctx.tokens.count(chunk.text)
    return chunks, ctx


def summarise(chunks: list[Chunk]) -> dict:
    tokens = [c.token_count for c in chunks]
    return {
        "chunks": len(chunks),
        "by_modality": dict(Counter(c.modality for c in chunks)),
        "tokens_min": min(tokens, default=0),
        "tokens_mean": round(statistics.fmean(tokens), 1) if tokens else 0,
        "tokens_max": max(tokens, default=0),
    }
