"""The `fixed`, `recursive` and `semantic` strategies: the text under each heading is split with a
LlamaIndex splitter.

- fixed: TokenTextSplitter (token windows, with `overlap`).
- recursive: SentenceSplitter (paragraphs, then sentences).
- semantic: SemanticSplitterNodeParser (a cut where neighbouring sentences differ most; the threshold
  is taken per section), then SentenceSplitter for pieces that are still over the limit.

The segmenting (headings, pages, tables) is ours (`segment` in chunk.py): the splitters only see text. Each
section becomes a Document whose one embed-visible metadata value is its headings, so the splitters
reserve room for them and a chunk's text is "headings, newline, body". Tables never go through a
splitter (`table_chunks`).
"""

import hashlib
import re
import time
from array import array
from pathlib import Path

from llama_index.core import Document
from llama_index.core.base.embeddings.base import BaseEmbedding
from llama_index.core.bridge.pydantic import PrivateAttr
from llama_index.core.node_parser import (
    SemanticSplitterNodeParser,
    SentenceSplitter,
    TokenTextSplitter,
)
from llama_index.core.schema import MetadataMode, TextNode

from rag_lab.core.config import EmbedConfig
from rag_lab.core.embed import OllamaEmbedder
from rag_lab.core.settings import DATA_DIR
from rag_lab.ingest.chunk import Chunk, ChunkContext, Section, TableBlock, segment


def section_chunks(ctx: ChunkContext) -> list[Chunk]:
    cfg = ctx.cfg.chunk
    blocks = segment(ctx.doc)
    sections = [b for b in blocks if isinstance(b, Section)]
    documents = [_document(ctx, f"{ctx.doc_id}:{i}", sec) for i, sec in enumerate(sections)]
    pieces_by_section = _split(ctx, documents)

    chunks: list[Chunk] = []
    next_section = 0
    for block in blocks:
        if isinstance(block, TableBlock):
            if cfg.table_handling != "skip":
                chunks.extend(table_chunks(ctx, block))
            continue
        for text, offset in pieces_by_section[documents[next_section].id_]:
            para = block.locate(offset)
            chunks.append(
                Chunk(ctx.doc_id, text, "text", cfg.strategy, para.page, block.headings, para.bbox)
            )
        next_section += 1
    return chunks


def _document(ctx: ChunkContext, doc_id: str, section: Section) -> Document:
    """A section as a Document. Its headings are the one metadata value the embedder sees."""
    headings = "\n".join(section.headings) if ctx.cfg.chunk.include_headings_in_text else ""
    return Document(
        id_=doc_id,
        text=section.text,
        metadata={"headings": headings} if headings else {},
        metadata_template="{value}",
        metadata_seperator="\n",  # (sic) LlamaIndex's alias for metadata_separator
        text_template="{metadata_str}\n{content}",
    )


def _split(ctx: ChunkContext, documents: list[Document]) -> dict[str, list[tuple[str, int]]]:
    """Run the strategy's splitters. Returns, per document id, each chunk's embedded text and the
    character offset of its body in the section (for its page and bounding box)."""
    cfg = ctx.cfg.chunk
    tokenizer = lambda text: ctx.tokens.hf.encode(text, add_special_tokens=False)  # noqa: E731
    text_of = {d.id_: d.text for d in documents}

    if cfg.strategy == "fixed":
        if cfg.overlap >= cfg.max_tokens:
            raise ValueError(f"overlap ({cfg.overlap}) must be smaller than max_tokens ({cfg.max_tokens})")
        nodes = TokenTextSplitter(
            chunk_size=cfg.max_tokens, chunk_overlap=cfg.overlap, tokenizer=tokenizer
        ).get_nodes_from_documents(documents)
    else:
        # chunk_overlap is explicit because SentenceSplitter's own default is 200 tokens
        sentence_splitter = SentenceSplitter(
            chunk_size=cfg.max_tokens,
            chunk_overlap=0,
            paragraph_separator="\n\n",
            tokenizer=tokenizer,
        )
        if cfg.strategy == "recursive":
            nodes = sentence_splitter.get_nodes_from_documents(documents)
        else:
            nodes = sentence_splitter(_semantic_nodes(ctx, documents))

    out: dict[str, list[tuple[str, int]]] = {d.id_: [] for d in documents}
    cursor: dict[str, int] = {}
    for node in nodes:
        body = node.get_content(MetadataMode.NONE).strip()
        if not body:
            continue
        section_id = node.ref_doc_id
        # LlamaIndex splitters copy the section's metadata to every node, so the offset is found
        # by searching for the body from the previous chunk's start (chunks can overlap).
        start = cursor.get(section_id, 0)
        found = text_of[section_id].find(body, start)
        offset = found if found >= 0 else start
        cursor[section_id] = offset
        out[section_id].append((node.get_content(MetadataMode.EMBED), offset))
    return out


def _semantic_nodes(ctx: ChunkContext, documents: list[Document]) -> list[TextNode]:
    s = ctx.cfg.chunk.semantic
    if ctx.embedder is None:
        raise ValueError("The semantic strategy needs an embedder (Ollama)")

    embed_model = OllamaLabEmbedding(ctx.embedder, ctx.cfg.embed)
    parser = SemanticSplitterNodeParser.from_defaults(
        embed_model=embed_model,
        buffer_size=s.buffer_size,
        breakpoint_percentile_threshold=round(s.breakpoint_threshold),
        sentence_splitter=lambda text: _sentence_pieces(text, s.sentence_pattern),
    )
    nodes = parser.get_nodes_from_documents(documents)
    stats = embed_model.stats
    ctx.stats.update(
        sentences=stats["texts"],  # one embedded group per sentence
        group_texts=stats["texts"],
        embed_cache_hits=stats["cache_hits"],
        embedded=stats["embedded"],
        embed_ms=round(stats["embed_ms"], 1),
    )
    return nodes


def _sentence_pieces(text: str, pattern: str) -> list[str]:
    """Cut `text` after each sentence boundary, keeping the boundary with the sentence before it,
    so the pieces join back into `text` (LlamaIndex's semantic splitter concatenates them)."""
    pieces, pos = [], 0
    for m in re.finditer(pattern, text):
        if text[pos : m.end()].strip():
            pieces.append(text[pos : m.end()])
            pos = m.end()
    if text[pos:].strip():
        pieces.append(text[pos:])
    return pieces


# --- tables ------------------------------------------------------------------------------------------
# Tables for the fixed, recursive and semantic strategies: a table becomes one chunk, or several
# according to `table_handling` and its size. A table never goes through a text splitter.

def table_chunks(ctx: ChunkContext, block: TableBlock) -> list[Chunk]:
    cfg = ctx.cfg.chunk
    prefix = "\n".join(block.headings) + "\n" if cfg.include_headings_in_text and block.headings else ""
    # Headings are part of the embedded text, so they use up part of the budget (but never more
    # than half of it, so a very long heading cannot starve the body).
    budget = max(cfg.max_tokens - ctx.tokens.count(prefix), cfg.max_tokens // 2) if prefix else cfg.max_tokens
    return [
        Chunk(ctx.doc_id, prefix + text, "table", cfg.strategy, block.page, block.headings, block.bbox)
        for text in _table_texts(ctx, block, budget)
    ]


def _table_texts(ctx: ChunkContext, block: TableBlock, budget: int) -> list[str]:
    if ctx.cfg.chunk.table_handling == "row-wise":
        df = block.item.export_to_dataframe(doc=ctx.doc)
        rows = [
            "; ".join(f"{col}: {val}" for col, val in row.items() if str(val).strip())
            for _, row in df.iterrows()
        ]
        rows = [r for r in rows if r]
        if rows:
            return rows
    markdown = block.markdown
    if ctx.tokens.count(markdown) <= budget:
        return [markdown]
    return _split_markdown_table(markdown, budget, ctx)


def _split_markdown_table(markdown: str, budget: int, ctx: ChunkContext) -> list[str]:
    """Split an oversize table by rows, repeating the header in every piece. A single row that is
    over budget stays whole: it cannot be split without cutting a cell."""
    lines = markdown.splitlines()
    header, rows = lines[:2], lines[2:]
    pieces: list[str] = []
    current: list[str] = []
    for row in rows:
        if current and ctx.tokens.count("\n".join(header + current + [row])) > budget:
            pieces.append("\n".join(header + current))
            current = []
        current.append(row)
    if current:
        pieces.append("\n".join(header + current))
    return pieces


# --- the embedder as LlamaIndex sees it --------------------------------------------------------------
# LlamaIndex's view of our Ollama embedder.
#
# Only LlamaIndex's semantic splitter embeds anything inside LlamaIndex. The stock OllamaEmbedding
# would lose Ollama's timings, the Matryoshka truncation and our retries, so this wraps OllamaEmbedder
# instead. Vectors go through the on-disk cache, so changing a breakpoint setting does not call Ollama
# again.

class OllamaLabEmbedding(BaseEmbedding):
    _embedder: OllamaEmbedder = PrivateAttr()
    _cfg: EmbedConfig = PrivateAttr()
    _stats: dict = PrivateAttr()

    def __init__(self, embedder: OllamaEmbedder, cfg: EmbedConfig):
        super().__init__(model_name=cfg.model, embed_batch_size=cfg.batch_size)
        self._embedder = embedder
        self._cfg = cfg
        self._stats = {"texts": 0, "cache_hits": 0, "embedded": 0, "embed_ms": 0.0}

    @classmethod
    def class_name(cls) -> str:
        return "OllamaLabEmbedding"

    @property
    def stats(self) -> dict:
        """Texts asked for, cache hits, texts sent to Ollama and their milliseconds, so far."""
        return dict(self._stats)

    def _get_text_embeddings(self, texts: list[str]) -> list[list[float]]:
        vectors, info = embed_cached(texts, self._embedder, self._cfg)
        for key in self._stats:
            self._stats[key] += info[key]
        return vectors

    def _get_text_embedding(self, text: str) -> list[float]:
        return self._get_text_embeddings([text])[0]

    def _get_query_embedding(self, query: str) -> list[float]:
        return self._embedder.embed([query], self._cfg, kind="query").vectors[0]

    async def _aget_query_embedding(self, query: str) -> list[float]:
        return self._get_query_embedding(query)


# --- the embedding cache -----------------------------------------------------------------------------
# On-disk cache of document embeddings, one small file per (model, dimension, text).
#
# Used by semantic chunking, which embeds every sentence group: changing only a breakpoint setting
# re-uses the vectors instead of calling Ollama again. Shared across experiments.

def _key(cfg: EmbedConfig, text: str) -> str:
    # The text as it is sent, so a vector made with another document template is not reused. With
    # Qwen3's template that is the text itself, which keeps the keys of the files already cached.
    sent = cfg.document_template.format(text=text)
    return hashlib.sha256(f"{cfg.model}|{cfg.dimension}|{sent}".encode()).hexdigest()


def _path(root: Path, key: str) -> Path:
    return root / key[:2] / f"{key}.f32"


def embed_cached(
    texts: list[str], embedder: OllamaEmbedder, cfg: EmbedConfig, root: Path | None = None
) -> tuple[list[list[float]], dict]:
    """Document-mode embeddings for `texts`, in order. Returns the vectors and cache statistics."""
    root = root or DATA_DIR / "artifacts" / "_cache" / "embeddings"
    keys = [_key(cfg, t) for t in texts]
    found: dict[str, list[float]] = {}
    missing: dict[str, str] = {}  # key -> text, de-duplicated
    for key, text in zip(keys, texts):
        if key in found or key in missing:
            continue
        path = _path(root, key)
        if path.exists():
            vec = array("f")
            vec.frombytes(path.read_bytes())
            found[key] = vec.tolist()
        else:
            missing[key] = text

    start = time.perf_counter()
    if missing:
        result = embedder.embed(list(missing.values()), cfg, kind="document")
        for key, vec in zip(missing, result.vectors):
            found[key] = vec
            path = _path(root, key)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(array("f", vec).tobytes())

    stats = {
        "texts": len(texts),
        "cache_hits": len(texts) - sum(1 for k in keys if k in missing),
        "embedded": len(missing),
        "embed_ms": (time.perf_counter() - start) * 1000 if missing else 0.0,
    }
    return [found[k] for k in keys], stats
