"""Chunking: what a chunk is, the blocks a Docling document is cut into, Docling's own chunkers (`hybrid`
and `hierarchical`) and the picture chunks. The strategies that split text are in split_text.py, and
`chunk_document` in documents.py chooses between them.

A Section is the text under one heading path (paragraphs joined by blank lines); a TableBlock is a
table. Pictures are skipped by `segment`. Strategies split Sections; tables never go through a text
splitter.
"""

import json
import statistics
from bisect import bisect_right
from collections import Counter
from dataclasses import asdict, dataclass, field
from functools import lru_cache
from pathlib import Path

from docling.chunking import HierarchicalChunker, HybridChunker
from docling_core.transforms.chunker.hierarchical_chunker import (
    ChunkingDocSerializer,
    ChunkingSerializerProvider,
)
from docling_core.transforms.chunker.tokenizer.huggingface import HuggingFaceTokenizer
from docling_core.transforms.serializer.markdown import MarkdownTableSerializer
from docling_core.types.doc import BoundingBox, DocItemLabel, DoclingDocument
from docling_core.types.doc.document import (
    PictureItem,
    SectionHeaderItem,
    TableItem,
    TextItem,
)
from llama_index.core import Document
from llama_index.node_parser.docling import DoclingNodeParser
from transformers import AutoTokenizer

from rag_lab.core.config import ExperimentConfig
from rag_lab.core.embed import OllamaEmbedder
from rag_lab.ingest.parse import kept_pictures, pictures_dir


@dataclass
class Paragraph:
    text: str
    page: int | None
    bbox: list[float] | None
    ref: str | None = None  # self_ref of the Docling item


@dataclass
class Section:
    headings: list[str]
    paragraphs: list[Paragraph] = field(default_factory=list)

    def __post_init__(self):
        self._text: str | None = None
        self._starts: list[int] = []

    @property
    def text(self) -> str:
        if self._text is None:
            self._text = "\n\n".join(p.text for p in self.paragraphs)
            pos = 0
            for p in self.paragraphs:
                self._starts.append(pos)
                pos += len(p.text) + 2
        return self._text

    def locate(self, offset: int) -> Paragraph:
        """The paragraph that contains a character offset of `text` (for page and bbox)."""
        _ = self.text
        return self.paragraphs[max(bisect_right(self._starts, offset) - 1, 0)]


@dataclass
class TableBlock:
    item: TableItem
    headings: list[str]
    page: int | None
    bbox: list[float] | None
    markdown: str  # the table as the chunkers see it


def _where(item) -> tuple[int | None, list[float] | None]:
    if not item.prov:
        return None, None
    prov = item.prov[0]
    return prov.page_no, list(prov.bbox.as_tuple())


def segment(doc: DoclingDocument) -> list[Section | TableBlock]:
    blocks: list[Section | TableBlock] = []
    by_level: dict[int, str] = {}  # heading text per level; a heading drops all deeper ones
    section: Section | None = None

    def headings() -> list[str]:
        return [by_level[k] for k in sorted(by_level)]

    for item, _ in doc.iterate_items():
        if isinstance(item, PictureItem):
            continue
        if isinstance(item, TableItem):
            section = None
            page, bbox = _where(item)
            blocks.append(
                TableBlock(item, headings(), page, bbox, item.export_to_markdown(doc=doc))
            )
        elif isinstance(item, SectionHeaderItem) or (
            isinstance(item, TextItem) and item.label == DocItemLabel.TITLE
        ):
            section = None
            level = item.level if isinstance(item, SectionHeaderItem) else 0
            for k in [k for k in by_level if k > level]:
                del by_level[k]
            by_level[level] = item.text.strip()
        elif isinstance(item, TextItem) and item.text.strip():
            if section is None:
                section = Section(headings())
                blocks.append(section)
            page, bbox = _where(item)
            section.paragraphs.append(Paragraph(item.text.strip(), page, bbox, item.self_ref))
    return blocks


# --- a chunk -----------------------------------------------------------------------------------------

@dataclass
class Chunk:
    doc_id: str
    text: str
    modality: str  # "text", "table" or "picture"
    strategy: str
    page: int | None = None
    headings: list[str] = field(default_factory=list)
    bbox: list[float] | None = None  # [left, top, right, bottom] of the first element, PDF points
    token_count: int = 0
    chunk_id: str = ""  # f"{doc_id}:{index}", assigned once the document's chunks are final
    # A picture chunk's file, relative to the parse folder: "<doc_id>.pictures/<n>.png"
    image: str | None = None


def write_chunks(chunks: list[Chunk], path: Path) -> None:
    with path.open("w", encoding="utf-8") as f:
        for chunk in chunks:
            f.write(json.dumps(asdict(chunk), ensure_ascii=False) + "\n")


def read_chunks(path: Path) -> list[Chunk]:
    with path.open(encoding="utf-8") as f:
        return [Chunk(**json.loads(line)) for line in f if line.strip()]


# --- counting tokens ---------------------------------------------------------------------------------

class Tokens:
    """Counts tokens with the embedding model's own tokenizer, so `max_tokens` means what it embeds."""

    def __init__(self, hf_tokenizer):
        self.hf = hf_tokenizer

    def count(self, text: str) -> int:
        return len(self.hf.encode(text, add_special_tokens=False))


@lru_cache(maxsize=4)
def load_tokenizer(name: str) -> Tokens:
    return Tokens(AutoTokenizer.from_pretrained(name))


# --- what a strategy is given ------------------------------------------------------------------------
# What every chunking strategy is given.

@dataclass
class ChunkContext:
    doc: DoclingDocument
    doc_id: str
    cfg: ExperimentConfig
    tokens: Tokens
    embedder: OllamaEmbedder | None = None  # only the semantic strategy needs it
    stats: dict = field(default_factory=dict)  # strategy-specific numbers, reported as metrics
    warnings: list[str] = field(default_factory=list)


# --- hybrid and hierarchical -------------------------------------------------------------------------
# The `hybrid` and `hierarchical` strategies: Docling's own chunkers, which follow the structure of
# the document (they do the segmenting and the splitting), run through LlamaIndex's DoclingNodeParser.
#
# Docling's default table serialisation is a "row, column = value" triplet text, which embeds poorly,
# so tables are serialised as Markdown instead.

class _MarkdownTables(ChunkingSerializerProvider):
    def get_serializer(self, doc):
        return ChunkingDocSerializer(doc=doc, table_serializer=MarkdownTableSerializer())


def _chunker(ctx: ChunkContext):
    cfg = ctx.cfg.chunk
    if cfg.strategy == "hybrid":
        return HybridChunker(
            tokenizer=HuggingFaceTokenizer(tokenizer=ctx.tokens.hf, max_tokens=cfg.max_tokens),
            merge_peers=cfg.hybrid.merge_peers,
            serializer_provider=_MarkdownTables(),
        )
    return HierarchicalChunker(serializer_provider=_MarkdownTables())


def docling_chunks(ctx: ChunkContext) -> list[Chunk]:
    cfg = ctx.cfg.chunk
    if cfg.table_handling == "row-wise":
        ctx.warnings.append(
            f"table_handling 'row-wise' is not supported by '{cfg.strategy}'; tables use markdown"
        )
    parser = DoclingNodeParser(chunker=_chunker(ctx))
    document = Document(id_=ctx.doc_id, text=json.dumps(ctx.doc.export_to_dict()))

    chunks = []
    for node in parser.get_nodes_from_documents([document]):
        items = node.metadata.get("doc_items") or []
        labels = [i["label"] for i in items]
        if cfg.table_handling == "skip" and labels and all(lab == "table" for lab in labels):
            continue
        headings = list(node.metadata.get("headings") or [])
        prov = items[0]["prov"][0] if items and items[0].get("prov") else None
        # Docling's contextualize(): the headings, one per line, then the chunk text
        text = "\n".join([*headings, node.text]) if cfg.include_headings_in_text else node.text
        chunks.append(
            Chunk(
                doc_id=ctx.doc_id,
                text=text,
                modality="table" if "table" in labels else "text",
                strategy=cfg.strategy,
                page=prov["page_no"] if prov else None,
                headings=headings,
                bbox=list(BoundingBox.model_validate(prov["bbox"]).as_tuple()) if prov else None,
            )
        )
    return chunks


# --- pictures ----------------------------------------------------------------------------------------
# One chunk per picture, added after the chosen strategy's chunks when `parse.pictures` is on.
#
# A picture chunk's `image` is its file (parse.py) and its text is the picture's caption, or
# "Figure on page N" when Docling linked none, so the text is never empty: BM25 and the rerankers read
# it, and the embedding does too unless `embed.picture_input` is `image`. Its headings are those of the
# last text or table before it.

def picture_chunks(ctx: ChunkContext) -> list[Chunk]:
    doc, cfg = ctx.doc, ctx.cfg
    # the headings every text and table item is under
    headings_of: dict[str, list[str]] = {}
    for block in segment(doc):
        if isinstance(block, TableBlock):
            headings_of[block.item.self_ref] = block.headings
        else:
            for paragraph in block.paragraphs:
                headings_of[paragraph.ref] = block.headings

    kept = {picture.self_ref: n for n, picture in kept_pictures(doc, cfg.parse)}
    headings: list[str] = []
    chunks = []
    for item, _ in doc.iterate_items():
        if item.self_ref in headings_of:
            headings = headings_of[item.self_ref]
        elif item.self_ref in kept:
            prov = item.prov[0]
            text = " ".join(item.caption_text(doc).split()) or f"Figure on page {prov.page_no}"
            if cfg.chunk.include_headings_in_text and headings:
                text = "\n".join(headings) + "\n" + text
            chunks.append(
                Chunk(
                    doc_id=ctx.doc_id,
                    text=text,
                    modality="picture",
                    strategy=cfg.chunk.strategy,
                    page=prov.page_no,
                    headings=list(headings),
                    bbox=list(prov.bbox.as_tuple()),
                    image=f"{pictures_dir(ctx.doc_id)}/{kept[item.self_ref]}.png",
                )
            )
    return chunks


# --- numbers for the run -----------------------------------------------------------------------------

def summarise(chunks: list[Chunk]) -> dict:
    tokens = [c.token_count for c in chunks]
    return {
        "chunks": len(chunks),
        "by_modality": dict(Counter(c.modality for c in chunks)),
        "tokens_min": min(tokens, default=0),
        "tokens_mean": round(statistics.fmean(tokens), 1) if tokens else 0,
        "tokens_max": max(tokens, default=0),
    }
