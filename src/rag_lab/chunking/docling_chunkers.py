"""The `hybrid` and `hierarchical` strategies: Docling's own chunkers, which follow the structure of
the document (they do the segmenting and the splitting), run through LlamaIndex's DoclingNodeParser.

Docling's default table serialisation is a "row, column = value" triplet text, which embeds poorly,
so tables are serialised as Markdown instead.
"""

import json

from docling.chunking import HierarchicalChunker, HybridChunker
from docling_core.transforms.chunker.hierarchical_chunker import (
    ChunkingDocSerializer,
    ChunkingSerializerProvider,
)
from docling_core.transforms.chunker.tokenizer.huggingface import HuggingFaceTokenizer
from docling_core.transforms.serializer.markdown import MarkdownTableSerializer
from docling_core.types.doc import BoundingBox
from llama_index.core import Document
from llama_index.node_parser.docling import DoclingNodeParser

from rag_lab.chunking.base import ChunkContext
from rag_lab.chunking.models import Chunk


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
