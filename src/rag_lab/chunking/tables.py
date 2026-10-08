"""Tables for the fixed, recursive and semantic strategies: a table becomes one chunk, or several
according to `table_handling` and its size. A table never goes through a text splitter."""

from rag_lab.chunking.base import ChunkContext
from rag_lab.chunking.models import Chunk
from rag_lab.chunking.segment import TableBlock


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
