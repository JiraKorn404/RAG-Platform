"""One chunk per picture, added after the chosen strategy's chunks when `parse.pictures` is on.

A picture chunk's `image` is its file (parsing/pictures.py) and its text is the picture's caption, or
"Figure on page N" when Docling linked none, so the text is never empty: BM25 and the rerankers read
it, and the embedding does too unless `embed.picture_input` is `image`. Its headings are those of the
last text or table before it."""

from rag_lab.chunking.base import ChunkContext
from rag_lab.chunking.models import Chunk
from rag_lab.chunking.segment import TableBlock, segment
from rag_lab.parsing.pictures import kept_pictures, pictures_dir


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
