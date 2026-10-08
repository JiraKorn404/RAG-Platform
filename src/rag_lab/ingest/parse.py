"""Parsing a PDF with Docling, and saving its pictures as files. OCR of regions without a text layer is
in ocr.py."""

import json
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path

import pypdfium2 as pdfium
from docling.datamodel.accelerator_options import AcceleratorDevice, AcceleratorOptions
from docling.datamodel.base_models import ConversionStatus, InputFormat
from docling.datamodel.pipeline_options import (
    PdfPipelineOptions,
    TableFormerMode,
    TableStructureOptions,
)
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling_core.types.doc import DoclingDocument
from docling_core.types.doc.document import PictureItem
from llama_index.readers.docling import DoclingReader

from rag_lab.core.config import ParseConfig
from rag_lab.ingest.ocr import fill_missing_text


@dataclass
class ParsedDocument:
    doc_id: str
    source_file: str
    status: str  # always "success": a parse that Docling fails or cuts short raises (see parse_pdf)
    pages: int = 0
    tables: int = 0
    table_cells: int = 0
    text_items: int = 0
    text_chars: int = 0
    chars_per_page: float = 0.0  # near zero means a scanned PDF parsed without OCR: there is no text layer
    markdown_chars: int = 0
    model_load_seconds: float = 0.0
    parse_seconds: float = 0.0  # Docling and, with OCR, the reading of the regions
    pages_per_second: float = 0.0
    markdown_preview: str = ""
    # Only with ParseConfig.ocr (see ocr.py: OcrStats)
    ocr_regions: int = 0
    ocr_pages: int = 0
    ocr_seconds: float = 0.0
    ocr_output_tokens: int = 0
    ocr_cut_regions: int = 0


class RecordingConverter(DocumentConverter):
    """Keeps the result of its last conversion. DoclingReader hands on only the document, so without
    this a parse that Docling stopped at its time limit (status `partial_success`, the pages it got
    to and nothing after them) would pass for a whole one."""

    last = None

    def convert(self, *args, **kwargs):
        self.last = super().convert(*args, **kwargs)
        return self.last


def build_converter(cfg: ParseConfig, pages: int = 0) -> RecordingConverter:
    """Docling's own OCR and its picture features are fixed off. With `cfg.ocr` the layout keeps the
    regions that have no text and the page images are made, for ocr.py to read them. Pictures
    are cropped by `save_pictures` from the boxes Docling finds either way."""
    options = PdfPipelineOptions(
        do_ocr=False,
        do_table_structure=cfg.do_table_structure,
        table_structure_options=TableStructureOptions(
            mode=TableFormerMode(cfg.table_mode),
            do_cell_matching=cfg.table_cell_matching,
        ),
        do_formula_enrichment=cfg.do_formula_enrichment,
        do_code_enrichment=cfg.do_code_enrichment,
        do_picture_classification=False,
        do_picture_description=False,
        generate_picture_images=False,
        generate_page_images=cfg.ocr,
        images_scale=cfg.ocr_scale if cfg.ocr else 1.0,
        document_timeout=max(cfg.document_timeout, cfg.page_timeout * pages),
        accelerator_options=AcceleratorOptions(
            num_threads=cfg.num_threads, device=AcceleratorDevice.CPU
        ),
    )
    options.layout_options.keep_empty_clusters = cfg.ocr
    return RecordingConverter(
        format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=options)}
    )


def parse_pdf(
    path: Path,
    doc_id: str,
    cfg: ParseConfig,
    out_dir: Path,
    ollama_url: str | None = None,
    on_ocr_page: Callable[[int, int], None] = lambda done, total: None,
) -> ParsedDocument:
    """Parse one PDF with LlamaIndex's DoclingReader (our converter inside) and write <doc_id>.json
    (Docling document), <doc_id>.md and <doc_id>.meta.json. Raises if Docling fails. With `cfg.ocr`
    the regions without text are then read by the OCR model on the Ollama at `ollama_url`."""
    if cfg.ocr and not ollama_url:
        raise ValueError("ParseConfig.ocr is on, so parsing needs the address of Ollama")
    pdf = pdfium.PdfDocument(path)
    page_count = len(pdf)
    pdf.close()
    t0 = time.perf_counter()
    converter = build_converter(cfg, page_count)
    converter.initialize_pipeline(InputFormat.PDF)  # loads the models, so parse time excludes it
    model_load_seconds = time.perf_counter() - t0

    def document_id(doc: DoclingDocument, file_path: str | Path) -> str:
        return doc_id  # the reader's Document id is our content-hash id, so Qdrant's `doc_id` matches

    reader = DoclingReader(
        export_type=DoclingReader.ExportType.JSON, doc_converter=converter, id_func=document_id
    )
    t1 = time.perf_counter()
    (li_doc,) = reader.load_data(path)

    doc = DoclingDocument.model_validate_json(li_doc.text)
    result = converter.last
    if result.status != ConversionStatus.SUCCESS:
        reached = len({item.prov[0].page_no for item, _ in doc.iterate_items() if item.prov})
        limit = max(cfg.document_timeout, cfg.page_timeout * page_count)
        reasons = "; ".join(sorted({e.error_message for e in result.errors})) or result.status.value
        raise RuntimeError(
            f"Docling did not finish {path.name} ({reasons}): it has content for {reached} of {page_count} "
            f"pages after {time.perf_counter() - t1:.0f} s, with a limit of {limit:.0f} s. Nothing was "
            "kept. Raise `parse.page_timeout` (seconds allowed for each page) or `parse.document_timeout`."
        )
    ocr = None
    if cfg.ocr:
        ocr = fill_missing_text(doc, cfg, ollama_url, on_ocr_page)
        for page in doc.pages.values():
            page.image = None  # only needed for the crops; they would make the JSON many megabytes
    if cfg.pictures:
        save_pictures(path, doc, out_dir, doc_id, cfg)
    parse_seconds = time.perf_counter() - t1
    markdown = doc.export_to_markdown()
    text_chars = sum(len(t.text) for t in doc.texts)
    parsed = ParsedDocument(
        doc_id=doc_id,
        source_file=path.name,
        status="success",
        model_load_seconds=round(model_load_seconds, 3),
        parse_seconds=round(parse_seconds, 3),
    )
    parsed.pages = len(doc.pages)
    parsed.tables = len(doc.tables)
    parsed.table_cells = sum(len(t.data.table_cells) for t in doc.tables)
    parsed.text_items = len(doc.texts)
    parsed.text_chars = text_chars
    parsed.chars_per_page = round(text_chars / parsed.pages, 1) if parsed.pages else 0.0
    parsed.markdown_chars = len(markdown)
    parsed.pages_per_second = round(parsed.pages / parse_seconds, 4) if parse_seconds else 0.0
    parsed.markdown_preview = markdown[:1500]
    if ocr:
        parsed.ocr_regions = ocr.regions
        parsed.ocr_pages = ocr.pages
        parsed.ocr_seconds = ocr.seconds
        parsed.ocr_output_tokens = ocr.output_tokens
        parsed.ocr_cut_regions = ocr.cut_regions

    doc.save_as_json(out_dir / f"{doc_id}.json")
    (out_dir / f"{doc_id}.md").write_text(markdown, encoding="utf-8")
    meta = {**asdict(parsed), "parse_config": cfg.model_dump(mode="json")}
    meta.pop("markdown_preview")
    (out_dir / f"{doc_id}.meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return parsed


# --- pictures ----------------------------------------------------------------------------------------
# The pictures of a parsed document as files: <doc_id>.pictures/<n>.png next to the parse files.
#
# Docling already finds every picture and its box, with or without a text layer. The picture itself is
# cropped here from the PDF page rendered with pypdfium2, so a document that is already parsed does not
# have to go through Docling again to get its pictures. `n` is the picture's index in `doc.pictures`.

def pictures_dir(doc_id: str) -> str:
    return f"{doc_id}.pictures"


def kept_pictures(doc: DoclingDocument, cfg: ParseConfig) -> list[tuple[int, PictureItem]]:
    """The pictures that are kept: those with a place on a page and no side shorter than
    `picture_min_side`. The parse stage saves exactly these and the chunk stage makes a chunk of each."""
    kept = []
    for n, picture in enumerate(doc.pictures):
        if not picture.prov:
            continue
        box = picture.prov[0].bbox
        if min(abs(box.r - box.l), abs(box.t - box.b)) >= cfg.picture_min_side:
            kept.append((n, picture))
    return kept


def save_pictures(pdf_path: Path, doc: DoclingDocument, out_dir: Path, doc_id: str, cfg: ParseConfig) -> int:
    """Write the kept pictures of `doc` under out_dir/<doc_id>.pictures/. Returns how many."""
    folder = out_dir / pictures_dir(doc_id)
    folder.mkdir(parents=True, exist_ok=True)
    kept = kept_pictures(doc, cfg)
    pdf = pdfium.PdfDocument(pdf_path)
    try:
        pages: dict[int, object] = {}  # page number -> rendered page, rendered once
        for n, picture in kept:
            prov = picture.prov[0]
            if prov.page_no not in pages:
                pages[prov.page_no] = pdf[prov.page_no - 1].render(scale=cfg.picture_scale).to_pil()
            image = pages[prov.page_no]
            page = doc.pages[prov.page_no]
            scale = image.width / page.size.width
            box = prov.bbox.to_top_left_origin(page.size.height)
            image.crop(
                (
                    max(int(box.l * scale), 0),
                    max(int(box.t * scale), 0),
                    min(int(box.r * scale) + 1, image.width),
                    min(int(box.b * scale) + 1, image.height),
                )
            ).save(folder / f"{n}.png")
    finally:
        pdf.close()
    return len(kept)
