"""Cut the Docling document into blocks, for the fixed, recursive and semantic strategies.

A Section is the text under one heading path (paragraphs joined by blank lines); a TableBlock is a
table. Pictures are skipped. Strategies split Sections; tables never go through a text splitter.
"""

from bisect import bisect_right
from dataclasses import dataclass, field

from docling_core.types.doc import DocItemLabel, DoclingDocument
from docling_core.types.doc.document import (
    PictureItem,
    SectionHeaderItem,
    TableItem,
    TextItem,
)


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
