"""OCR for the regions of a document that have no text layer, with a vision model on Ollama (glm-ocr).

Docling finds the regions: its layout model works on the page image, so a scanned page still gets its
headings, paragraphs, list items and tables with their boxes, only without text. Each such region is
cropped from the page image and read here. The model is used per region, never per page: a whole page
fails ("token repeat limit reached"), a cropped paragraph or table is read correctly.

glm-ocr:bf16 on Ollama 0.40 does not stop by itself. After the text it writes a newline and then a code
fence, a rule or the same text again, up to the token limit. So the reply is read as a stream and the
connection is closed as soon as `first_reading` sees where the first reading ends.
"""

import base64
import io
import json
import re
import time
from collections.abc import Callable
from dataclasses import dataclass

import httpx
from docling.utils.deepseekocr_utils import _parse_table_html
from docling_core.types.doc import DocItemLabel, DoclingDocument
from PIL import Image

from rag_lab.core.config import ParseConfig

TEXT_PROMPT = "Text Recognition:"
TABLE_PROMPT = "Table Recognition:"
FORMULA_PROMPT = "Formula Recognition:"
# Lines of these regions are kept apart; every other region is one piece of running text.
MULTILINE = {DocItemLabel.CODE, DocItemLabel.FORMULA}
# Points added around a region before it is cropped. More than this reads the neighbours' lines too
# (6 points did, on a page with 11 point line spacing).
PAD_POINTS = 1.0
SAME_START = 24  # a later line that starts with this much of the first line is the text coming round again
LIST_MARKER = re.compile(r"^[-*•·▪◦⦁]\s+")
HAN = re.compile(r"[⺀-鿿]")  # CJK radicals to unified ideographs
TIMEOUT = 300.0
RETRIES = 3


@dataclass
class OcrStats:
    regions: int = 0
    pages: int = 0
    seconds: float = 0.0
    output_tokens: int = 0
    cut_regions: int = 0  # reached the token limit, or the model gave up, before the reading ended
    empty_regions: int = 0  # nothing was read; a text region is then removed from the document


def first_reading(text: str, complete: bool = True, lenient: bool = False) -> tuple[list[str], bool]:
    """The lines of the model's first reading of a region, and whether its end was seen. What follows
    the reading has been seen in five shapes, and each ends it: a code fence; the text again (a line
    that starts like the first line); a line with Chinese characters after a first line that has none
    (the model drifts into them); a rule (a line with no letter or digit); one junk line over and over
    (a line seen before: the reading is then cut where that line first appeared). `lenient` leaves out
    the last two, for code and formulas, where a line such as `}` or a repeated line is normal.
    With `complete` false the last line is still being written: it can end the reading, but is not kept."""
    lines = text.split("\n")
    kept: list[str] = []
    for i, raw in enumerate(lines):
        line = raw.strip()
        growing = not complete and i == len(lines) - 1
        if not line:
            continue
        if line.startswith("```"):
            return kept, True
        if kept:
            first = kept[0]
            if len(first) >= SAME_START and line.startswith(first[:SAME_START]):
                return kept, True
            if HAN.search(line) and not HAN.search(first):
                return kept, True
            if not growing and line == first:
                return kept, True
            if not growing and not lenient:
                if not any(c.isalnum() for c in line):
                    return kept, True
                if line in kept:
                    return kept[: kept.index(line)], True
        if not growing:
            kept.append(line)
    return kept, False


def _ask(
    base_url: str, cfg: ParseConfig, image: Image.Image, prompt: str, lenient: bool = False
) -> tuple[str, int, bool]:
    """One region through /api/chat as a stream. Returns the reply as far as it was read, the output
    tokens, and whether it was cut (the token limit, or the model's own abort) before it ended."""
    table = prompt == TABLE_PROMPT
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    options = {"temperature": 0, "num_predict": cfg.ocr_max_tokens}
    if table:
        options["stop"] = ["</table>"]
    body = {
        "model": cfg.ocr_model,
        "messages": [
            {"role": "user", "content": prompt, "images": [base64.b64encode(buffer.getvalue()).decode()]}
        ],
        "stream": True,
        "keep_alive": cfg.ocr_keep_alive,
        "options": options,
    }
    for attempt in range(RETRIES):
        text, tokens = "", 0
        try:
            with httpx.stream("POST", f"{base_url.rstrip('/')}/api/chat", json=body, timeout=TIMEOUT) as reply:
                if reply.status_code >= 400:
                    reply.read()
                    raise RuntimeError(f"OCR with {cfg.ocr_model} failed: {reply.text[:300]}")
                for line in reply.iter_lines():
                    if not line:
                        continue
                    part = json.loads(line)
                    if part.get("error"):  # for example "prediction aborted, token repeat limit reached"
                        return text, tokens, True
                    text += part.get("message", {}).get("content", "")
                    tokens += 1
                    if part.get("done"):
                        return text, part.get("eval_count", tokens), part.get("done_reason") == "length"
                    if not table and first_reading(text, complete=False, lenient=lenient)[1]:
                        return text, tokens, False  # leaving the block closes the connection: Ollama stops
            return text, tokens, True  # the stream ended without saying it was done
        except httpx.TransportError:
            if attempt < RETRIES - 1:
                time.sleep(2**attempt)
    raise RuntimeError(f"OCR failed: Ollama at {base_url} could not be reached after {RETRIES} attempts")


def _crop(doc: DoclingDocument, item) -> tuple[int, Image.Image]:
    prov = item.prov[0]
    page = doc.pages[prov.page_no]
    image = page.image.pil_image
    scale = image.width / page.size.width
    box = prov.bbox.to_top_left_origin(page.size.height)
    return prov.page_no, image.crop(
        (
            max(int((box.l - PAD_POINTS) * scale), 0),
            max(int((box.t - PAD_POINTS) * scale), 0),
            min(int((box.r + PAD_POINTS) * scale) + 1, image.width),
            min(int((box.b + PAD_POINTS) * scale) + 1, image.height),
        )
    )


def fill_missing_text(
    doc: DoclingDocument,
    cfg: ParseConfig,
    base_url: str,
    on_page: Callable[[int, int], None] = lambda done, total: None,
) -> OcrStats:
    """Read every text item that has no text and every table that has no cells, in place. `doc` must
    have its page images. A region where nothing is read is removed. `on_page(n, total)` is called
    before the n-th of the pages that have such regions."""
    todo = [(item, False) for item in doc.texts if item.prov and not item.text.strip()]
    todo += [
        (item, True)
        for item in doc.tables
        if item.prov and not any(cell.text.strip() for cell in item.data.table_cells)
    ]
    todo.sort(key=lambda job: job[0].prov[0].page_no)
    stats = OcrStats(pages=len({item.prov[0].page_no for item, _ in todo}))
    started = time.perf_counter()
    seen: set[int] = set()
    empty = []
    for item, is_table in todo:
        page_no, image = _crop(doc, item)
        if page_no not in seen:
            seen.add(page_no)
            on_page(len(seen), stats.pages)
        if is_table:
            reply, tokens, cut = _ask(base_url, cfg, image, TABLE_PROMPT)
            item.data = _parse_table_html(reply + "</table>")  # the stop string is not part of the reply
            found = bool(item.data.table_cells)
        else:
            prompt = FORMULA_PROMPT if item.label == DocItemLabel.FORMULA else TEXT_PROMPT
            lenient = item.label in MULTILINE
            reply, tokens, cut = _ask(base_url, cfg, image, prompt, lenient)
            lines, ended = first_reading(reply, lenient=lenient)
            cut = cut and not ended
            text = ("\n" if lenient else " ").join(lines)
            if item.label == DocItemLabel.LIST_ITEM:
                text = LIST_MARKER.sub("", text)  # Docling writes the marker itself
            item.text = item.orig = text
            found = bool(text)
            if not found:
                empty.append(item)
        stats.regions += 1
        stats.output_tokens += tokens
        stats.cut_regions += cut
        stats.empty_regions += not found
    if empty:
        doc.delete_items(node_items=empty)
    stats.seconds = round(time.perf_counter() - started, 3)
    return stats
