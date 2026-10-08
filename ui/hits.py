"""The card that shows one retrieved chunk."""

import base64
import html
import io
from functools import lru_cache
from pathlib import Path

from PIL import Image

from rag_lab.paths import DATA_DIR

PICTURE_WIDTH = 560  # pixels; a picture is drawn no wider than this


@lru_cache(maxsize=256)
def picture_html(path: Path) -> str:
    """An <img> of a picture chunk's file, scaled down and inlined, or nothing when the file is gone
    (its experiment was deleted, for example)."""
    try:
        image = Image.open(path).convert("RGB")
        image.thumbnail((PICTURE_WIDTH, PICTURE_WIDTH))
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=80)  # a PNG of this size is ten times the bytes
    except OSError:
        return ""
    data = base64.b64encode(buffer.getvalue()).decode()
    return f'<img class="picture" src="data:image/jpeg;base64,{data}" alt="picture">'


def hit_picture(hit) -> str:
    """The picture of a picture hit. `hit.image` is relative to data/artifacts and must stay inside it."""
    if not getattr(hit, "image", None):
        return ""
    root = (DATA_DIR / "artifacts").resolve()
    path = (root / hit.image).resolve()
    return picture_html(path) if root in path.parents and path.suffix == ".png" else ""


def hit_card(hit, color: str, cited: bool = False, seen: bool = False) -> str:
    """`seen`: the chatbot's model was given this picture as an image, not only its caption."""
    body = " ".join(hit.text.split())
    snippet = html.escape(body[:330] + ("…" if len(body) > 330 else ""))
    where = f"{hit.source_file}, p. {hit.page}" if hit.page and hit.source_file else (
        f"p. {hit.page}" if hit.page else (hit.source_file or "page ?")
    )
    return (
        '<div class="hit"><div class="hit-top">'
        f'<span class="rank">{hit.rank}</span><span class="sim">{hit.similarity:.3f}</span>'
        f'<div class="bar"><span style="width:{max(0.0, min(1.0, hit.similarity)) * 100:.0f}%;background:{color}"></span></div>'
        "</div>"
        f'<div class="badges"><span class="badge {hit.modality}">{hit.modality}</span>'
        f'<span class="badge page">{html.escape(where)}</span>'
        + (f'<span class="badge agree">cited [{hit.rank}]</span>' if cited else "")
        + ('<span class="badge picture">image given to the model</span>' if seen else "")
        + "</div>"
        + (f'<div class="heading">{html.escape(" › ".join(hit.headings))}</div>' if hit.headings else "")
        + hit_picture(hit)
        + f'<div class="snippet">{snippet}</div>'
        f"<details><summary>Full text</summary><pre>{html.escape(hit.text)}</pre></details></div>"
    )
