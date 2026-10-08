"""The card that shows one retrieved chunk."""

import base64
import html
import io
from functools import lru_cache

import data
from PIL import Image

PICTURE_WIDTH = 560  # pixels; a picture is drawn no wider than this


@lru_cache(maxsize=256)
def picture_html(image: str) -> str:
    """An <img> of a picture chunk (`Hit.image`, fetched from the API), scaled down and inlined, or
    nothing when the file is gone (its experiment was deleted, for example)."""
    content = data.picture(image)
    if content is None:
        return ""
    try:
        picture = Image.open(io.BytesIO(content)).convert("RGB")
        picture.thumbnail((PICTURE_WIDTH, PICTURE_WIDTH))
        buffer = io.BytesIO()
        picture.save(buffer, format="JPEG", quality=80)  # a PNG of this size is ten times the bytes
    except OSError:
        return ""
    encoded = base64.b64encode(buffer.getvalue()).decode()
    return f'<img class="picture" src="data:image/jpeg;base64,{encoded}" alt="picture">'


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
        + (picture_html(hit.image) if getattr(hit, "image", None) else "")
        + f'<div class="snippet">{snippet}</div>'
        f"<details><summary>Full text</summary><pre>{html.escape(hit.text)}</pre></details></div>"
    )
