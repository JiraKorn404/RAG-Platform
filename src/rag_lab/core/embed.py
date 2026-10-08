"""Turning text and pictures into vectors: the Ollama embedder, the cutting of a vector to a smaller
size, and the BM25 sparse vectors of the hybrid search. A question is embedded here too, so both the
ingestion and the chatbots use it."""

import base64
import math
import re
import time
import zlib
from collections import Counter
from dataclasses import dataclass
from typing import Literal

import httpx

from rag_lab.core.config import EmbedConfig, embed_family


@dataclass
class EmbedResult:
    vectors: list[list[float]]
    wall_ms: float  # includes network and retries
    ollama_ms: float  # Ollama's own total_duration, summed over batches
    load_ms: float  # model load time; large only on a cold start
    prompt_tokens: int
    batches: int


class OllamaEmbedder:
    def __init__(self, base_url: str, timeout: float = 120.0, retries: int = 3):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.retries = retries

    def embed(
        self,
        texts: list[str | None],
        cfg: EmbedConfig,
        kind: Literal["document", "query"] = "document",
        images: list[bytes | None] | None = None,
    ) -> EmbedResult:
        """A query and a document are written differently, as the model's family expects: Qwen3 gives a
        query an instruction and sends a document as it is, EmbeddingGemma 2 prefixes both.

        `images` (PNG or JPEG bytes, one entry per text, None where there is none) makes an input an
        image, alone when its text is None, or together with its text as one vector. An image gets no
        template. It goes inside `input` as an object: a top-level `images` field, as the chat API has,
        is accepted by Ollama and silently ignored."""
        images = images or [None] * len(texts)
        if any(images):
            family = embed_family(cfg.model)
            if family is None or not family.images:
                raise ValueError(f"The embedding model '{cfg.model}' does not take images")
        if kind == "query":
            written = [cfg.query_template.format(instruction=cfg.query_instruction, text=t) for t in texts]
        else:
            written = [None if t is None else cfg.document_template.format(text=t) for t in texts]
        inputs: list[str | dict] = []
        for text, image in zip(written, images, strict=True):
            if image is None:
                inputs.append(text)
            else:
                item = {"image": base64.b64encode(image).decode()}
                inputs.append(item if text is None else {"text": text, **item})

        start = time.perf_counter()
        vectors: list[list[float]] = []
        ollama_ns = load_ns = tokens = batches = 0
        for i in range(0, len(inputs), cfg.batch_size):
            data = self._post_embed(inputs[i : i + cfg.batch_size], cfg)
            vectors.extend(data["embeddings"])
            ollama_ns += data.get("total_duration", 0)
            load_ns += data.get("load_duration", 0)
            tokens += data.get("prompt_eval_count", 0)
            batches += 1

        return EmbedResult(
            vectors=truncate_and_normalise(vectors, cfg.dimension),
            wall_ms=(time.perf_counter() - start) * 1000,
            ollama_ms=ollama_ns / 1e6,
            load_ms=load_ns / 1e6,
            prompt_tokens=tokens,
            batches=batches,
        )

    def _post_embed(self, batch: list[str | dict], cfg: EmbedConfig) -> dict:
        body = {"model": cfg.model, "input": batch, "keep_alive": cfg.keep_alive}
        for attempt in range(self.retries):
            try:
                resp = httpx.post(f"{self.base_url}/api/embed", json=body, timeout=self.timeout)
                if resp.status_code < 500:
                    resp.raise_for_status()
                    return resp.json()
            except httpx.TransportError:
                pass
            if attempt < self.retries - 1:
                time.sleep(2**attempt)
        raise RuntimeError(f"Ollama embed failed after {self.retries} attempts ({self.base_url})")


# --- truncation --------------------------------------------------------------------------------------

def truncate_and_normalise(vectors: list[list[float]], dimension: int | None) -> list[list[float]]:
    """Matryoshka truncation: keep the first `dimension` values, then L2-normalise again."""
    if dimension is None:
        return vectors
    out = []
    for v in vectors:
        cut = v[:dimension]
        norm = math.sqrt(sum(x * x for x in cut)) or 1.0
        out.append([x / norm for x in cut])
    return out


# --- BM25 --------------------------------------------------------------------------------------------
# BM25 sparse vectors, for the hybrid search methods. Plain Python, no model.
#
# A term is a lower-cased `\w+` token, hashed to a stable index. A document's weight for a term is
# BM25's term-frequency part; the IDF part is applied by Qdrant (the `bm25` vector is created with the
# IDF modifier), and a query's weight for a term is 1. Meant for languages that put spaces between
# words: other languages need a word segmenter first.

K1 = 1.2
B = 0.75

SparseVector = tuple[list[int], list[float]]  # indices, values


def tokens(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower())


def _index(term: str) -> int:
    return zlib.crc32(term.encode("utf-8"))  # stable between runs, unlike hash(); fits Qdrant's u32


def document_vectors(texts: list[str]) -> list[SparseVector]:
    """One vector per text. Length normalisation uses the mean length of these texts."""
    tokenised = [tokens(t) for t in texts]
    avg_len = (sum(len(t) for t in tokenised) / len(tokenised)) if tokenised else 0.0
    vectors = []
    for terms in tokenised:
        weights: dict[int, float] = {}
        norm = K1 * (1 - B + B * len(terms) / avg_len) if avg_len else K1
        for term, tf in Counter(terms).items():
            index = _index(term)
            weights[index] = weights.get(index, 0.0) + tf * (K1 + 1) / (tf + norm)
        vectors.append((list(weights), list(weights.values())))
    return vectors


def query_vector(text: str) -> SparseVector:
    indices = sorted({_index(t) for t in tokens(text)})
    return indices, [1.0] * len(indices)
