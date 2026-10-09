"""Searching an experiment's collection: `dense` or `hybrid` (dense and BM25 fused with RRF). The
reranker is here too: `rerank_hits` scores what a search found against the question."""

import math
import time
from contextlib import contextmanager
from dataclasses import dataclass, replace

import httpx

from rag_lab.core.config import ExperimentConfig, RerankerConfig, SearchConfig
from rag_lab.core.embed import EmbedResult, OllamaEmbedder, query_vector
from rag_lab.core.events import Hit
from rag_lab.core.qdrant import QdrantStore
from rag_lab.core.store import MetricsStore

# --- the reranker ------------------------------------------------------------------------------------
# Reranking with a Qwen3-Reranker model served by Ollama.
#
# Ollama has no rerank endpoint, so each (query, chunk) pair is one `/api/generate` call with the model's
# own yes/no prompt, sent raw (no chat template) for one token. The score is the probability of "yes"
# against "no" in that token's log-probabilities. Only a build that gives a sharp yes/no works: the 4B
# Q4_K_M build of `dengcao/Qwen3-Reranker` does, the 0.6B builds on Ollama did not.

_SYSTEM = (
    "Judge whether the Document meets the requirements based on the Query and the Instruct provided. "
    'Note that the answer can only be "yes" or "no".'
)


@dataclass
class RerankResult:
    scores: list[float]  # one per text, in the order given; higher is more relevant
    wall_ms: float


def prompt(query: str, text: str, instruction: str) -> str:
    return (
        f"<|im_start|>system\n{_SYSTEM}<|im_end|>\n"
        f"<|im_start|>user\n<Instruct>: {instruction}\n<Query>: {query}\n<Document>: {text}<|im_end|>\n"
        "<|im_start|>assistant\n<think>\n\n</think>\n\n"
    )


def yes_probability(top_logprobs: list[dict]) -> float:
    """P(yes) / (P(yes) + P(no)) from a token's top log-probabilities. 'yes', 'Yes' and ' yes' all count
    as yes, and likewise for no; any other token is ignored. 0.0 when neither is among them."""
    mass = {"yes": 0.0, "no": 0.0}
    for entry in top_logprobs:
        word = entry["token"].strip().lower()
        if word in mass:
            mass[word] += math.exp(entry["logprob"])
    total = mass["yes"] + mass["no"]
    return mass["yes"] / total if total else 0.0


class OllamaReranker:
    def __init__(self, base_url: str, timeout: float = 300.0, retries: int = 3):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.retries = retries

    def score(self, query: str, texts: list[str], cfg: RerankerConfig) -> RerankResult:
        start = time.perf_counter()
        scores = [self._score_one(query, text, cfg) for text in texts]
        return RerankResult(scores, (time.perf_counter() - start) * 1000)

    def _score_one(self, query: str, text: str, cfg: RerankerConfig) -> float:
        body = {
            "model": cfg.model,
            "prompt": prompt(query, text, cfg.instruction),
            "raw": True,
            "stream": False,
            "logprobs": True,
            "top_logprobs": 20,
            "keep_alive": cfg.keep_alive,
            "options": {"num_predict": 1, "temperature": 0, "num_ctx": cfg.num_ctx},
        }
        for attempt in range(self.retries):
            try:
                resp = httpx.post(f"{self.base_url}/api/generate", json=body, timeout=self.timeout)
                if resp.status_code < 500:
                    resp.raise_for_status()
                    logprobs = resp.json().get("logprobs")
                    if not logprobs:
                        raise RuntimeError(
                            f"Ollama returned no log-probabilities for '{cfg.model}'. "
                            "Reranking needs an Ollama with `logprobs` support."
                        )
                    return yes_probability(logprobs[0]["top_logprobs"])
            except httpx.TransportError:
                pass
            if attempt < self.retries - 1:
                time.sleep(2**attempt)
        raise RuntimeError(f"Ollama rerank failed after {self.retries} attempts ({self.base_url})")


def rerank_hits(
    question: str, hits: list[Hit], reranker: OllamaReranker, cfg: RerankerConfig, scored: list[Hit] = ()
) -> tuple[list[Hit], float]:
    """Score `hits` against the question and return every hit scored so far, best first, with how long
    the scoring took. A returned hit has the reranker's score as its `similarity` and its place as its
    `rank`. `scored` is what an earlier search for the same question got back from here: a chunk among
    them keeps its score and is not sent to the reranker again."""
    known = {hit.chunk_id for hit in scored}
    new = [hit for hit in hits if hit.chunk_id not in known]
    result = reranker.score(question, [hit.text for hit in new], cfg) if new else RerankResult([], 0.0)
    every = [*scored, *(replace(hit, similarity=s, distance=1 - s) for hit, s in zip(new, result.scores))]
    every.sort(key=lambda hit: -hit.similarity)
    return [replace(hit, rank=rank) for rank, hit in enumerate(every, start=1)], result.wall_ms


# --- timing ------------------------------------------------------------------------------------------

@dataclass
class Timer:
    ms: float = 0.0


@contextmanager
def timed():
    """`with timed() as t: ...` then read `t.ms`."""
    timer = Timer()
    start = time.perf_counter()
    try:
        yield timer
    finally:
        timer.ms = (time.perf_counter() - start) * 1000


@dataclass
class SearchResult:
    query: str
    hits: list[Hit]
    embed_ms: float
    search_ms: float  # Qdrant, including the BM25 branch and the fusion of a hybrid search
    total_ms: float
    method: str = "dense"
    rerank_ms: float = 0.0  # set by a caller that reranked the hits (`rerank_hits`)


def load_experiment(metrics: MetricsStore, name: str) -> tuple[str, ExperimentConfig]:
    """(config_hash, config) recorded for an ingested experiment, so a query is embedded with the
    same model, dimension and instruction as the documents were."""
    found = metrics.get_experiment(name)
    if found is None:
        raise ValueError(f"No experiment named '{name}' in rag_metrics. Ingest a document first.")
    return found[0], ExperimentConfig.model_validate(found[1])


def search(
    query: str,
    config: ExperimentConfig,
    embedder: OllamaEmbedder,
    store: QdrantStore,
    top_k: int = 5,
    filters: dict[str, str] | None = None,
    embedded: EmbedResult | None = None,
    options: SearchConfig | None = None,
) -> SearchResult:
    """Embed the query (with the instruction prefix) and search the experiment's collection.
    `filters` match payload fields exactly; only `doc_id`, `modality` and `source_file` have indexes.
    `embedded` is an already embedded query (same model, dimension and instruction as the experiment),
    so several experiments that share an embedding setup can reuse one call; its time counts as embed_ms.
    `options` picks the search method (dense when not given): `hybrid` needs an experiment made with
    `index.sparse`, and each of its two branches fetches `options.candidates` hits (at least top k)."""
    options = options or SearchConfig()
    reused = embedded is not None
    with timed() as total:
        if embedded is None:
            embedded = embedder.embed([query], config.embed, kind="query")
        with timed() as qdrant_time:
            try:
                points = store.query(
                    config.collection,
                    embedded.vectors[0],
                    top_k,
                    filters,
                    config.index.hnsw_ef,
                    sparse=query_vector(query) if options.hybrid else None,
                    branch_limit=max(options.candidates, top_k),
                )
            except Exception as e:  # Qdrant names the missing vector in its error
                if options.hybrid and "bm25" in str(e):
                    raise ValueError(
                        f"Experiment '{config.name}' has no BM25 vector, so it cannot be searched with "
                        f"'{options.method}'. It was made without `index.sparse`."
                    ) from e
                raise

    hits = [
        Hit(
            rank=rank,
            similarity=p.score,
            distance=1 - p.score,
            text=p.payload["text"],
            source_file=p.payload.get("source_file"),
            page=p.payload.get("page"),
            modality=p.payload["modality"],
            headings=p.payload.get("headings") or [],
            doc_id=p.payload["doc_id"],
            chunk_id=p.payload["chunk_id"],
            image=f"{config.name}/parse/{p.payload['image']}" if p.payload.get("image") else None,
        )
        for rank, p in enumerate(points, start=1)
    ]
    return SearchResult(
        query=query,
        hits=hits,
        embed_ms=embedded.wall_ms,
        search_ms=qdrant_time.ms,
        total_ms=total.ms + (embedded.wall_ms if reused else 0),
        method=options.method,
    )
