"""Reranking with a Qwen3-Reranker model served by Ollama.

Ollama has no rerank endpoint, so each (query, chunk) pair is one `/api/generate` call with the model's
own yes/no prompt, sent raw (no chat template) for one token. The score is the probability of "yes"
against "no" in that token's log-probabilities. Only a build that gives a sharp yes/no works: the 4B
Q4_K_M build of `dengcao/Qwen3-Reranker` does, the 0.6B builds on Ollama did not."""

import math
import time
from dataclasses import dataclass

import httpx

from rag_lab.config import SearchConfig

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

    def score(self, query: str, texts: list[str], cfg: SearchConfig) -> RerankResult:
        start = time.perf_counter()
        scores = [self._score_one(query, text, cfg) for text in texts]
        return RerankResult(scores, (time.perf_counter() - start) * 1000)

    def _score_one(self, query: str, text: str, cfg: SearchConfig) -> float:
        body = {
            "model": cfg.reranker.model,
            "prompt": prompt(query, text, cfg.reranker.instruction),
            "raw": True,
            "stream": False,
            "logprobs": True,
            "top_logprobs": 20,
            "keep_alive": cfg.reranker.keep_alive,
            "options": {"num_predict": 1, "temperature": 0, "num_ctx": cfg.reranker.num_ctx},
        }
        for attempt in range(self.retries):
            try:
                resp = httpx.post(f"{self.base_url}/api/generate", json=body, timeout=self.timeout)
                if resp.status_code < 500:
                    resp.raise_for_status()
                    logprobs = resp.json().get("logprobs")
                    if not logprobs:
                        raise RuntimeError(
                            f"Ollama returned no log-probabilities for '{cfg.reranker.model}'. "
                            "Reranking needs an Ollama with `logprobs` support."
                        )
                    return yes_probability(logprobs[0]["top_logprobs"])
            except httpx.TransportError:
                pass
            if attempt < self.retries - 1:
                time.sleep(2**attempt)
        raise RuntimeError(f"Ollama rerank failed after {self.retries} attempts ({self.base_url})")
