"""The clients of the services the system talks to, built from config/connections.yaml. Everything that
needs Ollama, Qdrant or the metrics database gets it here: the Dagster assets, the pages and the
command lines."""

from rag_lab.embedding.ollama import OllamaEmbedder
from rag_lab.metrics.store import MetricsStore
from rag_lab.reranking import OllamaReranker
from rag_lab.settings import load
from rag_lab.storage.qdrant import QdrantStore


def ollama_url() -> str:
    return load().connections.ollama.url


def embedder() -> OllamaEmbedder:
    return OllamaEmbedder(ollama_url())


def reranker() -> OllamaReranker:
    return OllamaReranker(ollama_url())


def qdrant() -> QdrantStore:
    return QdrantStore(load().connections.qdrant.url)


def metrics() -> MetricsStore:
    return MetricsStore(load().connections.app_database.url)
