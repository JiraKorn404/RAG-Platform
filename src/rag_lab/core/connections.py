"""The clients of the services the system talks to, built from config/connections.yaml. Everything that
needs Ollama, Qdrant or our tables gets it here: the Dagster assets, the API and the command lines."""

from rag_lab.core.embed import OllamaEmbedder
from rag_lab.core.qdrant import QdrantStore
from rag_lab.core.settings import load
from rag_lab.core.store import MetricsStore


def ollama_url() -> str:
    return load().connections.ollama.url


def embedder() -> OllamaEmbedder:
    return OllamaEmbedder(ollama_url())


def qdrant() -> QdrantStore:
    return QdrantStore(load().connections.qdrant.url)


def metrics() -> MetricsStore:
    database = load().connections.app_database
    return MetricsStore(database.url, database.schema_name)
