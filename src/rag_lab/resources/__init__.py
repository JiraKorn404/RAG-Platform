"""Thin Dagster wrappers around the plain clients. Search and other non-Dagster code use the clients directly."""

from dagster import ConfigurableResource

from rag_lab.config import (
    ChunkConfig,
    EmbedConfig,
    ExperimentConfig,
    IndexConfig,
    ParseConfig,
)
from rag_lab.embedding.ollama import OllamaEmbedder
from rag_lab.metrics.store import MetricsStore
from rag_lab.storage.qdrant import QdrantStore


class ExperimentResource(ConfigurableResource):
    """The experiment's settings. Set once per run (launchpad: `resources.experiment.config`) and
    shared by every asset, so a run never needs the same settings repeated per step."""

    name: str
    parse: ParseConfig = ParseConfig()
    chunk: ChunkConfig = ChunkConfig()
    embed: EmbedConfig = EmbedConfig()
    index: IndexConfig = IndexConfig()

    def config(self) -> ExperimentConfig:
        return ExperimentConfig(
            name=self.name,
            parse=self.parse,
            chunk=self.chunk,
            embed=self.embed,
            index=self.index,
        )


class OllamaResource(ConfigurableResource):
    base_url: str
    timeout: float = 120.0

    def embedder(self) -> OllamaEmbedder:
        return OllamaEmbedder(self.base_url, timeout=self.timeout)


class QdrantResource(ConfigurableResource):
    url: str
    grpc_port: int = 6334

    def store(self) -> QdrantStore:
        return QdrantStore(self.url, grpc_port=self.grpc_port)


class MetricsStoreResource(ConfigurableResource):
    database_url: str

    def store(self) -> MetricsStore:
        return MetricsStore(self.database_url)
