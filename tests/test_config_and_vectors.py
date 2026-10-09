import math

import pytest

from rag_lab.core.config import (
    ChunkConfig,
    EmbedConfig,
    ExperimentConfig,
    ParseConfig,
    SemanticSettings,
)
from rag_lab.core.embed import truncate_and_normalise
from rag_lab.core.settings import ConfigError, load


def test_config_hash_changes_with_the_name():
    assert ExperimentConfig(name="a").config_hash() != ExperimentConfig(name="b").config_hash()
    assert ExperimentConfig(name="a").config_hash() == ExperimentConfig(name="a").config_hash()


def test_config_hash_changes_with_a_setting():
    a = ExperimentConfig(name="a")
    assert a.config_hash() != ExperimentConfig(name="a", parse=ParseConfig(table_mode="fast")).config_hash()
    assert a.config_hash() != ExperimentConfig(name="a", embed=EmbedConfig(dimension=512)).config_hash()


def test_config_hash_ignores_settings_of_unused_strategies():
    hybrid = ExperimentConfig(name="a", chunk=ChunkConfig(strategy="hybrid"))
    tweaked = ExperimentConfig(
        name="a", chunk=ChunkConfig(strategy="hybrid", semantic=SemanticSettings(buffer_size=3))
    )
    assert hybrid.config_hash() == tweaked.config_hash()
    semantic = ExperimentConfig(
        name="a", chunk=ChunkConfig(strategy="semantic", semantic=SemanticSettings(buffer_size=3))
    )
    assert semantic.config_hash() != ExperimentConfig(
        name="a", chunk=ChunkConfig(strategy="semantic")
    ).config_hash()


def test_pictures_as_images_need_a_model_that_takes_images():
    qwen = EmbedConfig.for_model("qwen3-embedding:0.6b")
    with pytest.raises(ValueError, match="takes images"):
        ExperimentConfig(name="a", parse=ParseConfig(pictures=True), embed=qwen)
    caption = EmbedConfig.for_model("qwen3-embedding:0.6b", picture_input="caption")
    ExperimentConfig(name="a", parse=ParseConfig(pictures=True), embed=caption)  # text only: fine


def test_truncate_keeps_dimension_and_unit_length():
    out = truncate_and_normalise([[3.0, 4.0, 12.0, 0.5]], 2)[0]
    assert len(out) == 2
    assert math.isclose(math.sqrt(sum(x * x for x in out)), 1.0)
    assert out == [0.6, 0.8]


# --- the settings files (rag_lab/settings.py) ---

CONNECTIONS = """
ollama: {url: "http://ollama:11434"}
qdrant: {url: "http://qdrant:6333"}
app_database: {url: "postgresql://admin:${TEST_DB_PASSWORD}@postgres:5432/rag_metrics"}
tables_database:
  loader_url: postgresql://loader:x@postgres:5432/rag_data
  reader_url: postgresql://reader:x@postgres:5432/rag_data
"""


def write_config(folder, pipeline="name: auto", llm="", connections=CONNECTIONS):
    for name, text in (("pipeline.yaml", pipeline), ("llm.yaml", llm), ("connections.yaml", connections)):
        (folder / name).write_text(text, encoding="utf-8")
    return folder


def test_settings_replace_environment_variables_and_refuse_a_missing_one(tmp_path, monkeypatch):
    write_config(tmp_path)
    monkeypatch.setenv("TEST_DB_PASSWORD", "secret1")
    assert load(tmp_path).connections.app_database.url == "postgresql://admin:secret1@postgres:5432/rag_metrics"
    monkeypatch.delenv("TEST_DB_PASSWORD")
    with pytest.raises(ConfigError, match="connections.yaml.*TEST_DB_PASSWORD"):
        load(tmp_path)


def test_settings_name_the_file_and_the_key_of_a_mistake(tmp_path, monkeypatch):
    monkeypatch.setenv("TEST_DB_PASSWORD", "secret1")
    for pipeline, llm, message in (
        ("name: auto\nchunk: {stratgy: fixed}", "", "pipeline.yaml: chunk.stratgy: unknown key"),
        ("chunk: {strategy: fixed}", "", "pipeline.yaml: name"),
        ("name: auto", "chat: {modle: x}", "llm.yaml: chat.modle: unknown key"),
        ("name: auto", "documents_chat: {top_kk: 3}", "llm.yaml: documents_chat.top_kk: unknown key"),
        ("name: auto", 'embedding: {model: "qwen3-embeding:0.6b"}', "not of a known embedding family"),
        ("name: auto", "documents_chat: {reranker: {model: x}}", "set it in the `reranker` section"),
    ):
        with pytest.raises(ConfigError, match=message):
            load(write_config(tmp_path, pipeline, llm))


def test_settings_put_the_models_of_llm_yaml_where_they_are_used(tmp_path, monkeypatch):
    monkeypatch.setenv("TEST_DB_PASSWORD", "secret1")
    llm = """
embedding: {model: "embeddinggemma-2:740m"}
ocr: {model: other-ocr}
reranker: {model: other-reranker}
chat: {model: shared-model, num_ctx: 4096}
database_chat: {num_ctx: 32768}
"""
    settings = load(write_config(tmp_path, "name: auto\nparse: {ocr: true}", llm))
    experiment = settings.experiment
    assert (experiment.name, experiment.parse.ocr, experiment.parse.ocr_model) == ("auto", True, "other-ocr")
    assert experiment.chunk == ChunkConfig()  # a section left out keeps its defaults
    assert experiment.embed == EmbedConfig.for_model("embeddinggemma-2:740m")  # the family's templates
    assert settings.documents_chat.reranker.model == "other-reranker"
    assert settings.database_chat.embed.model == "embeddinggemma-2:740m"
    # `chat` is shared, and a chat's own section wins
    assert (settings.documents_chat.model, settings.documents_chat.num_ctx) == ("shared-model", 4096)
    assert (settings.database_chat.model, settings.database_chat.num_ctx) == ("shared-model", 32768)
    assert settings.database_chat.temperature == 0.0  # not set anywhere: the flow's own default
