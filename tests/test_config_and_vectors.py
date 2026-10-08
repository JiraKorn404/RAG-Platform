import math

import pytest

from rag_lab.config import (
    ChunkConfig,
    EmbedConfig,
    ExperimentConfig,
    ParseConfig,
    SemanticSettings,
    load_experiment_file,
)
from rag_lab.embedding.vectors import truncate_and_normalise


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


def test_experiment_file_fills_defaults_and_embedding_family(tmp_path):
    path = tmp_path / "ingest.yaml"
    path.write_text("name: auto\nembed:\n  model: embeddinggemma-2:740m\nindex:\n  sparse: false\n")
    config = load_experiment_file(path)
    assert (config.name, config.index.sparse) == ("auto", False)
    assert config.chunk == ChunkConfig()  # a section left out keeps its defaults
    assert config.embed == EmbedConfig.for_model("embeddinggemma-2:740m")

    path.write_text("name: auto\nembed:\n  model: embeddinggemma-2:740m\n  tokenizer: other/tokenizer\n")
    assert load_experiment_file(path).embed.tokenizer == "other/tokenizer"  # what the file sets wins


def test_experiment_file_refuses_unknown_keys_and_unknown_models(tmp_path):
    path = tmp_path / "ingest.yaml"
    path.write_text("name: auto\nchunk:\n  stratgy: fixed\n")
    with pytest.raises(ValueError, match="chunk.stratgy"):
        load_experiment_file(path)
    path.write_text("name: auto\nembed:\n  model: qwen3-embeding:0.6b\n")
    with pytest.raises(ValueError, match="not of a known embedding family"):
        load_experiment_file(path)
    path.write_text("chunk:\n  strategy: fixed\n")
    with pytest.raises(ValueError, match="name"):
        load_experiment_file(path)
