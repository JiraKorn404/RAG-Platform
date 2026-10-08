"""Reads the configuration: three YAML files in config/ (or the folder RAG_CONFIG_DIR names).

    connections.yaml   where Ollama, Qdrant and the PostgreSQL databases are
    pipeline.yaml      what a PDF or a CSV goes through: the collection, parsing, chunking, indexing
    llm.yaml           every model and how it is called: embedding, OCR, reranker, the chat models

A key left out keeps the default written in config.py. An unknown key is an error that names the file
and the key. `${NAME}` in a value is replaced with the environment variable of that name (passwords
stay in .env), and a variable that is not set is an error.

`load()` reads the files every time it is called, so Dagster picks up an edit at the next sensor tick or
run. The UI keeps its clients for as long as it runs, so a change of connections.yaml needs
`docker compose restart ui`."""

import os
import re
from dataclasses import dataclass
from pathlib import Path

import yaml
from pydantic import BaseModel, ValidationError

from rag_lab.config import (
    EMBED_FAMILIES,
    EXAMPLES_QUERY_INSTRUCTION,
    AgentConfig,
    ChatModelConfig,
    Connections,
    ExperimentConfig,
    ImportConfig,
    LlmFile,
    ParseConfig,
    PipelineFile,
    SqlAgentConfig,
    embed_family,
)

_VARIABLE = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")


class ConfigError(ValueError):
    """A settings file is missing or wrong. The message names the file and the key."""


@dataclass(frozen=True)
class Settings:
    connections: Connections
    experiment: ExperimentConfig  # pipeline.yaml, with the embedding and OCR models of llm.yaml
    tables: ImportConfig  # pipeline.yaml: tables
    documents_chat: AgentConfig  # llm.yaml: chat and documents_chat, with the reranker
    database_chat: SqlAgentConfig  # llm.yaml: chat and database_chat, with the embedding model


def config_dir() -> Path:
    return Path(os.environ.get("RAG_CONFIG_DIR", "config"))


def load(folder: Path | None = None) -> Settings:
    folder = Path(folder) if folder else config_dir()
    connections = _validate(Connections, _read(folder / "connections.yaml"), folder / "connections.yaml")
    pipeline = _validate(PipelineFile, _read(folder / "pipeline.yaml"), folder / "pipeline.yaml")
    llm = _validate(LlmFile, _llm(_read(folder / "llm.yaml"), folder / "llm.yaml"), folder / "llm.yaml")
    try:
        experiment = ExperimentConfig(
            name=pipeline.name,
            parse=ParseConfig(
                **pipeline.parse.model_dump(), ocr_model=llm.ocr.model, ocr_keep_alive=llm.ocr.keep_alive
            ),
            chunk=pipeline.chunk,
            embed=llm.embedding,
            index=pipeline.index,
        )
    except ValidationError as e:  # a rule that spans the two files (pictures need a model that takes images)
        raise ConfigError(f"{folder / 'pipeline.yaml'} and {folder / 'llm.yaml'}: {problems(e)}") from e
    examples_embedding = llm.embedding.model_copy(update={"query_instruction": EXAMPLES_QUERY_INSTRUCTION})
    return Settings(
        connections=connections,
        experiment=experiment,
        tables=pipeline.tables,
        documents_chat=llm.documents_chat.model_copy(update={"reranker": llm.reranker}),
        database_chat=llm.database_chat.model_copy(update={"embed": examples_embedding}),
    )


def _read(path: Path) -> dict:
    """A YAML file as a dict, with `${NAME}` replaced in every value."""
    if not path.exists():
        raise ConfigError(f"{path}: the file does not exist")
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError as e:
        raise ConfigError(f"{path}: not valid YAML: {e}") from e
    if not isinstance(data, dict):
        raise ConfigError(f"{path}: expected `key: value` settings, found {type(data).__name__}")
    return _expand(data, path)


def _expand(value, path: Path):
    if isinstance(value, dict):
        return {key: _expand(item, path) for key, item in value.items()}
    if isinstance(value, list):
        return [_expand(item, path) for item in value]
    if not isinstance(value, str):
        return value

    def variable(match: re.Match) -> str:
        name = match.group(1)
        if name not in os.environ:
            raise ConfigError(f"{path}: the environment variable {name} is not set (it belongs in .env)")
        return os.environ[name]

    return _VARIABLE.sub(variable, value)


def _llm(data: dict, path: Path) -> dict:
    """llm.yaml as LlmFile reads it. `chat` is what the two chats share, so it is put under each chat's
    own section, whose keys win. An embedding model brings its family's tokenizer and prompt templates
    unless the file sets them."""
    data = dict(data)
    chat = data.pop("chat", None) or {}
    for section, source, key in (("documents_chat", "reranker", "reranker"), ("database_chat", "embedding", "embed")):
        own = data.get(section) or {}
        if not isinstance(chat, dict) or not isinstance(own, dict):
            raise ConfigError(f"{path}: `chat` and `{section}` must be `key: value` settings")
        if key in own:
            raise ConfigError(f"{path}: {section}.{key}: set it in the `{source}` section instead")
        data[section] = {**chat, **own}
    unknown = sorted(set(chat) - set(ChatModelConfig.model_fields))
    if unknown:
        raise ConfigError(f"{path}: chat.{unknown[0]}: unknown key")

    embedding = data.get("embedding")
    if isinstance(embedding, dict) and "model" in embedding:
        family = embed_family(str(embedding["model"]))
        own = ("tokenizer", "query_template", "document_template")
        if family is not None:
            data["embedding"] = {**{key: getattr(family, key) for key in own}, **embedding}
        elif not all(key in embedding for key in own):
            known = ", ".join(f.prefix for f in EMBED_FAMILIES)
            raise ConfigError(
                f"{path}: embedding.model: '{embedding['model']}' is not of a known embedding family ({known}). "
                f"Check the name, or set embedding.{', embedding.'.join(own)} for it."
            )
    return data


def _validate[T: BaseModel](model: type[T], data: dict, path: Path) -> T:
    try:
        return model.model_validate(data)
    except ValidationError as e:
        raise ConfigError(f"{path}: {problems(e)}") from e


def problems(error: ValidationError) -> str:
    """Each problem as `key.path: what is wrong`."""
    found = []
    for err in error.errors():
        where = ".".join(str(part) for part in err["loc"])
        what = "unknown key" if err["type"] == "extra_forbidden" else err["msg"]
        found.append(f"{where}: {what}" if where else what)
    return "; ".join(found)
