"""What a front end can do, as plain functions: the API (api.py) and the command lines call these, and the
Streamlit UI and the platform's backend reach them through the API. So every front end answers a
question, lists a chat and deletes a collection in the same way.

A function returns plain data (dicts, lists, events) and raises `Refused` with a reason for the person
who asked. The clients are built once and kept for as long as the process runs, so an edit to
config/connections.yaml needs a restart; config/llm.yaml is read at every question."""

from collections.abc import Iterator
from functools import lru_cache
from pathlib import Path

import httpx

from rag_lab.core import connections
from rag_lab.core.config import ChatModelConfig, ExperimentConfig
from rag_lab.core.events import Event, Query, SqlRan, SqlWritten, from_dict
from rag_lab.core.settings import DATA_DIR, load
from rag_lab.serve import examples as good_answers
from rag_lab.serve import library as stored
from rag_lab.serve.chat_database import SqlFlow, remembered
from rag_lab.serve.chat_database import build_graph as build_database_graph
from rag_lab.serve.chat_documents import DocumentsFlow
from rag_lab.serve.chat_documents import build_graph as build_documents_graph
from rag_lab.serve.run import run
from rag_lab.serve.search import OllamaReranker

KINDS = ("documents", "database")  # what a chat searches: a collection, or a schema


class Refused(Exception):
    """Something that cannot be done as asked, with the reason."""


@lru_cache(maxsize=1)
def _clients() -> tuple:
    """The embedder, Qdrant, our tables and the reranker."""
    return (
        connections.embedder(),
        connections.qdrant(),
        connections.metrics(),
        OllamaReranker(connections.ollama_url()),
    )


def health() -> dict[str, bool]:
    """Whether each service answers."""
    _, qdrant, metrics, _ = _clients()

    def answers(call) -> bool:
        try:
            call()
        except Exception:  # noqa: BLE001  (any failure means it cannot be used now)
            return False
        return True

    return {
        "ollama": _ollama_models() is not None,
        "qdrant": answers(qdrant.client.get_collections),
        "database": answers(metrics.list_db_schemas),
    }


# --- what can be searched ----------------------------------------------------------------------------


def targets() -> dict[str, list[dict]]:
    """What a chat can search: the experiments that have a Qdrant collection with a BM25 vector (hybrid
    search needs it), each with its number of points, and the schemas."""
    _, qdrant, metrics, _ = _clients()
    existing = {c.name for c in qdrant.client.get_collections().collections}
    collections = [
        {**row, "points": qdrant.count(row["name"])}
        for row in metrics.list_experiments()
        if row["name"] in existing and row["config"].get("index", {}).get("sparse")
    ]
    return {"collections": collections, "schemas": metrics.list_db_schemas()}


def _ollama_models() -> dict[str, list[str]] | None:
    """The models Ollama has, each with its capabilities, or None when Ollama cannot be reached."""
    try:
        reply = httpx.get(f"{connections.ollama_url().rstrip('/')}/api/tags", timeout=5)
        return {m["name"]: m.get("capabilities", []) for m in reply.json()["models"]}
    except Exception:  # noqa: BLE001  (not reachable, or not Ollama: the caller says so)
        return None


def _prepare(kind: str, target: str) -> tuple[ChatModelConfig, ExperimentConfig | None, list[str], list[str]]:
    """What a chat of this kind runs with now: the settings of config/llm.yaml with what the chat model
    cannot do switched off, the experiment (for documents), what stops the chat (each with the key to
    change) and a note for each setting that was switched off."""
    _, _, metrics, _ = _clients()
    settings = load()
    experiment = None
    if kind == "documents":
        found = metrics.get_experiment(target)
        if found is None:
            raise Refused(f"There is no collection named '{target}'.")
        experiment = ExperimentConfig.model_validate(found[1])
        cfg: ChatModelConfig = settings.documents_chat
    elif kind == "database":
        if not metrics.get_db_schema(target):
            raise Refused(f"There is no schema '{target}'.")
        cfg = settings.database_chat
    else:
        raise Refused(f"A chat searches one of: {', '.join(KINDS)}.")

    models = _ollama_models()
    if models is None:
        problem = f"Ollama cannot be reached at {connections.ollama_url()} (`ollama.url` in config/connections.yaml)."
        return cfg, experiment, [problem], []

    def installed(name: str) -> bool:
        return name in models or f"{name}:latest" in models

    wanted = [(cfg.model, "the chat model (`chat.model` in config/llm.yaml)")]
    if kind == "documents":
        wanted.append((cfg.reranker.model, "the reranker (`reranker.model` in config/llm.yaml)"))
        wanted.append((experiment.embed.model, "the embedding model this collection was made with"))
    missing = [
        f"Ollama has no model `{name}`, which is {what}. Pull it with `ollama pull {name}`, or name one it has."
        for name, what in wanted
        if not installed(name)
    ]

    can = models.get(cfg.model) or models.get(f"{cfg.model}:latest") or []
    off, notes = {}, []
    if installed(cfg.model):
        if cfg.think and "thinking" not in can:
            off["think"] = False
            notes.append(f"`{cfg.model}` cannot think, so this chat runs without thinking (`chat.think` in config/llm.yaml).")
        if getattr(cfg, "show_pictures", False) and "vision" not in can:
            off["show_pictures"] = False
            notes.append(f"`{cfg.model}` cannot see images, so pictures are given to it as their captions (`documents_chat.show_pictures`).")
    return cfg.model_copy(update=off), experiment, missing, notes


def check(kind: str, target: str) -> dict:
    """Whether a chat can be asked now. `missing` is what stops it, `notes` what was switched off because
    the chat model cannot do it, `settings` what a question would run with."""
    cfg, _, missing, notes = _prepare(kind, target)
    return {"missing": missing, "notes": notes, "settings": cfg.model_dump(mode="json")}


# --- chats -------------------------------------------------------------------------------------------


def _target(session: dict) -> str | None:
    """What a chat searches: the experiment's name or the schema. None when the experiment is gone."""
    if session["kind"] == "database":
        return session["schema_name"]
    _, _, metrics, _ = _clients()
    return next((e["name"] for e in metrics.list_experiments() if e["config_hash"] == session["config_hash"]), None)


def _history(turns: list[dict]) -> list[tuple[str, str]]:
    """The answered turns as the agent's history; a failed turn has no answer to remember. A database
    answer is remembered with the SQL that produced it, so a follow-up can be understood."""
    history = []
    for turn in turns:
        if not turn["error"]:
            sql = next((e["sql"] for e in reversed(turn["events"]) if e["type"] == "SqlChecked" and e["ok"]), "")
            history += [("User", turn["question"]), ("Assistant", remembered(turn["answer"], sql))]
    return history


def chats() -> list[dict]:
    """The chats of both kinds, most recently used first."""
    return _clients()[2].list_chat_sessions()


def chat(chat_id: str) -> dict | None:
    """A chat with its turns, oldest first (each with the events it is shown from), or None."""
    metrics = _clients()[2]
    session = metrics.get_chat_session(chat_id)
    if session is None:
        return None
    return {
        "id": chat_id,
        "kind": session["kind"],
        "target": _target(session),
        "title": session["title"],
        "turns": metrics.get_chat_turns(chat_id),
    }


def delete_chat(chat_id: str) -> None:
    _clients()[2].delete_chat_session(chat_id)


def ask(chat_id: str, kind: str, target: str, question: str) -> Iterator[Event]:
    """Answer one question in a chat: the events as they happen, `Done` last. The turn is saved (a failed
    one too, before its error is raised from the iterator). The chat is made with its first turn and
    keeps what it searches; its earlier turns are the history. Whatever stops the question before it
    starts is raised here as `Refused`, not from the iterator."""
    embedder, qdrant, metrics, reranker = _clients()
    if not question.strip():
        raise Refused("There is no question.")
    session = metrics.get_chat_session(chat_id)
    if session and (session["kind"], _target(session)) != (kind, target):
        raise Refused("A chat keeps what it searches. Start a new chat to search somewhere else.")
    cfg, experiment, missing, _ = _prepare(kind, target)
    if missing:
        raise Refused(" ".join(missing))
    base_url = connections.ollama_url()
    if kind == "documents":
        graph = build_documents_graph(experiment, embedder, qdrant, reranker, base_url, cfg)
        flow = DocumentsFlow(experiment, cfg)
    else:
        graph = build_database_graph(metrics, target, base_url, cfg, embedder, qdrant)
        flow = SqlFlow(target, cfg)
    history = _history(metrics.get_chat_turns(chat_id))
    return run(graph, flow, question, history, metrics=metrics, session_id=chat_id)


# --- good answers ------------------------------------------------------------------------------------


def _example_of(turn: dict) -> tuple[str, str] | None:
    """What a good answer saves for a turn: its standalone question and the SQL the model wrote for the
    query that ran. None when the turn did not produce an answer from a result."""
    if turn["error"] or turn["abstained"]:
        return None
    standalone, written, ran = turn["question"], "", ""
    for event in map(from_dict, turn["events"]):
        if isinstance(event, Query):
            standalone = event.text
        elif isinstance(event, SqlWritten):
            written = event.sql
        elif isinstance(event, SqlRan):
            ran = written
    return (standalone, ran) if ran else None


def examples(schema: str) -> list[dict]:
    """The good answers saved for a schema, newest first."""
    return _clients()[2].list_sql_examples(schema)


def mark_good(turn_id: int) -> int:
    """Keep a database turn's question and SQL as an example for similar questions. Returns its id."""
    embedder, qdrant, metrics, _ = _clients()
    found = metrics.get_chat_turns(turn_id=turn_id)
    pair = _example_of(found[0]) if found and found[0]["kind"] == "database" else None
    if pair is None:
        raise Refused("Only a database answer that was written from a result can be kept as a good answer.")
    turn = found[0]
    schema = metrics.get_chat_session(turn["session_id"])["schema_name"]
    return good_answers.save(
        metrics, embedder, qdrant, load().database_chat, schema, turn["question"], pair[0], pair[1], turn_id
    )


def unmark_good(example_id: int) -> None:
    _, qdrant, metrics, _ = _clients()
    found = next((e for e in metrics.list_sql_examples() if e["id"] == example_id), None)
    if found is None:
        raise Refused(f"There is no example #{example_id}.")
    good_answers.remove(metrics, qdrant, found["schema_name"], example_id)


# --- what is stored ----------------------------------------------------------------------------------


def library() -> list[dict]:
    """Every experiment with its documents, then the collections that have no experiment row."""
    _, qdrant, metrics, _ = _clients()
    return stored.list_library(metrics, qdrant)


def delete_document(name: str, doc_id: str) -> dict:
    _, qdrant, metrics, _ = _clients()
    try:
        return stored.delete_document(name, doc_id, metrics, qdrant)
    except ValueError as e:
        raise Refused(str(e)) from e


def delete_experiment(name: str) -> dict:
    _, qdrant, metrics, _ = _clients()
    return stored.delete_experiment(name, metrics, qdrant)


def picture(relative: str) -> Path | None:
    """The file of a picture chunk (`Hit.image`, relative to data/artifacts), or None when it is gone
    or is not a picture inside that folder."""
    root = (DATA_DIR / "artifacts").resolve()
    path = (root / relative).resolve()
    return path if root in path.parents and path.suffix == ".png" and path.is_file() else None
