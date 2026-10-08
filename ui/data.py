"""Small helpers shared by the pages: the label of an embedding model, and whether Ollama has the
models config/llm.yaml names."""

import httpx
import streamlit as st

from rag_lab import clients
from rag_lab.config import ChatModelConfig, embed_model_label

model_label = embed_model_label  # 'qwen3-embedding:4b' -> '4b'; another family keeps its whole name


@st.cache_data(ttl=60)
def ollama_models() -> dict[str, list[str]] | None:
    """The models Ollama has, each with its capabilities, or None when Ollama cannot be reached."""
    try:
        reply = httpx.get(f"{clients.ollama_url().rstrip('/')}/api/tags", timeout=5)
        return {m["name"]: m.get("capabilities", []) for m in reply.json()["models"]}
    except Exception:  # noqa: BLE001  (not reachable, or not Ollama: the caller says so)
        return None


def check_models(cfg: ChatModelConfig, embedding: str | None = None) -> tuple[list[str], dict, list[str]]:
    """Whether Ollama can run a chat with these settings (`documents_chat` or `database_chat` of
    llm.yaml; `embedding` is the model the collection was made with). Returns what stops it, each with
    the key to change; the settings to switch off because the chat model cannot do them; and a note
    for each of those."""
    models = ollama_models()
    if models is None:
        return [f"Ollama cannot be reached at {clients.ollama_url()} (`ollama.url` in config/connections.yaml)."], {}, []

    def installed(name: str) -> bool:
        return name in models or f"{name}:latest" in models

    wanted = [(cfg.model, "the chat model (`chat.model` in config/llm.yaml)")]
    if hasattr(cfg, "reranker"):
        wanted.append((cfg.reranker.model, "the reranker (`reranker.model` in config/llm.yaml)"))
    if embedding:
        wanted.append((embedding, "the embedding model this collection was made with"))
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
    return missing, off, notes
