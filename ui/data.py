"""Small helpers shared by the pages: the label of an embedding model, and which models Ollama has."""

import os

import httpx
import streamlit as st

from rag_lab.config import embed_model_label

model_label = embed_model_label  # 'qwen3-embedding:4b' -> '4b'; another family keeps its whole name


def _ollama_models() -> list[dict]:
    try:
        reply = httpx.get(f"{os.environ['OLLAMA_BASE_URL'].rstrip('/')}/api/tags", timeout=5)
        return reply.json()["models"]
    except Exception:  # noqa: BLE001  (Ollama not reachable: callers fall back to the default model)
        return []


@st.cache_data(ttl=60)
def reranker_models() -> list[str]:
    """The reranker models Ollama has that can generate (a name with `reranker` in it; a build that only
    embeds cannot answer yes or no). Whether the answer is sharp enough: see reranking/ollama.py."""
    return sorted(
        m["name"]
        for m in _ollama_models()
        if "reranker" in m["name"].lower() and "completion" in m.get("capabilities", ["completion"])
    )


@st.cache_data(ttl=60)
def chat_models() -> dict[str, list[str]]:
    """The models Ollama has that can chat with tools (the chatbot's model), each with its capabilities
    (`thinking` and `vision` are what the page asks about). Rerankers are left out: they also report
    these capabilities but only answer yes or no."""
    return {
        m["name"]: m["capabilities"]
        for m in _ollama_models()
        if {"completion", "tools"} <= set(m.get("capabilities", []))
        and "reranker" not in m["name"].lower()
    }
