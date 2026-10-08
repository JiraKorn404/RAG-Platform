"""How the pages reach the system: the HTTP API (src/rag_lab/serve/api.py), the same way the platform's
backend does. The pages import nothing of the system but its events and its settings, so whatever they
show is known to be reachable over HTTP."""

import json
from collections.abc import Iterator
from datetime import datetime

import httpx
import streamlit as st

from rag_lab.core.config import embed_model_label
from rag_lab.core.events import Event, Failed, from_dict
from rag_lab.core.settings import load

model_label = embed_model_label  # 'qwen3-embedding:4b' -> '4b'; another family keeps its whole name


class ApiError(Exception):
    """The API refused a request, or cannot be reached. The message is for the person at the page."""


@st.cache_resource
def _client() -> httpx.Client:
    """Kept for as long as the UI runs: `docker compose restart ui` after editing config/connections.yaml."""
    api = load().connections.api
    return httpx.Client(base_url=api.url, headers={"X-API-Key": api.key} if api.key else {}, timeout=120)


def _refusal(reply: httpx.Response) -> str:
    try:
        return str(reply.json()["detail"])
    except (ValueError, KeyError, TypeError):
        return f"The API answered {reply.status_code}."


def _call(method: str, path: str, **kwargs):
    try:
        reply = _client().request(method, path, **kwargs)
    except httpx.HTTPError as e:
        raise ApiError(f"The API cannot be reached at {_client().base_url} (`api.url` in config/connections.yaml): {e}") from e
    if reply.status_code >= 400:
        raise ApiError(_refusal(reply))
    return reply.json() if reply.content else None


def get(path: str, **params):
    return _call("GET", path, params=params)


def find(path: str) -> dict | None:
    """`get`, with None for what the API does not have (404)."""
    reply = _client().get(path)
    return reply.json() if reply.status_code == 200 else None


def post(path: str):
    return _call("POST", path)


def delete(path: str):
    return _call("DELETE", path)


def ask(chat_id: str, kind: str, target: str, question: str) -> Iterator[Event]:
    """Ask a question and yield the events of the turn as they arrive, `Done` last. A turn the API
    refuses, or that fails on the way, raises `ApiError` with the reason."""
    body = {"kind": kind, "target": target, "question": question}
    try:
        with _client().stream("POST", f"/chats/{chat_id}/turns", json=body, timeout=None) as reply:
            if reply.status_code >= 400:
                reply.read()
                raise ApiError(_refusal(reply))
            for line in reply.iter_lines():
                if not line.startswith("data: "):
                    continue
                event = from_dict(json.loads(line.removeprefix("data: ")))
                if isinstance(event, Failed):
                    raise ApiError(event.message)
                yield event
    except httpx.HTTPError as e:
        raise ApiError(f"The connection to the API was lost: {e}") from e


def picture(image: str) -> bytes | None:
    """The PNG of a picture chunk (`Hit.image`), or None when it is gone."""
    try:
        reply = _client().get(f"/pictures/{image}")
    except httpx.HTTPError:
        return None
    return reply.content if reply.status_code == 200 else None


def when(value: str | None) -> datetime | None:
    """A time as the API sends it (ISO text), as a datetime."""
    return datetime.fromisoformat(value) if value else None
