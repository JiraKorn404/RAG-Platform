"""The HTTP API: what a front end can do (service.py), over HTTP. The Streamlit UI and the platform's
backend both call this.

    uvicorn rag_lab.serve.api:app

An answer is a stream of server-sent events (`text/event-stream`): each is one of the events of
core/events.py as JSON, tagged with its `type`, and the last is `Done`, or `Failed` when the turn stopped
with an error. Every other reply is JSON. With `api.key` set in config/connections.yaml a caller must send
it in the `X-API-Key` header. `/docs` shows every request, and `/openapi.json` is the contract a client
can be generated from."""

import json
import secrets
from typing import Literal

from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel

from rag_lab.core.events import Failed, to_dict
from rag_lab.core.settings import load
from rag_lab.serve import service


def _key(x_api_key: str | None = Header(None)) -> None:
    wanted = load().connections.api.key
    if wanted and not (x_api_key and secrets.compare_digest(x_api_key, wanted)):
        raise HTTPException(401, "Send the API key in the X-API-Key header.")


app = FastAPI(title="RAG-Platform", dependencies=[Depends(_key)])


@app.exception_handler(service.Refused)
def _refused(request, error: service.Refused) -> JSONResponse:
    """What cannot be done as asked: 409 with the reason, in the shape of FastAPI's own errors."""
    return JSONResponse({"detail": str(error)}, status_code=409)


class Question(BaseModel):
    kind: Literal["documents", "database"]  # what the chat searches
    target: str  # the collection (experiment) or the schema
    question: str


@app.get("/health")
def health() -> dict[str, bool]:
    return service.health()


@app.get("/targets")
def targets() -> dict:
    """The collections and the schemas a chat can search."""
    return service.targets()


@app.get("/check")
def check(kind: Literal["documents", "database"], target: str) -> dict:
    """Whether a chat can be asked now: what is missing, what was switched off, and its settings."""
    return service.check(kind, target)


@app.get("/chats")
def chats() -> list[dict]:
    return service.chats()


@app.get("/chats/{chat_id}")
def chat(chat_id: str) -> dict:
    found = service.chat(chat_id)
    if found is None:
        raise HTTPException(404, "There is no such chat.")
    return found


@app.delete("/chats/{chat_id}")
def delete_chat(chat_id: str) -> None:
    service.delete_chat(chat_id)


@app.post("/chats/{chat_id}/turns")
def ask(chat_id: str, body: Question) -> StreamingResponse:
    """Ask a question. `chat_id` is the caller's: a chat is made with its first turn. The reply is the
    stream of the turn's events."""
    events = service.ask(chat_id, body.kind, body.target, body.question)

    def stream():
        try:
            for event in events:
                yield _sse(to_dict(event))
        except Exception as e:  # noqa: BLE001  (Ollama or a database went away: the turn is saved with this error)
            yield _sse(to_dict(Failed(str(e))))

    return StreamingResponse(stream(), media_type="text/event-stream")


def _sse(data: dict) -> str:
    return f"event: {data['type']}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


@app.get("/schemas/{schema}/examples")
def examples(schema: str) -> list[dict]:
    """The good answers saved for a schema."""
    return service.examples(schema)


@app.post("/turns/{turn_id}/good")
def mark_good(turn_id: int) -> dict[str, int]:
    """The thumbs-up: keep this database turn's question and SQL as an example."""
    return {"id": service.mark_good(turn_id)}


@app.delete("/examples/{example_id}")
def unmark_good(example_id: int) -> None:
    service.unmark_good(example_id)


@app.get("/library")
def library() -> list[dict]:
    """Every experiment with its documents."""
    return service.library()


@app.delete("/collections/{name}")
def delete_experiment(name: str) -> dict:
    return service.delete_experiment(name)


@app.delete("/collections/{name}/documents/{doc_id}")
def delete_document(name: str, doc_id: str) -> dict:
    return service.delete_document(name, doc_id)


@app.get("/pictures/{path:path}")
def picture(path: str) -> FileResponse:
    """The image of a picture chunk: `path` is a hit's `image`."""
    found = service.picture(path)
    if found is None:
        raise HTTPException(404, "There is no such picture.")
    return FileResponse(found, media_type="image/png")
