"""What every chatbot flow shares: how a node reports itself, and the runner that streams a turn's events
and saves it. A flow is a LangGraph graph plus a small object that says what a saved turn needs from it
(see `Flow`), so the runner does not know what a chunk or a table is."""

import functools
import time
from collections.abc import Callable, Iterator
from dataclasses import asdict, dataclass, field, replace
from typing import Protocol

import httpx
from langchain_ollama import ChatOllama
from langgraph.config import get_stream_writer

from rag_lab.core.config import ChatModelConfig
from rag_lab.core.events import (
    AnswerToken,
    Done,
    Event,
    ModelState,
    Query,
    StepFinished,
    StepStarted,
    Thinking,
    to_dict,
)
from rag_lab.core.store import MetricsStore


def step(node):
    """Report when a node starts and how long it took."""

    @functools.wraps(node)
    def reported(state: dict) -> dict:
        emit = get_stream_writer()
        emit(StepStarted(node.__name__))
        start = time.perf_counter()
        update = node(state)
        emit(StepFinished(node.__name__, (time.perf_counter() - start) * 1000))
        return update

    return reported


@dataclass
class Summary:
    """What a saved turn needs from a flow."""

    query: str  # the last thing searched for (documents) or run (database)
    hits: list[dict] = field(default_factory=list)  # for the `hits` column
    answer: str = ""
    thinking: str = ""
    abstained: bool = False
    cited: list[int] = field(default_factory=list)
    unknown_citations: list[int] = field(default_factory=list)
    log: Callable[[MetricsStore], None] | None = None  # extra writes, such as the search log
    # How a documents turn went (see `Done`); a database flow leaves them empty.
    outcome: str = ""
    abstain_reason: str | None = None
    route: str = ""
    rewrites: int = 0
    top_score: float | None = None


class Flow(Protocol):
    kind: str  # "documents" or "database": what a chat that uses this flow searches
    config_hash: str | None  # the experiment, for a documents flow
    schema_name: str | None  # the schema, for a database flow
    cfg: ChatModelConfig

    def summarise(self, question: str, events: list[Event], final: dict) -> Summary:
        """What to save for a turn. `final` is the graph's final state; it is empty when the turn
        failed, and then `events` (what happened up to the failure) is all there is."""


def run(
    graph,
    flow: Flow,
    question: str,
    history: list[tuple[str, str]] | None = None,
    *,
    metrics: MetricsStore | None = None,
    session_id: str | None = None,
) -> Iterator[Event]:
    """Answer one question, yielding the events as they happen and `Done` last. With `metrics` and a
    `session_id` the turn is saved to `chat_turns` (a failed turn too, with its error, before the error
    is raised again) and whatever the flow logs besides. A save that fails does not hide the answer:
    `Done.saved` is false and `Done.save_error` says why."""
    cfg = flow.cfg
    start = time.perf_counter()
    timings: dict[str, float] = {}
    states: list[ModelState] = []
    stored: list[Event] = []  # what a saved turn replays from: every event but the token-by-token ones
    step_now = "starting"
    final: dict = {}
    turn_id: int | None = None

    def save(summary: Summary, **extra) -> str | None:
        """Save the turn. Returns why that failed, or None."""
        nonlocal turn_id
        if not (metrics and session_id):
            return None
        try:
            turn_id = metrics.add_chat_turn(
                flow.config_hash,
                flow.schema_name,
                session_id,
                question,
                summary.query,
                cfg.model,
                cfg.think,
                summary.answer,
                summary.thinking,
                summary.hits,
                timings,
                [asdict(s) for s in states],
                kind=flow.kind,
                events=[to_dict(e) for e in stored],
                total_ms=(time.perf_counter() - start) * 1000,
                settings=cfg.model_dump(mode="json"),
                **extra,
            )
        except Exception as e:  # noqa: BLE001  (a failed save is reported, never hides the answer)
            return str(e)
        return None

    try:
        for mode, payload in graph.stream(
            {"question": question, "history": history or []}, stream_mode=["custom", "values"]
        ):
            if mode == "values":
                final = payload
                continue
            if isinstance(payload, StepStarted):
                step_now = payload.node
            elif isinstance(payload, StepFinished):
                timings[payload.node] = timings.get(payload.node, 0.0) + payload.ms  # a retry repeats steps
            elif isinstance(payload, ModelState):
                states.append(payload)
            if not isinstance(payload, (Thinking, AnswerToken)):
                stored.append(payload)
            yield payload
    except Exception as e:
        save(flow.summarise(question, stored, {}), error=f"{step_now} failed: {e}")  # the original error is the one to raise
        raise

    summary = flow.summarise(question, stored, final)
    done = Done(
        answer=summary.answer,
        thinking=summary.thinking,
        cited=summary.cited,
        unknown_citations=summary.unknown_citations,
        timings=timings,
        total_ms=(time.perf_counter() - start) * 1000,
        abstained=summary.abstained,
        outcome=summary.outcome,
        abstain_reason=summary.abstain_reason,
        route=summary.route,
        rewrites=summary.rewrites,
        top_score=summary.top_score,
    )
    stored.append(done)
    failure = None
    if metrics and session_id:
        if summary.log:
            try:
                summary.log(metrics)
            except Exception as e:  # noqa: BLE001
                failure = str(e)
        failure = save(summary, abstained=summary.abstained, cited=summary.cited) or failure
    yield replace(done, saved=False, save_error=failure) if failure else replace(done, turn_id=turn_id)


# --- the chat model ----------------------------------------------------------------------------------
# The chat model: Ollama through langchain-ollama, with the state of every call reported.

def chat_model(base_url: str, cfg: ChatModelConfig, think: bool) -> ChatOllama:
    return ChatOllama(
        model=cfg.model,
        base_url=base_url,
        temperature=cfg.temperature,
        num_ctx=cfg.num_ctx,
        keep_alive=cfg.keep_alive,
        reasoning=think,
    )


def model_loaded(base_url: str, model: str) -> bool | None:
    """Whether Ollama already has the model in memory (a cold model makes the first call slow)."""
    try:
        listed = httpx.get(f"{base_url.rstrip('/')}/api/ps", timeout=5).json()["models"]
    except (httpx.HTTPError, ValueError, KeyError):
        return None
    return any(m.get("name") == model for m in listed)


def call_model(
    llm: ChatOllama,
    base_url: str,
    cfg: ChatModelConfig,
    node: str,
    think: bool,
    messages: list,
    stream: bool,
    tokens: bool = True,
) -> tuple[str, str]:
    """Run one call and report a ModelState. With `stream` the thinking and the answer are reported as
    they arrive; `tokens=False` reports only the thinking (for a reply that is not the answer shown to
    the user, such as the SQL). Returns (reply, thinking)."""
    emit = get_stream_writer()
    loaded = model_loaded(base_url, cfg.model)
    text, thinking, usage, meta = "", "", {}, {}
    if stream:
        for chunk in llm.stream(messages):
            piece = chunk.additional_kwargs.get("reasoning_content") or ""
            if piece:
                thinking += piece
                emit(Thinking(piece))
            if chunk.content:
                text += chunk.content
                if tokens:
                    emit(AnswerToken(chunk.content))
            if chunk.usage_metadata:  # only the chunk that ends the generation has the counts
                usage, meta = chunk.usage_metadata, chunk.response_metadata
    else:
        reply = llm.invoke(messages)
        text, thinking = reply.content, reply.additional_kwargs.get("reasoning_content") or ""
        usage, meta = reply.usage_metadata or {}, reply.response_metadata
    seconds = (meta.get("eval_duration") or 0) / 1e9
    prompt_tokens = usage.get("input_tokens", 0)
    output_tokens = usage.get("output_tokens", 0)
    emit(
        ModelState(
            node=node,
            model=cfg.model,
            think=think,
            loaded=loaded,
            prompt_tokens=prompt_tokens,
            output_tokens=output_tokens,
            num_ctx=cfg.num_ctx,
            tokens_per_s=output_tokens / seconds if seconds else None,
            context_full=prompt_tokens >= cfg.num_ctx,
        )
    )
    return text, thinking


def standalone_question(llm: ChatOllama, base_url: str, cfg: ChatModelConfig, state: dict, system: str, prompt) -> str:
    """Every flow's condense step: the question made standalone from the last `history_turns` pairs of
    the history (no call without history), reported as a Query. `prompt(question, history)` is the
    flow's own condense prompt."""
    question = state["question"]
    history = state.get("history", [])[-2 * cfg.history_turns :]
    text = question
    if history:
        text, _ = call_model(
            llm,
            base_url,
            cfg,
            "condense",
            think=False,
            stream=False,
            messages=[("system", system), ("human", prompt(question, history))],
        )
        text = text.strip() or question
    get_stream_writer()(Query(text, rewritten=text != question))
    return text
