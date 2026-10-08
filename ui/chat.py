import os
import uuid

import data
import streamlit as st
import style
from trace_view import (
    STEP_NAMES,
    apply,
    attempt_lines,
    example_of,
    history_of,
    messages_from,
    new_trace,
    show_answer,
    show_trace,
    sql_attempt_lines,
)

from rag_lab.agent import run
from rag_lab.agent.documents import DocumentsFlow, build_graph
from rag_lab.agent.events import (
    AnswerToken,
    ExamplesFound,
    Graded,
    Query,
    Repairing,
    Retrieved,
    Rewrote,
    SchemaShown,
    SqlChecked,
    SqlRan,
    SqlWritten,
    StepStarted,
    Thinking,
)
from rag_lab.agent.sql import SqlFlow
from rag_lab.agent.sql import build_graph as build_sql_graph
from rag_lab.config import AgentConfig, ExperimentConfig, SearchConfig, SqlAgentConfig
from rag_lab.embedding.ollama import OllamaEmbedder
from rag_lab.metrics.store import MetricsStore
from rag_lab.reranking import OllamaReranker
from rag_lab.sql import examples as good_answers
from rag_lab.storage.qdrant import QdrantStore

# A chat searches one kind of thing for its whole life, chosen before its first question.
SOURCES = {"documents": "Documents (vector database)", "database": "Database (tables)"}
ICONS = {"documents": ":material/description:", "database": ":material/database:"}


@st.cache_resource
def services() -> tuple[OllamaEmbedder, QdrantStore, MetricsStore, OllamaReranker]:
    base_url = os.environ["OLLAMA_BASE_URL"]
    return (
        OllamaEmbedder(base_url),
        QdrantStore(os.environ["QDRANT_URL"]),
        MetricsStore(os.environ["METRICS_DATABASE_URL"]),
        OllamaReranker(base_url),
    )


@st.cache_data(ttl=15)
def experiments() -> list[dict]:
    """The experiments that have a Qdrant collection and a BM25 vector (hybrid search needs it)."""
    _, qdrant, metrics, _ = services()
    existing = {c.name for c in qdrant.client.get_collections().collections}
    return [
        {**row, "points": qdrant.count(row["name"])}
        for row in metrics.list_experiments()
        if row["name"] in existing and row["config"].get("index", {}).get("sparse")
    ]


def describe(row: dict) -> str:
    config = row["config"]
    return (
        f"{row['name']} · {data.model_label(config['embed']['model'])} · {config['chunk']['strategy']} · "
        f"{row['documents']} document(s) · {row['points']} points"
    )


# A chat is the id in the page's URL (?chat=<id>), so a refresh or a bookmark finds it again. Its turns
# are in the database; the page shows them from the saved events, the same way it shows a live answer.


def new_chat() -> str:
    chat_id = uuid.uuid4().hex[:12]
    st.query_params["chat"] = chat_id
    return chat_id


def open_chat(chat_id: str) -> None:
    st.query_params["chat"] = chat_id


def delete_chat(chat_id: str) -> None:
    services()[2].delete_chat_session(chat_id)
    new_chat()


def where(hit) -> str:
    name = hit.source_file or hit.doc_id
    return f"{name}, p. {hit.page}" if hit.page else name


def database_progress(trace: dict) -> str:
    """What a database turn has done so far, for the status panel."""
    parts = []
    if trace["rewritten"]:
        parts.append(f"**Understood as:** {trace['standalone']}")
    if trace["schema"]:
        parts.append(f"**Schema:** {len(trace['schema'].tables)} table(s), {trace['schema'].chars:,} characters")
    if trace["examples"]:
        parts.append(f"**Similar good answers shown to the model:** {len(trace['examples'])}")
    if trace["sql_attempts"]:
        parts.append(sql_attempt_lines(trace))
    return "\n\n".join(parts)


def answer_turn(
    question: str, graph, flow, metrics: MetricsStore, chat_id: str, history: list[tuple[str, str]],
) -> dict:
    """Run one question, drawing each step as it happens. Returns what is shown for the turn."""
    kind = flow.kind
    trace = new_trace(question, flow.cfg.model_dump(mode="json"), kind)
    live, box = st.empty(), st.empty()
    text, step = "", "starting"
    with live.container():
        status = st.status("Starting", expanded=True)
        with status:
            attempts_box, chunks_box, thinking_box = st.empty(), st.empty(), st.empty()
    try:
        for event in run(graph, flow, question, history, metrics=metrics, session_id=chat_id):
            apply(trace, event)
            if isinstance(event, StepStarted):
                step = event.node
                label = STEP_NAMES[step] + "…"
                if step == "retrieve":
                    label = f"{STEP_NAMES[step]} (hybrid, {flow.cfg.search.candidates} candidates, then reranking)…"
                status.update(label=label)
            elif isinstance(event, Thinking):
                thinking_box.markdown(f"**Thinking**\n\n{trace['thinking']}")
            elif isinstance(event, AnswerToken):
                text += event.text
                box.markdown(text + " ▌")
            elif kind == "database":
                if isinstance(event, (Query, SchemaShown, ExamplesFound, SqlWritten, SqlChecked, SqlRan, Repairing)):
                    attempts_box.markdown(database_progress(trace))
            elif isinstance(event, (Query, Rewrote, Graded)):
                attempts_box.markdown(attempt_lines(trace))
            elif isinstance(event, Retrieved):
                chunks_box.markdown(
                    "**Found**\n\n" + "\n".join(
                        f"{h.rank}. `{h.similarity:.3f}` {where(h)}" + (f" › {h.headings[-1]}" if h.headings else "")
                        for h in event.hits
                    )
                )
    except Exception as e:  # noqa: BLE001  (Ollama or Qdrant not reachable: show where it failed, do not crash)
        trace["error"] = f"{STEP_NAMES.get(step, 'Starting')} failed: {e}"
        status.update(label=trace["error"], state="error")
        return trace
    live.empty()
    with box.container():
        show_answer(trace)
    return trace


style.hero("Chatbot", "Ask questions of your documents or your tables, and see how each answer was found")

rerankers = data.reranker_models()
chat_models = data.chat_models()
if not chat_models:
    st.info("Ollama has no chat model that can use tools installed, so the chatbot cannot run.")
    st.stop()

embedder, qdrant, metrics, reranker = services()
available = experiments()
by_name = {row["name"]: row for row in available}
schemas = metrics.list_db_schemas()
schema_by_name = {s["name"]: s for s in schemas}

# Which chat is this? The id in the URL, or a new one. A chat searches the one thing it started with.
chat_id = st.query_params.get("chat")
session = metrics.get_chat_session(chat_id) if chat_id else None
owner = None
if session:
    if session["kind"] == "documents":
        found = next((r for r in available if r["config_hash"] == session["config_hash"]), None)
        owner = found["name"] if found else None
    else:
        owner = session["schema_name"] if session["schema_name"] in schema_by_name else None
    if owner is None:
        st.warning("That chat searches something that is not available here, so a new chat was started.")
        chat_id, session = None, None
if not chat_id:
    chat_id = new_chat()
locked = session is not None  # a chat that has a turn keeps what it searches
if locked:
    st.session_state["source"] = session["kind"]
    st.session_state["experiment" if session["kind"] == "documents" else "schema"] = owner
elif st.session_state.get("source") not in SOURCES:
    st.session_state["source"] = "documents" if available or not schemas else "database"
for key, valid in (("experiment", by_name), ("schema", schema_by_name)):
    if st.session_state.get(key) not in valid:
        st.session_state.pop(key, None)

if st.session_state.get("loaded_chat") != chat_id:
    st.session_state["messages"] = messages_from(metrics.get_chat_turns(chat_id))
    st.session_state["loaded_chat"] = chat_id

defaults = AgentConfig()
models = list(chat_models)
target = None  # the experiment (a row) or the schema (its name) this chat searches
with st.sidebar:
    st.subheader("Chat settings")
    kind = st.radio("Search in", list(SOURCES), key="source", format_func=SOURCES.get, disabled=locked, help="Chosen before the first question, and fixed for the whole chat.")
    if locked:
        what = "the documents of" if kind == "documents" else "the tables of the schema"
        st.caption(f"This chat searches {what} `{owner}`. Start a new chat to search somewhere else.")
    if kind == "documents":
        if available:
            name = st.selectbox("Experiment", list(by_name), key="experiment", format_func=lambda n: describe(by_name[n]), disabled=locked)
            target = by_name[name]
        else:
            st.info("No experiment with a BM25 vector exists yet. Set `index.sparse: true` in config/ingest.yaml and put a PDF in data/raw.")
        if not rerankers:
            st.info("Ollama has no reranker installed, so documents cannot be searched.")
            target = None
    elif schemas:
        target = st.selectbox("Schema", list(schema_by_name), key="schema", format_func=lambda n: f"{n} · {schema_by_name[n]['tables']} table(s)", disabled=locked)
    else:
        st.info("No schema exists yet.")
    model = st.selectbox("Chat model", models, index=models.index(defaults.model) if defaults.model in models else 0)
    can_think, can_see = "thinking" in chat_models[model], "vision" in chat_models[model]
    think = st.toggle("Thinking", value=defaults.think and can_think, disabled=not can_think, help="The model thinks before it answers (or, for a database, before it writes the SQL), and the page shows it. Slower.")
    if kind == "documents":
        rerank_model = st.selectbox("Reranker", rerankers or ["none installed"], index=rerankers.index(defaults.search.reranker) if defaults.search.reranker in rerankers else 0)
        top_k = st.slider("Chunks given to the model (top k)", 1, 10, defaults.top_k)
        candidates = int(st.number_input("Candidates", min_value=1, max_value=100, value=defaults.search.candidates, help="Hits the reranker scores; one call to Ollama each."))
        show_pictures = st.toggle("Show pictures to the model", value=defaults.show_pictures and can_see, disabled=not can_see, help=f"When a retrieved chunk is a picture, the model is given the image and not only its caption (the first {defaults.max_pictures} of a turn). Only an experiment made with *Index the pictures* has picture chunks.")
        st.caption("Search: hybrid (dense and BM25 keywords) with reranking.")
    else:
        st.caption("The model is given the whole schema, writes one SELECT, and a check and the database confirm it before it runs, read-only.")
    st.button("New chat", on_click=new_chat, width="stretch")
    if st.session_state["messages"]:
        with st.popover("Delete this chat", width="stretch"):
            st.write("This deletes the chat and all its turns.")
            st.button("Yes, delete it", on_click=delete_chat, args=(chat_id,), key="delete-chat")
    past = metrics.list_chat_sessions()
    if past:
        st.subheader("Past chats")
        for s in past:
            title = s["title"] if len(s["title"]) <= 34 else s["title"][:34] + "…"
            what = "Documents" if s["kind"] == "documents" else "Database"
            st.button(
                f"{title} · {s['turns']}",
                key=f"chat-{s['session_id']}",
                icon=ICONS[s["kind"]],
                on_click=open_chat,
                args=(s["session_id"],),
                type="primary" if s["session_id"] == chat_id else "secondary",
                width="stretch",
                help=f"{what}: {s['target']}. {s['turns']} turn(s), last used {s['updated_at']:%d %b %Y %H:%M} UTC",
            )

def mark_good(trace: dict, schema: str, saved: dict, pair: tuple[str, str], key: str) -> None:
    """The thumbs-up under a database answer: save the question and its SQL as an example, or, when it is
    already saved, take it away again."""
    existing = saved.get(pair)
    pressed = existing is not None
    if st.button(
        "Marked as a good answer" if pressed else "Good answer",
        key=key,
        icon=":material/thumb_up:",
        type="primary" if pressed else "secondary",
        help="Click to remove it from the examples." if pressed else "Keep this question and its SQL as an example for similar questions.",
    ):
        try:
            if pressed:
                good_answers.remove(metrics, qdrant, schema, existing["id"])
            else:
                good_answers.save(metrics, embedder, qdrant, SqlAgentConfig(), schema, trace["question"], pair[0], pair[1], trace["turn_id"])
        except Exception as e:  # noqa: BLE001  (Ollama or Qdrant down: the table may be ahead of the index; `examples --reindex` fixes it)
            st.error(f"Could not change the example: {e}")
        else:
            st.rerun()


saved_examples = (
    {(e["standalone"], e["sql"]): e for e in metrics.list_sql_examples(owner)} if locked and kind == "database" else {}
)
for index, message in enumerate(st.session_state["messages"]):
    with st.chat_message(message["role"]):
        if message["role"] == "user":
            st.markdown(message["content"])
            continue
        trace = message["trace"]
        if trace["error"]:
            st.error(trace["error"])
        else:
            show_answer(trace)
        if locked and kind == "database" and (pair := example_of(trace)):
            mark_good(trace, owner, saved_examples, pair, f"good-{chat_id}-{index}")
        show_trace(trace)
        if trace["done"] and not trace["done"].saved:
            st.warning(f"This turn was not saved to the database: {trace['done'].save_error}")

if target is None:
    st.info("Choose what to search in the sidebar." if not locked else "What this chat searches is not available.")
    st.stop()

question = st.chat_input("Ask a question about the documents" if kind == "documents" else "Ask a question about the tables")
if question and question.strip():
    with st.chat_message("user"):
        st.markdown(question)
    base_url = os.environ["OLLAMA_BASE_URL"]
    if kind == "documents":
        experiment = ExperimentConfig.model_validate(target["config"])
        cfg = AgentConfig(
            model=model,
            think=think,
            top_k=top_k,
            search=SearchConfig(method="hybrid+rerank", reranker=rerank_model, candidates=candidates),
            show_pictures=show_pictures and can_see,
        )
        graph = build_graph(experiment, embedder, qdrant, reranker, base_url, cfg)
        flow = DocumentsFlow(experiment, cfg)
    else:
        cfg = SqlAgentConfig(model=model, think=think)
        graph = build_sql_graph(metrics, target, base_url, cfg, embedder, qdrant)
        flow = SqlFlow(target, cfg)
    with st.chat_message("assistant"):
        trace = answer_turn(question, graph, flow, metrics, chat_id, history_of(st.session_state["messages"]))
    st.session_state["messages"] += [
        {"role": "user", "content": question},
        {"role": "assistant", "trace": trace},
    ]
    st.rerun()  # draws the turn from what was kept, and brings the list of past chats up to date
