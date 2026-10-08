import uuid

import data
import streamlit as st
import style
from trace_view import (
    STEP_NAMES,
    apply,
    attempt_lines,
    example_of,
    messages_from,
    new_trace,
    show_answer,
    show_trace,
    sql_attempt_lines,
)

from rag_lab.core.events import (
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

# A chat searches one kind of thing for its whole life, chosen before its first question.
SOURCES = {"documents": "Vector database (documents)", "database": "Relational database (tables)"}
ICONS = {"documents": ":material/description:", "database": ":material/database:"}


@st.cache_data(ttl=15)
def targets() -> dict:
    """The experiments that have a Qdrant collection and a BM25 vector (hybrid search needs it), and the
    schemas."""
    return data.get("/targets")


@st.cache_data(ttl=60)
def check(kind: str, target: str) -> dict:
    """What a chat runs with now (config/llm.yaml, checked against what Ollama has)."""
    return data.get("/check", kind=kind, target=target)


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
    data.delete(f"/chats/{chat_id}")
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


def answer_turn(question: str, kind: str, target: str, settings: dict, chat_id: str) -> dict:
    """Ask one question, drawing each step as it happens. Returns what is shown for the turn."""
    trace = new_trace(question, settings, kind)
    live, box = st.empty(), st.empty()
    text, step = "", "starting"
    with live.container():
        status = st.status("Starting", expanded=True)
        with status:
            attempts_box, chunks_box, thinking_box = st.empty(), st.empty(), st.empty()
    try:
        for event in data.ask(chat_id, kind, target, question):
            apply(trace, event)
            if isinstance(event, StepStarted):
                step = event.node
                label = STEP_NAMES[step] + "…"
                if step == "retrieve":
                    label = f"{STEP_NAMES[step]} (hybrid, {settings['candidates']} candidates, then reranking)…"
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
    except data.ApiError as e:  # Ollama or Qdrant not reachable: show where it failed, do not crash
        trace["error"] = f"{STEP_NAMES.get(step, 'Starting')} failed: {e}"
        status.update(label=trace["error"], state="error")
        return trace
    live.empty()
    with box.container():
        show_answer(trace)
    return trace


style.hero("Chatbot", "Ask questions of your documents or your tables, and see how each answer was found")

try:
    found = targets()
    past = data.get("/chats")
except data.ApiError as e:
    st.error(str(e))
    st.stop()
available = found["collections"]
by_name = {row["name"]: row for row in available}
schemas = found["schemas"]
schema_by_name = {s["name"]: s for s in schemas}

# Which chat is this? The id in the URL, or a new one. A chat searches the one thing it started with.
chat_id = st.query_params.get("chat")
if chat_id and st.session_state.get("loaded_chat") != chat_id:
    saved = data.find(f"/chats/{chat_id}")
    st.session_state["messages"] = messages_from(saved["turns"] if saved else [])
    st.session_state["opened"] = {"kind": saved["kind"], "target": saved["target"]} if saved else None
    st.session_state["loaded_chat"] = chat_id
# A chat exists from its first saved turn: it is the one opened, or it is among the chats by now.
session = (st.session_state.get("opened") or next((s for s in past if s["session_id"] == chat_id), None)) if chat_id else None
owner = None
if session:
    owner = session["target"] if session["target"] in (by_name if session["kind"] == "documents" else schema_by_name) else None
    if owner is None:
        st.warning("That chat searches something that is not available here, so a new chat was started.")
        chat_id, session = None, None
if not chat_id:
    chat_id = new_chat()
    st.session_state["messages"], st.session_state["opened"], st.session_state["loaded_chat"] = [], None, chat_id
locked = session is not None  # a chat that has a turn keeps what it searches
if locked:
    st.session_state["source"] = session["kind"]
    st.session_state["experiment" if session["kind"] == "documents" else "schema"] = owner
elif st.session_state.get("source") not in SOURCES:
    st.session_state["source"] = "documents" if available or not schemas else "database"
for key, valid in (("experiment", by_name), ("schema", schema_by_name)):
    if st.session_state.get(key) not in valid:
        st.session_state.pop(key, None)

# The sidebar: a new chat, what this chat searches, and the chats so far. The models and every other
# setting are in config/llm.yaml.
target = None  # the experiment or the schema this chat searches, by name
with st.sidebar:
    st.button("New chat", on_click=new_chat, icon=":material/add:", width="stretch")
    kind = st.radio("Search in", list(SOURCES), key="source", format_func=SOURCES.get, disabled=locked, help="Chosen before the first question, and fixed for the whole chat.")
    if kind == "documents":
        if available:
            target = st.selectbox("Collection", list(by_name), key="experiment", format_func=lambda n: describe(by_name[n]), disabled=locked)
        else:
            st.info("No collection with a BM25 vector exists yet. Put a PDF in data/raw (with `index.sparse: true` in config/pipeline.yaml).")
    elif schemas:
        target = st.selectbox("Schema", list(schema_by_name), key="schema", format_func=lambda n: f"{n} · {schema_by_name[n]['tables']} table(s)", disabled=locked)
    else:
        st.info("No schema exists yet. Put a CSV file in data/tables/<schema>/.")
    if locked:
        st.caption("A chat keeps what it searches. Start a new chat to search somewhere else.")
    if past:
        st.subheader("Chat history")
        for s in past:
            title = s["title"] if len(s["title"]) <= 34 else s["title"][:34] + "…"
            what = "Vector database" if s["kind"] == "documents" else "Relational database"
            st.button(
                f"{title} · {s['turns']}",
                key=f"chat-{s['session_id']}",
                icon=ICONS[s["kind"]],
                on_click=open_chat,
                args=(s["session_id"],),
                type="primary" if s["session_id"] == chat_id else "secondary",
                width="stretch",
                help=f"{what}: {s['target']}. {s['turns']} turn(s), last used {data.when(s['updated_at']):%d %b %Y %H:%M} UTC",
            )
    if st.session_state["messages"]:
        with st.popover("Delete this chat", width="stretch"):
            st.write("This deletes the chat and all its turns.")
            st.button("Yes, delete it", on_click=delete_chat, args=(chat_id,), key="delete-chat")


def mark_good(trace: dict, saved: dict, pair: tuple[str, str], key: str) -> None:
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
                data.delete(f"/examples/{existing['id']}")
            else:
                data.post(f"/turns/{trace['turn_id']}/good")
        except data.ApiError as e:  # Ollama or Qdrant down: the table may be ahead of the index; `tables examples --reindex` fixes it
            st.error(f"Could not change the example: {e}")
        else:
            st.rerun()


saved_examples = (
    {(e["standalone"], e["sql"]): e for e in data.get(f"/schemas/{owner}/examples")} if locked and kind == "database" else {}
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
        # a turn that was not saved has no id, and a good answer is kept by its turn
        if locked and kind == "database" and trace["turn_id"] and (pair := example_of(trace)):
            mark_good(trace, saved_examples, pair, f"good-{chat_id}-{index}")
        show_trace(trace)
        if trace["done"] and not trace["done"].saved:
            st.warning(f"This turn was not saved to the database: {trace['done'].save_error}")

if target is None:
    st.info("Choose what to search in the sidebar." if not locked else "What this chat searches is not available.")
    st.stop()

# What this chat runs with: llm.yaml, checked against what Ollama has. A missing model stops here with
# the key to change, instead of failing in the middle of an answer.
try:
    ready = check(kind, target)
except data.ApiError as e:
    st.error(str(e))
    st.stop()
for problem in ready["missing"]:
    st.error(problem)
if ready["missing"]:
    st.stop()
for note in ready["notes"]:
    st.caption(note)

question = st.chat_input("Ask a question about the documents" if kind == "documents" else "Ask a question about the tables")
if question and question.strip():
    with st.chat_message("user"):
        st.markdown(question)
    with st.chat_message("assistant"):
        trace = answer_turn(question, kind, target, ready["settings"], chat_id)
    st.session_state["messages"] += [
        {"role": "user", "content": question},
        {"role": "assistant", "trace": trace},
    ]
    st.rerun()  # draws the turn from what was kept, and brings the list of past chats up to date
