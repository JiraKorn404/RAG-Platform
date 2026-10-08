"""How a chatbot turn is shown: what a turn's events add up to, how a saved turn is rebuilt, and the
"How this was answered" view, for a documents turn and for a database turn. A live answer and a saved
turn both go through `apply`."""

import pandas as pd
import streamlit as st
import style
from hits import hit_card

from rag_lab.agent.events import (
    Done,
    ExamplesFound,
    Graded,
    ModelState,
    Query,
    Repairing,
    Retrieved,
    Rewrote,
    SchemaShown,
    SqlChecked,
    SqlRan,
    SqlWritten,
    Thinking,
    from_dict,
)
from rag_lab.agent.sql import remembered

STEP_NAMES = {
    "condense": "Reading the question",
    "retrieve": "Searching",
    "grade": "Checking the chunks",
    "rewrite": "Trying a different query",
    "generate": "Writing the answer",
    "abstain": "Not found",
    # the database flow
    "schema": "Reading the schema",
    "examples": "Looking for similar questions",
    "write_sql": "Writing the SQL",
    "check": "Checking it",
    "run_sql": "Running it",
    "repair": "Fixing it",
    "answer": "Writing the answer",
}


def new_trace(question: str, settings: dict | None, kind: str = "documents") -> dict:
    return {
        "kind": kind, "question": question, "settings": settings, "attempts": [], "retrieved": None,
        "thinking": "", "states": [], "done": None, "error": None,
        # a database turn
        "standalone": "", "rewritten": False, "schema": None, "examples": None, "sql_attempts": [], "sql": "",
        "turn_id": None,
    }


def apply(trace: dict, event) -> None:
    """Fold one event into what is shown for a turn. A live answer and a saved turn both go through here."""
    if isinstance(event, Query) and trace["kind"] == "database":
        trace["standalone"], trace["rewritten"] = event.text, event.rewritten
    elif isinstance(event, (Query, Rewrote)):
        query = event.text if isinstance(event, Query) else event.query
        trace["attempts"].append({"query": query, "retrieved": None, "graded": None})
    elif isinstance(event, Graded):
        trace["attempts"][-1]["graded"] = event
    elif isinstance(event, Retrieved):
        trace["retrieved"] = trace["attempts"][-1]["retrieved"] = event
    elif isinstance(event, SchemaShown):
        trace["schema"] = event
    elif isinstance(event, ExamplesFound):
        trace["examples"] = event.examples
    elif isinstance(event, SqlWritten):
        trace["sql_attempts"].append({"attempt": event.attempt, "written": event.sql, "checked": None, "ran": None, "error": ""})
    elif isinstance(event, SqlChecked):
        trace["sql_attempts"][-1]["checked"] = event
        if event.ok:
            trace["sql"] = event.sql
    elif isinstance(event, SqlRan):
        trace["sql_attempts"][-1]["ran"] = event
    elif isinstance(event, Repairing):
        trace["sql_attempts"][-1]["error"] = event.error
    elif isinstance(event, Thinking):
        trace["thinking"] += event.text
    elif isinstance(event, ModelState):
        trace["states"].append(event)
    elif isinstance(event, Done):
        trace["done"], trace["thinking"] = event, event.thinking
        trace["turn_id"] = event.turn_id or trace["turn_id"]


def trace_from_turn(turn: dict) -> dict:
    """What is shown for a saved turn: its events, applied in order."""
    trace = new_trace(turn["question"], turn["settings"], turn["kind"])
    for saved in turn["events"]:
        apply(trace, from_dict(saved))
    trace["error"] = turn["error"]
    trace["turn_id"] = turn["id"]
    return trace


def messages_from(turns: list[dict]) -> list[dict]:
    messages = []
    for turn in turns:
        messages += [
            {"role": "user", "content": turn["question"]},
            {"role": "assistant", "trace": trace_from_turn(turn)},
        ]
    return messages


def history_of(messages: list[dict]) -> list[tuple[str, str]]:
    """The answered turns as the agent's history; a failed turn has no answer to remember. A database
    answer is remembered with the SQL that produced it, so a follow-up can be understood."""
    history = []
    for user, assistant in zip(messages[::2], messages[1::2]):
        trace = assistant["trace"]
        if not trace["error"]:
            history += [("User", user["content"]), ("Assistant", remembered(trace["done"].answer, trace["sql"]))]
    return history


def state_line(s: ModelState) -> str:
    loaded = {True: "already loaded", False: "had to be loaded", None: "load state unknown"}[s.loaded]
    speed = f" · {s.tokens_per_s:.1f} tok/s" if s.tokens_per_s else ""
    return (
        f"**{STEP_NAMES[s.node]}** · {s.model} ({loaded}) · thinking {'on' if s.think else 'off'} · "
        f"{s.prompt_tokens} of {s.num_ctx} context in, {s.output_tokens} out{speed}"
    )


def settings_line(settings: dict | None) -> str:
    """What a turn ran with; a turn saved before the settings were kept only has the model."""
    if not settings:
        return ""
    search = settings.get("search") or {}
    parts = [settings.get("model"), f"thinking {'on' if settings.get('think') else 'off'}"]
    if search:
        parts += [
            search["method"].replace("+", " + "),
            search["reranker"],
            f"{search['candidates']} candidates",
            f"top {settings['top_k']}",
            f"context {settings['num_ctx']}",
        ]
        if settings.get("show_pictures"):
            parts.append(f"up to {settings['max_pictures']} pictures shown to the model")
    elif "row_limit" in settings:  # a database turn
        parts += [
            f"context {settings['num_ctx']}",
            f"up to {settings['row_limit']} rows",
            f"{settings['max_repairs']} repairs",
        ]
    return "Settings: " + ", ".join(str(p) for p in parts)


def verdict(graded: Graded | None) -> str:
    if not graded:
        return ""
    answers = "the chunks answer it" if graded.enough else "the chunks do not answer it"
    return f" — best score {graded.best_score:.3f}, {answers} (decided by the {graded.by})"


def attempt_lines(trace: dict) -> str:
    return "\n".join(
        f"{n}. **Searched for:** {a['query']}{verdict(a['graded'])}" for n, a in enumerate(trace["attempts"], start=1)
    )


def _database_error(attempt: dict) -> str:
    """What the database said about an attempt; empty when it is the check's refusal, which is shown as that."""
    checked = attempt["checked"]
    refusal = checked is not None and not checked.ok and checked.reason == attempt["error"]
    return "" if refusal else attempt["error"]


def sql_attempt_lines(trace: dict) -> str:
    """The queries of a database turn so far, as markdown: what the model wrote, what the check said, what
    the database said. Used while the answer is being made."""
    parts = []
    for a in trace["sql_attempts"]:
        parts.append(f"**Attempt {a['attempt']}**\n\n```sql\n{a['written']}\n```")
        checked = a["checked"]
        if checked is not None:
            parts.append("Checked: ok" if checked.ok else f"Refused: {checked.reason}")
        if a["ran"] is not None:
            cut = " (cut at the row limit)" if a["ran"].truncated else ""
            parts.append(f"Ran in {a['ran'].ms:.0f} ms: {a['ran'].row_count} row(s){cut}")
        if error := _database_error(a):
            parts.append(f"The database said: {error}")
    return "\n\n".join(parts)


def show_answer(trace: dict) -> None:
    """The answer, and under a database answer the SQL that produced it."""
    done = trace["done"]
    (st.info if done.abstained else st.markdown)(done.answer)
    if trace["kind"] == "database" and trace["sql"] and not done.abstained:
        st.code(trace["sql"], language="sql")


def example_of(trace: dict) -> tuple[str, str] | None:
    """What a good answer saves for this turn: its standalone question and the SQL the model wrote for the
    query that ran. None when the turn did not produce an answer from a result."""
    if trace["error"] or trace["done"] is None or trace["done"].abstained:
        return None
    ran = [a for a in trace["sql_attempts"] if a["ran"] is not None]
    return (trace["standalone"] or trace["question"], ran[-1]["written"]) if ran else None


def _timeline(done: Done) -> None:
    rows = "".join(f"| {STEP_NAMES.get(node, node)} | {ms:.0f} ms |\n" for node, ms in done.timings.items())
    st.markdown(f"| Step | Time |\n|---|---|\n{rows}| **Total** | **{done.total_ms:.0f} ms** |")


def _models(trace: dict) -> None:
    for s in trace["states"]:
        st.markdown(state_line(s))
        if s.context_full:
            st.warning("The prompt filled the context window, so Ollama cut it. Raise `num_ctx`.")
    if trace["settings"]:
        st.caption(settings_line(trace["settings"]))


def _thinking(trace: dict) -> None:
    if trace["thinking"]:
        st.markdown(trace["thinking"])
    else:
        st.caption("The model did not think for this answer.")


def show_trace(trace: dict) -> None:
    """Everything the agent did for one answer."""
    if trace["kind"] == "database":
        return show_sql_trace(trace)
    done, retrieved = trace["done"], trace["retrieved"]
    with st.expander("How this was answered"):
        steps, thinking, chunks = st.tabs(
            ["Steps and model", "Thinking", f"Retrieved chunks ({len(retrieved.hits) if retrieved else 0})"]
        )
        with steps:
            st.markdown(f"**Question:** {trace['question']}")
            if trace["attempts"]:
                st.markdown(attempt_lines(trace))
            if done:
                _timeline(done)
            if retrieved and retrieved.method:
                st.caption(
                    f"Search: {retrieved.method.replace('+', ' + ')}, {retrieved.candidates} candidates. "
                    f"Embedding the query {retrieved.embed_ms:.0f} ms, Qdrant {retrieved.search_ms:.0f} ms, "
                    f"reranking {retrieved.rerank_ms:.0f} ms."
                )
            _models(trace)
            if done and done.unknown_citations:
                st.warning(f"The answer cites passage(s) {done.unknown_citations}, which do not exist.")
        with thinking:
            _thinking(trace)
        with chunks:
            if not retrieved or not retrieved.hits:
                st.caption("Nothing was retrieved.")
            else:
                st.caption("The number on a card is the passage number the answer cites. Score: the reranker's probability that the chunk answers the question.")
                cited = done.cited if done else []
                # the pictures the model was given: the rule of agent.documents.graph.shown_pictures
                settings = trace["settings"] or {}
                pictures = [h.rank for h in retrieved.hits if h.image] if settings.get("show_pictures") else []
                seen = set(pictures[: settings.get("max_pictures", 0)])
                st.markdown(
                    "".join(
                        hit_card(h, style.ACCENT,cited=h.rank in cited, seen=h.rank in seen)
                        for h in retrieved.hits
                    ),
                    unsafe_allow_html=True,
                )


def show_sql_trace(trace: dict) -> None:
    done, schema = trace["done"], trace["schema"]
    with st.expander("How this was answered"):
        steps, thinking, schema_tab, examples_tab, sql_tab = st.tabs(
            ["Steps and model", "Thinking", "Schema", "Examples", "SQL and result"]
        )
        with steps:
            st.markdown(f"**Question:** {trace['question']}")
            if trace["rewritten"]:
                st.markdown(f"**Understood as:** {trace['standalone']}")
            if done:
                _timeline(done)
            _models(trace)
        with thinking:
            _thinking(trace)
        with schema_tab:
            if not schema:
                st.caption("The schema was not read.")
            else:
                st.caption(f"The model was given {len(schema.tables)} table(s), {schema.chars:,} characters: {', '.join(schema.tables) or 'none'}.")
                if schema.text:
                    st.code(schema.text, language="text")
        with examples_tab:
            if trace["examples"] is None:
                st.caption("No good answers are saved for this schema, so none were looked for.")
            elif not trace["examples"]:
                st.caption("Good answers are saved for this schema, but none was similar enough to this question.")
            else:
                st.caption("Good answers to similar questions, shown to the model with the schema. They may not fit.")
                for e in trace["examples"]:
                    st.markdown(f"**{e['question']}** · similarity {e['score']:.2f}")
                    st.code(e["sql"], language="sql")
        with sql_tab:
            if not trace["sql_attempts"]:
                st.caption("No query was written.")
            for a in trace["sql_attempts"]:
                st.markdown(f"**Attempt {a['attempt']}**")
                st.code(a["written"], language="sql")
                checked = a["checked"]
                if checked is not None and checked.ok:
                    if checked.sql != a["written"]:
                        st.caption("What ran, after the check wrote it out in full:")
                        st.code(checked.sql, language="sql")
                elif checked is not None:
                    st.error(f"Refused: {checked.reason}")
                if error := _database_error(a):
                    st.error(f"The database said: {error}")
                ran = a["ran"]
                if ran is not None:
                    shown = len(ran.rows)
                    st.caption(
                        f"{ran.row_count} row(s) in {ran.ms:.0f} ms"
                        + (" (cut at the row limit)" if ran.truncated else "")
                        + (f"; the first {shown} are shown" if shown < ran.row_count else "")
                    )
                    st.dataframe(pd.DataFrame(ran.rows, columns=ran.columns), hide_index=True)
