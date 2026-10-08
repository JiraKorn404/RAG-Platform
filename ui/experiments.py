"""What is in each experiment, with delete buttons for a document and for a whole experiment.

The work is behind the API (serve/library.py); this page lists, asks for confirmation inline, and redraws."""

import data
import streamlit as st
import style


def _title(entry: dict) -> str:
    config = entry["config"]
    parts = [entry["name"]]
    if config:
        parts += [data.model_label(config["embed"]["model"]), config["chunk"]["strategy"]]
    parts.append(f"{len(entry['documents'])} document(s)")
    parts.append(f"{entry['points']} points")
    flags = []
    if not entry["has_row"]:
        flags.append("no experiment row")
    if not entry["has_collection"]:
        flags.append("no collection")
    return " · ".join(parts) + (f"  ({', '.join(flags)})" if flags else "")


def _ask(kind: str, name: str, doc_id: str | None = None) -> None:
    st.session_state["confirm_delete"] = {"kind": kind, "name": name, "doc_id": doc_id}


def _clear_ask() -> None:
    st.session_state.pop("confirm_delete", None)


def _confirm(entry: dict, pending: dict) -> None:
    """The inline 'are you sure' for this entry, and the delete itself."""
    name = entry["name"]
    if pending["kind"] == "doc":
        doc = next(d for d in entry["documents"] if d["doc_id"] == pending["doc_id"])
        label = doc["source_file"] or doc["doc_id"]
        st.warning(
            f"Remove “{label}” from “{name}”? This deletes its {doc['points']} points and its files in this "
            "experiment. The PDF in data/raw stays, and Dagster still counts it as ingested: to put it back, "
            "run its partition of `ingest_job` in Dagster. This cannot be undone."
        )
    elif entry["has_row"]:
        st.warning(
            f"Delete the experiment “{name}”? This deletes its collection ({entry['points']} points in "
            f"{len(entry['documents'])} document(s)), its files and its records. This cannot be undone."
        )
    else:
        st.warning(f"Delete the collection “{name}”? It has no experiment record. This cannot be undone.")
    yes, no, _ = st.columns([1, 1, 6])
    if no.button("Cancel", key=f"cancel-{name}"):
        _clear_ask()
        st.rerun()
    if yes.button("Delete", type="primary", key=f"yes-{name}"):
        try:
            if pending["kind"] == "doc":
                done = data.delete(f"/collections/{name}/documents/{pending['doc_id']}")
                message = (
                    f"Removed “{label}” from “{name}”: {done['points']} points, {done['files']} files, "
                    f"{done['rows']} metric rows."
                )
            else:
                done = data.delete(f"/collections/{name}")
                message = (
                    f"Deleted “{name}”: collection {'yes' if done['collection'] else 'was missing'}, "
                    f"folder {'yes' if done['folder'] else 'was missing'}, record {'yes' if done['row'] else 'was missing'}."
                )
        except data.ApiError as e:  # Qdrant or Postgres not reachable: show it, change nothing else
            st.error(f"The delete failed: {e}. Nothing after the failed step was changed; press Delete again to retry.")
            return
        _clear_ask()
        st.cache_data.clear()  # the Chatbot caches its experiment list
        st.session_state["experiments_message"] = message
        st.rerun()


def _render(entry: dict, pending: dict | None) -> None:
    name = entry["name"]
    mine = pending is not None and pending["name"] == name
    with st.expander(_title(entry), expanded=mine):
        if entry["created_at"]:
            st.caption(f"Created {data.when(entry['created_at']):%Y-%m-%d %H:%M}")
        if not entry["has_collection"]:
            st.caption("This experiment has no Qdrant collection, so it cannot be searched.")
        for doc in entry["documents"]:
            name_col, id_col, points_col, when_col, button_col = st.columns([4, 2, 1, 2, 1])
            name_col.markdown(f"**{doc['source_file'] or 'unknown file'}**")
            id_col.code(doc["doc_id"], language=None)
            points_col.caption(f"{doc['points']} points")
            when_col.caption((doc["ingested_at"] or "")[:16].replace("T", " "))
            button_col.button("Delete", key=f"doc-{name}-{doc['doc_id']}", on_click=_ask, args=("doc", name, doc["doc_id"]))
        if entry["has_collection"] and not entry["documents"]:
            st.caption("The collection holds no documents.")
        st.button(
            "Delete experiment" if entry["has_row"] else "Delete collection",
            key=f"exp-{name}",
            on_click=_ask,
            args=("exp", name),
        )
        if mine:
            _confirm(entry, pending)


style.hero("Experiments", "The collections of the vector database: what is in each, and delete what you no longer need")
st.caption(
    "An experiment is a Qdrant collection with the settings it was made with. A PDF put into `data/raw/` is "
    "ingested by Dagster into the experiment named in `config/pipeline.yaml`."
)

message = st.session_state.pop("experiments_message", None)
if message:
    st.success(message)

try:
    entries = data.get("/library")
except data.ApiError as e:
    st.error(str(e))
    st.stop()
if not entries:
    st.info("No experiments yet.")
    st.stop()

with_row = [e for e in entries if e["has_row"]]
without_row = [e for e in entries if not e["has_row"]]
total_docs = sum(len(e["documents"]) for e in entries)
st.caption(
    f"{len(with_row)} experiment(s), {total_docs} document(s) and {sum(e['points'] for e in entries)} points in "
    f"Qdrant." + (f" {len(without_row)} collection(s) have no experiment record." if without_row else "")
)
if st.button("Refresh"):
    st.rerun()

pending = st.session_state.get("confirm_delete")
if pending and not any(e["name"] == pending["name"] for e in entries):
    _clear_ask()  # what was being deleted is already gone
    pending = None
for entry in with_row:
    _render(entry, pending)
if without_row:
    style.section("Collections without an experiment record", "Leftovers in Qdrant that nothing in the app knows about.")
    for entry in without_row:
        _render(entry, pending)
