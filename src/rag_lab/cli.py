"""The command line: python -m rag_lab <command>

    search "query" --experiment <name> [--top-k 5] [--modality table] [--json] [--no-log]
    chat documents ["question"] --experiment <name> [--top-k 5] [--model <name>] [--no-think] [--no-pictures]
    chat database ["question"] --schema <name> [--model <name>] [--no-think] [--no-examples]
    tables schema --schema <name>
    tables examples [--schema <name>] [--reindex] [--remove <id>]
    tables drop --schema <name> [--table <name>]
    tables register
    eval --experiment <name> --label <label> [--against <label>]
    setup

`search` is a dense search of a collection. `chat` answers from an experiment's documents or a schema's
tables and prints every step as it happens; without a question it reads questions from the prompt (an
empty line ends) and keeps the conversation. Each turn is saved. The models and the other settings are
those of config/llm.yaml; the options change one for this run.
`tables schema` prints exactly what the text-to-SQL agent is given about a schema, with its size against
the budget. `tables examples` lists the good answers saved for a schema; with `--reindex` it rebuilds
their Qdrant collections from the table, and with `--remove` it deletes one. `tables drop` deletes a
table or a whole schema with its chats and good answers (the CSV files stay, and so do the tables of a
registered schema). `tables register` describes the tables that are already in the database
(ingest/register.py). `eval` asks the documents chatbot every case of data/eval/<experiment>.jsonl and
prints how it did, next to an earlier run with `--against` (serve/evaluate.py). `setup` checks the
databases and makes our tables (core/setup.py).
The connections and the settings are read from config/ (inside the containers this just works)."""

import argparse
import json
import sys
import uuid
from dataclasses import asdict

from rag_lab.core import connections
from rag_lab.core.database import drop_schema, drop_table
from rag_lab.core.events import (
    AnswerToken,
    Done,
    Event,
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
    StepFinished,
    StepStarted,
    Thinking,
)
from rag_lab.core.settings import load
from rag_lab.core.store import MetricsStore
from rag_lab.serve import catalog, evaluate, examples
from rag_lab.serve.chat_database import SqlFlow, remembered
from rag_lab.serve.chat_database import build_graph as build_database_graph
from rag_lab.serve.chat_documents import DocumentsFlow
from rag_lab.serve.chat_documents import build_graph as build_documents_graph
from rag_lab.serve.run import Flow, run
from rag_lab.serve.search import OllamaReranker, load_experiment, search

DIM, RESET = "\033[2m", "\033[0m"


class Printer:
    """Prints events as they arrive; thinking is dimmed and the answer follows it."""

    def __init__(self) -> None:
        self.streaming = ""  # "thinking" or "answer" while a stream is being printed

    def _stream(self, kind: str, text: str) -> None:
        if self.streaming != kind:
            self._end_stream()
            print(f"{DIM}thinking:{RESET}\n{DIM}" if kind == "thinking" else "answer:")
            self.streaming = kind
        print(text, end="", flush=True)

    def _end_stream(self) -> None:
        if self.streaming:
            print(RESET if self.streaming == "thinking" else "")
            self.streaming = ""

    def __call__(self, event: Event) -> None:
        if isinstance(event, Thinking):
            return self._stream("thinking", event.text)
        if isinstance(event, AnswerToken):
            return self._stream("answer", event.text)
        self._end_stream()
        if isinstance(event, StepStarted):
            print(f"\n-> {event.node}", flush=True)
        elif isinstance(event, StepFinished):
            print(f"   {event.node} took {event.ms:.0f} ms")
        elif isinstance(event, Query):
            print(f"   query{' (rewritten)' if event.rewritten else ''}: {event.text}")
        elif isinstance(event, Graded):
            verdict = "the chunks answer it" if event.enough else "the chunks do not answer it"
            print(f"   {verdict} (best score {event.best_score:.3f}, decided by the {event.by})")
        elif isinstance(event, Rewrote):
            print(f"   trying a different query: {event.query}")
        elif isinstance(event, SchemaShown):
            print(f"   the model is given {len(event.tables)} table(s), {event.chars:,} characters: {', '.join(event.tables)}")
        elif isinstance(event, ExamplesFound):
            shown = "".join(f"\n      {e['score']:.2f}  {e['question']}" for e in event.examples)
            print(f"   good answers shown to the model: {len(event.examples)}{shown}")
        elif isinstance(event, SqlWritten):
            print(f"   SQL, attempt {event.attempt}:")
            print("      " + event.sql.replace("\n", "\n      "))
        elif isinstance(event, SqlChecked):
            print("   checked: ok" if event.ok else f"   checked: refused - {event.reason}")
            if event.ok:
                print("      runs as: " + event.sql.replace("\n", " "))
        elif isinstance(event, SqlRan):
            cut = " (cut at the row limit)" if event.truncated else ""
            print(f"   ran in {event.ms:.0f} ms: {event.row_count} row(s){cut}")
            print("      " + " | ".join(event.columns))
            for row in event.rows[:10]:
                print("      " + " | ".join("NULL" if v is None else str(v) for v in row))
            if event.row_count > 10:
                print(f"      ... {event.row_count - 10} more")
        elif isinstance(event, Repairing):
            print(f"   repairing (attempt {event.attempt}): {event.error}")
        elif isinstance(event, Retrieved):
            print(
                f"   {event.method}, {event.candidates} candidates: embed {event.embed_ms:.0f} ms, "
                f"search {event.search_ms:.0f} ms, rerank {event.rerank_ms:.0f} ms"
            )
            for number, hit in enumerate(event.hits, start=1):
                where = f"{hit.source_file or hit.doc_id} p.{hit.page}" if hit.page else (hit.source_file or hit.doc_id)
                heading = f" > {' > '.join(hit.headings)}" if hit.headings else ""
                print(f"   [{number}] score {hit.similarity:.3f} [{hit.modality}] {where}{heading}")
        elif isinstance(event, ModelState):
            speed = f", {event.tokens_per_s:.1f} tok/s" if event.tokens_per_s else ""
            loaded = {True: "was loaded", False: "was cold", None: "load state unknown"}[event.loaded]
            print(
                f"   model {event.model} ({loaded}), thinking {'on' if event.think else 'off'}: "
                f"{event.prompt_tokens} of {event.num_ctx} context in, {event.output_tokens} out{speed}"
            )
            if event.context_full:
                print("   WARNING: the prompt filled the context window, so Ollama cut it")
        elif isinstance(event, Done):
            if event.outcome:  # a documents turn says how it went
                why = f" ({event.abstain_reason})" if event.abstain_reason else ""
                score = f", best score {event.top_score:.3f}" if event.top_score is not None else ""
                print(f"\noutcome: {event.outcome}{why}, {event.rewrites} rewrite(s){score}", end="")
            elif event.abstained:
                print("\nabstained: no answer was found", end="")
            print(f"\ncited: {event.cited or 'nothing'}", end="")
            print(f"; no such passage: {event.unknown_citations}" if event.unknown_citations else "")
            steps = ", ".join(f"{node} {ms:.0f}" for node, ms in event.timings.items())
            print(f"total {event.total_ms:.0f} ms ({steps})")
            if not event.saved:
                print(f"WARNING: this turn was not saved: {event.save_error}")


def chat_loop(graph, flow: Flow, metrics: MetricsStore, question: str | None, remember=None) -> None:
    """Answer `question` and stop, or, without one, read questions from the prompt (an empty line ends)
    and keep the conversation. `remember(answer, events)` gives what an answer leaves in the history when
    that is more than the answer."""
    session_id = uuid.uuid4().hex[:12]
    show = Printer()
    history: list[tuple[str, str]] = []
    once = bool(question)
    while True:
        if question is None:
            question = input("> ").strip()
            if not question:
                return
        events = []
        try:
            for event in run(graph, flow, question, history, metrics=metrics, session_id=session_id):
                show(event)
                events.append(event)
        except ValueError as e:  # for example an experiment without the BM25 vector, or a schema deleted meanwhile
            sys.exit(str(e))
        if once:
            return
        answer = events[-1].answer  # the last event is Done
        history += [("User", question), ("Assistant", remember(answer, events) if remember else answer)]
        question = None


# --- search ------------------------------------------------------------------------------------------


def search_command(args) -> None:
    metrics = connections.metrics()
    try:
        config_hash, config = load_experiment(metrics, args.experiment)
        result = search(
            args.query,
            config,
            connections.embedder(),
            connections.qdrant(),
            top_k=args.top_k,
            filters={"modality": args.modality} if args.modality else None,
        )
    except ValueError as e:
        sys.exit(str(e))

    if not args.no_log:
        metrics.add_search_log(
            config_hash,
            args.query,
            args.top_k,
            result.embed_ms,
            result.search_ms,
            result.total_ms,
            result.hits[0].similarity if result.hits else None,
        )

    if args.json:
        print(json.dumps(asdict(result), indent=2, ensure_ascii=False))
        return
    for h in result.hits:
        where = f"{h.source_file or h.doc_id} p.{h.page}" if h.page else (h.source_file or h.doc_id)
        print(f"{h.rank}. similarity {h.similarity:.3f}  distance {h.distance:.3f}  [{h.modality}]  {where}")
        if h.headings:
            print(f"   {' > '.join(h.headings)}")
        text = " ".join(h.text.split())
        print(f"   {text[:200]}{'...' if len(text) > 200 else ''}")
    if not result.hits:
        print("No hits.")
    print(
        f"embed {result.embed_ms:.0f} ms, search {result.search_ms:.0f} ms, "
        f"total {result.total_ms:.0f} ms"
    )


# --- chat --------------------------------------------------------------------------------------------


def chat_documents(args) -> None:
    overrides = {
        **({"top_k": args.top_k} if args.top_k else {}),
        **({"model": args.model} if args.model else {}),
        **({"think": False} if args.no_think else {}),
        **({"show_pictures": False} if args.no_pictures else {}),
    }
    cfg = load().documents_chat.model_copy(update=overrides)

    metrics = connections.metrics()
    try:
        _, experiment = load_experiment(metrics, args.experiment)
    except ValueError as e:
        sys.exit(str(e))
    base_url = connections.ollama_url()
    graph = build_documents_graph(
        experiment, connections.embedder(), connections.qdrant(), OllamaReranker(base_url), base_url, cfg
    )
    chat_loop(graph, DocumentsFlow(experiment, cfg), metrics, args.question)


def _remember(answer: str, events: list[Event]) -> str:
    sql = next((e.sql for e in reversed(events) if isinstance(e, SqlChecked) and e.ok), "")
    return remembered(answer, sql)


def chat_database(args) -> None:
    overrides = {
        **({"model": args.model} if args.model else {}),
        **({"think": False} if args.no_think else {}),
        **({"use_examples": False} if args.no_examples else {}),
    }
    cfg = load().database_chat.model_copy(update=overrides)
    metrics = connections.metrics()
    if not metrics.get_db_schema(args.schema):
        sys.exit(f"There is no schema '{args.schema}'.")
    graph = build_database_graph(
        metrics, args.schema, connections.ollama_url(), cfg, connections.embedder(), connections.qdrant()
    )
    chat_loop(graph, SqlFlow(args.schema, cfg), metrics, args.question, _remember)


# --- tables ------------------------------------------------------------------------------------------


def tables_schema(args) -> None:
    try:
        text = catalog.render(connections.metrics(), args.schema)
    except ValueError as e:
        sys.exit(str(e))
    budget = load().database_chat.schema_char_budget
    print(text.text)
    over = " - over the budget: the agent would refuse this schema" if text.chars > budget else ""
    print(f"\n{text.chars:,} of {budget:,} characters, {len(text.tables)} table(s){over}")


def tables_examples(args) -> None:
    metrics = connections.metrics()
    if args.remove is not None:
        found = next((e for e in metrics.list_sql_examples() if e["id"] == args.remove), None)
        if found is None:
            sys.exit(f"There is no example #{args.remove}.")
        examples.remove(metrics, connections.qdrant(), found["schema_name"], found["id"])
        print(f"removed #{found['id']} ({found['schema_name']}): {found['standalone']}")
        return
    if args.reindex:
        done = examples.reindex(metrics, connections.embedder(), connections.qdrant(), load().database_chat, args.schema)
        print("rebuilt: " + (", ".join(f"{name} ({n})" for name, n in done.items()) or "nothing to do"))
        return
    found = metrics.list_sql_examples(args.schema)
    for e in found:
        first_line = " ".join(e["sql"].split())[:90]
        print(f"#{e['id']} {e['schema_name']} {e['created_at']:%Y-%m-%d %H:%M}  {e['standalone']}\n      {first_line}")
    print(f"{len(found)} example(s)")


def tables_drop(args) -> None:
    metrics = connections.metrics()
    found = metrics.get_db_schema(args.schema)
    if not found:
        sys.exit(f"There is no schema '{args.schema}'.")
    if args.table:
        if not metrics.get_db_table(args.schema, args.table):
            sys.exit(f"There is no table {args.schema}.{args.table}.")
        drop_table(metrics, args.schema, args.table)
        print(f"dropped the table {args.schema}.{args.table}")
    else:
        drop_schema(metrics, args.schema, connections.qdrant())
        print(f"dropped the schema {args.schema}, with its tables, chats and good answers")
    if found["registered"]:
        print(
            "It was a registered schema, so only what the chatbot knew about it went: its tables in the database "
            "were not touched. It is described again at the next `tables register` while it is still listed in "
            "`existing_schemas` (config/connections.yaml)."
        )
    else:
        print(
            f"The CSV files in data/tables/{args.schema}/ were not touched. A file is imported again when it "
            "changes, or when its partition is started by hand in Dagster (tables_job)."
        )


def tables_register(args) -> None:
    from rag_lab.ingest.importer import CsvError  # the ingestion side: only this command needs it
    from rag_lab.ingest.register import register

    try:
        done = register(connections.metrics(), connections.qdrant(), load().tables)
    except (CsvError, RuntimeError) as e:
        sys.exit(str(e))
    for name, tables in done["schemas"].items():
        print(f"{name}: {len(tables)} table(s): {', '.join(tables) or 'none'}")
    for name in done["removed"]:
        print(f"{name}: no longer listed, removed from the registry")
    for warning in done["warnings"]:
        print(f"WARNING: {warning}")
    if not (done["schemas"] or done["removed"]):
        print("Nothing to do: `tables_database.existing_schemas` in config/connections.yaml names no schema.")


def eval_command(args) -> None:
    earlier = None
    if args.against:
        path = evaluate.results_path(args.experiment, args.against)
        if not path.exists():
            sys.exit(f"There is no run labelled '{args.against}': {path} does not exist.")
        earlier = json.loads(path.read_text(encoding="utf-8"))
    try:
        results = evaluate.evaluate(args.experiment, args.label, progress=lambda line: print(line, flush=True))
    except ValueError as e:
        sys.exit(str(e))
    print("\n" + evaluate.report(results, earlier))
    print(f"\nwritten to {evaluate.results_path(args.experiment, args.label)}")


def setup_command(args) -> None:
    from rag_lab.core.setup import main as setup

    setup()


def main() -> None:
    parser = argparse.ArgumentParser(prog="python -m rag_lab", description="Search, chat, look after the tables.")
    commands = parser.add_subparsers(dest="command", required=True)

    find = commands.add_parser("search", help="a dense search of a collection")
    find.add_argument("query")
    find.add_argument("--experiment", required=True)
    find.add_argument("--top-k", type=int, default=5)
    find.add_argument("--modality", choices=["text", "table", "picture"])
    find.add_argument("--json", action="store_true", help="print the result as JSON")
    find.add_argument("--no-log", action="store_true", help="do not write to search_log")
    find.set_defaults(main=search_command)

    flows = commands.add_parser("chat", help="ask questions, with every step printed").add_subparsers(
        dest="flow", required=True, metavar="flow"
    )
    documents = flows.add_parser("documents", help="ask questions of the documents in an experiment")
    documents.add_argument("question", nargs="?")
    documents.add_argument("--experiment", required=True)
    documents.add_argument("--top-k", type=int)
    documents.add_argument("--model")
    documents.add_argument("--no-think", action="store_true", help="write the answer without thinking")
    documents.add_argument(
        "--no-pictures", action="store_true", help="give the model the caption of a picture only, not the image"
    )
    documents.set_defaults(main=chat_documents)
    database = flows.add_parser("database", help="ask questions of the tables in a schema")
    database.add_argument("question", nargs="?")
    database.add_argument("--schema", required=True)
    database.add_argument("--model")
    database.add_argument("--no-think", action="store_true", help="write the SQL without thinking")
    database.add_argument("--no-examples", action="store_true", help="do not show the model good answers to similar questions")
    database.set_defaults(main=chat_database)

    tables = commands.add_parser("tables", help="the schemas the SQL chatbot answers from").add_subparsers(
        dest="action", required=True, metavar="action"
    )
    show = tables.add_parser("schema", help="print what the model is given about a schema")
    show.add_argument("--schema", required=True)
    show.set_defaults(main=tables_schema)
    saved = tables.add_parser("examples", help="list the good answers saved for a schema, or rebuild their index")
    saved.add_argument("--schema")
    saved.add_argument("--reindex", action="store_true", help="rebuild the Qdrant collection from the table")
    saved.add_argument("--remove", type=int, metavar="ID", help="delete one example")
    saved.set_defaults(main=tables_examples)
    dropped = tables.add_parser("drop", help="delete a table, or a whole schema")
    dropped.add_argument("--schema", required=True)
    dropped.add_argument("--table", help="only this table; without it the whole schema goes")
    dropped.set_defaults(main=tables_drop)
    tables.add_parser("register", help="describe the tables that are already in the database").set_defaults(
        main=tables_register
    )

    measured = commands.add_parser("eval", help="ask the documents chatbot every case of an eval set")
    measured.add_argument("--experiment", required=True)
    measured.add_argument("--label", required=True, help="the name of this run, for example phase0")
    measured.add_argument("--against", metavar="LABEL", help="an earlier run to show the numbers next to")
    measured.set_defaults(main=eval_command)

    commands.add_parser("setup", help="check the databases and make our tables").set_defaults(main=setup_command)

    args = parser.parse_args()
    args.main(args)


if __name__ == "__main__":
    main()
