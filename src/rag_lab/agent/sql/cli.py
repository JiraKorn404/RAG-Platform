"""The command line for the database flow: python -m rag_lab.agent sql ["question"] --schema <name>

Answers questions about the tables of an imported schema: the model writes a SELECT, a guard and the
database check it, it runs as the read-only role, and the answer is written from its rows. Every step is
printed as it happens, and each turn is saved to `chat_turns`. Without a question it reads questions from
the prompt (an empty line ends) and keeps the conversation, so follow-ups work.
The model and the other settings are those of config/llm.yaml (`chat`, `database_chat`); the options here
change one for this run. Good answers saved for the schema are shown to the model when a similar question
is asked; `--no-examples` leaves them out."""

import sys

from rag_lab import clients
from rag_lab.agent.events import Event, SqlChecked
from rag_lab.agent.printer import chat_loop
from rag_lab.agent.sql import SqlFlow, build_graph, remembered
from rag_lab.settings import load


def _remember(answer: str, events: list[Event]) -> str:
    sql = next((e.sql for e in reversed(events) if isinstance(e, SqlChecked) and e.ok), "")
    return remembered(answer, sql)


def add_parser(subparsers) -> None:
    parser = subparsers.add_parser("sql", help="ask questions of the tables in a schema")
    parser.add_argument("question", nargs="?")
    parser.add_argument("--schema", required=True)
    parser.add_argument("--model")
    parser.add_argument("--no-think", action="store_true", help="write the SQL without thinking")
    parser.add_argument("--no-examples", action="store_true", help="do not show the model good answers to similar questions")
    parser.set_defaults(main=main)


def main(args) -> None:
    overrides = {
        **({"model": args.model} if args.model else {}),
        **({"think": False} if args.no_think else {}),
        **({"use_examples": False} if args.no_examples else {}),
    }
    cfg = load().database_chat.model_copy(update=overrides)
    metrics = clients.metrics()
    if not metrics.get_db_schema(args.schema):
        sys.exit(f"There is no schema '{args.schema}'.")
    graph = build_graph(metrics, args.schema, clients.ollama_url(), cfg, clients.embedder(), clients.qdrant())
    chat_loop(graph, SqlFlow(args.schema, cfg), metrics, args.question, _remember)
