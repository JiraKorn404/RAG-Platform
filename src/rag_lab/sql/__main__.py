"""CLI: python -m rag_lab.sql schema --schema <name> | examples [--schema <name>] [--reindex]

`schema` prints exactly what the text-to-SQL agent is given about a schema, with its size against the
budget. `examples` lists the good answers saved for a schema (every schema without `--schema`); with
`--reindex` it rebuilds their Qdrant collections from the table (needs OLLAMA_BASE_URL and QDRANT_URL).
Both need METRICS_DATABASE_URL in the environment (already set inside the containers)."""

import argparse
import os
import sys

from rag_lab.config import SqlAgentConfig
from rag_lab.embedding.ollama import OllamaEmbedder
from rag_lab.metrics.store import MetricsStore
from rag_lab.sql import catalog, examples
from rag_lab.storage.qdrant import QdrantStore


def schema(args) -> None:
    metrics = MetricsStore(os.environ["METRICS_DATABASE_URL"])
    try:
        text = catalog.render(metrics, args.schema)
    except ValueError as e:
        sys.exit(str(e))
    budget = SqlAgentConfig().schema_char_budget
    print(text.text)
    over = " - over the budget: the agent would refuse this schema" if text.chars > budget else ""
    print(f"\n{text.chars:,} of {budget:,} characters, {len(text.tables)} table(s){over}")


def list_examples(args) -> None:
    metrics = MetricsStore(os.environ["METRICS_DATABASE_URL"])
    if args.reindex:
        base_url, qdrant = os.environ["OLLAMA_BASE_URL"], QdrantStore(os.environ["QDRANT_URL"])
        done = examples.reindex(metrics, OllamaEmbedder(base_url), qdrant, SqlAgentConfig(), args.schema)
        print("rebuilt: " + (", ".join(f"{name} ({n})" for name, n in done.items()) or "nothing to do"))
        return
    found = metrics.list_sql_examples(args.schema)
    for e in found:
        first_line = " ".join(e["sql"].split())[:90]
        print(f"#{e['id']} {e['schema_name']} {e['created_at']:%Y-%m-%d %H:%M}  {e['standalone']}\n      {first_line}")
    print(f"{len(found)} example(s)")


def main() -> None:
    parser = argparse.ArgumentParser(prog="python -m rag_lab.sql", description=__doc__.split("\n")[0])
    commands = parser.add_subparsers(dest="command", required=True)
    show = commands.add_parser("schema", help="print what the model is given about a schema")
    show.add_argument("--schema", required=True)
    show.set_defaults(main=schema)
    saved = commands.add_parser("examples", help="list the good answers saved for a schema, or rebuild their index")
    saved.add_argument("--schema")
    saved.add_argument("--reindex", action="store_true", help="rebuild the Qdrant collection from the table")
    saved.set_defaults(main=list_examples)
    args = parser.parse_args()
    args.main(args)


if __name__ == "__main__":
    main()
