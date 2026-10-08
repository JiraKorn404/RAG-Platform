"""CLI: python -m rag_lab.sql schema --schema <name>
                             | examples [--schema <name>] [--reindex] [--remove <id>]
                             | drop --schema <name> [--table <name>]

`schema` prints exactly what the text-to-SQL agent is given about a schema, with its size against the
budget. `examples` lists the good answers saved for a schema (every schema without `--schema`); with
`--reindex` it rebuilds their Qdrant collections from the table, and with `--remove` it deletes one.
`drop` deletes a table, or a whole schema with its chats and good answers; the CSV files in data/tables
stay. The connections and the settings are read from config/ (inside the containers this just works)."""

import argparse
import sys

from rag_lab import clients
from rag_lab.settings import load
from rag_lab.sql import catalog, examples
from rag_lab.sql.database import drop_schema, drop_table


def schema(args) -> None:
    try:
        text = catalog.render(clients.metrics(), args.schema)
    except ValueError as e:
        sys.exit(str(e))
    budget = load().database_chat.schema_char_budget
    print(text.text)
    over = " - over the budget: the agent would refuse this schema" if text.chars > budget else ""
    print(f"\n{text.chars:,} of {budget:,} characters, {len(text.tables)} table(s){over}")


def list_examples(args) -> None:
    metrics = clients.metrics()
    if args.remove is not None:
        found = next((e for e in metrics.list_sql_examples() if e["id"] == args.remove), None)
        if found is None:
            sys.exit(f"There is no example #{args.remove}.")
        examples.remove(metrics, clients.qdrant(), found["schema_name"], found["id"])
        print(f"removed #{found['id']} ({found['schema_name']}): {found['standalone']}")
        return
    if args.reindex:
        done = examples.reindex(metrics, clients.embedder(), clients.qdrant(), load().database_chat, args.schema)
        print("rebuilt: " + (", ".join(f"{name} ({n})" for name, n in done.items()) or "nothing to do"))
        return
    found = metrics.list_sql_examples(args.schema)
    for e in found:
        first_line = " ".join(e["sql"].split())[:90]
        print(f"#{e['id']} {e['schema_name']} {e['created_at']:%Y-%m-%d %H:%M}  {e['standalone']}\n      {first_line}")
    print(f"{len(found)} example(s)")


def drop(args) -> None:
    metrics = clients.metrics()
    if not metrics.get_db_schema(args.schema):
        sys.exit(f"There is no schema '{args.schema}'.")
    if args.table:
        if not metrics.get_db_table(args.schema, args.table):
            sys.exit(f"There is no table {args.schema}.{args.table}.")
        drop_table(metrics, args.schema, args.table)
        print(f"dropped the table {args.schema}.{args.table}")
    else:
        drop_schema(metrics, args.schema, clients.qdrant())
        print(f"dropped the schema {args.schema}, with its tables, chats and good answers")
    print(
        f"The CSV files in data/tables/{args.schema}/ were not touched. A file is imported again when it "
        "changes, or when its partition is started by hand in Dagster (tables_job)."
    )


def main() -> None:
    parser = argparse.ArgumentParser(prog="python -m rag_lab.sql", description=__doc__.split("\n")[0])
    commands = parser.add_subparsers(dest="command", required=True)
    show = commands.add_parser("schema", help="print what the model is given about a schema")
    show.add_argument("--schema", required=True)
    show.set_defaults(main=schema)
    saved = commands.add_parser("examples", help="list the good answers saved for a schema, or rebuild their index")
    saved.add_argument("--schema")
    saved.add_argument("--reindex", action="store_true", help="rebuild the Qdrant collection from the table")
    saved.add_argument("--remove", type=int, metavar="ID", help="delete one example")
    saved.set_defaults(main=list_examples)
    dropped = commands.add_parser("drop", help="delete a table, or a whole schema")
    dropped.add_argument("--schema", required=True)
    dropped.add_argument("--table", help="only this table; without it the whole schema goes")
    dropped.set_defaults(main=drop)
    args = parser.parse_args()
    args.main(args)


if __name__ == "__main__":
    main()
