"""CLI: python -m rag_lab.search "query" --experiment <name> [--top-k 5] [--modality table] [--json] [--no-log]

A dense search. The connections are read from config/connections.yaml, and the experiment's settings
from the `experiments` table.
"""

import argparse
import json
import sys
from dataclasses import asdict

from rag_lab import clients
from rag_lab.search import load_experiment, search


def main() -> None:
    parser = argparse.ArgumentParser(prog="python -m rag_lab.search", description=__doc__.split("\n")[0])
    parser.add_argument("query")
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--modality", choices=["text", "table", "picture"])
    parser.add_argument("--json", action="store_true", help="print the result as JSON")
    parser.add_argument("--no-log", action="store_true", help="do not write to search_log")
    args = parser.parse_args()

    metrics = clients.metrics()
    try:
        config_hash, config = load_experiment(metrics, args.experiment)
        result = search(
            args.query,
            config,
            clients.embedder(),
            clients.qdrant(),
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


if __name__ == "__main__":
    main()
