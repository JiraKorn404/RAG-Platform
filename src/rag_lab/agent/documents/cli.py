"""The command line for the documents flow: python -m rag_lab.agent documents ["question"] --experiment <name>

Answers from the experiment's documents with hybrid search and reranking, printing every step as it
happens: the query, the retrieved chunks, the model's thinking, the answer and the model's state. Each
turn is saved to `chat_turns`. Without a question it reads questions from the prompt (an empty line
ends) and keeps the conversation, so follow-ups work.
Needs OLLAMA_BASE_URL, QDRANT_URL and METRICS_DATABASE_URL in the environment (already set inside the
dagster-code container). The experiment must have been made with `index.sparse`."""

import sys

from rag_lab.agent.documents.graph import DocumentsFlow, build_graph
from rag_lab.agent.printer import chat_loop, env
from rag_lab.config import AgentConfig
from rag_lab.embedding.ollama import OllamaEmbedder
from rag_lab.metrics.store import MetricsStore
from rag_lab.reranking import OllamaReranker
from rag_lab.search import load_experiment
from rag_lab.storage.qdrant import QdrantStore


def add_parser(subparsers) -> None:
    parser = subparsers.add_parser("documents", help="ask questions of the documents in an experiment")
    parser.add_argument("question", nargs="?")
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--top-k", type=int)
    parser.add_argument("--model")
    parser.add_argument("--no-think", action="store_true", help="write the answer without thinking")
    parser.add_argument(
        "--no-pictures", action="store_true", help="give the model the caption of a picture only, not the image"
    )
    parser.set_defaults(main=main)


def main(args) -> None:
    overrides = {
        **({"top_k": args.top_k} if args.top_k else {}),
        **({"model": args.model} if args.model else {}),
        **({"think": False} if args.no_think else {}),
        **({"show_pictures": False} if args.no_pictures else {}),
    }
    cfg = AgentConfig(**overrides)

    base_url = env("OLLAMA_BASE_URL")
    database_url = env("METRICS_DATABASE_URL")
    metrics = MetricsStore(database_url)
    try:
        _, experiment = load_experiment(metrics, args.experiment)
    except ValueError as e:
        sys.exit(str(e))
    graph = build_graph(
        experiment,
        OllamaEmbedder(base_url),
        QdrantStore(env("QDRANT_URL")),
        OllamaReranker(base_url),
        base_url,
        cfg,
    )
    chat_loop(graph, DocumentsFlow(experiment, cfg), metrics, args.question)
