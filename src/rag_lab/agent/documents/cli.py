"""The command line for the documents flow: python -m rag_lab.agent documents ["question"] --experiment <name>

Answers from the experiment's documents with hybrid search and reranking, printing every step as it
happens: the query, the retrieved chunks, the model's thinking, the answer and the model's state. Each
turn is saved to `chat_turns`. Without a question it reads questions from the prompt (an empty line
ends) and keeps the conversation, so follow-ups work.
The models and the other settings are those of config/llm.yaml (`chat`, `documents_chat`, `reranker`);
the options here change one for this run. The experiment must have been made with `index.sparse`."""

import sys

from rag_lab import clients
from rag_lab.agent.documents.graph import DocumentsFlow, build_graph
from rag_lab.agent.printer import chat_loop
from rag_lab.search import load_experiment
from rag_lab.settings import load


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
    cfg = load().documents_chat.model_copy(update=overrides)

    metrics = clients.metrics()
    try:
        _, experiment = load_experiment(metrics, args.experiment)
    except ValueError as e:
        sys.exit(str(e))
    graph = build_graph(
        experiment, clients.embedder(), clients.qdrant(), clients.reranker(), clients.ollama_url(), cfg
    )
    chat_loop(graph, DocumentsFlow(experiment, cfg), metrics, args.question)
