"""The eval of the documents chatbot: a file of cases, each asked of the real graph, and the numbers that
are compared from one change of the flow to the next (PLAN.md).

    data/eval/<experiment>.jsonl                   the cases, one per line
    data/eval/results/<experiment>-<label>.json    every case as it went, and the summary

A case has `id`, `question`, `history` (pairs of role and text, for a follow-up), `expect` (`answer`,
`abstain` or `direct`), `facts` (strings the answer must contain) and `source` (`file` and `snippets`:
words copied from the passage that holds the answer). A case runs with the settings of config/llm.yaml
and is not saved to the chats."""

import json
import math
import re
import statistics
from collections.abc import Callable
from pathlib import Path

from rag_lab.core import connections
from rag_lab.core.events import Done, Graded, ModelState, Query, Retrieved, Rewrote, StepStarted
from rag_lab.core.settings import DATA_DIR, load
from rag_lab.serve.chat_documents import DocumentsFlow, build_graph
from rag_lab.serve.run import Flow, run
from rag_lab.serve.search import OllamaReranker, load_experiment

EVAL_DIR = DATA_DIR / "eval"

# An answer that was given but says the documents do not hold it. It counts as answered: this only
# finds them, so that they can be read.
_NOT_IN_DOCUMENTS = re.compile(r"\b(do|does|did) not (contain|mention|provide|include|cover|address)\b", re.IGNORECASE)


def squash(text: str) -> str:
    return " ".join(text.split())


def load_cases(experiment: str) -> list[dict]:
    path = EVAL_DIR / f"{experiment}.jsonl"
    if not path.exists():
        raise ValueError(f"There are no cases for '{experiment}': {path} does not exist.")
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def found_source(source: dict | None, hits: list) -> bool:
    """Whether a hit of the source's file contains one of its snippets."""
    if not source:
        return False
    snippets = [squash(s) for s in source["snippets"]]
    return any(
        hit.source_file == source["file"] and any(s in squash(hit.text) for s in snippets) for hit in hits
    )


def has_facts(facts: list[str], answer: str) -> bool:
    return all(fact.lower() in answer.lower() for fact in facts)


def run_case(graph, flow: Flow, case: dict) -> dict:
    """Ask one case and say how it went. A case that fails on the way has `error` and no outcome."""
    row = {key: case[key] for key in ("id", "expect", "question")} | {"tags": case.get("tags", [])}
    steps, searched, scores, hits, calls, rerank_ms, done = [], [], [], [], 0, 0.0, None
    try:
        for event in run(graph, flow, case["question"], [tuple(pair) for pair in case.get("history", [])]):
            if isinstance(event, StepStarted):
                steps.append(event.node)
            elif isinstance(event, ModelState):
                calls += 1
            elif isinstance(event, Query):
                searched.append(event.text)
            elif isinstance(event, Rewrote):
                searched.append(event.query)
            elif isinstance(event, Graded):
                scores.append(event.best_score)
            elif isinstance(event, Retrieved):
                hits = event.hits
                rerank_ms += event.rerank_ms
            elif isinstance(event, Done):
                done = event
    except Exception as e:  # noqa: BLE001  (one case that fails must not lose the others)
        return row | {"error": f"{steps[-1] if steps else 'starting'} failed: {e}"}

    outcome = done.outcome
    wanted = {"answer": "answered", "abstain": "abstained", "direct": "direct"}[case["expect"]]
    return row | {
        "route": done.route,
        "outcome": outcome,
        "abstain_reason": done.abstain_reason,
        "ok": outcome == wanted and has_facts(case.get("facts", []), done.answer),
        "found": found_source(case.get("source"), hits),
        "first_score": scores[0] if scores else None,  # what the thresholds are set from
        "top_score": done.top_score,
        "rewrites": done.rewrites,
        "searches": steps.count("retrieve"),
        "searched": searched,
        "rerank_ms": rerank_ms,
        "model_calls": calls,
        "total_ms": done.total_ms,
        "timings": done.timings,
        "says_not_in_documents": outcome == "answered" and bool(_NOT_IN_DOCUMENTS.search(done.answer)),
        "answer": done.answer,
        "cited": done.cited,
        # each kept hit: file, page, score, and whether it is a passage that holds the answer
        "hits": [
            [h.source_file, h.page, round(h.similarity, 4), found_source(case.get("source"), [h])] for h in hits
        ],
    }


def _p95(values: list[float]) -> float:
    return sorted(values)[math.ceil(0.95 * len(values)) - 1]


def summarise(rows: list[dict]) -> dict:
    """The numbers of a run. A share is kept as [how many, of how many], so a small set stays readable."""
    ran = [r for r in rows if "error" not in r]
    answerable = [r for r in ran if r["expect"] == "answer"]
    abstained = [r for r in ran if r["outcome"] == "abstained"]
    unanswerable = [r for r in ran if r["expect"] == "abstain"]
    knowledge = [r for r in ran if r["expect"] != "direct"]

    def share(of: list[dict], test: Callable[[dict], bool]) -> list[int]:
        return [sum(1 for r in of if test(r)), len(of)]

    nodes = sorted({node for r in ran for node in r["timings"]})
    seconds = [r["total_ms"] / 1000 for r in ran]
    return {
        "cases": len(rows),
        "failed": [r["id"] for r in rows if "error" in r],
        "hit_rate": share(answerable, lambda r: r["found"]),
        "correct": share(answerable, lambda r: r["ok"]),
        "abstain_precision": share(abstained, lambda r: r["expect"] == "abstain"),
        "abstain_recall": share(unanswerable, lambda r: r["outcome"] == "abstained"),
        "route_accuracy": share(ran, lambda r: r["route"] == ("direct" if r["expect"] == "direct" else "retrieve")),
        "knowledge_sent_direct": [r["id"] for r in knowledge if r["route"] == "direct"],
        "says_not_in_documents": [r["id"] for r in ran if r["says_not_in_documents"]],
        "median_s": statistics.median(seconds) if seconds else None,
        "p95_s": _p95(seconds) if seconds else None,
        "node_median_ms": {
            node: statistics.median(r["timings"][node] for r in ran if node in r["timings"]) for node in nodes
        },
        "model_calls_median": statistics.median(r["model_calls"] for r in ran) if ran else None,
        "searches_max": max((r["searches"] for r in ran), default=None),
    }


def results_path(experiment: str, label: str) -> Path:
    return EVAL_DIR / "results" / f"{experiment}-{label}.json"


def evaluate(experiment_name: str, label: str, progress: Callable[[str], None] = print) -> dict:
    """Run every case of an experiment and write the results. Returns {"summary", "cases", "settings"}."""
    cases = load_cases(experiment_name)
    cfg = load().documents_chat
    _, experiment = load_experiment(connections.metrics(), experiment_name)
    base_url = connections.ollama_url()
    graph = build_graph(
        experiment, connections.embedder(), connections.qdrant(), OllamaReranker(base_url), base_url, cfg
    )
    flow = DocumentsFlow(experiment, cfg)
    rows = []
    for number, case in enumerate(cases, start=1):
        row = run_case(graph, flow, case)
        rows.append(row)
        went = row.get("error") or f"{row['outcome']}, {'ok' if row['ok'] else 'NOT ok'}, {row['total_ms'] / 1000:.1f} s"
        progress(f"{number}/{len(cases)} {row['id']} ({row['expect']}): {went}")
    results = {"summary": summarise(rows), "settings": cfg.model_dump(mode="json"), "cases": rows}
    path = results_path(experiment_name, label)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    return results


def _shown(value) -> str:
    if isinstance(value, list) and len(value) == 2 and all(isinstance(v, int) for v in value):
        return f"{value[0]}/{value[1]}"
    if isinstance(value, float):
        return f"{value:.1f}"
    if isinstance(value, list):
        return ", ".join(value) or "none"
    return str(value)


def report(results: dict, earlier: dict | None = None) -> str:
    """The summary as text, with the numbers of an earlier run next to those that changed, then the cases."""
    summary, before = results["summary"], (earlier or {}).get("summary", {})
    lines = []
    for key, value in summary.items():
        if key == "node_median_ms":
            value = ", ".join(f"{node} {ms:.0f}" for node, ms in value.items())
            was = ", ".join(f"{node} {ms:.0f}" for node, ms in before.get(key, {}).items())
        else:
            value, was = _shown(value), _shown(before[key]) if key in before else ""
        lines.append(f"{key:<24}{value}" + (f"   (was {was})" if was and was != value else ""))
    lines.append("")
    lines.append(f"{'id':<5}{'expect':<9}{'outcome':<11}{'ok':<5}{'source':<8}{'score':<8}{'searches':<10}{'calls':<7}seconds")
    for r in results["cases"]:
        if "error" in r:
            lines.append(f"{r['id']:<5}{r['expect']:<9}{r['error']}")
            continue
        score = f"{r['first_score']:.3f}" if r["first_score"] is not None else "-"
        source = ("found" if r["found"] else "missed") if r["expect"] == "answer" else "-"
        lines.append(
            f"{r['id']:<5}{r['expect']:<9}{r['outcome']:<11}{'yes' if r['ok'] else 'NO':<5}{source:<8}{score:<8}"
            f"{r['searches']:<10}{r['model_calls']:<7}{r['total_ms'] / 1000:.1f}"
        )
    return "\n".join(lines)
