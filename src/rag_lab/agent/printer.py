"""What the command-line interfaces of every flow share: the printing of a turn's events as they arrive,
and the chat loop."""

import sys
import uuid

from rag_lab.agent.events import (
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
from rag_lab.agent.run import Flow, run
from rag_lab.metrics.store import MetricsStore

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
            if event.abstained:
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
