"""The text-to-SQL agent as a LangGraph graph:

    condense -> schema -> examples -> write_sql -> check -> run_sql -> answer
                                         ^        |
                                         +- repair <-+   (a check or the database refused the query)
    schema too large, "CANNOT", or repairs used up -> abstain

The model is given the whole schema (agent/../sql/catalog.py), writes one query, and a guard and the
database check it before it runs as the read-only role. It thinks while it writes and repairs the SQL and
does not when it rewrites the question or words the answer. Every node reports what it does as events
(agent/events.py); `agent.run()` is the way to use the graph, with a `SqlFlow`."""

from dataclasses import asdict, dataclass
from typing import TypedDict

from langgraph.config import get_stream_writer
from langgraph.graph import END, START, StateGraph

from rag_lab.agent.events import (
    ROWS_KEPT,
    AnswerToken,
    Event,
    ExamplesFound,
    Repairing,
    SchemaShown,
    SqlChecked,
    SqlRan,
    SqlWritten,
)
from rag_lab.agent.model import call_model, chat_model, standalone_question
from rag_lab.agent.run import Summary, step
from rag_lab.agent.sql.prompts import (
    ANSWER_SYSTEM,
    CANNOT,
    CONDENSE_SYSTEM,
    WRITE_SYSTEM,
    answer_prompt,
    condense_prompt,
    extract_sql,
    not_answered,
    repair_prompt,
    write_prompt,
)
from rag_lab.config import SqlAgentConfig
from rag_lab.embedding.ollama import OllamaEmbedder
from rag_lab.metrics.store import MetricsStore
from rag_lab.sql import catalog, examples as good_answers, execute
from rag_lab.sql.catalog import SchemaText, SchemaTooLarge
from rag_lab.sql.execute import QueryResult, SqlError
from rag_lab.sql.guard import Refused, guard
from rag_lab.storage.qdrant import QdrantStore


class SqlState(TypedDict, total=False):
    question: str
    history: list[tuple[str, str]]  # ("User" or "Assistant", text); an answer ends with its SQL (`remembered`)
    standalone: str  # the question made standalone; what the SQL and the answer are written for
    schema: SchemaText
    examples: list  # good answers to similar questions (sql/examples.py), when there are any
    stop: str  # a reason to give up at once (the schema is too large, or has no tables)
    sql: str  # the latest query the model wrote, or CANNOT
    to_run: str  # what the guard made of it: what runs
    error: str  # why the latest query failed; empty when it did not
    repairs: int
    result: QueryResult
    answer: str
    thinking: str  # the model's thinking while it wrote and repaired the SQL
    abstained: bool


def remembered(answer: str, sql: str) -> str:
    """An answer as it goes into the history: with the SQL that produced it, so a follow-up can be understood."""
    return f"{answer}\nSQL used:\n{sql}" if sql else answer


def build_graph(
    metrics: MetricsStore,
    schema_name: str,
    base_url: str,
    cfg: SqlAgentConfig,
    embedder: OllamaEmbedder,
    store: QdrantStore,
):
    """`embedder` and `store` are for the good answers; with `use_examples` off none are looked for."""
    quick_llm = chat_model(base_url, cfg, think=False)  # condense and answer never think
    sql_llm = chat_model(base_url, cfg, think=cfg.think)

    def give_up_or_repair(state: SqlState) -> str:
        return "repair" if state["repairs"] < cfg.max_repairs else "abstain"

    def write(node: str, prompt: str, attempt: int) -> tuple[str, str]:
        """Ask for a query and report it. Returns (sql, thinking)."""
        reply, thinking = call_model(
            sql_llm,
            base_url,
            cfg,
            node,
            think=cfg.think,
            stream=True,
            tokens=False,  # the SQL is not the answer: it is reported as SqlWritten
            messages=[("system", WRITE_SYSTEM), ("human", prompt)],
        )
        sql = extract_sql(reply)
        get_stream_writer()(SqlWritten(sql, attempt=attempt))
        return sql, thinking

    @step
    def condense(state: SqlState) -> dict:
        question = standalone_question(quick_llm, base_url, cfg, state, CONDENSE_SYSTEM, condense_prompt)
        return {"standalone": question, "repairs": 0, "error": "", "stop": ""}

    @step
    def schema(state: SqlState) -> dict:
        text = catalog.render(metrics, schema_name)
        get_stream_writer()(SchemaShown(text.tables, text.chars, text.text))
        try:
            text.check(cfg.schema_char_budget)
        except SchemaTooLarge as e:
            return {"schema": text, "stop": str(e)}
        if not text.tables:
            return {"schema": text, "stop": f"The schema '{schema_name}' has no tables."}
        return {"schema": text}

    @step
    def examples(state: SqlState) -> dict:
        """Good answers to questions like this one, when the schema has any: no call is made when it has none."""
        if not (cfg.use_examples and good_answers.has_examples(store, schema_name)):
            return {"examples": []}
        found = good_answers.retrieve(embedder, store, cfg, schema_name, state["standalone"])
        get_stream_writer()(ExamplesFound([asdict(e) for e in found]))
        return {"examples": found}

    @step
    def write_sql(state: SqlState) -> dict:
        prompt = write_prompt(state["schema"].text, state["standalone"], state.get("examples"))
        sql, thinking = write("write_sql", prompt, 1)
        return {"sql": sql, "thinking": thinking}

    @step
    def check(state: SqlState) -> dict:
        emit = get_stream_writer()
        try:
            to_run = guard(state["sql"], schema_name, state["schema"].tables, cfg.row_limit + 1)
            execute.explain(schema_name, to_run, cfg.statement_timeout_s)
        except (Refused, SqlError) as e:
            emit(SqlChecked(False, state["sql"], str(e)))
            return {"error": str(e), "to_run": ""}
        emit(SqlChecked(True, to_run))
        return {"error": "", "to_run": to_run}

    @step
    def run_sql(state: SqlState) -> dict:
        try:
            result = execute.run(schema_name, state["to_run"], cfg.row_limit, cfg.statement_timeout_s)
        except SqlError as e:
            return {"error": str(e)}
        get_stream_writer()(SqlRan(result.columns, result.rows[:ROWS_KEPT], result.row_count, result.truncated, result.ms))
        return {"result": result, "error": ""}

    @step
    def repair(state: SqlState) -> dict:
        attempt = state["repairs"] + 1
        get_stream_writer()(Repairing(state["error"], attempt))
        prompt = repair_prompt(state["schema"].text, state["standalone"], state["sql"], state["error"], state.get("examples"))
        sql, thinking = write("repair", prompt, attempt + 1)
        earlier = state.get("thinking", "")
        return {"sql": sql, "repairs": attempt, "thinking": f"{earlier}\n\n{thinking}".strip() if thinking else earlier}

    @step
    def answer(state: SqlState) -> dict:
        text, _ = call_model(
            quick_llm,
            base_url,
            cfg,
            "answer",
            think=False,
            stream=True,
            messages=[
                ("system", ANSWER_SYSTEM),
                ("human", answer_prompt(state["standalone"], state["to_run"], state["result"], cfg.answer_rows)),
            ],
        )
        return {"answer": text}

    @step
    def abstain(state: SqlState) -> dict:
        if state.get("stop"):
            reason, sql, error = state["stop"], "", ""
        elif state.get("sql") == CANNOT:
            reason, sql, error = "The tables do not seem to hold what this question asks for.", "", ""
        else:
            reason, sql, error = f"No working query was found after {cfg.max_repairs} repair(s).", state["sql"], state["error"]
        text = not_answered(reason, sql, error)
        get_stream_writer()(AnswerToken(text))
        return {"answer": text, "abstained": True}

    graph = StateGraph(SqlState)
    for node in (condense, schema, examples, write_sql, check, run_sql, repair, answer, abstain):
        graph.add_node(node.__name__, node)
    graph.add_edge(START, "condense")
    graph.add_edge("condense", "schema")
    graph.add_conditional_edges("schema", lambda s: "abstain" if s.get("stop") else "examples", ["abstain", "examples"])
    graph.add_edge("examples", "write_sql")
    for node in ("write_sql", "repair"):
        graph.add_conditional_edges(node, lambda s: "abstain" if s["sql"] == CANNOT else "check", ["abstain", "check"])
    graph.add_conditional_edges("check", lambda s: give_up_or_repair(s) if s["error"] else "run_sql", ["repair", "abstain", "run_sql"])
    graph.add_conditional_edges("run_sql", lambda s: give_up_or_repair(s) if s["error"] else "answer", ["repair", "abstain", "answer"])
    graph.add_edge("answer", END)
    graph.add_edge("abstain", END)
    return graph.compile()


@dataclass
class SqlFlow:
    """What a saved database turn needs: the SQL that ran (or the last one tried), the answer, the thinking."""

    schema_name: str
    cfg: SqlAgentConfig
    kind = "database"
    config_hash = None

    def summarise(self, question: str, events: list[Event], final: dict) -> Summary:
        sql = ""
        for event in events:
            if isinstance(event, SqlWritten):
                sql = event.sql
            elif isinstance(event, SqlChecked) and event.ok:
                sql = event.sql
        if not final:  # the turn failed: what was written until then
            return Summary(query=sql or question)
        return Summary(
            query=final.get("to_run") or sql or question,
            answer=final["answer"],
            thinking=final.get("thinking", ""),
            abstained=final.get("abstained", False),
        )
