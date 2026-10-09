"""What a chatbot reports while it works, and what a search finds (`Hit`). This is the contract with a
front end: the API sends these as JSON (`to_dict`), a saved turn keeps them, and a page or a CLI draws
them without knowing the graph. Plain dataclasses with no other import, so a front end can use them."""

import re
from dataclasses import asdict, dataclass, field


@dataclass
class StepStarted:
    node: str


@dataclass
class StepFinished:
    node: str
    ms: float


@dataclass
class Query:
    text: str
    rewritten: bool  # false when the question was already standalone


@dataclass
class Graded:
    """Whether the retrieved chunks answer the question, and how that was decided."""

    enough: bool
    best_score: float  # the best chunk's reranker score
    by: str  # "score" or "model" (the score was in between, so the model was asked)


@dataclass
class Rewrote:
    query: str  # the different query tried next
    previous: str


@dataclass
class SchemaShown:
    """What the text-to-SQL flow gives the model: every table of the schema, and how long that text is."""

    tables: list[str]
    chars: int
    text: str = ""  # the text itself, as the model saw it


@dataclass
class ExamplesFound:
    """Good answers to similar questions that were put in front of the model. Empty when the schema has
    examples but none was similar enough."""

    examples: list[dict]  # each: id, question, sql, score


@dataclass
class SqlWritten:
    sql: str  # what the model wrote, as it wrote it
    attempt: int  # 1 for the first query, 2 for the first repair


@dataclass
class SqlChecked:
    ok: bool
    sql: str  # what will run when ok (schema-qualified, with its limit), else what was refused
    reason: str = ""  # why not, worded for the model


@dataclass
class SqlRan:
    columns: list[str]
    rows: list[list]  # the first `ROWS_KEPT` of them
    row_count: int  # how many were read (at most the row limit)
    truncated: bool  # there were more
    ms: float


@dataclass
class Repairing:
    error: str  # what the guard or the database said about the query
    attempt: int  # the repair that follows: 1 for the first


@dataclass
class Hit:
    rank: int
    # The score of what ranked the hit: Qdrant's cosine similarity for a `dense` search, a rank-fusion
    # score for a `hybrid` one, the reranker's probability of "yes" once it is reranked (a chatbot's
    # hits). Only the cosine score is comparable between experiments, and `distance` means something
    # only for it.
    similarity: float
    distance: float  # 1 - similarity
    text: str
    source_file: str | None
    page: int | None
    modality: str
    headings: list[str]
    doc_id: str
    chunk_id: str
    # A picture chunk's file, relative to data/artifacts: "<experiment>/parse/<doc_id>.pictures/<n>.png"
    image: str | None = None


@dataclass
class Retrieved:
    hits: list[Hit]
    method: str
    candidates: int
    embed_ms: float
    search_ms: float
    rerank_ms: float


@dataclass
class Thinking:
    text: str  # a piece of the model's thinking


@dataclass
class AnswerToken:
    text: str  # a piece of the answer


@dataclass
class ModelState:
    """One model call: what ran, how full its context was and how fast it generated."""

    node: str
    model: str
    think: bool
    loaded: bool | None  # was the model already in memory before the call; None when Ollama could not say
    prompt_tokens: int
    output_tokens: int  # includes the thinking
    num_ctx: int
    tokens_per_s: float | None
    context_full: bool  # the prompt reached num_ctx, so Ollama cut it


@dataclass
class Done:
    answer: str
    thinking: str
    cited: list[int]  # the passage numbers the answer cites that exist
    unknown_citations: list[int]  # cited numbers that no passage has
    timings: dict[str, float] = field(default_factory=dict)  # ms per step, summed over retries
    total_ms: float = 0.0
    abstained: bool = False  # the chunks did not answer it, so the answer is the fixed "not found" message
    saved: bool = True  # false when saving the turn failed; the answer is still good
    save_error: str | None = None
    turn_id: int | None = None  # the saved turn, when it was saved
    # How a documents turn went, in one place (a database turn leaves these empty).
    outcome: str = ""  # "answered", "direct" or "abstained"
    abstain_reason: str | None = None  # "retrieval" or "grounding", when it abstained
    route: str = ""  # "retrieve" or "direct"
    rewrites: int = 0  # other queries tried
    top_score: float | None = None  # the best reranker score of the turn


@dataclass
class Failed:
    """A turn that stopped with an error. The API sends it in place of `Done`; the turn is saved with
    the same message, so it is never among a saved turn's events."""

    message: str


ROWS_KEPT = 100  # rows of a result kept in a turn's events

Event = (
    StepStarted | StepFinished | Query | Graded | Rewrote | Retrieved | Thinking | AnswerToken | ModelState | Done
    | SchemaShown | ExamplesFound | SqlWritten | SqlChecked | SqlRan | Repairing | Failed
)


_TYPES = {
    cls.__name__: cls
    for cls in (
        StepStarted, StepFinished, Query, Graded, Rewrote, Retrieved, Thinking, AnswerToken, ModelState, Done,
        SchemaShown, ExamplesFound, SqlWritten, SqlChecked, SqlRan, Repairing, Failed,
    )
}


def to_dict(event: Event) -> dict:
    """The event as plain data (JSON-able), tagged with its class."""
    return {"type": type(event).__name__, **asdict(event)}


def from_dict(data: dict) -> Event:
    """The event `to_dict` made, read back."""
    fields = dict(data)
    cls = _TYPES[fields.pop("type")]
    if cls is Retrieved:
        fields["hits"] = [Hit(**hit) for hit in fields["hits"]]
    return cls(**fields)


def citations(answer: str, passages: int) -> tuple[list[int], list[int]]:
    """The passage numbers an answer cites as [n]: (those that exist, those that do not)."""
    numbers = sorted({int(n) for n in re.findall(r"\[(\d+)\]", answer)})
    return [n for n in numbers if 1 <= n <= passages], [n for n in numbers if not 1 <= n <= passages]
