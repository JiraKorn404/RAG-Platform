"""The settings, as pydantic models. Every stage choice is a value here; nothing is hard-coded in the
stages. The values come from the YAML files in config/ (core/settings.py reads them): `PipelineFile`,
`LlmFile` and `Connections` are those files, key for key, and a key a file leaves out keeps the default
written here."""

import hashlib
import json
from dataclasses import dataclass
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

NAME_PATTERN = r"^[a-z0-9][a-z0-9_-]*$"


class _Section(BaseModel):
    # A typo in a settings file should fail loudly, and settings do not change once they are read.
    model_config = ConfigDict(extra="forbid", frozen=True)


class ParseSettings(_Section):
    """pipeline.yaml: parse."""

    # Docling's own OCR engines and its picture options are fixed off in the parser and not exposed.
    do_table_structure: bool = True
    table_mode: Literal["fast", "accurate"] = "accurate"
    table_cell_matching: bool = True
    do_formula_enrichment: bool = False
    do_code_enrichment: bool = False
    num_threads: int = 8
    # How long Docling may take on one document: `document_timeout`, or `page_timeout` for each of its
    # pages when that is longer (a 444-page book took 3 to 7 s a page here, more when the machine is
    # busy, so 600 s cut it at page 38). A parse that runs out of time fails; it is never kept in part.
    document_timeout: float = 600.0
    page_timeout: float = 20.0
    # Our OCR (ingest/ocr.py): a region Docling's layout found but that has no text layer is cropped
    # from the page image and read by a vision model on Ollama. A page with a text layer keeps it.
    ocr: bool = False
    ocr_scale: float = 2.0  # page image pixels per PDF point (2.0 is 144 dpi)
    ocr_max_tokens: int = 4096  # per region; a region that reaches it is counted as cut
    # Pictures (ingest/parse.py): each picture Docling found is cropped from the rendered page and
    # kept as a file, and becomes a chunk of its own (ingest/chunk.py).
    pictures: bool = False
    picture_scale: float = 2.0  # pixels per PDF point in the saved picture
    picture_min_side: float = 50.0  # PDF points; a picture with a shorter side (a rule, a logo) is left out


class OcrModel(_Section):
    """llm.yaml: ocr."""

    model: str = "glm-ocr:bf16"
    keep_alive: str = "30m"


class ParseConfig(ParseSettings):
    """The parse settings of an experiment: what pipeline.yaml sets, and the OCR model of llm.yaml."""

    ocr_model: str = OcrModel().model
    ocr_keep_alive: str = OcrModel().keep_alive


class HybridSettings(_Section):
    merge_peers: bool = True  # merge undersized neighbours that share the same headings


class SemanticSettings(_Section):
    buffer_size: int = 1  # sentences on each side that are embedded together with a sentence
    # A cut is made where the distance between neighbouring sentences is above this percentile
    # (0-100) of the distances in the section.
    breakpoint_threshold: float = 90.0
    # Sentence boundary: end punctuation plus whitespace, or a blank line. Not suited to
    # languages without sentence-final punctuation (Thai, for example).
    sentence_pattern: str = r"(?<=[.!?])\s+|\n{2,}"


class ChunkConfig(_Section):
    """pipeline.yaml: chunk."""

    strategy: Literal["hybrid", "hierarchical", "fixed", "recursive", "semantic"] = "hybrid"
    max_tokens: int = 512
    overlap: int = 0  # tokens; only the `fixed` strategy uses it
    # For hybrid and hierarchical, Docling serialises tables itself: "row-wise" falls back to
    # markdown there, and "skip" drops table-only chunks.
    table_handling: Literal["markdown", "row-wise", "skip"] = "markdown"
    include_headings_in_text: bool = True
    # Settings for the chosen strategy only; the others are ignored (and left out of the hash).
    hybrid: HybridSettings = HybridSettings()
    semantic: SemanticSettings = SemanticSettings()


@dataclass(frozen=True)
class EmbedFamily:
    """What the embedding models of one family share. A model belongs to the family its Ollama name
    starts with. Only a source of defaults: an experiment records every value it was made with."""

    prefix: str
    tokenizer: str  # Hugging Face id
    query_template: str  # filled with {instruction} and {text}
    document_template: str  # filled with {text}
    dimensions: tuple[int, ...]  # sizes the vectors can be cut to (Matryoshka)
    images: bool = False  # whether an image can be embedded, into the same space as text


EMBED_FAMILIES = (
    # Asymmetric: a query carries an instruction, a document is embedded as it is.
    EmbedFamily(
        prefix="qwen3-embedding",
        tokenizer="Qwen/Qwen3-Embedding-0.6B",
        query_template="Instruct: {instruction}\nQuery: {text}",
        document_template="{text}",
        dimensions=(256, 512, 768, 1024),
    ),
    # Both sides carry a fixed prefix (the model card's "search result" task); 768 is its native size.
    EmbedFamily(
        prefix="embeddinggemma-2",
        tokenizer="google/embeddinggemma-2",
        query_template="task: search result | query: {text}",
        document_template="title: none | text: {text}",
        dimensions=(128, 256, 512, 768),
        images=True,
    ),
)


def embed_family(model: str) -> EmbedFamily | None:
    return next((f for f in EMBED_FAMILIES if model.startswith(f.prefix)), None)


def embed_model_label(model: str) -> str:
    """'qwen3-embedding:4b' -> '4b'; a model of another family keeps its whole name, so its size is not
    read as a Qwen3 size."""
    return model.split(":")[-1] if model.startswith("qwen3-embedding") else model


class EmbedConfig(_Section):
    """llm.yaml: embedding. Part of an experiment: a collection holds the vectors of one setup."""

    model: str = "qwen3-embedding:0.6b"
    tokenizer: str = "Qwen/Qwen3-Embedding-0.6B"  # Hugging Face id; chunk sizes are counted with it
    dimension: int | None = None  # None keeps the model's native size (1024); smaller truncates
    batch_size: int = 32
    query_instruction: str = (
        "Given a question, retrieve relevant passages from the documents that answer it"
    )
    # How a query and a document are written before they are embedded. The defaults are Qwen3's.
    # Another family needs its own (see EMBED_FAMILIES and `for_model`): the wrong ones still give
    # vectors, only worse ones.
    query_template: str = "Instruct: {instruction}\nQuery: {text}"
    document_template: str = "{text}"
    # What a picture chunk is embedded from: the image with its text (its caption) as one input, the
    # image alone, or the text alone (which needs no model that takes images, and is the baseline that
    # says whether the image adds anything). Only used with `parse.pictures`.
    picture_input: Literal["image+caption", "image", "caption"] = "image+caption"
    keep_alive: str = "30m"

    @classmethod
    def for_model(cls, model: str, **settings) -> "EmbedConfig":
        """The settings for an Ollama model of a known family: its tokenizer and its templates."""
        family = embed_family(model)
        if family is None:
            known = ", ".join(f.prefix for f in EMBED_FAMILIES)
            raise ValueError(f"'{model}' is not of a known embedding family ({known})")
        return cls(
            model=model,
            tokenizer=family.tokenizer,
            query_template=family.query_template,
            document_template=family.document_template,
            **settings,
        )


class IndexConfig(_Section):
    """pipeline.yaml: index."""

    hnsw_m: int = 16
    hnsw_ef_construct: int = 100
    hnsw_ef: int | None = None  # search-time ef; None uses the Qdrant default
    # Also store a BM25 sparse vector on every point, which the hybrid search methods and the chatbot
    # need. It cannot be added to a collection later.
    sparse: bool = True


SearchMethod = Literal["dense", "hybrid", "dense+rerank", "hybrid+rerank"]
SEARCH_METHODS: tuple[str, ...] = ("dense", "hybrid", "dense+rerank", "hybrid+rerank")


# The instruction a question is embedded with when it looks for good answers to similar questions.
EXAMPLES_QUERY_INSTRUCTION = "Given a question about a dataset, retrieve questions that ask for the same kind of information"


class RerankerConfig(_Section):
    """llm.yaml: reranker. The 4B builds work; the 0.6B builds give every chunk a score of 0."""

    model: str = "dengcao/Qwen3-Reranker-4B:Q8_0"
    instruction: str = "Given a question, retrieve relevant passages from the documents that answer it"
    keep_alive: str = "30m"
    # The reranker's context window. Ollama's default (40k) makes the model take about 9 GB of memory,
    # which pushes the embedding model out. A prompt is the instruction, the query and one chunk.
    num_ctx: int = 2048


class SearchConfig(_Section):
    """How an experiment is searched. Not part of ExperimentConfig or its hash: changing how you search
    does not make a new experiment."""

    method: SearchMethod = "dense"
    # What the first stage hands on: each branch of a hybrid search fetches this many hits, and a
    # reranker scores this many. At least top k is always fetched.
    candidates: int = 20
    reranker: RerankerConfig = RerankerConfig()

    @property
    def hybrid(self) -> bool:
        return self.method.startswith("hybrid")

    @property
    def rerank(self) -> bool:
        return self.method.endswith("+rerank")


class ChatModelConfig(_Section):
    """llm.yaml: chat. What both chatbot flows share: the chat model and how it is called. A flow's own
    section may set any of these again for itself."""

    model: str = "gemma4:e4b-mlx"
    temperature: float = 0.2
    # Ollama's own default window is small and silently cuts a long prompt, so it is always set. The
    # prompt is the instructions, the history, what the flow retrieved and, when `think` is on, the thinking.
    num_ctx: int = 8192
    think: bool = True  # the answer is written with the model's thinking on; the other steps never think
    keep_alive: str = "30m"
    history_turns: int = 6  # earlier question-and-answer pairs the question is condensed with


class AgentConfig(ChatModelConfig):
    """llm.yaml: documents_chat. The chatbot agent for documents. Not part of ExperimentConfig or its
    hash: it changes how an experiment is asked, not what is stored in it."""

    top_k: int = 5  # chunks given to the model
    # Judging the retrieval by the best chunk's reranker score (a probability of "yes"): at or above
    # `enough_score` the chunks answer it, below `missing_score` they do not, and in between the model is
    # asked. Chosen on one document: questions it answers scored 0.99 or more, questions it does not
    # cover 0.0 to 0.05 (carburetor icing, which it never mentions, 0.051).
    enough_score: float = 0.5
    missing_score: float = 0.1
    max_rewrites: int = 1  # different queries tried when the chunks do not answer it, before giving up
    # The documents are searched with hybrid search and reranking: the reranker scores this many hits
    # (one call to Ollama each) and the best `top_k` are kept. The reranker is llm.yaml's `reranker`.
    candidates: int = 20
    reranker: RerankerConfig = RerankerConfig()
    # A picture among the chunks is given to the answer model as an image, not only as its caption. The
    # model must take images. Each one costs context, so only the first `max_pictures` by rank are given.
    show_pictures: bool = True
    max_pictures: int = 2

    @property
    def search(self) -> SearchConfig:
        return SearchConfig(method="hybrid+rerank", candidates=self.candidates, reranker=self.reranker)


class SqlAgentConfig(ChatModelConfig):
    """llm.yaml: database_chat. The chatbot agent for a database schema (text-to-SQL)."""

    # Exact answers want no randomness, and the prompt (the whole schema, the repairs, the rows and, when
    # `think` is on, the thinking) is longer than a documents prompt.
    temperature: float = 0.0
    num_ctx: int = 16384
    # The model is given the description of every table in the schema. A schema whose description is
    # longer than this is refused with its size, not cut: a silently cut schema gives silently wrong SQL.
    schema_char_budget: int = 12_000
    max_repairs: int = 2  # a query the guard or the database refused is rewritten this many times
    row_limit: int = 200  # rows a query may return; more are cut, and the answer says so
    answer_rows: int = 30  # rows given to the model to write the answer from
    statement_timeout_s: float = 10
    # Good answers (serve/examples.py): a question like the one asked, answered correctly before, is
    # shown to the model with its SQL. They are found by the similarity of the questions (cosine), embedded
    # with llm.yaml's `embedding`. `examples_min_score` was chosen with qwen3-embedding:0.6b: on six saved
    # examples the best match scored 0.56 to 0.78 for a paraphrase or the same kind of question, 0.36 to
    # 0.67 for another kind of question about the same table, and 0.26 to 0.51 for an unrelated one.
    # Another embedding model has another scale, so look at the scores again after changing it.
    use_examples: bool = True
    examples_k: int = 3
    examples_min_score: float = 0.55
    embed: EmbedConfig = EmbedConfig(query_instruction=EXAMPLES_QUERY_INSTRUCTION)


class ImportConfig(_Section):
    """pipeline.yaml: tables. Importing a CSV file into the database for imported tables."""

    max_bytes: int = 50 * 1024 * 1024
    max_rows: int = 1_000_000
    sample_rows: int = 10_000  # rows read to guess the column types
    sample_bytes: int = 4 * 1024 * 1024  # the start of the file that is read for that
    sample_values: int = 10  # example values kept for a text column with few distinct values
    max_distinct_for_samples: int = 50


class ExperimentConfig(_Section):
    """What a collection is made with: pipeline.yaml, with the embedding and OCR models of llm.yaml.
    It is recorded in full with the experiment, so a search reads the settings the collection was
    made with and not the files as they are now."""

    # The name doubles as the Qdrant collection name, so keep it simple.
    name: str = Field(pattern=NAME_PATTERN)
    parse: ParseConfig = ParseConfig()
    chunk: ChunkConfig = ChunkConfig()
    embed: EmbedConfig = EmbedConfig()
    index: IndexConfig = IndexConfig()

    @model_validator(mode="after")
    def _pictures_need_a_model_that_takes_images(self):
        if self.parse.pictures and self.embed.picture_input != "caption":
            family = embed_family(self.embed.model)
            if family is None or not family.images:
                raise ValueError(
                    f"`parse.pictures` with `embedding.picture_input` = '{self.embed.picture_input}' needs an "
                    f"embedding model that takes images, and '{self.embed.model}' does not. Use one that "
                    "does (embeddinggemma-2), or set `embedding.picture_input` to 'caption'."
                )
        return self

    @property
    def collection(self) -> str:
        return self.name

    def config_hash(self) -> str:
        """Identifies the experiment: its name and every setting. The settings of the chunking
        strategies that are not chosen are left out, so editing them does not make a new experiment."""
        data = self.model_dump(mode="json")
        for strategy in ("hybrid", "semantic"):
            if strategy != self.chunk.strategy:
                data["chunk"].pop(strategy)
        payload = json.dumps(data, sort_keys=True)
        return hashlib.sha256(payload.encode()).hexdigest()[:16]


class PipelineFile(_Section):
    """config/pipeline.yaml: what a PDF or a CSV goes through."""

    name: str = Field(pattern=NAME_PATTERN)  # the experiment, which is also the Qdrant collection
    parse: ParseSettings = ParseSettings()
    chunk: ChunkConfig = ChunkConfig()
    index: IndexConfig = IndexConfig()
    tables: ImportConfig = ImportConfig()


class LlmFile(_Section):
    """config/llm.yaml: every model, and how it is called. The file's `chat` section is not a field here:
    it is merged under both chat sections before they are read."""

    embedding: EmbedConfig = EmbedConfig()
    ocr: OcrModel = OcrModel()
    reranker: RerankerConfig = RerankerConfig()
    documents_chat: AgentConfig = AgentConfig()
    database_chat: SqlAgentConfig = SqlAgentConfig()


class Url(_Section):
    url: str


class AppDatabase(_Section):
    # Chats, metrics and the registry of tables. Our tables are made in `schema`, which must exist and
    # belong to the role of `url` (scripts/provision.sql makes it).
    url: str
    schema_name: str = Field("public", alias="schema", pattern=r"^[a-z_][a-z0-9_]*$")


class TablesDatabase(_Section):
    # The tables the SQL chatbot answers from. The reader role can only select. The loader role creates
    # and fills the tables of the CSV import; without it the import is off. `existing_schemas` are
    # schemas that are already in the database (not imported) and that the chatbot may be asked about.
    reader_url: str
    loader_url: str | None = None
    existing_schemas: tuple[str, ...] = ()


class Api(_Section):
    # The HTTP API (serve/api.py): where the Streamlit UI finds it, and the key a caller must send in
    # the `X-API-Key` header. An empty key asks for none.
    url: str = "http://api:8000"
    key: str = ""


class Connections(_Section):
    """config/connections.yaml: where every service is."""

    ollama: Url
    qdrant: Url
    app_database: AppDatabase
    tables_database: TablesDatabase
    api: Api = Api()
