# RAG-Platform

A chatbot over two kinds of data, with ingestion run by Dagster:

- **PDF documents** (text, tables, optionally scanned pages and pictures) are parsed, chunked, embedded and indexed into a vector database. The chatbot answers from the retrieved chunks, with citations.
- **Tables** in a relational database: CSV files that are imported, or tables that are already there and are only described. The chatbot writes a SQL query, checks it, runs it read-only and answers from the rows.

Docling parses, LlamaIndex splits, Ollama embeds, reranks and chats, Qdrant stores the vectors, PostgreSQL holds the tables, the chats and the metrics, Dagster orchestrates. An HTTP API (FastAPI) is the door for every front end: the Streamlit UI and the platform's backend both use it. Everything is configured in three YAML files in `config/`.

The system was cut down from a larger lab in October 2026 (`REFACTORPLAN.md`) and then restructured into three packages so that it can move to a server that already has PostgreSQL and Qdrant (`RESTRUCTUREPLAN.md`). `docs/history/` holds the plans of the earlier phases (they describe features that no longer exist). Update this file when the code changes, so that it describes what is in the repo.

## Architecture

```
data/raw/*.pdf              -> Dagster ingest_job: parse -> chunk -> embed -> index -> Qdrant collection
data/tables/<schema>/*.csv  -> Dagster tables_job: import                             -> PostgreSQL schema
tables already in PostgreSQL-> Dagster register_job: describe                         -> the registry

HTTP API:   targets, chats, a turn as a stream of events, good answers, the library, pictures
Streamlit:  Experiments (the collections, with deletes)    Chatbot (vector database | relational database)
Config:     config/pipeline.yaml   config/llm.yaml   config/connections.yaml   (.env for passwords)
Metrics:    stage metrics, the search log and every chat turn -> PostgreSQL (our schema)
```

| Service | What it is | Image | Address |
|---|---|---|---|
| `setup` | Runs once at start and exits: checks the connections, applies the migrations | serve | |
| `api` | The HTTP API (`rag_lab.serve.api`) | serve | http://localhost:8000 (`/docs`) |
| `ui` | Streamlit: Experiments and Chatbot. It only talks to the API | serve | http://localhost:8501 |
| `dagster-code`, `dagster-webserver`, `dagster-daemon` | Dagster; runs execute in `dagster-code` | ingest | http://localhost:3000 |
| `postgres`, `qdrant`, `provision` | Local only (`docker-compose.local.yml`): the databases, and the run of `scripts/provision.sql` | | `5432`, `6333`, `6334` |
| Ollama (not in compose) | Embedding, reranking, chat and OCR models | | `OLLAMA_BASE_URL` in `.env` |

The local PostgreSQL has the databases `dagster` (Dagster's own), `rag_metrics` (our tables, in the schema `rag`) and `rag_data` (the imported tables). This machine has no NVIDIA GPU, so Docling runs on CPU. Ports are bound to `127.0.0.1` only.

## Scope

- **In:** PDF documents (text and tables); CSV files imported into PostgreSQL, one schema per dataset; schemas that are already in PostgreSQL, registered for the SQL chatbot.
- **In, off by default:** OCR of scanned pages (`parse.ocr`) and the pictures of a PDF as chunks (`parse.pictures`, needs `embeddinggemma-2`).
- **Out:** other document types, audio and video, a login or users (every chat is visible to every caller of the API). Docling's own OCR engines and picture features stay off; the only OCR is ours (`ingest/ocr.py`) and pictures are cropped by us (`ingest/parse.py`).
- **Not built yet:** PDFs from an object store (SeaweedFS). `ingest/source.py` is where it goes: today it only knows the folder `data/raw`.

## Working style

This project is kept minimal. Prefer the smallest change that works, and do not add code, config, abstractions or docs that nothing needs yet.

- **Few tests.** Write tests only for pure logic where a silent bug would corrupt results: vector truncation and re-normalisation, BM25 weights and the reranker score, the decisions of the documents graph (answer, rewrite or abstain), config hashing and the settings loader, CSV import (cleaning of names, guessing of column types, delimiter detection) and the SQL guard (the one place where a silent bug is a security hole). No tests for Dagster assets, Docling, Docker or anything needing Qdrant, Postgres or Ollama, and no mocks or fakes of them. Do not add test coverage to a change unless asked.
- **No re-testing after minor edits.** After a small change (rename, config tweak, comment, log line, doc edit, small refactor) do not run the test suite, rebuild containers or re-materialise assets. Run something only when a change alters the logic of one of the tested areas above, or when asked. Never rerun a passing check "to be sure".
- **Verify by running the real thing, once.** When a piece of work is finished, confirm it with a single real run on the stack. Do not build extra verification around it.
- **Keep dependencies and services few.** Do not add a library, container or tool without a stated need.

## Design rules

- **Three packages, by where the code runs.** `core` is what both sides need, `ingest` is the batch side (Dagster), `serve` answers questions (the API). **`ingest` and `serve` import `core`, never each other**; only `cli.py` reaches into both. `core` and `serve` must not import Docling, torch, LlamaIndex or Dagster: the serve image does not have them.
- **One door for front ends.** `serve/service.py` is what a front end can do, as plain functions; `serve/api.py` is those functions over HTTP. The Streamlit pages call the API and import nothing of the system but `core.events`, `core.config` and `core.settings`. A new thing the UI shows needs a function in `service.py` and an endpoint first.
- **Every choice is a value in `config/*.yaml`**, never a constant in code. A key left out keeps the default written in `core/config.py`.
- **Stages are plain Python** with no Dagster or Streamlit imports. The Dagster assets in `ingest/definitions.py`, the endpoints in `serve/api.py` and the pages in `ui/` are thin callers.
- **One Qdrant collection per experiment**, named after it. An experiment is the collection plus the settings it was made with, recorded in full in the `experiments` table. **Settings cannot change under a name:** a name is unique in the database, and every writer refuses a name that already has other settings.
- **A search reads the settings stored with the collection**, not the files as they are now, so a question is always embedded the way the documents were.
- **Vectors are supplied by us.** Qdrant only stores and searches; embeddings come from Ollama.
- **Our code is never a database admin.** `scripts/provision.sql` makes the databases, the roles and our schema, and a person (or, locally, the `provision` service) runs it. `python -m rag_lab setup` only checks the connections and applies our migrations inside our schema. Never write to Dagster's database. The tables database is written only by the loader role and read only by the reader role.
- **Passwords and private addresses stay in `.env`.** `connections.yaml` refers to them as `${NAME}`. Every container gets the whole of `.env`, so a new variable is added there only.
- **Intermediates are kept on disk** under `data/artifacts/<experiment>/<stage>/`, so a stage can be re-run without redoing the ones before it. The API reads pictures from there, so `dagster-code` and `api` share `data/`.

## Configuration

`rag_lab/core/settings.py` reads three files from `config/` (`RAG_CONFIG_DIR`). An unknown key is an error naming the file and the key; `${NAME}` is replaced from the environment and a missing variable is an error.

| File | Holds | Read |
|---|---|---|
| `connections.yaml` | Ollama, Qdrant, `app_database` (URL and `schema`), `tables_database` (reader URL, optional loader URL, `existing_schemas`), `api` (URL and key) | The API and the UI keep their clients while they run: `docker compose restart api ui` after an edit |
| `pipeline.yaml` | `name` (the experiment and collection), `parse`, `chunk`, `index`, and `tables` (CSV limits) | By Dagster at every sensor tick and every run |
| `llm.yaml` | `embedding`, `ocr`, `reranker`, `chat` (shared), `documents_chat`, `database_chat` | By Dagster at every run, by the API at every question |

- `settings.load()` returns `connections`, `experiment` (an `ExperimentConfig`: `pipeline.yaml` plus the embedding and OCR models of `llm.yaml`), `tables` (`ImportConfig`), `documents_chat` (`AgentConfig`, with the reranker) and `database_chat` (`SqlAgentConfig`, with the embedding model for good answers).
- `chat` is merged under both chat sections; a chat's own section wins. `documents_chat.reranker` and `database_chat.embed` are refused: set `reranker` and `embedding`.
- `core/connections.py` builds the embedder, the Qdrant store and the store of our tables from `connections.yaml`. Nothing else reads an address. The only environment variables the code reads are `DATA_DIR` and `RAG_CONFIG_DIR`.
- **`app_database.schema`**: our tables are made and found in this schema (the store sets it as the connection's search path, so the migrations name no schema). It must exist and belong to the role of `app_database.url`.
- **`tables_database.loader_url` is optional.** Without it the CSV import is off (the sensor says so) and nothing of ours can write to that database. `existing_schemas` names the schemas that are already there.
- The database and role names of the tables database are taken from its URLs.
- **Dagster's own database is not in `connections.yaml`**: Dagster reads only `docker/dagster.yaml`, which takes `POSTGRES_HOST`, `POSTGRES_PORT`, `POSTGRES_USER`, `POSTGRES_PASSWORD` and `DAGSTER_POSTGRES_DB` from `.env`. So moving to another server means `connections.yaml` and `.env`.
- `core/config.py` holds the models as plain pydantic (`extra="forbid"`, frozen): change a value with `model_copy(update=...)`.
- `config/connections.vm.example.yaml` is `connections.yaml` for a server whose databases already run.

## Key technical facts

### Ingestion of PDFs

- **Sensor** (`ingest/definitions.py: new_pdf_sensor`, every 30 s): a PDF that is new in `data/raw` gets a partition and an `ingest_job` run. New means not yet a partition: the same bytes under another name, or a PDF whose run failed, start nothing; run that partition by hand. A file is taken only when it can be read and its last kilobyte has `%%EOF`. When the settings do not load, or name an experiment that exists with other settings, the tick fails with the reason and the PDF stays new.
- **No run config and no Dagster resources.** Each of the four assets (`parsed_document`, `chunks`, `embeddings`, `qdrant_index`) starts with `configured_experiment()` (`ingest/documents.py`), which loads the files, refuses a conflict and registers the experiment. `ingest_job` uses the in-process executor: with the default one every step imported Docling again (about 45 s each). `dagster.yaml` limits the queue to one run at a time.
- A document's id is the first 16 hex characters of the SHA-256 of its bytes (`ingest/source.py`); it is also the partition key.
- `ExperimentConfig.config_hash()` covers the name and every setting, except the settings of the chunking strategies that are not chosen.
- **A parse is whole or it fails.** Docling returns `partial_success` when it hits its time limit; `parse_pdf` raises, with the pages reached and the settings to raise. The limit is `document_timeout` (600 s) or `page_timeout` (20 s) per page, whichever is longer.
- **Chunking**: `hybrid` and `hierarchical` are Docling's chunkers (`ingest/chunk.py`, tables as Markdown, `row-wise` unsupported); `fixed`, `recursive` and `semantic` split the text under each heading with LlamaIndex splitters and handle tables by `table_handling` (`ingest/split_text.py`). `chunk_document` in `ingest/documents.py` chooses. `overlap` applies to `fixed` only. Sizes are counted with the embedding family's Hugging Face tokenizer, and headings are part of each chunk's text. `semantic` calls Ollama for sentence embeddings, cached in `data/artifacts/_cache/embeddings/`.
- **Embedding families** (`core/config.py: EMBED_FAMILIES`): a model belongs to the family its Ollama name starts with, which gives its tokenizer, its query and document templates and the sizes its vectors can be cut to. `qwen3-embedding` (0.6b, 4b, 8b): asymmetric, queries get `Instruct: <task>\nQuery: <text>`, sizes 256 to 1024. `embeddinggemma-2` (740m, 768 dimensions): `task: search result | query: {text}` and `title: none | text: {text}`, sizes 128 to 768, and it takes images. `settings.py` fills these in from the model name; a model of no known family needs them set in `llm.yaml`. Truncation and re-normalisation are ours (`core/embed.py`). Use Ollama's `/api/embed`.
- **OCR** (`ingest/ocr.py`, `glm-ocr:bf16`): Docling runs with `do_ocr=False` and keeps empty layout regions; each region with no text is cropped and read through `/api/chat`, so a page with a text layer keeps it. The model is used per region, never per page. It does not stop by itself on Ollama 0.40, so the reply is read as a stream and cut at the end of the first reading. There is no fallback: with OCR on, a parse needs Ollama.
- **Pictures** (`ingest/parse.py: save_pictures`, `ingest/chunk.py: picture_chunks`): each kept picture is cropped into `<doc_id>.pictures/<n>.png` and becomes a chunk whose text is its caption (or "Figure on page N"). `embedding.picture_input` is `image+caption`, `image` or `caption`. **An image goes inside `input` as an object (`{"image": "<base64>"}`); a top-level `images` field is silently ignored by `/api/embed`.** A family that takes no images is refused when the settings are loaded.
- **BM25** (`index.sparse`, on by default; `core/embed.py`): a named sparse vector `bm25` with Qdrant's IDF modifier, written after the dense vector. It cannot be added to a collection later, and the Chatbot only offers collections that have it. The dense vector is the unnamed one.
- Writes go through LlamaIndex's `QdrantVectorStore` into a collection `QdrantStore.ensure_collection` created. Point ids are deterministic UUIDs of the chunk ids, so re-ingesting overwrites. Qdrant's storage must be a named volume: a Windows bind mount can corrupt it.

### Tables: imported or registered

The SQL chatbot reads a description of every table from the registry (`db_schemas`, `db_tables`, among our tables). A schema gets there in one of two ways, and `db_schemas.registered` says which.

- **Folder** (`ingest/folder.py`): `data/tables/<schema>/<table>.csv`. The folder is the schema, the cleaned file name is the table. `_schema.yaml` in the folder is optional: the dataset's description, tables' and columns' descriptions, and a column type to use instead of the guessed one. A column it names that the file does not have is an error.
- **Sensor** (`new_csv_sensor`): asks for a run of `tables_job` for every settled file at every tick, keyed by the content hash of the file and of `_schema.yaml`; Dagster skips the keys it has seen. So the same file is imported once, an edit imports it again, and a failed import waits for the file to change (or for its partition, `schema.table`, to be run by hand). Without `tables_database.loader_url` it skips with that reason.
- **Import** (`ingest/importer.py`): UTF-8 only; the delimiter is sniffed; names are cleaned (letters of any language are kept); types are guessed from up to 10,000 rows. One transaction as the loader role: create the table (replacing it), `COPY` the file, profile every column. A bad row leaves the old table and the error names the line.
- **Register** (`ingest/register.py`, the asset `registered_tables`, `register_job`, or `python -m rag_lab tables register`): for each schema of `existing_schemas`, the tables, views and column types are read from PostgreSQL's catalog and each column is profiled with the import's own queries. It runs as the reader, so it changes nothing in that database. Descriptions go in `config/schemas/<schema>.yaml` (the format of `_schema.yaml`, without `type`). A profile query that fails or runs into the reader's 15 s limit leaves that column's profile empty and is reported as a warning. Run it again when the tables change; a schema that is no longer listed leaves the registry with its chats and good answers. No sensor watches a database: the job is started by hand.
- **Roles** (`scripts/provision.sql`): the loader may create schemas and tables; the reader may only `SELECT`, with `default_transaction_read_only`, a 15 s statement timeout and 10 connections. The grants are the boundary, not the read-only setting. For an existing schema the reader is granted with `scripts/grant_read.sql`, one schema at a time, and must not be granted our own schema.
- Dropping (`python -m rag_lab tables drop`) never touches the CSV files, and for a registered schema never touches its tables: only what the chatbot knew about it goes.

### Search and the chatbot

- **Search** (`serve/search.py`): `dense` or `hybrid` (dense and BM25 prefetches fused with RRF). A hit's `similarity` is a cosine similarity only for `dense`.
- **Reranker** (`serve/search.py: OllamaReranker`, `rerank_hits`): one `/api/generate` call per (question, chunk) with the Qwen3-Reranker prompt, one token and `logprobs`, scored `P(yes)/(P(yes)+P(no))`. `dengcao/Qwen3-Reranker-4B` works (`Q8_0`, `Q4_K_M`); the 0.6B builds give every chunk 0. It is about 0.8 s a chunk, so it is most of a search's time.
- **Documents agent** (`serve/chat_documents.py`, LangGraph): `condense -> retrieve -> rerank -> grade -> generate`, with `grade -> rewrite -> retrieve` while the chunks do not answer and `grade -> abstain` when the retries are used up. `retrieve` is a hybrid search for `candidates` hits. `rerank` scores them **against the standalone question, not the query that found them**, keeps every score of the turn (a chunk a rewrite finds again is not scored again) and hands on the best `top_k`. `grade` makes no model call: the best score at or above `enough_score` answers, from the kept chunks at or above `min_doc_score`. The answer cites `[n]`. Retrieved pictures are attached to the answer prompt with `Passage [n]` drawn above them. A rewrite that repeats a query already searched uses its attempt and is not searched again, so a turn searches at most `max_rewrites + 1` times (`after_grade`, `after_rewrite`). `Done` says how the turn went: `outcome` (`answered` or `abstained`), `abstain_reason`, `route`, `rewrites` and `top_score`. `PLAN.md` is the plan this flow is being rebuilt by.
- **SQL agent** (`serve/chat_database.py`): `condense -> schema -> examples -> write_sql -> check -> run_sql -> answer`, with `repair` up to `max_repairs` times and `abstain` when the schema is too large, the model replies `CANNOT`, or the repairs are used up. The model is given the whole schema as text (`serve/catalog.py`, from the registry: descriptions, types, example values, ranges, likely joins); there is no retrieval. A schema over `schema_char_budget` is refused, not cut.
- **The SQL guard** (`serve/guard.py`, sqlglot): accepts exactly one `SELECT`, refuses any write, DDL, locking clause, unknown function or table outside the chat's schema, qualifies tables and adds a `LIMIT`. What runs is the SQL sqlglot writes out, not the model's text. **Known limit: the guard is the only thing keeping one dataset from another**, since the reader may select from every schema it was granted.
- **Good answers** (`serve/examples.py`): the thumbs-up under a database answer saves the question and its SQL; they are embedded with `llm.yaml`'s `embedding` into `sqlexamples__<schema>` in Qdrant and shown to the model for similar questions. `examples_min_score` depends on the embedding model (0.8 for `embeddinggemma-2`, about 0.55 for `qwen3-embedding:0.6b`). Changes go table first, then Qdrant; `python -m rag_lab tables examples --reindex` rebuilds.
- **Events** (`core/events.py`): every node reports typed events through LangGraph's custom stream. `run(graph, flow, question, history, metrics=, session_id=)` (`serve/run.py`) yields them, ends with `Done`, and saves the turn (also a failed one, with `error`). A saved turn keeps its events, so the page replays a past chat through the same `apply(trace, event)` that draws a live one. `Hit` lives there too, so the events import nothing.
- **Eval** (`serve/evaluate.py`, `python -m rag_lab eval`): the cases of `data/eval/<experiment>.jsonl` (a question, what to expect: `answer`, `abstain` or `direct`, the facts the answer must contain, and snippets of the passage that holds it) are asked of the documents graph through `run()` without a chat, so nothing is saved. It writes `data/eval/results/<experiment>-<label>.json` and prints the hit rate, the correct answers, abstaining, latency and each case's best reranker score. `PLAN.md` has the run of every phase.
- **Chats**: a chat is a row in `chat_sessions` with a `kind` (`documents` or `database`) for its whole life; a composite foreign key refuses a turn of the other kind. It belongs to its experiment or its schema and is deleted with it. The caller chooses the chat's id; the chat is made with its first turn.
- `num_ctx` is always set, because Ollama's default silently cuts long prompts.

### The API and the UI

- **Service** (`serve/service.py`): `targets`, `check`, `ask`, `chats`, `chat`, `delete_chat`, `examples`, `mark_good`, `unmark_good`, `library`, `delete_document`, `delete_experiment`, `picture`, `health`. It raises `Refused` with a reason for the person who asked. `ask` checks everything before it returns the iterator: that the chat still searches the same thing, and that Ollama has the models (a missing model stops the turn with the key to change; a chat model that cannot think or see images has that setting switched off, with a note). The history of a turn is read from the chat's saved turns, never sent by the caller.
- **API** (`serve/api.py`): `GET /health`, `/targets`, `/check`, `/chats`, `/chats/{id}`, `/schemas/{schema}/examples`, `/library`, `/pictures/{path}`; `POST /chats/{id}/turns`, `/turns/{id}/good`; `DELETE /chats/{id}`, `/examples/{id}`, `/collections/{name}`, `/collections/{name}/documents/{doc_id}`. `Refused` is a 409 with `{"detail": ...}`. With `api.key` set, a caller sends it as `X-API-Key`. `/docs` and `/openapi.json` are FastAPI's.
- **A turn is a stream** of server-sent events: `event: <type>` and `data: <the event as JSON>`, `Done` last. A turn that fails on the way ends with `Failed` (and is saved with that error). The question is sent with `POST`, so a browser reads the stream with `fetch`, not `EventSource`.
- **Chatbot page** (`ui/chat.py`, `ui/trace_view.py`, `ui/data.py`): the sidebar has *New chat*, *Search in* with the collection or schema, and *Chat history*. The chat id is in the URL (`?chat=<id>`). `ui/data.py` is the page's HTTP client; pictures are fetched from the API too.

### Operations

- The stack is a compose project named after its folder, with two images: `rag-platform-serve` (built by the `setup` service, from `docker/serve.Dockerfile`) and `rag-platform-ingest` (built by `dagster-code`, from `docker/ingest.Dockerfile`).
- **Three compose files.** `docker-compose.yml` has our services only; `docker-compose.local.yml` adds `postgres`, `qdrant` and `provision`; `docker-compose.vm.yml` joins the Docker network of a server's own PostgreSQL and Qdrant (`PLATFORM_NETWORK`). `.env` names the files in `COMPOSE_FILE` (with `COMPOSE_PATH_SEPARATOR=:` so the same line works on Windows and Linux), so the command is always `docker compose up -d`.
- `src/`, `ui/` and `config/` are bind-mounted, so code and settings changes need no rebuild. After editing code under `src/` that the API runs, or `connections.yaml`: `docker compose restart api`. After editing a UI module other than a page script: `docker compose restart ui`. After changing the Dagster definitions (assets, jobs, sensors): `docker compose restart dagster-code dagster-daemon dagster-webserver`. A new dependency goes in the right extra of `pyproject.toml` and needs `docker compose up -d --build`.
- `dagster-code` takes about 50 s to start because it imports Docling; the daemon logs "could not reach user code server" until then. The API starts in a few seconds.
- The `model_cache` volume (`/root/.cache`) holds Docling's models, the tokenizers and NLTK data. Ad hoc `uv pip install` in a container needs `UV_NO_CACHE=1`.
- Schema changes go through a new numbered file in `src/rag_lab/core/migrations/`, never by editing an applied one. A migration names no schema.
- `sqlalchemy<2.1` is pinned: 2.1 switches the default Postgres driver to psycopg 3, which breaks dagster-postgres.
- Ollama must be reachable from the containers. On another machine it must listen on all interfaces (`OLLAMA_HOST=0.0.0.0`); use its IP address, since MagicDNS names may not resolve inside containers. The reranker needs an Ollama with `logprobs` (0.35 or newer).
- `scripts/*.sql` are run by psql inside a Linux container, so they keep LF line ends (`.gitattributes`).
- Streamlit's app tester (`streamlit.testing.v1.AppTest`) can drive the pages inside the `ui` container.
- Do not run `uv run` in the repo without `--no-project`: it builds a `.venv` with every dependency (Docling and torch) and rewrites `uv.lock`.

## Layout

```
docker-compose.yml           # our services
docker-compose.local.yml     # + postgres, qdrant and the provision run (this machine)
docker-compose.vm.yml        # + the network of a server's own databases
.env.example                 # the compose files to use, addresses and passwords (copy to .env)
config/                      # connections.yaml, pipeline.yaml, llm.yaml; connections.vm.example.yaml; schemas/<schema>.yaml
docker/                      # ingest.Dockerfile, serve.Dockerfile, dagster.yaml (mounted), workspace.yaml
scripts/                     # provision.sql (databases, roles, our schema), grant_read.sql (an existing schema to the reader)
src/rag_lab/
  cli.py  __main__.py        # python -m rag_lab search | chat | tables | eval | setup
  core/                      # what both sides need
    config.py                # the settings as pydantic models
    settings.py              # reads config/*.yaml; DATA_DIR and the folders
    connections.py           # Ollama, Qdrant and store clients from connections.yaml
    qdrant.py                # the Qdrant store
    store.py                 # our tables in PostgreSQL, and the migrations
    migrations/              # 0001_init.sql, 0002_registered_schemas.sql
    setup.py                 # check the connections, apply the migrations
    embed.py                 # the Ollama embedder, truncation, BM25
    database.py              # the tables database: roles, schemas, drops
    events.py                # the events of a turn, and Hit: the contract with a front end
  ingest/                    # the batch side (Dagster image)
    source.py                # the PDFs in data/raw
    parse.py  ocr.py         # Docling, the pictures; OCR of regions
    chunk.py  split_text.py  # chunks, Docling's chunkers, picture chunks; the text splitters, tables, the embedding cache
    documents.py             # the stage bodies the assets call, and chunk_document
    folder.py  importer.py   # CSV files -> tables
    register.py              # describe tables that are already in the database
    definitions.py           # Dagster: assets, jobs, sensors
  serve/                     # the serving side (API image)
    search.py                # the four search methods and the reranker
    run.py                   # the runner of a turn, and the chat model
    chat_documents.py  chat_database.py   # the two agents with their prompts
    catalog.py  guard.py  execute.py  examples.py   # the SQL side: schema as text, the guard, running, good answers
    evaluate.py              # the eval of the documents chatbot: cases in, numbers out
    library.py               # what is in each experiment; delete a document or an experiment
    service.py               # what a front end can do
    api.py                   # the same, over HTTP
ui/                          # app.py, experiments.py, chat.py, trace_view.py, hits.py, data.py (the API client), style.py
data/raw/                    # drop PDFs here (git-ignored)
data/tables/<schema>/        # drop CSV files here (git-ignored)
data/artifacts/              # per-stage outputs (git-ignored)
data/eval/                   # the eval cases, <experiment>.jsonl, and results/ (git-ignored)
tests/                       # the few pure-logic tests
docs/history/                # the plans of the earlier phases
```

## Commands

```powershell
docker compose up -d --build          # start the stack; the UI is at http://localhost:8501, the API at http://localhost:8000/docs
docker compose logs -f dagster-code   # code location logs
docker compose logs -f api            # the API
docker compose down                   # stop; add -v to wipe every volume, the model cache included
docker compose exec postgres psql -U <user> -d rag_metrics   # inspect chats and metrics (SET search_path = rag;)

docker compose exec api python -m rag_lab search "query text" --experiment <name> --top-k 5
docker compose exec api python -m rag_lab chat documents "question" --experiment <name>   # no question = a chat loop
docker compose exec api python -m rag_lab chat database "question" --schema <name>
docker compose exec api python -m rag_lab tables schema --schema <name>      # what the SQL agent is given
docker compose exec api python -m rag_lab tables examples [--schema <name>] [--reindex] [--remove <id>]
docker compose exec api python -m rag_lab tables drop --schema <name> [--table <name>]
docker compose exec api python -m rag_lab tables register                    # describe the schemas of existing_schemas
docker compose exec api python -m rag_lab eval --experiment <name> --label <label> [--against <label>]   # the cases of data/eval/<name>.jsonl

# the tests, in a container (in Git Bash prefix with MSYS_NO_PATHCONV=1)
docker compose run --rm --no-deps -e UV_NO_CACHE=1 -v ./tests:/app/tests api sh -c "uv pip install --system -q pytest && python -m pytest /app/tests -q"
```

## Conventions

- Python 3.12, managed with `uv`. Ruff for lint and format.
- The host is Windows: use PowerShell syntax in docs and scripts, and forward slashes in paths inside containers.
- Never hard-code an address or a credential: they belong in `config/connections.yaml` and `.env`.
