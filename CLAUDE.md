# RAG-Platform

A chatbot over two kinds of data, with ingestion run by Dagster:

- **PDF documents** (text, tables, optionally scanned pages and pictures) are parsed, chunked, embedded and indexed into a vector database. The chatbot answers from the retrieved chunks, with citations.
- **CSV files** become tables in a relational database. The chatbot writes a SQL query, checks it, runs it read-only and answers from the rows.

Docling parses, LlamaIndex splits, Ollama embeds, reranks and chats, Qdrant stores the vectors, PostgreSQL holds the tables, the chats and the metrics, Dagster orchestrates, Streamlit is the UI. Everything is configured in three YAML files in `config/`.

The system was cut down from a larger lab in October 2026. `REFACTORPLAN.md` records what was removed and why; `docs/history/` holds the plans of the earlier phases (they describe features that no longer exist). Update this file when the code changes, so that it describes what is in the repo.

## Architecture

```
data/raw/*.pdf              -> Dagster ingest_job: parse -> chunk -> embed -> index -> Qdrant collection
data/tables/<schema>/*.csv  -> Dagster tables_job: import                             -> PostgreSQL schema

Streamlit:  Experiments (the collections, with deletes)    Chatbot (vector database | relational database)
Config:     config/pipeline.yaml   config/llm.yaml   config/connections.yaml   (.env for passwords)
Metrics:    stage metrics, the search log and every chat turn -> PostgreSQL (rag_metrics)
```

| Service (docker-compose.yml) | What it is | Address |
|---|---|---|
| `setup` | Runs once at start and exits: creates the databases, applies the migration, makes the roles | |
| `dagster-code`, `dagster-webserver`, `dagster-daemon` | Dagster; runs execute in `dagster-code` | http://localhost:3000 |
| `ui` | Streamlit: Experiments and Chatbot | http://localhost:8501 |
| `qdrant` | Vector database | HTTP `6333` (dashboard at `/dashboard`), gRPC `6334` |
| `postgres` | Databases `dagster` (Dagster's own), `rag_metrics` (ours), `rag_data` (the imported tables) | `5432` |
| Ollama (not in compose) | Embedding, reranking, chat and OCR models | `OLLAMA_BASE_URL` in `.env` |

This machine has no NVIDIA GPU, so Docling runs on CPU. Ports are bound to `127.0.0.1` only.

## Scope

- **In:** PDF documents (text and tables), and CSV files imported into PostgreSQL, one schema per dataset.
- **In, off by default:** OCR of scanned pages (`parse.ocr`) and the pictures of a PDF as chunks (`parse.pictures`, needs `embeddinggemma-2`).
- **Out:** other document types, audio and video, a login. Docling's own OCR engines and picture features stay off; the only OCR is ours (`parsing/ocr.py`) and pictures are cropped by us (`parsing/pictures.py`).

## Working style

This project is kept minimal. Prefer the smallest change that works, and do not add code, config, abstractions or docs that nothing needs yet.

- **Few tests.** Write tests only for pure logic where a silent bug would corrupt results: vector truncation and re-normalisation, BM25 weights and the reranker score, config hashing and the settings loader, CSV import (cleaning of names, guessing of column types, delimiter detection) and the SQL guard (the one place where a silent bug is a security hole). No tests for Dagster assets, Docling, Docker or anything needing Qdrant, Postgres or Ollama, and no mocks or fakes of them. Do not add test coverage to a change unless asked.
- **No re-testing after minor edits.** After a small change (rename, config tweak, comment, log line, doc edit, small refactor) do not run the test suite, rebuild containers or re-materialise assets. Run something only when a change alters the logic of one of the tested areas above, or when asked. Never rerun a passing check "to be sure".
- **Verify by running the real thing, once.** When a piece of work is finished, confirm it with a single real run on the stack. Do not build extra verification around it.
- **Keep dependencies and services few.** Do not add a library, container or tool without a stated need.

## Design rules

- **Every choice is a value in `config/*.yaml`**, never a constant in code. A key left out keeps the default written in `config.py`.
- **Stages are modular.** Parsing, chunking, embedding, storage, search, the agents and the SQL side live in separate packages under `src/rag_lab/` as plain Python with no Dagster or Streamlit imports. The Dagster assets in `assets/` and the pages in `ui/` are thin callers.
- **One Qdrant collection per experiment**, named after it. An experiment is the collection plus the settings it was made with, recorded in full in `rag_metrics.experiments`. **Settings cannot change under a name:** a name is unique in the database, and every writer refuses a name that already has other settings.
- **A search reads the settings stored with the collection**, not the files as they are now, so a question is always embedded the way the documents were.
- **Vectors are supplied by us.** Qdrant only stores and searches; embeddings come from Ollama.
- **The databases have owners.** Never write to `dagster`. Our tables live in `rag_metrics`. `rag_data` is written only by the loader role and read only by the reader role.
- **Passwords and private addresses stay in `.env`.** `connections.yaml` refers to them as `${NAME}`.
- **Intermediates are kept on disk** under `data/artifacts/<experiment>/<stage>/`, so a stage can be re-run without redoing the ones before it.

## Configuration

`rag_lab/settings.py` reads three files from `config/` (`RAG_CONFIG_DIR`). An unknown key is an error naming the file and the key; `${NAME}` is replaced from the environment and a missing variable is an error.

| File | Holds | Read |
|---|---|---|
| `connections.yaml` | Ollama, Qdrant, the app database, the tables database (loader and reader URLs) | The UI keeps its clients while it runs: `docker compose restart ui` after an edit |
| `pipeline.yaml` | `name` (the experiment and collection), `parse`, `chunk`, `index`, and `tables` (CSV limits) | By Dagster at every sensor tick and every run |
| `llm.yaml` | `embedding`, `ocr`, `reranker`, `chat` (shared), `documents_chat`, `database_chat` | By Dagster at every run, by the Chatbot at every question |

- `settings.load()` returns `connections`, `experiment` (an `ExperimentConfig`: `pipeline.yaml` plus the embedding and OCR models of `llm.yaml`), `tables` (`ImportConfig`), `documents_chat` (`AgentConfig`, with the reranker) and `database_chat` (`SqlAgentConfig`, with the embedding model for good answers).
- `chat` is merged under both chat sections; a chat's own section wins. `documents_chat.reranker` and `database_chat.embed` are refused: set `reranker` and `embedding`.
- `rag_lab/clients.py` builds the embedder, the reranker, the Qdrant store and the metrics store from `connections.yaml`. Nothing else reads an address. The only environment variables the code reads are `DATA_DIR` and `RAG_CONFIG_DIR`.
- The database and role names of the tables database are taken from its two URLs.
- Dagster's own database is not in `connections.yaml`: Dagster reads `docker/dagster/dagster.yaml`, which uses the same `POSTGRES_USER` and `POSTGRES_PASSWORD`.
- `config.py` holds the models as plain pydantic (`extra="forbid"`, frozen): change a value with `model_copy(update=...)`.

## Key technical facts

### Ingestion of PDFs

- **Sensor** (`assets/sensors.py: new_pdf_sensor`, every 30 s): a PDF that is new in `data/raw` gets a partition and an `ingest_job` run. New means not yet a partition: the same bytes under another name, or a PDF whose run failed, start nothing; run that partition by hand. A file is taken only when it can be read and its last kilobyte has `%%EOF`. When the settings do not load, or name an experiment that exists with other settings, the tick fails with the reason and the PDF stays new.
- **No run config and no Dagster resources.** Each of the four assets (`parsed_document`, `chunks`, `embeddings`, `qdrant_index`) starts with `ingest.configured_experiment()`, which loads the files, refuses a conflict and registers the experiment. `ingest_job` uses the in-process executor: with the default one every step imported Docling again (about 45 s each). `dagster.yaml` limits the queue to one run at a time.
- A document's id is the first 16 hex characters of the SHA-256 of its bytes; it is also the partition key.
- `ExperimentConfig.config_hash()` covers the name and every setting, except the settings of the chunking strategies that are not chosen.
- **A parse is whole or it fails.** Docling returns `partial_success` when it hits its time limit; `parse_pdf` raises, with the pages reached and the settings to raise. The limit is `document_timeout` (600 s) or `page_timeout` (20 s) per page, whichever is longer.
- **Chunking** (`chunking/`): `hybrid` and `hierarchical` are Docling's chunkers (`docling_chunkers.py`, tables as Markdown, `row-wise` unsupported); `fixed`, `recursive` and `semantic` split the text under each heading with LlamaIndex splitters (`text_splitters.py`) and handle tables by `table_handling` (`tables.py`). `overlap` applies to `fixed` only. Sizes are counted with the embedding family's Hugging Face tokenizer, and headings are part of each chunk's text. `semantic` calls Ollama for sentence embeddings, cached in `data/artifacts/_cache/embeddings/`.
- **Embedding families** (`config.EMBED_FAMILIES`): a model belongs to the family its Ollama name starts with, which gives its tokenizer, its query and document templates and the sizes its vectors can be cut to. `qwen3-embedding` (0.6b, 4b, 8b): asymmetric, queries get `Instruct: <task>\nQuery: <text>`, sizes 256 to 1024. `embeddinggemma-2` (740m, 768 dimensions): `task: search result | query: {text}` and `title: none | text: {text}`, sizes 128 to 768, and it takes images. `settings.py` fills these in from the model name; a model of no known family needs them set in `llm.yaml`. Truncation and re-normalisation are ours (`embedding/vectors.py`). Use Ollama's `/api/embed`.
- **OCR** (`parsing/ocr.py`, `glm-ocr:bf16`): Docling runs with `do_ocr=False` and keeps empty layout regions; each region with no text is cropped and read through `/api/chat`, so a page with a text layer keeps it. The model is used per region, never per page. It does not stop by itself on Ollama 0.40, so the reply is read as a stream and cut at the end of the first reading. There is no fallback: with OCR on, a parse needs Ollama.
- **Pictures** (`parsing/pictures.py`, `chunking/pictures.py`): each kept picture is cropped into `<doc_id>.pictures/<n>.png` and becomes a chunk whose text is its caption (or "Figure on page N"). `embedding.picture_input` is `image+caption`, `image` or `caption`. **An image goes inside `input` as an object (`{"image": "<base64>"}`); a top-level `images` field is silently ignored by `/api/embed`.** A family that takes no images is refused when the settings are loaded.
- **BM25** (`index.sparse`, on by default; `embedding/sparse.py`): a named sparse vector `bm25` with Qdrant's IDF modifier, written after the dense vector. It cannot be added to a collection later, and the Chatbot only offers collections that have it. The dense vector is the unnamed one.
- Writes go through LlamaIndex's `QdrantVectorStore` into a collection `QdrantStore.ensure_collection` created. Point ids are deterministic UUIDs of the chunk ids, so re-ingesting overwrites. Qdrant's storage must be a named volume: a Windows bind mount can corrupt it.

### Import of CSV files

- **Folder** (`sql/folder.py`): `data/tables/<schema>/<table>.csv`. The folder is the schema, the cleaned file name is the table. `_schema.yaml` in the folder is optional: the dataset's description, tables' and columns' descriptions, and a column type to use instead of the guessed one. A column it names that the file does not have is an error.
- **Sensor** (`new_csv_sensor`): asks for a run of `tables_job` for every settled file at every tick, keyed by the content hash of the file and of `_schema.yaml`; Dagster skips the keys it has seen. So the same file is imported once, an edit imports it again, and a failed import waits for the file to change (or for its partition, `schema.table`, to be run by hand).
- **Import** (`sql/importer.py`): UTF-8 only; the delimiter is sniffed; names are cleaned (letters of any language are kept); types are guessed from up to 10,000 rows. One transaction as the loader role: create the table (replacing it), `COPY` the file, profile every column. A bad row leaves the old table and the error names the line.
- **Roles** (`sql/bootstrap.py`): the loader may create schemas and tables; the reader may only `SELECT`, with `default_transaction_read_only`, a 15 s statement timeout and 10 connections. The grants are the boundary, not the read-only setting.
- Dropping (`python -m rag_lab.sql drop`) never touches the CSV files.

### Search and the chatbot

- **Search** (`search/engine.py`): `dense`, `hybrid` (dense and BM25 prefetches fused with RRF), `dense+rerank`, `hybrid+rerank`. A hit's `similarity` is a cosine similarity only for `dense`.
- **Reranker** (`reranking/ollama.py`): one `/api/generate` call per (query, chunk) with the Qwen3-Reranker prompt, one token and `logprobs`, scored `P(yes)/(P(yes)+P(no))`. `dengcao/Qwen3-Reranker-4B` works (`Q8_0`, `Q4_K_M`); the 0.6B builds give every chunk 0.
- **Documents agent** (`agent/documents/`, LangGraph): `condense -> retrieve -> grade -> generate`, with `grade -> rewrite -> retrieve` while the chunks do not answer and `grade -> abstain` when the retries are used up. Retrieval is always `hybrid+rerank`. `grade` uses the best reranker score: at or above `enough_score` it answers, below `missing_score` it does not, in between the model is asked. The answer cites `[n]`. Retrieved pictures are attached to the answer prompt with `Passage [n]` drawn above them.
- **SQL agent** (`agent/sql/`): `condense -> schema -> examples -> write_sql -> check -> run_sql -> answer`, with `repair` up to `max_repairs` times and `abstain` when the schema is too large, the model replies `CANNOT`, or the repairs are used up. The model is given the whole schema as text (`sql/catalog.py`, from the registry: descriptions, types, example values, ranges, likely joins); there is no retrieval. A schema over `schema_char_budget` is refused, not cut.
- **The SQL guard** (`sql/guard.py`, sqlglot): accepts exactly one `SELECT`, refuses any write, DDL, locking clause, unknown function or table outside the chat's schema, qualifies tables and adds a `LIMIT`. What runs is the SQL sqlglot writes out, not the model's text. **Known limit: the guard is the only thing keeping one dataset from another**, since the reader may select from every schema.
- **Good answers** (`sql/examples.py`): the thumbs-up under a database answer saves the question and its SQL; they are embedded with `llm.yaml`'s `embedding` into `sqlexamples__<schema>` in Qdrant and shown to the model for similar questions. `examples_min_score` depends on the embedding model (0.8 for `embeddinggemma-2`, about 0.55 for `qwen3-embedding:0.6b`). Changes go table first, then Qdrant; `python -m rag_lab.sql examples --reindex` rebuilds.
- **Events** (`agent/events.py`): every node reports typed events through LangGraph's custom stream. `agent.run(graph, flow, question, history, metrics=, session_id=)` yields them, ends with `Done`, and saves the turn (also a failed one, with `error`). A saved turn keeps its events, so the page replays a past chat through the same `apply(trace, event)` that draws a live one.
- **Chats**: a chat is a row in `chat_sessions` with a `kind` (`documents` or `database`) for its whole life; a composite foreign key refuses a turn of the other kind. It belongs to its experiment or its schema and is deleted with it.
- **Chatbot page** (`ui/chat.py`, `ui/trace_view.py`): the sidebar has *New chat*, *Search in* with the collection or schema, and *Chat history*. The chat id is in the URL (`?chat=<id>`). Before a question the page checks that Ollama has the models (`ui/data.py: check_models`): a missing model stops the chat with the key to change; a chat model that cannot think or see images has that setting switched off with a note.
- `num_ctx` is always set, because Ollama's default silently cuts long prompts.

### Operations

- The stack is a compose project named after its folder, with the image `rag-platform:latest` (built by the `setup` service) and its own volumes.
- `src/`, `ui/` and `config/` are bind-mounted, so code and settings changes need no rebuild. After editing a UI module other than a page script: `docker compose restart ui`. After changing the Dagster definitions (assets, jobs, sensors): `docker compose restart dagster-code dagster-daemon dagster-webserver`. A new dependency needs `docker compose up -d --build`.
- `dagster-code` takes about 50 s to start because it imports Docling; the daemon logs "could not reach user code server" until then.
- The `model_cache` volume (`/root/.cache`) holds Docling's models, the tokenizers and NLTK data. Ad hoc `uv pip install` in a container needs `UV_NO_CACHE=1`.
- Schema changes go through a new numbered file in `src/rag_lab/metrics/migrations/`, never by editing an applied one.
- `sqlalchemy<2.1` is pinned: 2.1 switches the default Postgres driver to psycopg 3, which breaks dagster-postgres.
- Ollama must be reachable from the containers. On another machine it must listen on all interfaces (`OLLAMA_HOST=0.0.0.0`); use its IP address, since MagicDNS names may not resolve inside containers. The reranker needs an Ollama with `logprobs` (0.35 or newer).
- Streamlit's app tester (`streamlit.testing.v1.AppTest`) can drive the pages inside the `ui` container.

## Layout

```
docker-compose.yml
.env.example                 # OLLAMA_BASE_URL and the passwords (copy to .env)
config/                      # connections.yaml, pipeline.yaml, llm.yaml
docker/dagster/              # Dockerfile, dagster.yaml (mounted), workspace.yaml
src/rag_lab/
  config.py                  # the settings as pydantic models
  settings.py                # reads config/*.yaml
  clients.py                 # Ollama, Qdrant and metrics clients from connections.yaml
  setup.py                   # python -m rag_lab.setup: databases, migration, roles
  ingest.py                  # the stage bodies the assets call
  library.py                 # what is in each experiment; delete a document or an experiment
  documents.py  paths.py     # the PDFs in data/raw; DATA_DIR, the folders
  parsing/                   # parse.py (Docling), ocr.py, pictures.py
  chunking/                  # docling_chunkers.py, text_splitters.py, tables.py, segment.py, pictures.py, tokens.py
  embedding/                 # ollama.py, vectors.py, sparse.py (BM25), cache.py, llamaindex.py
  reranking/                 # ollama.py
  storage/                   # qdrant.py
  search/                    # engine.py, __main__.py (CLI)
  sql/                       # bootstrap.py, database.py, folder.py, importer.py, catalog.py, guard.py, execute.py, examples.py, __main__.py
  agent/                     # events.py, model.py, run.py, printer.py; documents/ and sql/ (graph.py, prompts.py, cli.py)
  metrics/                   # store.py, timing.py, migrate.py, migrations/0001_init.sql
  assets/                    # parsing.py, chunking.py, indexing.py, tables.py, jobs.py, sensors.py, partitions.py
  definitions.py             # Dagster entry point
ui/                          # app.py, experiments.py, chat.py, trace_view.py, hits.py, data.py, style.py
data/raw/                    # drop PDFs here (git-ignored)
data/tables/<schema>/        # drop CSV files here (git-ignored)
data/artifacts/              # per-stage outputs (git-ignored)
tests/                       # the few pure-logic tests
docs/history/                # the plans of the earlier phases
```

## Commands

```powershell
docker compose up -d --build          # start the stack; the UI is at http://localhost:8501
docker compose logs -f dagster-code   # code location logs
docker compose down                   # stop; add -v to wipe every volume, the model cache included
docker compose exec postgres psql -U <user> -d rag_metrics   # inspect chats and metrics

docker compose exec dagster-code python -m rag_lab.search "query text" --experiment <name> --top-k 5
docker compose exec dagster-code python -m rag_lab.agent documents "question" --experiment <name>   # no question = a chat loop
docker compose exec dagster-code python -m rag_lab.agent sql "question" --schema <name>
docker compose exec dagster-code python -m rag_lab.sql schema --schema <name>      # what the SQL agent is given
docker compose exec dagster-code python -m rag_lab.sql examples [--schema <name>] [--reindex] [--remove <id>]
docker compose exec dagster-code python -m rag_lab.sql drop --schema <name> [--table <name>]

# the tests, in a container (in Git Bash prefix with MSYS_NO_PATHCONV=1)
docker compose run --rm --no-deps -e UV_NO_CACHE=1 -v ./tests:/app/tests dagster-code sh -c "uv pip install --system -q pytest && python -m pytest /app/tests -q"
```

## Conventions

- Python 3.12, managed with `uv`. Ruff for lint and format.
- The host is Windows: use PowerShell syntax in docs and scripts, and forward slashes in paths inside containers.
- Never hard-code an address or a credential: they belong in `config/connections.yaml` and `.env`.
