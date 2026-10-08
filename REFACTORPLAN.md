# Refactor plan

Written 2026-10-08. **Phases 1 to 6 are done** (each has a status note with what was checked and what was built differently); Phase 7 is optional and not started. The current system is described in `CLAUDE.md` and `README.md`.

## Goal

A smaller system that does two things and is configured in YAML:

- **Ingestion is Dagster only.** Put a PDF in a folder and it is parsed, chunked, embedded and indexed into the vector database. Put a CSV in a folder and it becomes a table in the relational database.
- **The UI is two pages.** *Experiments* (the vector database collections and what is in them, with deletes) and *Chatbot* (new chat, chat history, and a choice of what to search: the vector database or the relational database).
- **Three YAML files** hold the configuration: the pipeline, the models, the connections.

## What you decided

| Question | Answer |
|---|---|
| How do tables get into the relational database without the Database page? | A Dagster sensor on a folder, like PDFs. Descriptions go in a small YAML next to the CSV files. |
| Must existing experiments, collections and chats keep working? | No. Start from empty databases and re-ingest. |
| Optional features | All four stay: OCR, pictures as chunks, good answers (the thumbs-up), and "How this was answered" under each answer. |
| Where do passwords live? | In `.env`. `connections.yaml` is committed and refers to them as `${POSTGRES_PASSWORD}`. |

## The system afterwards

```
data/raw/*.pdf            -> Dagster: parse -> chunk -> embed -> index -> Qdrant collection
data/tables/<schema>/*.csv -> Dagster: import                           -> PostgreSQL schema

Streamlit:  Experiments (collections, delete)   Chatbot (vector database | relational database)
Config:     config/pipeline.yaml   config/llm.yaml   config/connections.yaml   (+ .env for passwords)
```

### What goes and what stays

| | Goes | Stays |
|---|---|---|
| UI pages | Upload, Try a query, Database, Benchmark (and `preview.py`, `charts.py`) | Experiments, Chatbot (`trace_view.py`, `hits.py`, `style.py`, a smaller `data.py`) |
| Ingestion | Uploading in the UI, `data/uploads/`, the upload parse cache | The Dagster pipeline: four assets, `ingest_job`, the PDF sensor |
| Benchmark | All of it: `benchmark/`, `metrics/retrieval.py`, its tables, its store methods, `reportlab` | |
| Chunking | The `native` engine (our own `fixed`, `recursive`, `semantic` splitters), chunk spans (only the Upload preview and the benchmark read them) | The five strategies, through the LlamaIndex engine |
| Search | `search/quick.py`, `notebook/`, `ipykernel` | `search()` with its four methods, the search CLI |
| Tables | The CSV editor in the UI, the copy of each CSV under `data/csv/` | The importer, the catalog, the guard, the executor, the text-to-SQL agent, good answers |
| Config | `tag`, `settings_hash()`, `chunk.engine`, every "keeps the hashes of older experiments" rule, `ExperimentResource` and run config, environment variables for addresses | The stage settings, now read from YAML |
| Database | Migrations `0001` to `0011`, the Postgres init script, the code that rebuilds chat turns saved before `0008` | One migration with the final tables, the migration runner |

A rough count: about 11,100 lines of code and config today, about 3,500 to 4,000 of them removed and about 400 added. This is an estimate from file sizes, not a measurement.

### Layout afterwards

```
config/
  pipeline.yaml  llm.yaml  connections.yaml
src/rag_lab/
  config.py        the settings as pydantic models, one top-level model per YAML file
  settings.py      reads config/*.yaml into those models
  clients.py       the Ollama, Qdrant and PostgreSQL clients, built from connections.yaml
  setup.py         databases, roles and migrations; run once when the stack starts
  ingest.py        the stage bodies the assets call (register, embed, index, metric rows)
  library.py       what is in each collection; delete a document or a collection
  documents.py  paths.py
  parsing/         parse.py  ocr.py  pictures.py
  chunking/        the LlamaIndex engine, segment.py, tables, pictures.py, tokens.py, models.py
  embedding/  reranking/  storage/  search/
  sql/             database.py  bootstrap.py  importer.py  catalog.py  guard.py  execute.py  examples.py
  agent/           unchanged: shared runner and events, documents/ and sql/ flows
  metrics/         store.py  timing.py  migrate.py  migrations/0001_init.sql
  assets/          the document assets, tables.py (CSV import), jobs.py, sensors.py, partitions.py
ui/                app.py  experiments.py  chat.py  trace_view.py  hits.py  data.py  style.py
tests/             config, vectors, SQL guard, CSV import
```

The rule that stage code has no Dagster or Streamlit imports stays. After Phase 3 `config.py` no longer imports Dagster either.

## The three YAML files

A key left out keeps the default written in `config.py`. An unknown key is an error that names the file and the key. `${NAME}` is replaced with the environment variable of that name, and a missing variable is an error that names it.

**`config/pipeline.yaml`**: what a PDF or a CSV goes through.

```yaml
name: handbook            # the Qdrant collection. Settings cannot change under a name: change the name too.

parse:                    # Docling
  table_mode: accurate    # fast or accurate
  ocr: false              # read scanned pages with the OCR model in llm.yaml
  pictures: false         # each picture becomes a chunk; needs an embedding model that takes images
chunk:
  strategy: hybrid        # hybrid, hierarchical, fixed, recursive or semantic
  max_tokens: 512
index:
  sparse: true            # the BM25 vector the chatbot's hybrid search needs

tables:                   # CSV import limits
  max_bytes: 52428800
  max_rows: 1000000
```

**`config/llm.yaml`**: every model, and how it is called.

```yaml
embedding:                # part of a collection's identity, like the pipeline settings
  model: qwen3-embedding:0.6b
  dimension: null         # null keeps the model's own size
ocr:
  model: glm-ocr:bf16
reranker:
  model: dengcao/Qwen3-Reranker-4B:Q8_0
chat:                     # shared by both kinds of chat
  model: gemma4:e4b-mlx
  think: true
  num_ctx: 8192
  history_turns: 6
documents_chat:           # the vector database
  top_k: 5
  candidates: 20          # hits the reranker scores
  enough_score: 0.5
  missing_score: 0.1
  max_rewrites: 1
  show_pictures: true
database_chat:            # the relational database
  temperature: 0
  num_ctx: 16384
  max_repairs: 2
  row_limit: 200
  use_examples: true
```

**`config/connections.yaml`**: where everything is. Swapping a server is an edit here.

```yaml
ollama:
  url: ${OLLAMA_BASE_URL}   # a private address, so it stays in .env
qdrant:
  url: http://qdrant:6333
app_database:             # chats, metrics, the registry of tables
  url: postgresql://${POSTGRES_USER}:${POSTGRES_PASSWORD}@postgres:5432/rag_metrics
tables_database:          # the imported CSV tables
  loader_url: postgresql://rag_loader:${SQL_LOADER_PASSWORD}@postgres:5432/rag_data
  reader_url: postgresql://rag_reader:${SQL_READER_PASSWORD}@postgres:5432/rag_data
```

Two things that follow from this split:

- **The embedding model is in `llm.yaml` but belongs to a collection.** A collection is made from `pipeline.yaml` plus the `embedding` section. Changing either without changing `name` is refused by the sensor with the reason, as today. A question is always embedded with the settings stored for the collection it searches, so an edit to `llm.yaml` cannot mismatch an older collection.
- **Dagster's own database is not in `connections.yaml`.** Dagster reads its storage from `docker/dagster/dagster.yaml`, which only takes environment variables. It uses the same `POSTGRES_USER` and `POSTGRES_PASSWORD` from `.env`. A comment in `connections.yaml` will say so.

## Phases

### Phase 1: cut the pages and the code only they used

No data changes; what is in Qdrant and Postgres still works after this phase.

- [x] 1. `ui/`: delete `upload.py`, `query.py`, `database.py`, `benchmark.py`, `preview.py`, `charts.py`. `app.py` lists Experiments and Chatbot.
- [x] 2. Delete `src/rag_lab/benchmark/`, `metrics/retrieval.py`, `tests/test_retrieval.py`, `search/quick.py`, `notebook/`.
- [x] 3. `metrics/store.py`: remove the report methods (`create_report` to `get_results`) and the `benchmarked` column of `list_experiments`. `library.py` and `experiments.py`: remove `in_report`.
- [x] 4. `ingest.py`: remove `save_upload`, `ensure_parsed`, `ingest_document`, `check_new_experiment_name`, `experiments_with_settings`, `clean_experiment_name`.
- [x] 5. `ui/data.py` and `ui/style.py`: remove the chart helpers and the model lists nothing calls any more.
- [x] 6. `pyproject.toml`: remove `reportlab` and `ipykernel`.

**Done when:** after `docker compose restart ui` the UI has two pages, Experiments lists the collections, and a chat question is answered. Looked at once.

**Status (2026-10-08):** the six steps are done. The stack runs from this folder as its own compose project (`rag-platform`, image `rag-platform:latest`, its own empty volumes; `Desktop\RAG-Dagster` and its stack are not touched). Driven with Streamlit's app tester in the `ui` container: the app has two pages and both load without an exception (Chatbot shows "No experiment with a BM25 vector exists yet", Experiments shows "No experiments yet"). Then with a real document: `flying_handbook.pdf` put into `data/raw/` was ingested by the sensor (51 points in `test-ingest`), the Experiments page lists it, and a question on the Chatbot page was answered with a citation and saved. **Phase 1 is done.** Found on the way: `config/ingest.yaml` did not parse (a comment line under `embed:` had lost its `#`), so the sensor could not have started a run; fixed. `uv.lock` was left as it was: it was already behind `pyproject.toml`, and re-locking rewrote about 2,000 lines of it.

### Phase 2: one chunking engine, simpler settings, a fresh database

This is the phase that needs the wipe: removing settings keys makes the stored settings of old experiments unreadable.

- [x] 1. Chunking: delete `fixed.py`, `recursive.py`, `semantic.py`, `spans.py` and the strategy registry in `base.py`. Keep what the LlamaIndex engine uses (`segment.py`, the table chunks from `sectioned.py`, the Docling chunker builder from `native.py`) and fold the leftovers into two readable modules. Remove `Chunk.span` and `span_approx`. Delete `metrics/stats.py`.
- [x] 2. Settings: remove `chunk.engine`, `recursive.separators`, `semantic.breakpoint_type`, `semantic.min_tokens` (the LlamaIndex engine never supported them), `tag` and `settings_hash()`. `index.sparse` defaults to true.
- [x] 3. `config_hash()` becomes the hash of the name and every setting, with one exception kept: the settings of the chunking strategies that are not chosen. All the rules that keep older hashes go.
- [x] 4. One migration, `0001_init.sql`, with the final tables: `experiments`, `stage_metrics`, `search_log`, `chat_sessions`, `chat_turns`, `db_schemas`, `db_tables`, `sql_examples` (without `enabled`). No benchmark tables, no summary view, no backfills.
- [x] 5. `ui/trace_view.py`: remove the rebuilding of turns saved without events.
- [x] 6. `rag_lab/setup.py` (`python -m rag_lab.setup`): create the two databases if missing, apply the migrations, create the roles. A one-shot `setup` service in compose runs it, and the UI and Dagster wait for it. Remove the migration calls from `ui/app.py`, the Dagster resource and the CLIs, and delete `docker/postgres/init/`.
- [x] 7. Tests: the hash tests shrink to three (the name changes it, a setting changes it, an unused strategy's setting does not). Run the suite once, because the hash logic changed.
- [x] 8. **Wipe, after you confirm:** remove the `postgres_data` and `qdrant_data` volumes only (not `model_cache`, which holds several GB of Docling models), and empty `data/artifacts/`, `data/uploads/` and `data/csv/`. Dagster's partitions go with its database, so every PDF in `data/raw/` counts as new and is ingested again by the sensor.

**Done when:** on the empty stack a PDF in `data/raw/` is ingested by the sensor, the Experiments page shows it, and a chat question is answered with a citation. Checked once.

**Status (2026-10-08): Phase 2 is done.** The suite passes (42 tests, run once in a container). After the wipe (`rag-platform_postgres_data`, `rag-platform_qdrant_data` and `data/artifacts/`), the `setup` service created both databases, applied `0001_init` and made the roles; the sensor ingested `flying_handbook.pdf` again in about 3 minutes (51 chunks, 51 points with the BM25 vector); the Experiments page lists it; a chat question was answered with a citation, and the chat reopened from its saved events. The pages were driven with the app tester, not looked at in a browser.

Built differently from the plan:

- The chunking package is `docling_chunkers.py` (`hybrid`, `hierarchical`), `text_splitters.py` (`fixed`, `recursive`, `semantic`) and `tables.py`, with `segment.py`, `pictures.py`, `tokens.py`, `models.py` and `base.py` (the context) kept.
- The stage metrics table keeps its name, `ingestion_stage_metrics`; renaming it would only have changed three queries for no gain.
- `experiments.name` is unique in the database, and `register_experiment` refuses a name that already has other settings, so every writer is covered and not only the sensor (the plan had this in Phase 3).
- `chat_turns.events` is required: every turn is saved with its events.
- `python -m rag_lab.sql bootstrap` is gone; `python -m rag_lab.setup` does it together with the migrations.
- The image is built by the `setup` service, the first to start, so `docker compose up` works without `--build` on a machine that has no image yet.

### Phase 3: configuration in three YAML files

- [x] 1. `config.py`: plain pydantic models instead of `dagster.Config`, one top-level model per file (`Pipeline`, `Llm`, `Connections`) whose fields are the file's keys. `ExperimentConfig` is built from `pipeline.yaml` plus `llm.embedding` and `llm.ocr`, so the stage code does not change.
- [x] 2. `settings.py`: `load()` reads the three files from `RAG_CONFIG_DIR` (default `config/`), replaces `${NAME}`, and reports a bad key with its file. The embedding model's tokenizer and prompt templates are filled from its family, as `load_experiment_file` does today.
- [x] 3. `clients.py`: `embedder()`, `reranker()`, `qdrant()`, `metrics()` and the two table-database URLs, from `connections.yaml`. Every `os.environ[...]` read of an address goes (about 25 places in `ui/`, the CLIs, `sql/database.py`, `sql/bootstrap.py`) along with the `services()` copies in the pages.
- [x] 4. Dagster: delete `resources/`. Each asset gets its clients from `clients.py` and its settings from `settings.load()` when the run starts, and refuses to run when the name already exists with other settings. The sensor keeps its check, so a bad file fails the tick with the reason and no PDF is registered. Runs no longer carry run config: re-running a document is one click, with no launchpad form to fill.
- [x] 5. Write the three files from today's defaults. Delete `config/ingest.yaml`.
- [x] 6. Compose: mount `./config` read-only into the UI and the Dagster services. The environment block shrinks to the passwords, `DATA_DIR` and `RAG_CONFIG_DIR`. `.env.example` lists only passwords.
- [x] 7. Tests: three for the loader (`${NAME}` is replaced and a missing one is an error, an unknown key names its file, an embedding model brings its family's templates). Run the suite once.

**Done when:** changing `reranker.model` in `llm.yaml` and restarting the UI changes the reranker recorded in the next chat turn's settings; a misspelled key stops the start with a message naming the file and the key; and a PDF dropped after changing `chunk.strategy` and `name` lands in the new collection. Checked once.

Reading rule: Dagster reads the files at every sensor tick and every run, so an edit needs no restart. The Chatbot reads `llm.yaml` at every question. The UI keeps its clients while it runs, so an edit of `connections.yaml` needs `docker compose restart ui`.

**Status (2026-10-08): Phase 3 is done.** The suite passes (43 tests, run once). Checked on the running stack:

- The experiment built from the three files has the same hash as the one stored for `test-ingest`, so nothing had to be ingested again.
- With `reranker.model` set to the 0.6B build, the Chatbot page offered it by default and the next turn was saved with it in its settings (and abstained with a best score of 0.000, as that build does). The file was put back.
- A misspelled `chunk.stratgy` stopped the `setup` service with `/app/config/pipeline.yaml: chunk.stratgy: unknown key`, exit code 1.
- With `name: refactor-check` and `chunk.strategy: recursive`, a new one-page PDF put into `data/raw/` was started by the sensor (no run config) and landed in the collection `refactor-check`. The PDF, the experiment and its partition were removed afterwards and the file was put back.
- The same name with another `max_tokens` is refused with the reason by every stage.
- The search, agent and `rag_lab.sql` command lines work with the new clients.

Built differently from the plan:

- `candidates` is under `documents_chat`, not `reranker`: it belongs to the search, and the classes map to the file key for key. `chat` is the shared section; a chat's own section wins.
- The Ollama address is `${OLLAMA_BASE_URL}` in `connections.yaml`, since it can be a private address; `.env.example` therefore still lists it next to the passwords.
- The database and role names of the tables database are read from the URLs in `connections.yaml`; nothing in the code names `rag_data`, `rag_loader` or `rag_reader` any more.
- Good answers are embedded with `llm.yaml`'s `embedding` (before: always `qwen3-embedding:0.6b`). `examples_min_score` (0.55) was chosen with that model; look at the scores again when examples exist.
- `./config` is mounted into `setup`, `dagster-code` and `ui`, the three that run our code; the Dagster webserver and daemon do not need it.
- `RerankerConfig` is a section of its own inside `SearchConfig`, and `AgentConfig` holds `candidates` and `reranker` directly, so a saved turn's settings name the reranker model.

### Phase 4: CSV files through Dagster

- [x] 1. Folder: `data/tables/<schema>/<table>.csv`. The folder name is the schema, the file name (cleaned) is the table. An optional `data/tables/<schema>/_schema.yaml` holds the dataset's description, each table's and column's description, and a column type to use instead of the guessed one.
- [x] 2. `assets/tables.py`: an asset `imported_table`, one partition per `schema.table`, and `tables_job`. The body creates the schema if missing, reads names and types with `parse_sample`, imports with `import_csv(replace=True)` and writes the descriptions. A bad row fails the run with the importer's message (`COPY orders, line 3001, column total`) and leaves the old table.
- [x] 3. `new_csv_sensor`: a run for each CSV that is new or whose content (or `_schema.yaml`) changed, keyed by the content hash, so the same file is never imported twice. A file still being copied waits for a later tick.
- [x] 4. `sql/`: remove the copy under `data/csv/` (the file in `data/tables/` is the source). `ImportConfig` loses `preview_rows`.
- [x] 5. `python -m rag_lab.sql`: add `drop --schema <name> [--table <name>]` and `examples --remove <id>`, since the page that did these is gone.

**Done when:** a CSV put into `data/tables/sales/` becomes a table; `python -m rag_lab.sql schema --schema sales` prints it with a description from `_schema.yaml`; editing the CSV imports it again; a CSV with a bad row fails its run and the old table is still there. Checked once.

**Status (2026-10-08): Phase 4 is done.** Checked on the running stack with a sample dataset, `data/tables/sales/` (`customers.csv`, `orders.csv` and a `_schema.yaml`), which is still there for Phase 5:

- The sensor imported both files (6 and 12 rows). `python -m rag_lab.sql schema --schema sales` prints the dataset's, the tables' and the columns' descriptions from `_schema.yaml`, the guessed types and a likely join.
- A row appended to `orders.csv` imported the file again: 13 rows.
- A row with one field too many failed its run with `extra data after last expected column. COPY orders, line 15`, and the table kept its 13 rows.
- `drop --schema scratch --table my_notes` and `drop --schema scratch` removed a throwaway table and schema (the file `My Notes.csv` had become the table `my_notes`); `examples --remove 99` says there is no such example.
- `python -m rag_lab.agent sql "What is the total of shipped orders for each country?" --schema sales` wrote a join, ran it as the reader and answered with the right totals.

Built differently from the plan:

- The folder logic is plain Python in `sql/folder.py` (the files, `_schema.yaml`, `import_file`); `assets/tables.py` is only the asset. The partitions, the job and the sensor are in the existing `partitions.py`, `jobs.py` and `sensors.py`.
- The sensor asks for a run of every settled file at every tick and Dagster skips the run keys it has already seen, so there is no cursor to keep.
- `_schema.yaml` may describe a table whose file is not there yet (so the YAML can be written first); a column it names that the file does not have is an error.
- A folder whose name cannot be a schema, or two files that give the same table name, are reported by the sensor and skipped; the other files are still imported.
- No tests were added or run: the importer's tested logic (names, types, delimiter) did not change.

### Phase 5: the Chatbot sidebar

- [x] 1. The sidebar holds only: **New chat**; **Search in** (*Vector database* or *Relational database*, then which collection or schema, both locked after the first question as today); **Chat history** (the past chats, with the delete of the open chat).
- [x] 2. The chat model, thinking, reranker, top k, candidates and the picture toggle leave the page. Both agents are built from `llm.yaml`.
- [x] 3. One check when the page loads: the models named in `llm.yaml` are installed in Ollama, and `chat.think` and `documents_chat.show_pictures` are only on for a model that can do them. A problem is shown with the YAML key to change, not a failed turn.
- [x] 4. Experiments page: wording for the new flow (a PDF goes into `data/raw/`). A document deleted from a collection keeps its Dagster partition, so it is ingested again by re-running that partition, not by the sensor; the page says so.

**Done when:** in the browser, a new chat on each source answers, the sidebar shows only the three things, a refresh reopens the chat, and a thumbs-up on a database answer is remembered. Looked at once.

**Status (2026-10-08): Phase 5 is done, and looked at in Chrome** (this also closes the by-eye checks left open in Phases 1 and 2):

- The sidebar holds *New chat*, *Search in* with its collection or schema, and *Chat history* with *Delete this chat*. No model, reranker, top k or toggle is left on the page.
- A new chat on the vector database answered with citations while the status panel showed the search step; its source was then greyed out with the note that a chat keeps what it searches.
- A new chat on the relational database (`sales`) answered "Lena Fischer" with the SQL under it, which is right.
- A refresh on the same `?chat=` reopened each chat. *Good answer* was saved (a row in `sql_examples`, a point in `sqlexamples__sales`) and showed as *Marked as a good answer* after a refresh.
- The Experiments page shows the new wording and the one collection.

Found while checking: **`examples_min_score` was too low for the embedding model in use.** With `embeddinggemma-2:740m`, questions of the same kind as the saved one scored 0.89 to 0.94, other questions about the same tables 0.66 to 0.72, and an unrelated one 0.55, so 0.55 showed the example to everything. `config/llm.yaml` now sets 0.8 (one example and nine questions, so a small sample); the default in `config.py` stays 0.55, which was chosen with `qwen3-embedding:0.6b`.

Built differently from the plan:

- A model named in `llm.yaml` that Ollama does not have stops the chat with the model and the key to change. A model that cannot think, or cannot see images, does not stop it: that setting is switched off for the chat and a note under the messages says so.
- The check also covers the embedding model the collection was made with, since a question is embedded with it.

Left in the stack: the chats made while checking (in *Chat history*), the `sales` sample dataset and its one good answer.

### Phase 6: documents and leftovers

- [x] 1. Rewrite `CLAUDE.md` and `README.md` to describe what is in the repo (both become much shorter). Move `PLAN.md` and `COMPLETED_PLAN.md` to `docs/history/`.
- [x] 2. One pass for dead names (`benchmark`, `upload`, `tag`, `engine`, `INGEST_CONFIG`) and unused imports (`ruff check`).
- [x] 3. Compose: pin `qdrant/qdrant` to a version instead of `latest`, and `restart: unless-stopped` on the services.

**Status (2026-10-08): Phase 6 is done.**

- `CLAUDE.md` and `README.md` are rewritten for the system as it is (`CLAUDE.md` went from 63 KB to 21 KB). `PLAN.md` and `COMPLETED_PLAN.md` are in `docs/history/`, unchanged.
- The pass for dead names found three stale comments (compose, `pyproject.toml`) and nothing in the code; `ruff check --select F` is clean.
- Compose: Qdrant is pinned to `v1.19.1` (the version `latest` was), and every service but the one-shot `setup` has `restart: unless-stopped`. The stack was recreated with it and both pages load.

Also changed, since nothing uses them any more: the UI no longer mounts the model cache or waits for `dagster-code`.

Not done, and worth knowing:

- `uv.lock` still lists `reportlab` and `ipykernel`. It was already behind `pyproject.toml` before the refactor, and the image is built from `pyproject.toml`, not from the lock. `uv lock` brings it up to date (it rewrites about 2,000 lines).
- The image carries about 2 GB of `uv`'s download cache (the Dockerfile installs with `uv` and leaves it in `/root/.cache`). `ENV UV_NO_CACHE=1` in the Dockerfile removes it at the next build; it belongs with Phase 7.
- `restart: unless-stopped` means this stack comes back when Docker starts. It binds the same ports as the `rag-dagster` stack, so stop one before starting the other (`docker compose stop`).

### Phase 7 (optional): a small image for the UI

After Phase 3 the UI imports no Dagster, Docling, LlamaIndex or torch, but it still runs in the image that contains them. A second Dockerfile of about ten lines with only the UI's dependencies would make its image a few hundred MB instead of several GB. The cost is two images to build and a dependency list split in two. Only if you want it.

## Choices I made that you can overrule

1. **The `native` chunking engine goes.** It duplicates the LlamaIndex one. Note that `semantic` chunking through LlamaIndex scored lower than our own on the one paper they were compared on; if `semantic` matters to you, say so and the native one stays instead.
2. **The pipeline is read by the run, not passed as run config.** One `pipeline.yaml` means one collection is being filled at a time. To fill another, change `name`.
3. **Tables and schemas are dropped from the command line**, since the Experiments page stays about the vector database. A "Tables" section on that page is a small addition if you prefer it.
4. **"Delete this chat" stays** in the sidebar as part of chat history.
5. **The sidebar keeps a selector for which collection or schema**, because there can be several of each.
6. **The search CLI and the two agent CLIs stay** (about 350 lines with the printer). They are the only way to try retrieval or an agent without the UI.
7. **Names stay**: the package is `rag_lab`, and an "experiment" is a collection with its settings.
8. **Prompts and the SQL safety constants stay in code** (the guard's allowed functions, the reader role's limits). A change to these should be a reviewed code change.

## Not in this plan

Say if you want any of them: a login for the UI, baking the source into the image instead of mounting it, a CI workflow, document types other than PDF, and the text-to-SQL evaluation that was Phase 28 in `PLAN.md`.

## Testing

As in `CLAUDE.md`: tests only for pure logic where a silent bug corrupts results. After the refactor that is four files: config hashing and the YAML loader, vector truncation with BM25 and the reranker score, the SQL guard, and the CSV import. The suite is run once in Phase 2 and once in Phase 3, where tested logic changes. Every other phase is confirmed by its "Done when" line, once, on the real stack. No test is added for Dagster, Streamlit, Qdrant, Postgres or Ollama.
