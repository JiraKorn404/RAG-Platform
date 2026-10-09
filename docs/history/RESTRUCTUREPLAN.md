# Restructure plan

Written 2026-10-08. **Option B was chosen and is done**, except reading PDFs from SeaweedFS, which you left for later. The status note below says what was built, what was built differently from this plan, and what was checked. The rest of the file is the plan as it was proposed. The current system is described in `CLAUDE.md` and `README.md`.

## Status (2026-10-08)

**Built:** steps 1, 2, 3, 5, 6, 7 and 8 of the list at the end. `src/rag_lab/` went from 72 Python files in 13 packages to 32 in three (`core`, `ingest`, `serve`), plus `cli.py`.

**Checked with one real run on the stack** (a fresh local PostgreSQL and Qdrant, the host's Ollama):

- `scripts/provision.sql` made the roles, the three databases and the schema `rag`, and ran a second time without error, making nothing new. `python -m rag_lab setup` then made our nine tables in `rag`, none in `public`.
- Dagster's sensors found a PDF in `data/raw` and a CSV in `data/tables/shop`; `ingest_job` and `tables_job` both succeeded.
- Two tables made by hand in `public` were granted with `scripts/grant_read.sql`, registered, and the SQL chatbot answered a question about them correctly. Before the grant, `register` said that no table could be read.
- Through the API: a documents turn and two database turns as event streams (the second a follow-up, understood from the saved history), a thumbs-up, the chat list, a refused request (409).
- The Streamlit pages, driven with the app tester: two saved chats, the Experiments page, and one question typed into the Chatbot page.
- The four test files: 43 tests pass.

**Not checked:** `docker-compose.vm.yml` on a real server. Compose reads it correctly, but our containers have not yet joined another project's network.

**Built differently from the plan:**

| Plan | Built | Why |
|---|---|---|
| `provision: true/false` in `connections.yaml` | No such key. `scripts/provision.sql` is the only thing that creates databases and roles; locally a `provision` service runs it at every start | You asked for a script to run by hand. One script for both machines means there are not two versions of the same SQL |
| `documents_source` and `ingest/source.py` with a bucket | `ingest/source.py` is the old `documents.py` (the folder). No config key, no `boto3` | SeaweedFS is not decided |
| `ingest/chunk.py` holds the entry point | `chunk_document` is in `ingest/documents.py` | `chunk.py` and `split_text.py` would import each other |
| The reranker client is built in `core/connections.py` | It is built in `serve/service.py` | `core` must not import `serve`, and only a search uses the reranker |
| `setup` runs on the ingest image | It runs on the serve image | It needs nothing of Docling, and starts in seconds |
| `api.cors_origins` | Left out | The backend calls the API, not a browser |
| A profile sample for large tables | Not built. A profile query that fails or times out leaves that column's profile empty and is reported | Nothing shows yet that it is needed; check on the server's real tables |
| | New: `db_schemas.registered` (migration 0002) | So a registered schema can leave the registry without its tables being touched |
| | New: `Failed`, an event the API sends when a turn stops with an error | A stream has no status code once it has started |

**For you to do:**

- **Your `config/` does not load as it is.** `config/llm.yaml` (your uncommitted edit) names `qwen3-embedding:0.6b`, which cannot embed pictures, while `config/pipeline.yaml` has `parse.pictures: true`. Use `embeddinggemma-2:740m`, or set `pictures: false`. The run above used a copy of `config/` with pictures off.
- **`OLLAMA_BASE_URL` in `.env` is `http://localhost:...`**, which inside a container is the container itself. For an Ollama on this machine use `http://host.docker.internal:11434`.
- `config/llm.yaml` names the reranker build `Q8_0`; this machine's Ollama has `Q4_K_M`.
- `uv.lock` was not regenerated for the new extras in `pyproject.toml`. The images do not use it.
- The command lines moved: `python -m rag_lab search | chat | tables | setup`, run in the `api` container.

## Goal

1. **Fewer places to look.** Today `src/rag_lab/` has 72 Python files in 13 packages, and 33 of them are under 50 lines.
2. **Move to the VM by editing `config/connections.yaml`.** The VM already has PostgreSQL and Qdrant, in Docker. The code should use them as it uses the local ones, with no code change.
3. **Two front ends.** The Streamlit UI stays as the PoC. The real JavaScript frontend on the VM must be able to do the same things.
4. **Nothing is lost.** Every feature stays: PDF ingestion, OCR, pictures, CSV import, both chatbots, good answers, chat history, the trace under each answer, the command lines.

## What you decided

| Question | Answer | What it means for the plan |
|---|---|---|
| Who creates the databases and roles on the VM? | An admin. You want a script in the repo to run by hand. | `scripts/provision.sql`. Our code never acts as a database admin again. |
| Where do our tables go? | In a schema inside the database. | `app_database.schema`. |
| Qdrant on the VM | Only this project uses it. No API key, no HTTPS. | Nothing changes for Qdrant but its URL. |
| Tables on the VM | They are already in PostgreSQL. Locally you still import CSV files. | Two ways to get tables: import (local) and **register** (VM). Registering is new. |
| PDFs on the VM | They arrive in SeaweedFS and Dagster finds the new ones. Locally the folder stays. | Two sources of PDFs: the folder and a bucket. |
| Who calls the API? | The platform's backend. Every chat is visible to everyone for now. | No CORS, no users. An optional shared key. |
| Dagster on the VM | Yes, in Docker. PostgreSQL and Qdrant there are Docker containers too. | Our containers join the platform's Docker network. |

## What is in the way today

These were found by reading the code. They are the same whichever structure you choose.

### Moving to the VM

| # | Problem | Where |
|---|---|---|
| 1 | Setup acts as a database admin: it connects to the `postgres` maintenance database and runs `CREATE DATABASE`, `CREATE ROLE`, `ALTER ROLE ... PASSWORD`, `REVOKE ALL ON DATABASE ... FROM PUBLIC` and `REVOKE ALL ON SCHEMA public FROM PUBLIC`. On a shared server the last two take access away from the platform's own users. | `setup.py:23`, `sql/bootstrap.py:46-73` |
| 2 | Our tables are created without a schema, so they land in `public`. | `metrics/migrations/0001_init.sql` |
| 3 | Dagster's own database is not in `connections.yaml`. Its host, port and name are written in another file. | `docker/dagster/dagster.yaml:6-8` |
| 4 | `docker-compose.yml` always starts `postgres` and `qdrant`, and has no way to join the network of containers that already run. | `docker-compose.yml:17-42` |
| 5 | A new `${NAME}` must be added in three places: `.env`, the `x-dagster-env` block and `connections.yaml`. | `docker-compose.yml:10-14` |
| 6 | PDFs can only come from the folder `data/raw`. | `documents.py`, `assets/sensors.py` |
| 7 | The SQL chatbot only knows tables that the CSV import made: the registry it reads (`db_schemas`, `db_tables`) is filled nowhere else. A schema named `public` is refused. | `sql/importer.py:318`, `sql/database.py: check_schema_name` |

Qdrant is not in this list: with no key, no HTTPS and no other project on it, only its URL changes.

### A second front end

| # | Problem | Where |
|---|---|---|
| 8 | There is no HTTP API. The Streamlit pages import `rag_lab` and call it in the same process. | `ui/*.py` |
| 9 | Part of the chatbot's logic is in the page: which collections can be chosen, checking that Ollama has the models, switching off `think` or pictures, building the graph and the flow, making the history. | `ui/chat.py:55-64`, `ui/chat.py:286-312`, `ui/data.py:23-55`, `ui/trace_view.py:110` |
| 10 | Pictures are read from `data/artifacts` on disk by the page. A backend needs a URL. | `ui/hits.py:31-37` |
| 11 | One image holds everything. A service that only answers questions would carry Docling and torch, and take about 50 s to start. | `docker/dagster/Dockerfile` |

One thing already helps: every chatbot event is plain data with a `type` (`agent/events.py: to_dict`), and `agent.run()` yields them one by one. Sending them over HTTP is a small step.

## What changes in all three options

The three options differ in **where the files go**. The work below is the same in each.

### 1. `connections.yaml` is the file that differs between your machine and the VM

```yaml
ollama:
  url: ${OLLAMA_BASE_URL}

qdrant:
  url: http://qdrant:6333

app_database:                    # chats, metrics, the registry of tables
  url: postgresql://${POSTGRES_USER}:${POSTGRES_PASSWORD}@postgres:5432/rag_metrics
  schema: rag                    # our tables are made in this schema

tables_database:                 # the tables the SQL chatbot answers from
  reader_url: postgresql://rag_reader:${SQL_READER_PASSWORD}@postgres:5432/rag_data
  loader_url: postgresql://rag_loader:${SQL_LOADER_PASSWORD}@postgres:5432/rag_data   # local. Left out: CSV import is off
  existing_schemas: []           # VM: the schemas already in the database that the chatbot may read

documents_source:
  type: folder                   # folder: data/raw.  s3: a bucket (SeaweedFS), copied into data/raw
  # endpoint: ${S3_ENDPOINT}     # s3 only
  # bucket: documents
  # prefix: ""
  # access_key: ${S3_ACCESS_KEY}
  # secret_key: ${S3_SECRET_KEY}

api:
  key: ${RAG_API_KEY}            # empty: no key is asked for
```

A filled-in example for the VM is kept in the repo beside it (`config/connections.vm.example.yaml`).

### 2. The provision script

`scripts/provision.sql`, run with `psql` by whoever is the admin. It can be run again without harm. It makes, only where missing:

- the database (when you are given a server, not a database),
- our schema, and a role that owns it,
- the reader role, with what it has today: read-only, a statement timeout, a connection limit,
- locally only, the loader role.

A second small script, `scripts/grant_read.sql`, gives the reader `USAGE` and `SELECT` on one existing schema. It is run once for each schema of `existing_schemas`.

- **The same script is used locally.** The local PostgreSQL container runs it when its volume is first made, so there is one script and not two versions of the same SQL. `sql/bootstrap.py` goes.
- **`python -m rag_lab.setup` no longer creates anything outside our schema.** It connects with each URL, applies our migrations inside `app_database.schema`, and says what is missing, naming the script to run.
- **The script revokes nothing from `PUBLIC`.** That is only safe in a database of our own.
- **On the VM the grants are the boundary, as now.** The reader is granted only the schemas you list, and not ours, so the SQL chatbot cannot read the chats. With no loader role there, nothing of ours can change or drop the platform's tables.

### 3. Tables: import or register

The SQL chatbot reads a description of each table from the registry: column types, example values, ranges, likely joins. Today only the CSV import writes it.

- **Local, as now:** a CSV file in `data/tables/<schema>/` is imported by Dagster as the loader role.
- **VM, new:** each schema of `existing_schemas` is **registered**: its tables and column types are read from PostgreSQL's catalog, and each column is profiled with the code the import already uses (`sql/importer.py: _profile`). It runs as the reader, so it cannot change anything. It is a Dagster job (`register_job`), started by hand or by `python -m rag_lab tables register`, and run again when the tables change.
- **Descriptions** for a registered schema go in `config/schemas/<schema>.yaml`, in the format `_schema.yaml` has today.
- **A risk to check in the first real run:** profiling counts the distinct values of every column, which is one full scan per column. On a large platform table that can pass the reader's 15 s limit. If it does, the profile is taken from a sample of the rows, with the sample size as a setting.

### 4. PDFs: the folder or a bucket

Everything after the folder stays as it is. With `documents_source.type: s3`, the sensor first copies the objects that are new in the bucket into `data/raw`, and then does what it does today: a file that is not yet a partition gets a run. So a document's id is still the hash of its bytes, and the four assets do not change.

- This adds one dependency, `boto3`; the need is SeaweedFS. It is used through SeaweedFS's S3 API.
- The PDFs are copied to the VM's disk. Docling needs the file locally to parse it, and the stage outputs in `data/artifacts` are local already.
- `data/` on the VM is a Docker volume shared by Dagster and the API (pictures are read from it).
- Removing an object from the bucket removes nothing from a collection, the same as removing a file from the folder today.

### 5. Compose in three files

```
docker-compose.yml          our services only: setup, dagster-code, dagster-webserver, dagster-daemon, api, ui
docker-compose.local.yml    adds postgres and qdrant, and makes our services wait for them
docker-compose.vm.yml       joins the platform's Docker network (its name comes from .env)
```

`.env` names the files to use in `COMPOSE_FILE`, so `docker compose up -d` is the command on both machines. On the platform's network our containers reach PostgreSQL and Qdrant by their container names. Every service gets `env_file: .env`, so a new variable is added in one place.

**Dagster's own database** cannot be read from `connections.yaml`, because Dagster reads only its own `dagster.yaml`. That file will take its host, port and database name from `.env`, as it already takes the user and password. So the VM move is: `connections.yaml` and `.env`.

### 6. One module that says what a front end can do

The logic in the pages (problem 9) moves into one plain Python module, called `service` below. Streamlit, the API and the command lines all call it, so they cannot drift apart.

```python
targets()                                   # the collections and schemas a chat can search
check(kind, target)                         # is Ollama there, are the models there, what is switched off
ask(chat_id, kind, target, question)        # yields the events of one turn; the history is read from the saved turns
chats()   chat(chat_id)   delete_chat(chat_id)
mark_good(turn_id)   unmark_good(example_id)
library()   delete_document(name, doc_id)   delete_experiment(name)
```

### 7. An HTTP API over that module

FastAPI, one file. This adds two dependencies (`fastapi`, `uvicorn`); the need is the platform's backend.

| Request | Does |
|---|---|
| `GET /health` | Ollama, Qdrant and PostgreSQL can be reached; the models are there |
| `GET /targets` | The collections and schemas a chat can search |
| `GET /chats`, `GET /chats/{id}`, `DELETE /chats/{id}` | Chat history; a chat with its turns and their events |
| `POST /chats/{id}/turns` | Ask a question. The reply is a stream (`text/event-stream`): one event per step, `Done` last |
| `POST /turns/{id}/good`, `DELETE /examples/{id}` | The thumbs-up |
| `GET /library`, `DELETE /collections/{name}`, `DELETE /collections/{name}/documents/{doc_id}` | The Experiments page |
| `GET /pictures/...` | A picture chunk's image |

The events on the stream are the ones `agent/events.py` already defines, as JSON. The backend passes the stream on to the browser, or reads it to the end and sends the answer whole. FastAPI publishes the whole contract at `/openapi.json`, which the backend team can generate a client from.

### What does not change

The keys of `pipeline.yaml` and `llm.yaml`, our tables, the four ingestion assets, the two agents and their prompts, the SQL guard, the Qdrant store, the tests (only their imports).

---

## Option A: one package, fewer files

Keep one package and one image. Merge the small files and group by the two things the system does: **documents** and **tables**, plus **chat**. 72 files in 13 packages become about 32 files in 3 folders.

```
src/rag_lab/
  config.py            # the settings as pydantic models (as now)
  settings.py          # reads config/*.yaml; DATA_DIR and the folders       <- settings.py, paths.py
  connections.py       # builds every client from connections.yaml           <- clients.py
  qdrant.py            # the Qdrant store                                    <- storage/qdrant.py
  store.py             # our tables in PostgreSQL, and the migrations        <- metrics/store.py, migrate.py, timing.py
  migrations/
  setup.py             # check the connections, apply the migrations
  documents/           # PDF -> Qdrant, and searching it
    source.py          # NEW: the folder, or a bucket copied into it         <- documents.py
    parse.py           #                                                     <- parsing/parse.py, pictures.py
    ocr.py
    chunk.py           # the entry point, Docling's chunkers, picture chunks <- chunking/__init__, base, models, tokens, docling_chunkers, pictures
    split_text.py      # fixed, recursive, semantic; tables                  <- chunking/segment, tables, text_splitters
    embed.py           # Ollama embedder, truncation, BM25, the cache        <- embedding/* (5 files)
    ingest.py          # the stage bodies
    search.py          # the four methods and the reranker                   <- search/engine.py, reranking/ollama.py
    library.py
  tables/              # tables -> the registry, and querying them
    folder.py  importer.py                    # CSV import (local)
    register.py        # NEW: describe tables that already exist (VM)
    catalog.py  guard.py  examples.py
    database.py        # roles, schemas, running a query                     <- sql/database.py, execute.py
  chat/
    events.py
    run.py             # the runner and the model calls                      <- agent/run.py, model.py
    documents.py       # the graph and its prompts                           <- agent/documents/graph.py, prompts.py
    database.py        #                                                     <- agent/sql/graph.py, prompts.py
  service.py           # NEW: what a front end can do
  api.py               # NEW: FastAPI
  definitions.py       # Dagster: assets, jobs, sensors                      <- assets/* (7 files), definitions.py
  cli.py               # python -m rag_lab search | chat | tables | setup     <- the three __main__.py, two cli.py, printer.py
scripts/               # NEW: provision.sql, grant_read.sql
ui/                    # Streamlit, as now: it imports rag_lab and calls service.py
```

- **Streamlit:** imports `rag_lab.service` in the same process, as it imports `rag_lab` now.
- **The backend:** calls `api.py`, which calls the same `service.py`.
- **Images and services:** one image. One new service, `api`, from the same image.

| For | Against |
|---|---|
| The smallest change: files move and merge | The `api` service carries Docling and torch it never uses (several GB, slow to build) |
| One image, one `pyproject.toml`, as now | Nothing stops chat code from importing ingestion code, so the two can grow together again |
| Streamlit needs almost no change | Streamlit does not use the API, so the API is only proven by the backend |

**Size of the work:** small, on top of what all three share.

## Option B: split by what runs (recommended)

The same merged files as option A, sorted into three folders by **where they run**: `ingest` is the batch worker, `serve` answers questions, `core` is what both need. Two images. Streamlit becomes a client of the API, the same way the backend is.

```
src/rag_lab/
  core/                # needed by both sides. No Docling, no Dagster, no web framework
    config.py  settings.py  connections.py
    qdrant.py  store.py  migrations/  setup.py
    embed.py           # Ollama embedder, truncation, BM25 (a question is embedded too)
    database.py        # the tables database: roles and schemas
    events.py          # the contract with a front end: the events, and Hit
  ingest/              # image "ingest": Docling, torch, LlamaIndex, Dagster, boto3
    source.py          # NEW: the folder, or a bucket copied into it
    parse.py  ocr.py  chunk.py  split_text.py
    documents.py       # the stage bodies; the embedding cache
    folder.py  importer.py                    # CSV import (local)
    register.py        # NEW: describe tables that already exist (VM)
    definitions.py     # Dagster: assets, jobs, sensors
  serve/               # image "serve": LangGraph, sqlglot, FastAPI. Starts in seconds
    search.py          # the four methods and the reranker
    chat_documents.py  chat_database.py  run.py
    catalog.py  guard.py  execute.py  examples.py
    library.py
    service.py         # NEW
    api.py             # NEW
  cli.py
scripts/               # NEW: provision.sql, grant_read.sql
ui/                    # Streamlit: calls the API over HTTP. Imports only rag_lab.core.events
docker/
  ingest.Dockerfile  serve.Dockerfile  dagster.yaml  workspace.yaml
```

The rule that keeps it clean: `ingest` and `serve` import `core`, never each other.

- **Streamlit:** calls the API with `httpx` and reads the event stream. `trace_view.apply(trace, event)` already draws a turn from saved events, so it draws one from the stream the same way. `ui/chat.py`, `ui/experiments.py` and `ui/data.py` are rewritten to fetch instead of import (about 150 lines change); `trace_view.py`, `hits.py` and `style.py` stay.
- **The backend:** calls the same API. Whatever the PoC shows is known to be reachable over HTTP, because the PoC got it the same way.
- **Images and services:** `ingest` (today's image) runs `setup` and the three Dagster services; `serve` (small) runs `api` and `ui`. `pyproject.toml` gets two optional dependency groups.

| For | Against |
|---|---|
| Matches the VM as you described it: Dagster in Docker fills the databases, and a service is called by the backend. Each can be deployed and restarted alone | Two Dockerfiles and two dependency groups to keep |
| The API service has no Docling or torch: small image, starts in seconds | The Streamlit pages are partly rewritten |
| Streamlit tests the API every time you use it | Every new thing the UI shows needs an endpoint first |
| The import rule keeps the chat side and the ingestion side apart | `ingest` and `serve` share the `data` volume (pictures, deletes) |

**Size of the work:** medium. It is option A's merges, plus the two images and the Streamlit change.

## Option C: interfaces and adapters

Every outside service gets an interface, and the code for one product sits behind it in an adapter. `connections.yaml` chooses the adapter with a `type`. This is the only option where you could replace Qdrant with pgvector, or Ollama with another model server, by writing one new file.

```
src/rag_lab/
  config.py  settings.py
  ports.py             # the interfaces: VectorStore, Embedder, Reranker, ChatModel, AppStore, TableDatabase, DocumentSource
  adapters/
    qdrant.py  ollama.py  postgres.py  folder.py  s3.py
    registry.py        # "type: qdrant" -> the class
  documents/  tables/  chat/        # as option A, but they import ports.py only
  service.py  api.py  definitions.py  cli.py
scripts/
ui/                    # as option A or B
```

| For | Against |
|---|---|
| A product can be replaced, not only an address | Your answers need two addresses changed and one real choice of source (folder or bucket). Options A and B already have that one choice as `documents_source.type` |
| The clearest boundary around each service | Much of the code is tied to one product on purpose: BM25 with Qdrant's IDF modifier and RRF fusion, the reranker's `logprobs`, pictures inside `/api/embed`, the roles of PostgreSQL. An interface over them either leaks or drops features |
| | With one adapter each, the interfaces are guesses. They are usually wrong until a second adapter is written |
| | The most code and the most files, against this project's rule of adding no abstraction that nothing needs yet |

**Size of the work:** large. Every place that uses the Qdrant client directly (`ui/chat.py`, `library.py`, `sql/examples.py`) and both agents (LangChain's `ChatOllama`) is rewritten.

---

## Side by side

| | A: fewer files | B: split by what runs | C: adapters |
|---|---|---|---|
| Python files in `src/` (now 72) | about 32 | about 34 | about 40 |
| Folders (now 13) | 3 | 3 | 4 |
| Move to the VM by editing `connections.yaml` and `.env` | yes | yes | yes |
| Folder or bucket for PDFs, import or register for tables | yes | yes | yes |
| Replace Qdrant or Ollama with another product | rewrite | rewrite | one new adapter |
| Streamlit | imports the code | calls the API | either |
| The platform's backend | calls the API | calls the API | calls the API |
| Images | 1 (large) | 2 (large for ingestion, small for the API) | 1 or 2 |
| Size of the work | small | medium | large |

## Recommendation

**Option B.** Your answers describe two things on the VM: Dagster in Docker that fills the databases, and a service the backend calls. Option B gives the code that shape, and gives the backend and Streamlit the same door, so the PoC keeps proving the API.

Option A is the right choice if you want the least change now. It is not wasted: B uses the same merged files, so A can become B later by sorting the files into three folders and adding the second image.

I would not choose option C. The one place where your answers need two implementations is the source of PDFs, and that is a single `type` key in every option.

## Answered later

| Question | Answer |
|---|---|
| Which option? | B. |
| Dagster's own tables | The admin gives Dagster a database of its own. `scripts/provision.sql` makes it when given `dagster_db`. |
| Which schemas may the SQL chatbot read on the VM? | `public`, for now. A registered schema may be named `public`; an imported one may not. |
| SeaweedFS | Not decided. Left out: PDFs come from the folder on both machines. |
| Locally, use the schema too? | Yes: `schema: rag`. |

## Steps, once an option is chosen

Each step leaves a working system and is checked with one real run on the stack.

1. **Connections.** The new keys of `connections.yaml`, our schema, `scripts/provision.sql` in place of `bootstrap.py`, the three compose files, `env_file`. No file moves. Checked by pointing the stack at a second PostgreSQL and Qdrant (another compose project standing in for the VM) that was prepared only with the script.
2. **Move and merge the files** for the chosen option, with `git mv` so the history follows. No logic changes. The four test files get new imports and are run once.
3. **Register.** `existing_schemas`, `register_job`, `scripts/grant_read.sql`. Checked by asking the SQL chatbot about a schema that was made by hand, not by the import.
4. **The bucket.** `documents_source.type: s3`. Checked with a SeaweedFS container: a PDF put in the bucket is ingested.
5. **`service`.** The logic leaves the pages. Streamlit calls it and works as before.
6. **`api`.** FastAPI over `service`. Checked by streaming one documents turn and one database turn with `curl`.
7. **Option B only:** the two images, and Streamlit over HTTP.
8. **Docs.** `CLAUDE.md` and `README.md` describe the new layout; this file gets a status note.

Not in this plan: a login or users, several people asking at once (there is one Ollama), another model server.
