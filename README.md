# RAG-Platform

A chatbot over your documents and your tables.

- **PDF documents** go into a vector database, and the chatbot answers from the passages it finds, with citations.
- **Tables** in a relational database: CSV files that are imported, or tables that are already there. The chatbot writes a SQL query, checks it, runs it read-only and answers from the rows.

Ingestion is automatic: put a file in a folder and Dagster does the rest. An HTTP API is the door for every front end: the Streamlit UI here uses it, and so can another backend. Everything is configured in three YAML files, and moving to a server that already has PostgreSQL and Qdrant means editing `config/connections.yaml` and `.env`.

| Part | Uses |
|---|---|
| Parsing | Docling |
| Chunking | Docling's chunkers and LlamaIndex splitters (`hybrid`, `hierarchical`, `fixed`, `recursive`, `semantic`) |
| Embedding, reranking, chat, OCR | Ollama |
| Vector database | Qdrant (dense vectors and a BM25 keyword vector) |
| Relational database, chats, metrics | PostgreSQL |
| Orchestration | Dagster |
| API | FastAPI |
| UI | Streamlit (a client of the API) |

```
data/raw/*.pdf              -> parse -> chunk -> embed -> index -> Qdrant collection
data/tables/<schema>/*.csv  -> import                             -> PostgreSQL schema
tables already in PostgreSQL-> register (describe them)            -> the chatbot's registry

Chatbot, vector database:      question -> hybrid search + rerank -> answer with citations
Chatbot, relational database:  question -> SQL -> check -> run read-only -> answer
```

## Requirements

- Docker with Compose.
- An Ollama server, version 0.35 or newer (the reranker needs `logprobs`), reachable from the containers. If it is on another machine it must listen on all interfaces (`OLLAMA_HOST=0.0.0.0`); use its IP address, not a name.
- These Ollama models (the ones `config/llm.yaml` names; change the file to use others):

  ```
  ollama pull embeddinggemma-2:740m                  # embedding (or qwen3-embedding:0.6b, :4b, :8b)
  ollama pull dengcao/Qwen3-Reranker-4B:Q8_0         # the reranker; the 0.6B builds do not work
  ollama pull gemma4:e4b-mlx                         # the chat model
  ollama pull glm-ocr:bf16                           # only for OCR of scanned PDFs
  ```

## Getting started

```powershell
copy .env.example .env      # then set OLLAMA_BASE_URL and the passwords (letters and digits only)
docker compose up -d --build
```

`.env` names two compose files: `docker-compose.yml` (our services) and `docker-compose.local.yml` (PostgreSQL and Qdrant for this machine, prepared by `scripts/provision.sql`). The first start is slow: the ingestion image is several GB, the Dagster code container takes about 50 s to start, and Docling downloads its models on the first PDF. The API and the UI start in seconds.

| Open | For |
|---|---|
| http://localhost:8501 | The UI: Experiments and Chatbot |
| http://localhost:8000/docs | The API, with every request it takes |
| http://localhost:3000 | Dagster: the ingestion runs and the two sensors |
| http://localhost:6333/dashboard | Qdrant |

## Adding documents

Put a PDF in `data/raw/`. Within about a minute a sensor starts `ingest_job` for it, which parses, chunks, embeds and indexes it into the collection named in `config/pipeline.yaml`. Watch the run in Dagster.

- Only a new PDF starts a run. A document is known by its content, so a renamed copy is not new.
- A PDF whose run failed is not retried automatically: fix the cause, then run its partition of `ingest_job` in Dagster.
- A scanned PDF needs `parse.ocr: true` in `config/pipeline.yaml`. To make the pictures of a PDF searchable, set `parse.pictures: true` (needs `embeddinggemma-2`).

## Adding tables

Put a CSV file in `data/tables/<schema>/`. The folder is the dataset (a PostgreSQL schema) and each file becomes a table named after it. A sensor imports every file that is new or has changed.

- The file must be UTF-8 with a header row. Column types are guessed; a file with a bad row fails its run with the line number and leaves the existing table as it was.
- The limits (50 MB, 1,000,000 rows) are in `config/pipeline.yaml`.
- An optional `_schema.yaml` in the folder describes the data for the model, which makes its SQL better, and can set a column's type:

  ```yaml
  description: Orders of the web shop, 2023 to 2024
  tables:
    orders:
      description: One row per order
      columns:
        order_total: {description: "Order total in baht, with VAT", type: double precision}
        status: {description: "Shipped, Cancelled or Open"}
  ```

To see what the model is given about a dataset, and to remove one:

```powershell
docker compose exec api python -m rag_lab tables schema --schema <name>
docker compose exec api python -m rag_lab tables drop --schema <name> [--table <name>]
```

Dropping does not touch the CSV files.

### Tables that are already in PostgreSQL

Where the tables are already in the database, nothing is imported: the chatbot is told about them instead.

1. Name their schemas in `config/connections.yaml`: `tables_database.existing_schemas: [public]`. Leave `loader_url` out if nothing should be imported there.
2. Let the chatbot's read-only role read each schema, once: `psql ... -v schema=public -v reader=rag_reader -f scripts/grant_read.sql`.
3. Describe the tables: `docker compose exec api python -m rag_lab tables register` (or run `register_job` in Dagster). It reads the columns and their types from PostgreSQL and looks at the values of each column. It changes nothing in that database. Run it again when the tables change.

Descriptions are optional and go in `config/schemas/<schema>.yaml`, in the format of `_schema.yaml` above (without `type`).

## The UI

**Experiments** lists the collections of the vector database with the documents in each. Delete a document from a collection, or a whole collection.

**Chatbot.** The sidebar has three things:

- **New chat.**
- **Search in:** the vector database (and which collection) or the relational database (and which schema). The choice is fixed once the chat has its first question.
- **Chat history:** every past chat; opening one shows it as it was answered.

Under each answer, *How this was answered* shows the steps, the timings, the model's thinking, and the retrieved passages or the SQL attempts. Under a database answer, **Good answer** keeps the question and its SQL as an example the model is shown for similar questions later.

When the documents do not answer a question the chatbot tries one other query and otherwise says it found nothing.

## Configuration

| File | What is in it |
|---|---|
| `config/connections.yaml` | Where Ollama, Qdrant, the databases and the API are; the schema our own tables go in; the schemas that are already in the database. Passwords are written as `${NAME}` and come from `.env`. |
| `config/pipeline.yaml` | The collection's name, Docling's options, the chunking strategy, the index, the CSV limits. |
| `config/llm.yaml` | Every model and how it is called: embedding, OCR, reranker, the chat model, and the settings of each kind of chat. |

A key left out keeps its default. A misspelled key stops the start with a message that names the file and the key.

- A collection keeps the settings it was made with. To change a pipeline setting or the embedding model, change `name` in `pipeline.yaml` too: new PDFs then go into the new collection. To move the existing PDFs, run their partitions of `ingest_job` in Dagster.
- An edit to `pipeline.yaml` or `llm.yaml` needs no restart. After an edit to `connections.yaml`, run `docker compose restart api ui`.

## Command line

```powershell
# search a collection
docker compose exec api python -m rag_lab search "query text" --experiment <name> --top-k 5

# the two chatbots, with every step printed (no question = a chat loop)
docker compose exec api python -m rag_lab chat documents "question" --experiment <name>
docker compose exec api python -m rag_lab chat database "question" --schema <name>

# the good answers saved for a schema
docker compose exec api python -m rag_lab tables examples [--schema <name>] [--reindex] [--remove <id>]
```

## The API

Everything the UI does goes through the HTTP API, so another front end can do the same. http://localhost:8000/docs lists the requests, and `/openapi.json` is the contract a client can be generated from.

| Request | Does |
|---|---|
| `GET /health` | Whether Ollama, Qdrant and PostgreSQL answer |
| `GET /targets` | The collections and schemas a chat can search |
| `GET /check?kind=&target=` | Whether a chat can be asked now, and the settings it would run with |
| `GET /chats`, `GET /chats/{id}`, `DELETE /chats/{id}` | Chat history; a chat with its turns |
| `POST /chats/{id}/turns` | Ask a question: `{"kind": "documents" or "database", "target": "<collection or schema>", "question": "..."}` |
| `POST /turns/{id}/good`, `DELETE /examples/{id}`, `GET /schemas/{schema}/examples` | The thumbs-up under a database answer |
| `GET /library`, `DELETE /collections/{name}`, `DELETE /collections/{name}/documents/{doc_id}` | The Experiments page |
| `GET /pictures/{path}` | The image of a picture passage |

- **An answer is a stream** (`text/event-stream`). Each event is JSON with a `type`: the steps, the query, the passages or the SQL, the thinking, the answer piece by piece, and `Done` last with the whole answer. A turn that fails ends with `Failed`.
- **The caller chooses a chat's id.** The chat is made with its first question, keeps what it searches, and its earlier turns are the history of the next one.
- What cannot be done as asked is answered with `409` and `{"detail": "the reason"}`.
- To ask for a key, set `api.key` in `config/connections.yaml`; a caller then sends it in the `X-API-Key` header. There are no users: every chat is visible to every caller.

## Moving to a server

For a server where PostgreSQL and Qdrant already run in Docker. No code changes: two files do.

1. **PostgreSQL.** An admin runs `scripts/provision.sql` once with `psql` (its first lines say how). It makes the roles, our schema, and the databases that are missing; it can be run again. It does not change a database that is already there, apart from the grants to our roles.
2. **`.env`.** `COMPOSE_FILE=docker-compose.yml:docker-compose.vm.yml`, `PLATFORM_NETWORK` (the Docker network the databases are on), `POSTGRES_HOST`, the passwords, `OLLAMA_BASE_URL`.
3. **`config/connections.yaml`.** Start from `config/connections.vm.example.yaml`: the Qdrant URL, the database our schema is in, the reader's URL, `existing_schemas`.
4. `docker compose up -d --build`, then the three steps of *Tables that are already in PostgreSQL* above.

PDFs still arrive in the folder `data/raw` there; reading them from an object store is not built yet.

## Common commands

```powershell
docker compose up -d --build          # start the stack (rebuild after changing dependencies)
docker compose restart api            # after editing code the API runs, or connections.yaml
docker compose restart ui             # after editing a UI module other than a page script
docker compose logs -f dagster-code   # ingestion logs
docker compose logs -f api            # the API
docker compose down                   # stop; add -v to wipe every volume, the model cache included
docker compose exec postgres psql -U <user> -d rag_metrics   # inspect chats and metrics (they are in the schema `rag`)
```

`./src`, `./ui` and `./config` are mounted into the containers, so changes show up without a rebuild.

## How it is organised

```
config/           connections.yaml, pipeline.yaml, llm.yaml
scripts/          provision.sql, grant_read.sql: what an admin runs on PostgreSQL
docker/           the two images: ingest (Dagster, Docling) and serve (API, UI)
src/rag_lab/
  core/           what both sides need: the settings, the clients, our tables, the embedder, the events
  ingest/         the batch side: parsing, chunking, the stage bodies, CSV import, register, the Dagster definitions
  serve/          the serving side: search, the two chatbots, the SQL guard, good answers, the service and the API
  cli.py          python -m rag_lab search | chat | tables | setup
ui/               the Streamlit app: experiments.py and chat.py, which call the API
data/             raw/ (PDFs), tables/ (CSV files), artifacts/ (per-stage outputs); git-ignored
tests/            a few pure-logic tests
```

`ingest` and `serve` both use `core` and never each other, so each has its own image.

`CLAUDE.md` is the technical reference: the design rules, how each part works, and the facts that are easy to get wrong.
