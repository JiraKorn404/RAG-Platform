# RAG-Platform

A chatbot over your documents and your tables.

- **PDF documents** go into a vector database, and the chatbot answers from the passages it finds, with citations.
- **CSV files** become tables in a relational database, and the chatbot writes a SQL query, checks it, runs it read-only and answers from the rows.

Ingestion is automatic: put a file in a folder and Dagster does the rest. Everything is configured in three YAML files.

| Part | Uses |
|---|---|
| Parsing | Docling |
| Chunking | Docling's chunkers and LlamaIndex splitters (`hybrid`, `hierarchical`, `fixed`, `recursive`, `semantic`) |
| Embedding, reranking, chat, OCR | Ollama |
| Vector database | Qdrant (dense vectors and a BM25 keyword vector) |
| Relational database, chats, metrics | PostgreSQL |
| Orchestration | Dagster |
| UI | Streamlit |

```
data/raw/*.pdf              -> parse -> chunk -> embed -> index -> Qdrant collection
data/tables/<schema>/*.csv  -> import                             -> PostgreSQL schema

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
copy .env.example .env      # then set OLLAMA_BASE_URL and the four passwords (letters and digits only)
docker compose up -d --build
```

The first start is slow: the image is several GB, the Dagster code container takes about 50 s to start, and Docling downloads its models on the first PDF.

| Open | For |
|---|---|
| http://localhost:8501 | The UI: Experiments and Chatbot |
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
docker compose exec dagster-code python -m rag_lab.sql schema --schema <name>
docker compose exec dagster-code python -m rag_lab.sql drop --schema <name> [--table <name>]
```

Dropping does not touch the CSV files.

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
| `config/connections.yaml` | Where Ollama, Qdrant and the databases are. Passwords are written as `${NAME}` and come from `.env`. |
| `config/pipeline.yaml` | The collection's name, Docling's options, the chunking strategy, the index, the CSV limits. |
| `config/llm.yaml` | Every model and how it is called: embedding, OCR, reranker, the chat model, and the settings of each kind of chat. |

A key left out keeps its default. A misspelled key stops the start with a message that names the file and the key.

- A collection keeps the settings it was made with. To change a pipeline setting or the embedding model, change `name` in `pipeline.yaml` too: new PDFs then go into the new collection. To move the existing PDFs, run their partitions of `ingest_job` in Dagster.
- An edit to `pipeline.yaml` or `llm.yaml` needs no restart. After an edit to `connections.yaml`, run `docker compose restart ui`.

## Command line

```powershell
# search a collection
docker compose exec dagster-code python -m rag_lab.search "query text" --experiment <name> --top-k 5

# the two chatbots, with every step printed (no question = a chat loop)
docker compose exec dagster-code python -m rag_lab.agent documents "question" --experiment <name>
docker compose exec dagster-code python -m rag_lab.agent sql "question" --schema <name>

# the good answers saved for a schema
docker compose exec dagster-code python -m rag_lab.sql examples [--schema <name>] [--reindex] [--remove <id>]
```

## Common commands

```powershell
docker compose up -d --build          # start the stack (rebuild after changing dependencies)
docker compose restart ui             # after editing a UI module other than a page script
docker compose logs -f dagster-code   # ingestion logs
docker compose down                   # stop; add -v to wipe every volume, the model cache included
docker compose exec postgres psql -U <user> -d rag_metrics   # inspect chats and metrics
```

`./src`, `./ui` and `./config` are mounted into the containers, so changes show up without a rebuild.

## How it is organised

```
config/           connections.yaml, pipeline.yaml, llm.yaml
src/rag_lab/
  config.py  settings.py  clients.py     the settings and the clients built from them
  parsing/  chunking/  embedding/  reranking/  storage/  search/     the document pipeline
  sql/            the tables side: import, the schema as text, the SQL guard, good answers
  agent/          the two chatbots (documents/ and sql/) and what they share
  metrics/        the Postgres store and its migration
  assets/         the Dagster assets, jobs and sensors
ui/               the Streamlit app: experiments.py and chat.py
data/             raw/ (PDFs), tables/ (CSV files), artifacts/ (per-stage outputs); git-ignored
tests/            a few pure-logic tests
```

`CLAUDE.md` is the technical reference: the design rules, how each part works, and the facts that are easy to get wrong.
