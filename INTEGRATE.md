# Integrating RAG-Platform into the VM

How to move this project from the PoC machine to the VM, where PostgreSQL and Qdrant already run in Docker, and how the platform's backend talks to it. No code changes are needed: you run one SQL script, write two files (`.env` and `config/connections.yaml`) and start the containers.

The commands here are for the VM's shell (bash on Linux). Words in `<angle brackets>` are values you fill in.

**Contents**

1. [What ends up on the VM](#1-what-ends-up-on-the-vm)
2. [What to find out first](#2-what-to-find-out-first)
3. [Prepare PostgreSQL (the admin)](#3-prepare-postgresql-the-admin)
4. [Put the project on the VM](#4-put-the-project-on-the-vm)
5. [Write `.env`](#5-write-env)
6. [Write `config/connections.yaml`](#6-write-configconnectionsyaml)
7. [Check the other two settings files](#7-check-the-other-two-settings-files)
8. [Start it](#8-start-it)
9. [Tell the chatbot about the tables](#9-tell-the-chatbot-about-the-tables)
10. [Add documents](#10-add-documents)
11. [Connect the backend to the API](#11-connect-the-backend-to-the-api)
12. [Look at the UI and Dagster from your own machine](#12-look-at-the-ui-and-dagster-from-your-own-machine)
13. [Day to day](#13-day-to-day)
14. [When something is wrong](#14-when-something-is-wrong)
15. [Removing it again](#15-removing-it-again)
16. [What is not built yet](#16-what-is-not-built-yet)

---

## 1. What ends up on the VM

```
                 the platform's Docker network
 ┌──────────────────────────────────────────────────────────────────┐
 │  already there                       added by this project        │
 │  ─────────────                       ─────────────────────        │
 │  PostgreSQL  ◄───────────────────────  setup (runs once, exits)   │
 │  Qdrant      ◄───────────────────────  api  :8000   ◄── backend   │
 │  backend ────────────────────────────► ui   :8501  (the PoC UI)   │
 │                                        dagster-code               │
 │                                        dagster-webserver :3000    │
 │                                        dagster-daemon             │
 └──────────────────────────────────────────────────────────────────┘
   Ollama: wherever it runs, reachable from the containers
```

**What this project adds to the platform's servers:**

| Where | What | Made by |
|---|---|---|
| PostgreSQL | A role for our tables (called `rag` below) and a read-only role (`rag_reader`) | `scripts/provision.sql` |
| PostgreSQL, the platform's database | One schema, `rag`, with our nine tables (chats, metrics, the registry of tables) | The schema: the script. The tables: our `setup` container |
| PostgreSQL | One database of Dagster's own, `dagster` | The script |
| Qdrant | One collection for each experiment (named in `config/pipeline.yaml`), and one `sqlexamples__<schema>` for the good answers of each schema | Our containers |
| The VM's disk | The project folder with `data/` (PDFs and per-stage outputs), and two Docker volumes (`model_cache`, `dagster_storage`) | Our containers |

**What it does not do there:**

- It creates no database and no role by itself. Only the script does, and a person runs it.
- It does not write to the platform's tables. It reads them with a role that can only `SELECT`.
- The CSV import is off on the VM (there is no loader role).

## 2. What to find out first

Collect these before you start. The commands run on the VM.

| # | What | How to find it | Example used below |
|---|---|---|---|
| 1 | The PostgreSQL container's name | `docker ps --format '{{.Names}}\t{{.Image}}'` | `platform-postgres` |
| 2 | The Qdrant container's name | the same list | `platform-qdrant` |
| 3 | The Docker network both are on | `docker inspect <container> --format '{{range $k, $v := .NetworkSettings.Networks}}{{$k}} {{end}}'` | `platform_default` |
| 4 | A PostgreSQL superuser, for the script | from the platform's admin | `postgres` |
| 5 | The platform's database, where the tables are | from the admin | `platform` |
| 6 | The schema the chatbot may read | you said `public` for now | `public` |
| 7 | Where Ollama runs | see [section 5](#5-write-env) | |
| 8 | The names of the containers already on that network | `docker network inspect <network> --format '{{range .Containers}}{{.Name}} {{end}}'` | |

**Check 8 for a clash of names.** Our containers join the platform's network under their service names: `api`, `ui`, `setup`, `dagster-code`, `dagster-webserver`, `dagster-daemon`. If the platform already has a container or service called `api` or `ui` on that network, both would answer to the same name and requests would go to either one. If you see such a name in the list, stop and rename our service first: in `docker-compose.yml` change the service's name (for example `api` to `rag-api`), and change `api.url` in `config/connections.yaml` to match.

If PostgreSQL and Qdrant are on **two different** networks, `docker-compose.vm.yml` (which joins one) is not enough. Add the second network to that file and list both under each of our services, or ask the platform's admin to put both databases on one network.

## 3. Prepare PostgreSQL (the admin)

One script, run once by a superuser: `scripts/provision.sql`. It only makes what is missing, so running it again is safe. It does not change a database that already exists, apart from adding our schema to it and granting our roles.

Choose three passwords first (the app role's, the reader's, and nothing for a loader: there is none on the VM). **Use letters and digits only**: the passwords are put into connection URLs.

Copy the script into the PostgreSQL container and run it there:

```bash
docker cp scripts/provision.sql platform-postgres:/tmp/provision.sql

docker exec -it platform-postgres psql -U postgres -d postgres \
  -v app_role=rag            -v app_password=<app password> \
  -v app_db=platform         -v app_schema=rag \
  -v tables_db=platform \
  -v reader=rag_reader       -v reader_password=<reader password> \
  -v dagster_db=dagster \
  -f /tmp/provision.sql
```

| Variable | Meaning | On the VM |
|---|---|---|
| `app_role`, `app_password` | The role that owns our schema. It is also the role Dagster uses | A new role, `rag` |
| `app_db` | The database our schema goes into | The platform's database |
| `app_schema` | The schema our tables go into | `rag` |
| `tables_db` | The database the SQL chatbot answers from | The same platform database |
| `reader`, `reader_password` | The role that runs the chatbot's SQL. It can only select | A new role, `rag_reader` |
| `loader`, `loader_password` | The role of the CSV import | **Leave both out** |
| `dagster_db` | Dagster's own database | `dagster` |

What you should see: `CREATE ROLE` twice, `CREATE DATABASE` once (for `dagster`), `GRANT`, `CREATE SCHEMA`, then several `ALTER ROLE` and `GRANT`. There must be **no** `REVOKE` line: the script only revokes in a database it created itself, and the platform's database already exists.

What the script gives the reader role, so the admin knows what is being asked for:

- It may connect to the platform's database, with at most 10 connections.
- Its transactions are read-only by default, a statement is stopped after 15 seconds, and an idle transaction after 30 seconds.
- It is granted nothing to read yet. That is the next script, in [section 9](#9-tell-the-chatbot-about-the-tables).

## 4. Put the project on the VM

```bash
git clone <the repository> rag-platform
cd rag-platform
```

Everything below is run from this folder. The folder's name becomes the name of the Docker Compose project, so the containers are called `rag-platform-api-1` and so on.

The first build needs the internet: Python packages (the ingestion image is about 3 GB), and on the first PDF Docling downloads its models into the `model_cache` volume.

## 5. Write `.env`

```bash
cp .env.example .env
```

Then edit it to this (every line matters):

```ini
# Which compose files are used: ours, and the one that joins the platform's network.
COMPOSE_PATH_SEPARATOR=:
COMPOSE_FILE=docker-compose.yml:docker-compose.vm.yml
PLATFORM_NETWORK=platform_default

# Ollama: see below.
OLLAMA_BASE_URL=http://<address>:11434

# PostgreSQL: the container's name on the network, and its port inside the network (not a published one).
POSTGRES_HOST=platform-postgres
POSTGRES_PORT=5432
# On the VM this is OUR role from the script (app_role), not the server's admin.
POSTGRES_USER=rag
POSTGRES_PASSWORD=<app password>

# Dagster's own database (dagster_db in the script).
DAGSTER_POSTGRES_DB=dagster

# The reader role's password (reader_password in the script). There is no loader on the VM.
SQL_READER_PASSWORD=<reader password>

# The key the backend sends to our API. Any long random string of letters and digits.
RAG_API_KEY=<a long random string>
```

Remove or leave out `SQL_LOADER_PASSWORD`: nothing refers to it on the VM.

**`OLLAMA_BASE_URL`** depends on where Ollama runs:

| Ollama is | Use | Also needed |
|---|---|---|
| A container on the platform's network | `http://<its container name>:11434` | |
| Installed on the VM itself | `http://<the VM's IP address>:11434` | Ollama must listen on all interfaces: start it with `OLLAMA_HOST=0.0.0.0`. `localhost` does not work: inside a container it means the container |
| On another machine | `http://<that machine's IP address>:11434` | The same `OLLAMA_HOST=0.0.0.0`, and use the IP address, not a name |

Ollama must be version 0.35 or newer (the reranker needs `logprobs`).

## 6. Write `config/connections.yaml`

Start from the example, which is this file as it should look on the VM:

```bash
cp config/connections.vm.example.yaml config/connections.yaml
```

Then put the names right:

```yaml
ollama:
  url: ${OLLAMA_BASE_URL}

qdrant:
  url: http://platform-qdrant:6333        # the Qdrant container's name

# Our tables, in our own schema inside the platform's database.
app_database:
  url: postgresql://${POSTGRES_USER}:${POSTGRES_PASSWORD}@${POSTGRES_HOST}:${POSTGRES_PORT}/platform
  schema: rag

# The platform's tables. No loader_url: nothing of ours can write to them.
tables_database:
  reader_url: postgresql://rag_reader:${SQL_READER_PASSWORD}@${POSTGRES_HOST}:${POSTGRES_PORT}/platform
  existing_schemas: [public]

api:
  url: http://api:8000
  key: ${RAG_API_KEY}
```

- `platform` appears twice: it is the database name (`app_db` and `tables_db` of the script).
- `rag_reader` is the `reader` of the script.
- Qdrant needs only its URL: no key, no HTTPS. Our code also uses Qdrant's gRPC port 6334, which every container on the same network can reach.
- An unknown or misspelled key stops the start with a message that names the file and the key.

## 7. Check the other two settings files

**`config/pipeline.yaml`**

- `name` is the experiment, which is also the name of the Qdrant collection. Choose the name the VM should use, for example `platform-docs`. Settings cannot change under a name: to change the embedding model or a parsing or chunking setting later, change `name` too.
- `parse.pictures: true` needs an embedding model that takes images (`embeddinggemma-2`). With any other embedding model, set it to `false`, or the start stops with that message.

**`config/llm.yaml`** names every model. Each must be pulled on the VM's Ollama, under exactly the name in the file:

```bash
ollama pull <embedding.model>       # for example qwen3-embedding:0.6b
ollama pull <reranker.model>        # for example dengcao/Qwen3-Reranker-4B:Q4_K_M  (the 0.6B builds do not work)
ollama pull <chat.model>            # for example gemma4:e4b
ollama pull glm-ocr:bf16            # only with parse.ocr: true
```

The API checks this before every question and names the missing model and the key to change.

## 8. Start it

```bash
docker compose up -d --build
docker compose ps -a
```

The order is: `setup` runs and exits, then the other five start.

| Container | Should be | If not |
|---|---|---|
| `setup` | `Exited (0)` | `docker compose logs setup` says which connection or which schema is missing |
| `api`, `ui` | `Up` | `docker compose logs api` |
| `dagster-code` | `Up`; it needs about 50 seconds before it answers | `docker compose logs dagster-code` |
| `dagster-webserver`, `dagster-daemon` | `Up`. The daemon logs "could not reach user code server" until `dagster-code` is ready | |

`setup` connected with each role, and made our tables in the schema `rag`. Check the whole chain with one request:

```bash
curl -s -H "X-API-Key: <RAG_API_KEY>" http://127.0.0.1:8000/health
```

```json
{"ollama": true, "qdrant": true, "database": true}
```

All three must be `true` before going on. `false` for one means that address in `config/connections.yaml` or `.env` is wrong, or the service does not answer from inside the containers.

## 9. Tell the chatbot about the tables

The SQL chatbot can only be asked about tables it has a description of. For tables that already exist, three steps.

**a. Let the reader role read the schema** (the admin, once for each schema):

```bash
docker cp scripts/grant_read.sql platform-postgres:/tmp/grant_read.sql

docker exec -it platform-postgres psql -U postgres -d platform \
  -v schema=public -v reader=rag_reader \
  -f /tmp/grant_read.sql
```

This grants `SELECT` on **every** table of the schema. Think about that before running it on `public`:

- Whatever the reader can read, the chatbot can be asked about, and its rows can end up in an answer.
- To offer only some tables, do not run the script. Grant them one by one instead: `GRANT USAGE ON SCHEMA public TO rag_reader;` then `GRANT SELECT ON public.<table> TO rag_reader;` for each. Only the tables the reader can read are described in the next step.
- Never grant the reader our own schema `rag`: it holds the chats.

**b. Describe the tables:**

```bash
docker compose exec api python -m rag_lab tables register
```

```
public: 2 table(s): machines, readings
```

For each table it reads the columns and their types from PostgreSQL's catalog, counts the rows, and looks at each column: how many values are missing, how many are distinct, the range of numbers and dates, the most common values of a text column. That is what lets the model write `status = 'Shipped'` and not `'shipped'`. It runs as the reader and changes nothing in the database.

A `WARNING` line means one query was refused or took longer than the reader's 15 seconds (a very large table, or a column type that cannot be compared). That column is then described by its name and type only. The table is still usable.

**c. Look at what the model will be given:**

```bash
docker compose exec api python -m rag_lab tables schema --schema public
```

The last line is the size against the budget, for example `917 of 12,000 characters, 2 table(s)`.

- **If it says "over the budget", the chatbot refuses every question about that schema.** The model is given the whole schema as text, and a schema that is too long is refused, not cut. A `public` with many tables will hit this. Either grant fewer tables (step a) and register again, or raise `database_chat.schema_char_budget` in `config/llm.yaml` together with `database_chat.num_ctx`, if the chat model's context allows it.

**Descriptions (optional, and worth it).** Create `config/schemas/public.yaml`:

```yaml
description: Machines of the plant and their hourly readings
tables:
  readings:
    description: One row per machine and hour
    columns:
      temperature: {description: "Degrees Celsius"}
      state: {description: "running, idle or alarm"}
```

Then run `tables register` again. A column the file names that the table does not have is an error.

**When the tables change** (a new table, a new column, very different values): run `tables register` again. Nothing watches the database. It can also be started from Dagster: the job `register_job`.

## 10. Add documents

Until reading from SeaweedFS is built ([section 16](#16-what-is-not-built-yet)), PDFs arrive the same way as on the PoC machine: as files in `data/raw/` inside the project folder on the VM.

```bash
cp <a file>.pdf data/raw/
```

Within about a minute Dagster's sensor starts `ingest_job` for it: parse, chunk, embed, index into the collection named in `config/pipeline.yaml`. Watch it in Dagster ([section 12](#12-look-at-the-ui-and-dagster-from-your-own-machine)) or with `docker compose logs -f dagster-code`.

- Only a new PDF starts a run. A document is known by its content, so a renamed copy is not new.
- A PDF whose run failed is not retried by itself: fix the cause, then run its partition of `ingest_job` in Dagster.
- `data/` must stay where it is: the ingestion writes its outputs under `data/artifacts/`, and the API reads pictures from there.

## 11. Connect the backend to the API

The API is the only thing the backend talks to. It never needs our database or Qdrant directly.

### Where it is

| The backend runs | Address |
|---|---|
| In a container on the platform's network | `http://api:8000` (or the container name, `http://rag-platform-api-1:8000`) |
| On the VM itself, outside Docker | `http://127.0.0.1:8000` |
| On another machine | Not reachable as it is: the port is bound to `127.0.0.1`. Put it behind the platform's reverse proxy, or change `127.0.0.1:8000:8000` in `docker-compose.yml` |

Every request carries the key: the header `X-API-Key: <RAG_API_KEY>`. Without it the answer is `401`.

`http://127.0.0.1:8000/docs` lists every request with its fields, and `http://127.0.0.1:8000/openapi.json` is the same as a file a client can be generated from.

### The requests

| Request | Returns |
|---|---|
| `GET /health` | `{"ollama": bool, "qdrant": bool, "database": bool}` |
| `GET /targets` | `{"collections": [...], "schemas": [...]}`: what a chat can search. A collection has `name`, `documents`, `points`, `config`; a schema has `name`, `description`, `tables` (a count), `registered` |
| `GET /check?kind=<kind>&target=<name>` | `{"missing": [...], "notes": [...], "settings": {...}}`. `missing` is what stops a question now (a model Ollama does not have, with the key to change); empty means ready |
| `GET /chats` | The 30 most recent chats: `session_id`, `title`, `updated_at`, `kind`, `target`, `turns` (a count) |
| `GET /chats/{id}` | One chat: `id`, `kind`, `target`, `title`, and `turns`, each with `id`, `question`, `answer`, `error`, `events` and more. `404` when there is none |
| `DELETE /chats/{id}` | Deletes a chat with its turns |
| `POST /chats/{id}/turns` | Asks a question. The reply is a stream, see below |
| `POST /turns/{turn id}/good` | Keeps a database answer as an example for similar questions. `{"id": <example id>}` |
| `GET /schemas/{schema}/examples` | The saved examples of a schema |
| `DELETE /examples/{id}` | Removes one |
| `GET /library` | Every collection with its documents |
| `DELETE /collections/{name}/documents/{doc_id}` | Removes one document from a collection |
| `DELETE /collections/{name}` | Removes a whole collection, with its files, records and chats |
| `GET /pictures/{path}` | A picture passage as PNG. `path` is the `image` of a hit |

`kind` is `documents` (a collection of the vector database) or `database` (a schema). `target` is the collection's or the schema's name, from `GET /targets`.

**Errors**

| Status | Meaning | Body |
|---|---|---|
| `401` | The key is missing or wrong | `{"detail": "..."}` |
| `404` | No such chat or picture | `{"detail": "..."}` |
| `409` | It cannot be done as asked: the target does not exist, a model is missing, the chat searches something else | `{"detail": "<the reason, in a sentence for a person>"}` |
| `422` | The request is malformed (a missing field, a wrong `kind`) | FastAPI's own list of problems |

### Chats

- **The backend chooses a chat's id**: any short string that is safe in a URL, for example a UUID. There is no "create chat" request. The chat is made by its first question.
- **A chat keeps what it searches.** Its first question fixes `kind` and `target`; a later question with another one is answered with `409`.
- **The backend does not send the history.** A follow-up ("and last year?") is understood from the chat's saved turns, by its id.
- **There are no users.** Every chat is visible to every caller, and `GET /chats` lists them all. If the platform has users, the backend must keep which chat belongs to whom and only pass on what a user may see.

### Asking a question

```bash
curl -N -X POST http://127.0.0.1:8000/chats/3f2a9c1e7b10/turns \
  -H "X-API-Key: <RAG_API_KEY>" -H "Content-Type: application/json" \
  -d '{"kind": "database", "target": "public", "question": "Which machine had the highest temperature?"}'
```

The reply has the type `text/event-stream`. It is a series of blocks, each two lines and a blank line:

```
event: StepStarted
data: {"type": "StepStarted", "node": "condense"}

event: Query
data: {"type": "Query", "text": "Which machine had the highest temperature?", "rewritten": false}

...

event: Done
data: {"type": "Done", "answer": "The machine with the highest temperature was Press B, at 88.9.", "turn_id": 1, ...}
```

The `data` line is one JSON object, and its `type` repeats the `event` line. The stream ends after `Done`, or after `Failed`.

**If the question cannot start**, there is no stream: the reply is `409` with the reason. So check the status before reading.

**The events a front end needs**

| Event | Fields | Use |
|---|---|---|
| `StepStarted` | `node` | Show what is happening now (see the two lists below) |
| `AnswerToken` | `text` | A piece of the answer. Join them in order to show the answer as it is written |
| `Thinking` | `text` | A piece of the model's reasoning, if you want to show it |
| `Done` | `answer`, `cited`, `abstained`, `turn_id`, `total_ms`, `saved`, `save_error` | The end. `answer` is the whole answer. `abstained: true` means "not found" or "could not be answered", and `answer` then says so. `turn_id` is what `POST /turns/{id}/good` takes |
| `Failed` | `message` | The turn stopped with an error (Ollama or a database went away). There is no `Done` |

**A documents turn**, in order: `condense` (then `Query`: the question as it is searched for) → `retrieve` (then `Retrieved`, with `hits`) → `grade` (then `Graded`) → `generate` (`Thinking`, `AnswerToken`) → `Done`. When the passages do not answer the question it tries one other query (`rewrite`, `Rewrote`, then `retrieve` and `grade` again), and if that fails too, `abstain`.

- A hit has `rank`, `similarity` (the reranker's score, 0 to 1), `text`, `source_file`, `page`, `modality` (`text`, `table` or `picture`), `headings`, `doc_id`, `chunk_id` and, for a picture, `image`.
- The answer cites passages as `[1]`, `[2]`: the number is a hit's `rank`. `Done.cited` lists the numbers that were cited.

**A database turn**, in order: `condense` (`Query`) → `schema` (`SchemaShown`) → `examples` (`ExamplesFound`, only when the schema has saved examples) → `write_sql` (`Thinking`, then `SqlWritten`) → `check` (`SqlChecked`) → `run_sql` (`SqlRan`) → `answer` (`AnswerToken`) → `Done`. A query that is refused or fails is rewritten (`Repairing`, `SqlWritten` again), at most `max_repairs` times.

- `SqlChecked` has `ok`, `sql` (what will run when `ok`) and `reason` (why not).
- `SqlRan` has `columns`, `rows` (at most the first 100), `row_count`, `truncated` and `ms`. This is the table to show under the answer.

The other events (`StepFinished`, `ModelState`) carry timings and model details for a debug view. A front end can ignore any event it does not know.

### Reading the stream in JavaScript

`EventSource` cannot be used, because the question is sent with `POST`. Read the body of a `fetch` instead. This works in Node 18 or newer and in a browser:

```js
async function ask(chatId, kind, target, question, onEvent) {
  const reply = await fetch(`${API}/chats/${chatId}/turns`, {
    method: "POST",
    headers: { "Content-Type": "application/json", "X-API-Key": API_KEY },
    body: JSON.stringify({ kind, target, question }),
  });
  if (!reply.ok) {
    throw new Error((await reply.json()).detail);      // 409: the reason, for the person who asked
  }
  const reader = reply.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    const blocks = buffer.split("\n\n");
    buffer = blocks.pop();                              // the last one may be incomplete
    for (const block of blocks) {
      const data = block.split("\n").find((line) => line.startsWith("data: "));
      if (data) onEvent(JSON.parse(data.slice(6)));
    }
  }
}

let answer = "";
await ask("3f2a9c1e7b10", "database", "public", "Which machine had the highest temperature?", (event) => {
  if (event.type === "AnswerToken") answer += event.text;          // show it growing
  else if (event.type === "Done") answer = event.answer;           // the final text
  else if (event.type === "Failed") console.error(event.message);
});
```

**Keep the API key in the backend.** The browser should call the platform's backend, and the backend calls our API and passes the stream on. The key must not be in the frontend's code.

### Three things that go wrong with streams

- **Timeouts.** An answer takes from a few seconds to a few minutes, depending on the models and the machine. The backend's HTTP client must not give up after 30 seconds.
- **Buffering.** A reverse proxy between the two holds the stream back until it ends, unless told not to. For nginx: `proxy_buffering off;` and a long `proxy_read_timeout` on that location.
- **One answer at a time.** There is one Ollama. Two questions asked together are both accepted, and the second waits for the model. Nothing is lost, it is only slower.

### A backend that does not want a stream

Read the stream to its end and use only the last event: `Done.answer` is the whole answer. A chat can also be read back afterwards with `GET /chats/{id}`: each turn has its `answer` and its `events`.

## 12. Look at the UI and Dagster from your own machine

The three web pages are bound to `127.0.0.1` on the VM, so they cannot be opened from outside. Forward them through SSH instead:

```bash
ssh -L 8501:127.0.0.1:8501 -L 3000:127.0.0.1:3000 -L 8000:127.0.0.1:8000 <user>@<the VM>
```

While that connection is open, on your own machine:

| Open | For |
|---|---|
| http://localhost:8501 | The Streamlit UI: Experiments and Chatbot. It uses the same API as the backend, so what works here works for the backend |
| http://localhost:3000 | Dagster: the ingestion runs, the sensors, `register_job` |
| http://localhost:8000/docs | The API's own page, where each request can be tried |

## 13. Day to day

| After | Do |
|---|---|
| Editing `config/pipeline.yaml` or `config/llm.yaml` | Nothing. They are read at the next run and the next question |
| Editing `config/connections.yaml` or `.env` | `docker compose up -d` (for `.env`), then `docker compose restart api ui` |
| `git pull` with code changes only | `docker compose restart api ui dagster-code dagster-daemon dagster-webserver` |
| `git pull` that changed `pyproject.toml` or a Dockerfile | `docker compose up -d --build` |
| `git pull` with a new file in `src/rag_lab/core/migrations/` | `docker compose up -d`: `setup` runs again and applies it |
| The platform's tables changed | `docker compose exec api python -m rag_lab tables register` |

Logs:

```bash
docker compose logs -f api            # questions and answers
docker compose logs -f dagster-code   # ingestion
docker compose logs setup             # the last start's check
```

The command line, for trying things without a front end:

```bash
docker compose exec api python -m rag_lab search "query text" --experiment <name>
docker compose exec api python -m rag_lab chat documents "question" --experiment <name>
docker compose exec api python -m rag_lab chat database "question" --schema public
```

**What to back up:** the schema `rag` in the platform's database (the chats and the good answers), Qdrant's storage (the vectors), and the folder `data/` (the PDFs and the outputs of each stage). Dagster's database and the two Docker volumes can be made again.

## 14. When something is wrong

| What you see | Cause | Do |
|---|---|---|
| `setup` exits with `app_database.url: cannot connect` | Wrong host, port, role or password | Check `POSTGRES_HOST` (the container's name), `POSTGRES_USER`, `POSTGRES_PASSWORD` in `.env`, and that our containers are on the right network (`PLATFORM_NETWORK`) |
| `setup` exits with `there is no schema 'rag'` | The script was not run for this database, or `app_schema` was another name | Run `scripts/provision.sql` ([section 3](#3-prepare-postgresql-the-admin)) |
| `setup` exits with `the environment variable ... is not set` | `connections.yaml` refers to a `${NAME}` that `.env` does not have | Add it to `.env`, or remove the line that uses it |
| `setup` exits naming a key of a YAML file | A misspelled or unknown key | Fix that key |
| `docker compose up` says the network is not found | `PLATFORM_NETWORK` is not the network's exact name | `docker network ls` |
| `/health` has `"ollama": false` | Ollama is not reachable from the containers | See the table in [section 5](#5-write-env). Test: `docker compose exec api python -c "import httpx; print(httpx.get('<OLLAMA_BASE_URL>/api/tags').status_code)"` |
| `/health` has `"qdrant": false` | Wrong name in `qdrant.url`, or another network | Check the Qdrant container's name and network |
| A question is answered with `409` and "Ollama has no model ..." | The model named in `config/llm.yaml` is not pulled | `ollama pull <name>`, or name one that is there |
| `tables register` says "no table can be read" | The reader was not granted the schema | [Section 9, step a](#9-tell-the-chatbot-about-the-tables) |
| Every database question is answered "The schema ... is ... characters; the model is given at most ..." | The schema's description is over the budget | Grant fewer tables and register again, or raise `schema_char_budget` and `num_ctx` |
| The daemon logs "could not reach user code server" | `dagster-code` is still starting (about 50 seconds) | Wait. If it goes on, `docker compose logs dagster-code` |
| A PDF in `data/raw` is not ingested | The sensor's tick failed, or the run failed | Dagster, *Automation* for the sensor's message and *Runs* for the run. A failed PDF is run again by hand: its partition of `ingest_job` |
| The answer arrives all at once, not piece by piece | A proxy buffers the stream | [Three things that go wrong with streams](#three-things-that-go-wrong-with-streams) |
| The platform's own `api` (or `ui`) stops answering reliably after our start | Two containers with the same name on one network | [Section 2, check 8](#2-what-to-find-out-first) |

## 15. Removing it again

```bash
docker compose down -v          # our containers and our two volumes (the Docling models, Dagster's logs)
```

That leaves what was made on the platform's servers. To remove that too:

```sql
-- as a superuser, connected to the platform's database
DROP SCHEMA rag CASCADE;                    -- our tables: chats, metrics, the registry
REVOKE ALL ON ALL TABLES IN SCHEMA public FROM rag_reader;
REVOKE ALL ON SCHEMA public FROM rag_reader;
ALTER DEFAULT PRIVILEGES IN SCHEMA public REVOKE SELECT ON TABLES FROM rag_reader;

-- connected to another database, for example postgres
DROP DATABASE dagster;
REVOKE ALL ON DATABASE platform FROM rag, rag_reader;
DROP ROLE rag_reader;
DROP ROLE rag;
```

In Qdrant, delete the collections named after the experiments and the ones that start with `sqlexamples__`. Only this project uses that Qdrant, so that is every collection in it.

## 16. What is not built yet

- **PDFs from SeaweedFS.** Today a PDF must be a file in `data/raw/` on the VM. The plan is that Dagster's sensor first copies what is new in a bucket into that folder and then goes on as it does now, so nothing after the folder changes. It is waiting for the decision on how SeaweedFS is set up. The place for it is `src/rag_lab/ingest/source.py`.
- **Users.** The API has one shared key and no idea of who is asking. Which user may see which chat is the backend's job.
- **Sampling of very large tables.** `tables register` reads each column in full. On a table too large for the reader's 15 seconds, that column's description stays without its values. If that happens on the platform's real tables, a sample of the rows is the next step.
- **The join to the platform's network has not been run on a real server.** `docker-compose.vm.yml` is read correctly by Docker Compose, and everything else in this guide was run on the PoC machine against a PostgreSQL prepared only by `scripts/provision.sql`. The first start on the VM is where a name or a network that differs from this guide will show.
