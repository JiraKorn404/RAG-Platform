# Completed plan: Phases 0 to 12

> **Archive.** This is the plan as it was written and checked off while the project was built, with
> the results of each phase. It is history, not the current state: Phases 6, 6a, 7 and 8 describe a
> benchmark dashboard, a matrix runner, a query set and a model-by-strategy comparison that Phase 11
> removed, and Phase 1's `rag_metrics` schema lists benchmark tables that no longer exist. For what
> the repo is now, read `CLAUDE.md`. The plan for new work is in `PLAN.md`.

Build a modular, experiment-friendly ingestion pipeline: PDFs in, searchable text and table vectors out, with similarity scores and a full set of performance metrics stored in PostgreSQL.

Each phase ends with something that runs. Do them in order; Phase 0 exists to find network problems before any pipeline code is written.

## Working approach

Keep the project minimal. Testing is deliberately light:

- **Tests:** only the two small files listed in Phases 1 and 6 (config hashing and vector truncation, retrieval metric formulas). Nothing else gets a test file.
- **Verification:** each phase's "Done when" is checked once, by a real run against the real stack. No mocks, no fake embedders, no integration test suite.
- **After small edits:** do not rerun tests, rebuild containers or re-materialise assets. Rerun only when the logic of a tested area changed.
- **Optional items** (in Phase 5, under "Ideas" and under "Deferred") are built only when you ask for them.

## Scope

- **Documents:** PDF.
- **Content:** text and tables. Image and figure ingestion is deferred (see "Deferred").
- **Hardware:** no NVIDIA GPU on this machine. Docling runs on CPU; embedding runs on the MacBook.

## Decisions taken

| Topic | Decision | Reason |
|---|---|---|
| Docling placement | In-process in the Dagster code-location container, CPU-only torch | Fewest moving parts. `docling-serve` as a separate container is a later experiment. |
| Experiment isolation | One Qdrant collection per experiment name | Different dimensions or chunkers cannot share a collection cleanly. |
| Partitioning | One Dagster dynamic partition per document | Re-run a single document; failures are isolated. |
| Embedding | Computed by our Ollama client; Qdrant only stores and searches vectors | Keeps embedding swappable and measurable. |
| Tables | Serialised to Markdown and embedded as text, tagged `modality = table` | The embedding model is text-only; tagging allows filtering and per-modality metrics. |
| Metrics storage | PostgreSQL in Docker, database `rag_metrics` | Queryable history across experiments. |
| Postgres instances | One container, two databases: `dagster` and `rag_metrics` | One less service to run. Separate databases keep Dagster's tables apart from ours. Splitting into two containers later only means changing a connection string. |
| Intermediates | Files under `data/artifacts/<experiment>/<stage>/` | Stages re-run independently; outputs are inspectable. |
| Search interface | Python module + CLI first; UI is optional later | Search must work without Dagster. |
| Quality metrics | Built now, computed once `eval/queries.yaml` has expected results | The query set will be filled in later; latency and score metrics work without it. |

---

## Phase 0 — Connectivity and skeleton

Goal: every service is reachable from where it needs to be reached.

- [x] On the Mac: `ollama pull qwen3-embedding:0.6b`; set `OLLAMA_HOST=0.0.0.0` (`launchctl setenv OLLAMA_HOST "0.0.0.0:11434"`, then restart Ollama); note the Tailscale IP.
- [x] From Windows PowerShell: `curl http://<mac-ip>:11434/api/tags` lists the model.
- [x] `git init`, `.gitignore` (`.env`, `data/`, `.venv/`), `pyproject.toml` with `uv`, `.env.example`.
- [x] `docker-compose.yml` with `qdrant`, `postgres`, `dagster-webserver`, `dagster-daemon`, `dagster-code`. Named volumes for Qdrant storage (never a bind mount on Windows), Postgres, and the Docling/HuggingFace model cache. Bind-mount `./data` and `./src`.
- [x] `docker/postgres/init/` script that creates the `rag_metrics` database beside `dagster`. Health check on Postgres; Dagster services wait for it.
- [x] **From inside a container**, call the Mac's `/api/embed`. This is the main risk in the project: Docker Desktop normally routes container traffic to the tailnet through the host, but it must be confirmed.
  - Confirmed working (2026-10-02): a container reaches the Mac at its Tailscale IP, so the fallback is not needed. If it ever fails: a Tailscale sidecar container, with `dagster-code` using `network_mode: service:tailscale`.
- [x] Trivial Dagster asset that embeds one string, writes one point to Qdrant, and inserts one row into `rag_metrics`.

Done when: the Dagster UI is at `localhost:3000` and the smoke asset materialises.

**Status: done (2026-10-02).** The smoke asset ran through the webserver and succeeded: 1024-dimension embedding in about 2.5 s (cold model), one point in Qdrant collection `smoke`, one row in `rag_metrics.smoke_check`. `smoke.py` and the `smoke_check` table are throwaway and go in Phase 1. Notes:
- `sqlalchemy<2.1` is pinned in `pyproject.toml`: SQLAlchemy 2.1 defaults to psycopg 3, which breaks dagster-postgres.
- Docling and the Hugging Face cache volume are wired but Docling is not installed yet (Phase 2).
- `uv` is not installed on this machine; install it (`winget install astral-sh.uv`) before running `uv run` commands or tests locally. Containers do not need it locally.

## Phase 1 — Resources, config and metrics store

- [x] `config.py`: `ExperimentConfig` holding `name`, `ParseConfig`, `ChunkConfig`, `EmbedConfig`, `IndexConfig`; a stable `config_hash()` that ignores `name`. `ChunkConfig` has the shared settings only; per-strategy settings are added in Phase 3.
- [x] `OllamaResource`: batch `embed(texts, kind="document"|"query")`, applies the query instruction prefix, optional dimension truncation with re-normalisation, retries, timeout. Returns vectors plus timings (Ollama reports `total_duration`, `load_duration`, `prompt_eval_count`).
- [x] `QdrantResource`: client lifecycle (gRPC preferred), `ensure_collection(experiment)` including payload indexes, batch upsert, delete-by-document (filter on `doc_id`).
- [x] `MetricsStoreResource`: connection to `rag_metrics`, one write method per table below. Plain Python underneath so the search CLI can use it without Dagster.
- [x] Migrations for the `rag_metrics` schema: numbered SQL files in `src/rag_lab/metrics/migrations/`, applied by `MetricsStoreResource` at start-up under an advisory lock (no Alembic).
- [x] `metrics/`: a `timed()` context manager and a percentile helper.
- [x] One small test file covering only config hashing and truncation with re-normalisation.

Done when: each piece works against the live stack. **Status: done (2026-10-02).** One real check covered migrations (idempotent), document and query embedding with timings, truncation to 256 dimensions at unit length, Qdrant over gRPC (create, double upsert gives no duplicates, payload indexes, delete by document, vector-size mismatch rejected), and a write to every `rag_metrics` table. Dagster loads the resources. The two config/truncation tests pass. The throwaway smoke asset and its table are removed, and `OLLAMA_MODEL` is gone from the env files because the model is now `EmbedConfig.model`. `definitions.py` has no assets until Phase 2.

### `rag_metrics` schema

| Table | One row per | Main columns |
|---|---|---|
| `experiments` | experiment config | `config_hash` (key), `name`, `config` (jsonb), `created_at` |
| `ingestion_stage_metrics` | document × stage × run | `config_hash`, `doc_id`, `stage` (`parse` / `chunk` / `embed` / `index`), `dagster_run_id`, `duration_ms`, `items` (pages, chunks, vectors or points), `throughput`, `details` (jsonb), `created_at` |
| `eval_queries` | query in the query set | `query_id`, `query_text`, `expected` (jsonb, nullable), `query_set_version` |
| `benchmark_runs` | benchmark execution | `benchmark_run_id`, `config_hash`, `dagster_run_id`, `top_k`, `repeats`, `query_set_version`, `collection_point_count`, `started_at`, `finished_at` |
| `query_results` | query × repeat in a benchmark | `benchmark_run_id`, `query_id`, `repeat`, `embed_ms`, `search_ms`, `total_ms`, `hits` (jsonb: chunk id, doc, page, modality, distance, similarity, rank), `first_relevant_rank` (nullable) |
| `benchmark_metrics` | metric value in a benchmark | `benchmark_run_id`, `metric`, `k` (nullable), `modality` (nullable), `value` |
| `search_log` | ad-hoc search from the CLI | `config_hash`, `query_text`, `top_k`, `embed_ms`, `search_ms`, `total_ms`, `top_similarity`, `created_at` |

`benchmark_metrics` is deliberately long-format so a new metric needs no schema change.

## Phase 2 — Parsing with Docling

- [x] `ParseConfig` fields, each mapped onto Docling's PDF pipeline options:
  - OCR is fixed off (`do_ocr = False`) and not exposed. Only PDFs with a text layer are supported; scanned PDFs will parse to little or no text.
  - `do_table_structure`, `table_mode` (`fast` / `accurate`), `table_cell_matching`
  - `do_formula_enrichment`, `do_code_enrichment`
  - `num_threads` (CPU), `document_timeout`
  - Picture options (image generation, description, classification) are fixed off and not exposed.
- [x] `parsing/`: `build_converter(ParseConfig)` and `parse(path) -> ParsedDocument`. Writes the Docling JSON and a Markdown export to the artifacts folder.
- [x] Sensor on `data/raw/*.pdf` that registers a dynamic partition per new file (keyed by content hash, so a renamed file is not re-ingested).
- [x] `parsed_document` asset. Metrics to Postgres and asset metadata: pages, tables found, parse seconds, pages per second, characters extracted per page, Markdown preview. The per-page character count makes a scanned PDF (near zero) easy to spot.
- [x] Confirm Docling's model downloads land in the cache volume, not the container layer.

Done when: dropping a PDF in `data/raw/` produces parsed output, and changing `table_mode` in the launchpad changes the result.

**Status: done (2026-10-02).** Checked once against the live stack with two generated PDFs (`data/raw/sample-report.pdf`: 2 pages, headings, one table; `sample-scanned.pdf`: image only):
- The sensor registered both files as partitions within one tick, keyed by content hash.
- Launched from the webserver with run config, `parsed_document` wrote `<doc_id>.json`, `.md` and `.meta.json` under `data/artifacts/<experiment>/parse/` and one `parse` row per run in `ingestion_stage_metrics`. The table came out as a clean 5 by 4 Markdown table.
- `table_mode` `accurate` versus `fast` gave different config hashes and timings (18.4 s versus 12.4 s for the same PDF) but identical Markdown for this simple table. Expect differences only on harder tables.
- The scanned PDF parsed to 0 characters and logged the scanned-PDF warning.
- Models downloaded into the `model_cache` volume (about 500 MB), not the image. The first run took 218 s to load models; later runs 13 to 22 s.

How to use it: add a PDF to `data/raw/`, wait for the sensor (30 s), then in the Dagster UI materialise `parsed_document` for that partition. Run config is set once per run on the shared `experiment` resource (see Phase 3): `resources.experiment.config` with `name` and optional `parse`, `chunk`, ... sections.

Notes:
- The sample PDFs were generated because the earlier `report.pdf` and `scan.pdf` had disappeared from `data/raw/`. Replace them with real documents.
- The code location takes about 50 s to start (it imports Docling), so the Dagster UI can show a brief "could not reach user code server" error after `docker compose up`; reload the location.
- Config sections now inherit `dagster.Config` so one definition serves Python code and the launchpad (plain pydantic models are not accepted as nested Dagster config).

## Phase 3 — Chunking

The `fixed`, `recursive` and `semantic` strategies run in two steps, so that tables stay whole and every chunk keeps its page and headings:

1. **Segment** the Docling document into blocks: text blocks (with their headings and page) and table blocks. Picture items are skipped.
2. **Split** the text blocks with the chosen strategy. Table blocks bypass the strategy and follow `table_handling`.

`hybrid` and `hierarchical` are Docling's own chunkers and do both steps themselves. Docling's default table text ("row, column = value" triplets) embeds poorly, so both are given a Markdown table serialiser. They cannot honour `row-wise` (it falls back to Markdown, with a warning), `skip` drops table-only chunks, and `hybrid` with `merge_peers` may merge a small table into its neighbouring text. Set `merge_peers` to false for isolated tables.

| Strategy | How it splits | Own settings |
|---|---|---|
| `hybrid` | Docling `HybridChunker`: follows document structure, then splits or merges to fit the token limit | `merge_peers` |
| `hierarchical` | Docling `HierarchicalChunker`: one chunk per structural element | none |
| `fixed` | Fixed token windows | none |
| `recursive` | Tries separators in order (paragraph, line, sentence, word), recursing to the next one only for pieces still over the limit | `separators` |
| `semantic` | Splits into sentences, embeds them, and breaks where the distance between neighbouring sentence groups jumps | `buffer_size` (sentences per group), `breakpoint_type` (`percentile` / `stddev` / `absolute`), `breakpoint_threshold`, `min_tokens` |

- [x] `ChunkConfig`: `strategy`, shared settings (`max_tokens`, `overlap`, `table_handling` (`markdown` / `row-wise` / `skip`), `include_headings_in_text`), and nested settings for `hybrid`, `recursive` and `semantic`. Only the chosen strategy's settings count: the others are left out of `config_hash()`. `overlap` is used by `fixed` only; the other strategies ignore it and log a warning. Headings are part of the embedded text and use up part of the `max_tokens` budget.
- [x] `chunking/base.py`: a `Chunker` protocol and a registry keyed by strategy name. Adding a strategy means adding one file and one registry entry.
- [x] `chunking/segment.py`: the shared segmenter.
- [x] `recursive`: own implementation measured in tokens, not characters. No splitter library dependency.
- [x] `semantic`:
  - Sentence embeddings come from `OllamaResource` in document mode (no query prefix), batched per document.
  - A semantic chunk over `max_tokens` is split again with `recursive`; one under `min_tokens` is merged into its neighbour.
  - The breakpoint threshold is taken over the whole document, not per section, so a short section is not forced to split. The threshold's meaning depends on `breakpoint_type` (percentile 0-100, standard-deviation multiples, or an absolute cosine distance).
  - Sentence-group embeddings are cached on disk (`data/artifacts/_cache/embeddings/`), keyed by model, dimension and text, and shared across experiments, so changing only a breakpoint setting does not call Ollama again.
- [x] Chunk model: `chunk_id` (deterministic), `doc_id`, `text`, `modality` (`text` / `table`), `page`, `headings`, `bbox`, `token_count`, `strategy`.
- [x] Token counting uses the Qwen3 tokenizer so `max_tokens` means what the embedder sees.
- [x] `chunks` asset. Metrics: chunk count, counts per modality, token-length min/mean/max, chunking seconds, a sample chunk. For `semantic`, also sentences embedded, embedding seconds and breakpoints found.
- [x] No per-strategy tests. Check strategies by running them on a real PDF and reading the chunk metrics.
- [x] One `ExperimentResource` (nested config) shared by every asset, so `resources.experiment.config` is set once per run instead of once per asset. `parsed_document` was switched to it.

Done when: the same PDF chunked with all five strategies gives five comparable rows in `ingestion_stage_metrics`.

**Status: done (2026-10-02).** Checked once against the live stack with a generated 2-page PDF (`data/raw/handbook.pdf`: five topic sections of two paragraphs, a closing 12-row table). Parse plus chunk ran for five experiments with `max_tokens` 128:

| Strategy | Chunks (text / table) | Tokens min / mean / max |
|---|---|---|
| hybrid | 13 (10 / 3) | 59 / 85 / 127 |
| hierarchical | 12 (11 / 1) | 13 / 88 / 294 |
| fixed (overlap 16) | 14 (11 / 3) | 13 / 85 / 128 |
| recursive | 14 (11 / 3) | 13 / 80 / 120 |
| semantic (percentile 75) | 15 (12 / 3) | 13 / 75 / 120 |

- Every strategy gave a `chunk` row in `ingestion_stage_metrics`, with counts, token statistics and (for `semantic`) sentences, embeddings, cache hits, threshold and breakpoints.
- No chunk is over the limit except `hierarchical`, which has no size limit and keeps the whole table as one chunk (294 tokens).
- The oversize table was split by rows with the header repeated in each piece. Headings and page numbers are on every chunk.
- Semantic: 41 sentence groups embedded in 5.7 s. Rerunning with threshold 90 embedded none (41 cache hits) and found different breakpoints (4 instead of 9). `fixed` overlap was confirmed by the last words of one chunk reappearing at the start of the next.
- A `semantic` run needs the Mac; the other four do not.

Notes:
- Each experiment has its own `parse/` folder, so comparing chunkers means re-parsing the PDF once per experiment (about 20 s each). Fine for a lab; if it becomes annoying, let a chunk run read another experiment's parse output.
- Docling treated a larger first test table (60 rows across a page break) as a picture, and found no table. This is Docling's layout model, not the chunker; a table that Docling does not detect is never chunked as a table.
- `config_hash` changed with the new settings, so earlier experiment rows have different hashes. `EmbedConfig` gained `tokenizer` (Hugging Face id of the Qwen3 tokenizer used to count tokens).
- One extra test assertion covers the new hashing rule (3 tests pass); the chunkers have none, as planned.

## Phase 4 — Embedding and indexing

- [x] `embeddings` asset: batches chunks through `OllamaResource`, stores vectors as `embed/<doc_id>.npy` (float32, one row per chunk) in the experiment's artifacts folder. Metrics: batch size, total seconds, chunks per second, tokens per second, model load time, dimension.
- [x] Collection setup: vector size from `EmbedConfig`, cosine distance, HNSW parameters (`m`, `ef_construct`). Quantisation is not built; it is a Phase 7 option. Payload holds the chunk model fields plus `experiment`, `config_hash`, `source_file`, `ingested_at`. Payload indexes on `doc_id`, `modality`, `source_file`.
- [x] `qdrant_index` asset: deletes the document's existing points in that collection, then batch-upserts with deterministic UUIDs (uuid5 of the chunk ID) and `wait=True` so re-runs are idempotent and timings are honest. Metrics: points written, batches, upsert seconds, points per second, collection size. A failed batch raises out of the asset, so there is no partial-failure count.
- [x] `ingest_job` covering parse through index for selected partitions.

Done when: one job run takes a PDF from `data/raw/` to queryable points, running it twice leaves the same point count, and each stage has a row in `ingestion_stage_metrics`.

**Status: done (2026-10-02).** Checked against the live stack with `handbook.pdf`, launching `ingest_job` through the webserver:
- One run produced a row for each of `parse`, `chunk`, `embed` and `index` in `ingestion_stage_metrics`. The embed row has dimension, tokens, tokens per second and model load time; the index row has points written and collection size.
- The collection was created with 1024-dimension cosine vectors, HNSW m=16 and ef_construct=100, and payload indexes on `doc_id`, `modality` and `source_file`.
- Running the same job again left the same point count (6).
- An experiment with `embed.dimension` 256 and the `recursive` chunker gave a 256-dimension collection with 14 points, so truncation works through the assets.
- A query embedded with the instruction prefix found the right chunks: the Dagster section for a Dagster question (similarity 0.64) and the table chunk for a benchmark question (0.71).

Notes:
- To ingest: in the Dagster UI launch `ingest_job` for a document partition and set `resources.experiment.config` (`name` plus any `parse`, `chunk`, `embed`, `index` settings). Individual assets can still be run alone, in order.
- Changing `embed.dimension` under an existing experiment name fails at indexing (vector size mismatch) after the embeddings are rewritten, so use a new experiment name for a new dimension.
- `ingest_job` is partitioned per document; selecting several partitions in the UI launches a backfill with one run each.

## Phase 5 — Search with scores

- [x] `search/`: `search(query, experiment, top_k, filters)` embeds the query with the instruction prefix and runs `query_points`, with `hnsw_ef` taken from config. Each hit returns text, source file, page, modality, `similarity` (Qdrant's cosine score) and `distance` (`1 - similarity`).
- [x] Per-query timing breakdown: embed ms, Qdrant ms, total ms.
- [x] CLI: `python -m rag_lab.search "..." --experiment <name> --top-k 5 [--modality table] [--json] [--no-log]`. Writes to `search_log` unless `--no-log`.
- [ ] Optional: `--exact` flag (brute-force search) to measure how much recall the HNSW index gives up.
- [ ] Optional: hybrid mode for comparison. In Qdrant this needs a sparse (BM25) vector stored on each point at index time, fused with the dense result at query time, so it touches Phase 4 as well.

Done when: the CLI returns ranked hits with similarity and distance, filters by modality, and logs to `search_log`. **Status: done (2026-10-02).** Checked once against the live stack with the `docs` experiment (`aiayn.pdf`):
- "How does multi-head attention work?" returned the Multi-Head Attention section first (similarity 0.732, 339 ms embed, 16 ms Qdrant, 355 ms total) and wrote one `search_log` row.
- `--modality table --json --no-log` returned only table chunks (BLEU query: Table 2 first, 0.636) as JSON and wrote no row.
- `--exact` and hybrid mode (the optional items) are not built.

How to use it: `docker compose exec dagster-code python -m rag_lab.search "..." --experiment <name> --top-k 5`. The experiment's settings (embedding model, dimension, query instruction, `hnsw_ef`) are read from the newest row for that name in `rag_metrics.experiments`, so ingest at least `parsed_document` and `chunks` under the name first. `rag_lab.search.search(query, config, embedder, store, ...)` is plain Python for the Phase 6 benchmark to reuse. Running outside the container needs `OLLAMA_BASE_URL`, `QDRANT_URL` and `METRICS_DATABASE_URL` set.

## Phase 5a — Notebook search helper

Goal: try similarity searches from a Jupyter notebook with one function call, instead of the CLI's `docker compose exec ... --experiment ... --top-k ...` line. Nothing is rebuilt: the notebook is a thin front end over the Phase 5 `search()`.

Where it runs: the notebook kernel runs on the Windows host using the existing `.venv`, not in a container (a Jupyter container would be one more service). The host already reaches everything: Qdrant on `localhost:6333/6334` and Postgres on `localhost:5432` are published by `docker-compose.yml`, and Ollama is reached at the Tailscale IP from `.env`. Open the notebook in VS Code (or `uv run jupyter lab`) and pick the `.venv` kernel.

- [x] `search/quick.py` with one function, `ask(query, experiment, top_k=5, modality=None, log=False)`:
  - Builds the Ollama, Qdrant and metrics clients itself, so the notebook needs no setup cell. Settings come from the environment, with host defaults when unset: `QDRANT_URL` falls back to `http://localhost:6333`, `METRICS_DATABASE_URL` is built from `POSTGRES_USER` and `POSTGRES_PASSWORD` at `localhost:5432/rag_metrics`, and `OLLAMA_BASE_URL` has no default. The three values are read from `.env` if the variables are not already set (a few lines of parsing, no `python-dotenv`).
  - Looks the experiment's settings up in `rag_metrics.experiments` (the same lookup as the CLI), so the embedding model, dimension and query instruction always match what was ingested. The lookup moves from `__main__.py` into a shared helper so the CLI and `ask()` use one copy.
  - Calls `search()`, prints one compact line per hit (rank, similarity, modality, file and page, first 100 characters) followed by the three timings, and returns the `SearchResult` so cells can inspect `result.hits[0].text`.
  - `log=True` writes to `search_log`; the default is off, so notebook experiments do not fill the log.
- [x] `notebook/search.ipynb`: a first cell `from rag_lab.search.quick import ask`, then example cells: a plain query, a `modality="table"` query, and the same query against two experiments to compare scores. Outputs are cleared before committing.
- [x] Add `ipykernel` to the `dev` dependency group in `pyproject.toml`. This is the stated need for it; nothing else is added.
- [x] `CLAUDE.md`: add `notebook/` to the layout and the `ask()` line to the commands.

Not included: a rewrite of the CLI (it stays as it is), `--exact` or hybrid search, filters other than `modality`, pandas tables or charts, and tests (the function is glue over code that is already checked).

Done when: after `docker compose up`, `ask("...", "docs")` in the notebook prints ranked hits from the host with no extra setup, and the same query gives the same top hit and similarity as the CLI. One real run, no more.

Open points to confirm before building:
- The kernel runs on the host. If you would rather run it in a container, that needs a `jupyter` service (or an install in `dagster-code`) and adds weight to the Docker setup.
- `ask()` reads `.env` for the Ollama address, because the Tailscale IP is not in the host's environment by default.

**Status: done (2026-10-02).** Checked once: from the host `.venv`, `ask("How does multi-head attention work?", "docs")` returned the same top hit as the CLI (similarity 0.732, 62 ms total) and the `modality="table"` query returned only table chunks (0.636 first). The CLI still works after the shared-lookup refactor. Notes:
- `ipykernel` ended up in the main `dependencies` of `pyproject.toml`, not the dev group.
- `.env` has `OLLAMA_BASE_URL=http://host.docker.internal:11434` (Ollama on this machine), which only resolves inside containers. `ask()` swaps that name for `localhost`.
- Postgres and Qdrant default to `127.0.0.1`, not `localhost`: Docker publishes them on IPv4 only and Postgres connections hung on the IPv6 attempt.
- A hit whose text has characters the Windows console cannot encode (such as `ϵ`) makes `print` fail in a plain cp1252 terminal. Notebooks print UTF-8, so it only matters outside them.

## Phase 6 — Metrics and benchmarking

All metrics below are implemented in this phase. Quality metrics are computed only for queries that have expected results; otherwise they are stored as null and the run still completes.

- [x] `eval/queries.yaml` format: `id`, `query`, optional `expected` list of `{source_file, page}` or `{chunk_id}`, optional `modality`. Ship it with placeholder queries and empty `expected`. Loading it syncs `eval_queries`.
- [x] `metrics/retrieval.py`, with one small test of the formulas against a hand-computed case (the one place a silent error would look plausible).
- [x] `search_benchmark` asset per experiment: warm-up pass, then N repeats of the query set. Writes `benchmark_runs`, `query_results`, `benchmark_metrics`.
- [x] `experiment_summary` asset: a SQL view over `rag_metrics` giving one row per experiment, shown as a Markdown table in the Dagster UI.

| Group | Metrics |
|---|---|
| Search latency | p50, p95, p99, mean, min, max for embed, Qdrant search and total; queries per second; cold versus warm first query |
| Similarity scores | top-1 similarity, mean similarity over top-k, gap between rank 1 and rank k, score distribution per experiment |
| Retrieval quality (needs expected results) | Hit rate@k, Recall@k, Precision@k, MRR, MAP, nDCG@k for k in 1, 3, 5, 10; also split by modality (text versus table) |
| Ingestion performance | parse seconds and pages per second; chunks per document; embed chunks per second and tokens per second; model load time; index points per second; end-to-end seconds per document |
| Index state | point count, indexed vector count, vector dimension, segment count |

Done when: two experiments (for example table mode fast versus accurate, or two chunking strategies) can be compared with one SQL query, and filling in `expected` then re-running the benchmark produces quality metrics with no code change.

**Status: done (2026-10-02).** Checked against the live stack with the `docs` experiment (`aiayn.pdf`, 41 points):
- `search_benchmark` for `docs` (default 10 top-k, 5 repeats, the 3 shipped queries) wrote 1 `benchmark_runs` row, 15 `query_results` and 69 `benchmark_metrics`. Total latency p50 35 ms, p95 56 ms, 26.9 queries per second; the first query cost 172 ms cold and 37 ms warm. Top-1 similarity averaged 0.704. The 36 quality metrics (overall and for `table`) were stored as null because the shipped queries have no expected results.
- `SELECT * FROM experiment_summary` gives one row per experiment with ingestion performance (15 pages parsed at 0.47 pages per second, 41 chunks, 3.3 chunks and 887 tokens per second embedding, 10.5 s model load, 47 s per document) next to the benchmark columns.
- Filling in `expected` and re-running gave quality metrics with no code change, checked with a temporary query file: `attention` pointing at page 4 got rank 1, and a `bleu-table` query with one real and one made-up page got recall 0.5 and nDCG@5 0.613, which matches the hand calculation. Those test rows were deleted afterwards.
- The retrieval formulas have one test (`tests/test_retrieval.py`, hand-computed case including duplicate hits on one expected page); all 4 tests pass.

How to use it:
- Launch `search_benchmark` (and `experiment_summary`) from the Dagster UI with run config `resources.experiment.config.name: <experiment>`; optional `ops.search_benchmark.config` with `top_k`, `repeats`, `queries_file`. From a shell: `docker compose exec dagster-code dagster asset materialize -m rag_lab.definitions --select search_benchmark,experiment_summary --config-json '{"resources":{"experiment":{"config":{"name":"docs"}}}}'`.
- The benchmark takes only the experiment name from the run config. Its settings are read from the `experiments` table (as the search CLI does), so a benchmark always shares the config hash of the ingestion it measures. Benchmarking with a changed `index.hnsw_ef` is therefore not possible from the run config.
- To get quality metrics, add `expected` to entries in `eval/queries.yaml` (`{source_file, page}` or `{chunk_id}`) and re-run. The query set version is a hash of the file, so each edit is a new version. A query's `modality` restricts its search and also splits the metrics: the `table` rows cover queries marked `modality: table`.

Notes:
- Each expected item is matched by at most one hit (the best ranked), so two chunks from the same expected page cannot push a score above 1. MRR and MAP cover the whole retrieved top-k list, so their `k` is null. Precision, recall, hit rate and nDCG are stored for k in 1, 3, 5, 10 up to `top_k`.
- "Cold versus warm first query": `first_query_cold_ms` is the first query of the warm-up pass, `first_query_warm_ms` the mean of the same query across the timed repeats.
- `experiment_summary` is a SQL view (migration `0002_experiment_summary.sql`) over the latest run of each stage per document and the latest finished benchmark. The asset shows it transposed (one column per experiment) as Markdown.
- `indexed_vectors_count` is 0 for a small collection: Qdrant does not build the HNSW index below its indexing threshold, so a small lab collection is searched by brute force. This also means `--exact` would show no difference until the collection is large.
- `docker-compose.yml` mounts `./eval` into `dagster-code`; recreate that container (`docker compose up -d dagster-code`) once after pulling this change.
- A Dagster asset takes its run config only through a parameter named `config`, so the benchmark settings are `config: BenchmarkConfig`.

## Phase 6a — Notebook summary helper

Goal: see the `experiment_summary` view from the notebook with one call, `summary()`, instead of the `docker compose exec postgres psql ...` line or opening the Dagster UI. Same idea as Phase 5a: a thin helper over code that already exists, run from the host `.venv`.

- [x] `metrics/summary.py`: `summary_markdown(columns, rows)` builds the transposed Markdown table (one column per experiment, one row per metric). It moves out of the `experiment_summary` asset, which then calls it, so the notebook and the Dagster UI show the same table from one copy.
- [x] `search/quick.py`: `summary(names=None)`. It fetches the view through `MetricsStore.experiment_summary()` and renders the table in the notebook (`IPython.display.Markdown`; IPython comes with `ipykernel`). `names` is an optional list of experiment names to show; the default is all. The connection settings code that `ask()` already has (`.env`, `127.0.0.1` defaults) is split out into one helper that both functions use. It returns nothing, so the table is not shown twice.
- [x] `notebook/search.ipynb`: a final section with `from rag_lab.search.quick import ask, summary` and a `summary()` cell. Existing cells stay as they are.
- [x] `CLAUDE.md`: mention `summary()` next to `ask()`.

Not included: a pandas DataFrame (no new dependency), charts, or a way to run a benchmark from the notebook (benchmarks stay Dagster runs).

Done when: `summary()` in the notebook shows the same numbers as `SELECT * FROM experiment_summary`. One real run, no tests (the table builder moves unchanged, and there is no new logic to check).

**Status: done (2026-10-02).** Checked once: from the host `.venv`, the table `summary()` builds has the same numbers as `SELECT * FROM experiment_summary` for `docs` (15 pages, 41 chunks, search p50 35.2 ms, top-1 similarity 0.704, quality columns empty), and `summary(["nope"])` raises a clear error. The `experiment_summary` asset still materialises after the table builder moved. I did not open the notebook itself; in Jupyter `display(Markdown(...))` renders the table, while a plain script only prints the object.

How to use it: in `notebook/search.ipynb`, `from rag_lab.search.quick import ask, summary`, then `summary()` or `summary(["docs", "other"])`. It reads the view directly, so it is always current and needs no re-materialising.

## Phase 7 — Embedding model × chunking strategy comparison

Goal: find out which embedding model and which chunking strategy retrieve best on your documents, and see the answer in one place. The matrix is 3 models (`qwen3-embedding:0.6b`, `:4b`, `:8b`) × 5 strategies (`hybrid`, `hierarchical`, `fixed`, `recursive`, `semantic`) = 15 experiments over the same documents and the same query set. Two UIs sit on top: a dashboard over `rag_metrics`, and a page for trying a query against many experiments at once.

### Decisions

| Topic | Decision | Reason |
|---|---|---|
| What varies | Only the model and the strategy. Everything else (`max_tokens`, `overlap`, `table_handling`, HNSW, batch size, query instruction, parse settings) is one shared block in `experiments/matrix.yaml`. | A difference in the results must come from the model or the chunker, not from a stray setting. |
| Experiment names | `<model label>-<strategy>`, with labels set in the matrix file: `q3-0-6b-hybrid`, `q3-4b-fixed`, `q3-8b-semantic`, and so on. | Names double as Qdrant collection names, so they must match the existing name pattern. |
| Dimension | Each model at its native size (1024, 2560, 4096). | One collection per experiment, so sizes never clash. A fixed-dimension axis (Matryoshka) is an idea, not part of this phase. |
| Tokenizer | The same Qwen3 tokenizer for all three. | Same model family, so `max_tokens` means the same thing everywhere. |
| How relevance is judged | Expected items are `{source_file, page}`, optionally with `contains: "<text>"`. `chunk_id` is not used. | Chunk ids differ between chunkers, so they cannot be compared. A page is coarse (two chunkers both hit the right page), so `contains` adds a case-insensitive text check on the hit. |
| What is compared across models | Quality metrics (recall, MRR, nDCG) and ranks. Similarity scores only within one model. | Cosine scores of different models are on different scales; a higher score from the 8b model does not mean a better result. |
| Parsing | Each experiment parses the PDF again, as today. | Simplest, and keeps every experiment's metrics complete. About 20 to 30 s per experiment per document. Reusing one parse across experiments is an idea. |
| How the matrix runs | A command inside `dagster-code` that calls `dagster.materialize` once per experiment and document. | Runs show up in the Dagster UI, with no new service and no GraphQL client. |
| UI | One Streamlit app with two pages (dashboard, query), run as a Docker service `ui` built from the shared image, at `http://localhost:8501`. | One new dependency and one new service. In the container it already has the Qdrant, Postgres and Ollama addresses, the same as `dagster-code`, so it needs no `.env` parsing. |
| Documents | `aiayn.pdf` only ("Attention Is All You Need", 15 pages, 4 tables). | The chosen test document. One paper means the results describe how the models and chunkers behave on this kind of paper, not on documents in general. |

### 7.1 Groundwork

- [x] On the Mac: `ollama pull qwen3-embedding:4b` and `ollama pull qwen3-embedding:8b`. Check the 8b model fits in the Mac's memory next to the others; Ollama loads one at a time, but a swap-heavy Mac will distort the latency numbers. Embed one string with each and confirm the dimensions (1024, 2560, 4096).
- [x] Document: `data/raw/aiayn.pdf`, already there and ingested as the `docs` experiment.
- [x] Query set: about 30 queries in `eval/queries.yaml` replacing the three placeholders, with `expected` filled in: roughly 20 about the text (architecture, attention, training, results) and 10 marked `modality: table` (the paper's complexity, BLEU, model-variation and parsing tables). Each expected item is `{source_file, page}` plus a `contains` snippet taken from the paper. Without expected results the comparison has only latency and scores, which cannot rank models (see the decisions).
- [x] I draft the queries from the parsed paper and check every expected item against it (the page exists and the `contains` text is on that page) before handing the file to you; an item that matches nothing would silently score zero. You review the file. The whole comparison rests on it.
- [x] `metrics/retrieval.py`: `relevance()` also accepts a `contains` key in an expected item (a substring of the hit's text). Hits carry their text into the matching only; `query_results` still stores no text. One added assertion in `tests/test_retrieval.py`, since this is the formula area the project does test.

### 7.2 Matrix runner

- [x] `experiments/matrix.yaml`: `models` (label to Ollama model), `strategies`, `shared` (the fixed settings above), and `documents` (all partitions, or a list of file names). Mounted into `dagster-code` like `eval/`.
- [x] `src/rag_lab/experiments/`: `matrix.py` expands the file into a list of `ExperimentConfig` (plain Python, no Dagster). `__main__.py` runs them: `docker compose exec -d dagster-code python -m rag_lab.experiments run experiments/matrix.yaml [--only <pattern>] [--dry-run]`.
- [x] Run order is model by model, so each model is loaded on the Mac once. For each experiment: `parsed_document`, `chunks`, `embeddings`, `qdrant_index` for every document, then `search_benchmark` once.
- [x] Resumable: an experiment that already has a finished benchmark for the current query-set version is skipped. A failed experiment is logged and the run moves on; a summary of failures is printed at the end.
- [x] `--dry-run` prints the 15 experiment names and config hashes without running anything.

### 7.3 Dashboard page

- [x] Migration `0003`: recreate `experiment_summary` with `model`, `strategy` and `dimension` columns read from `experiments.config`, so the dashboard and SQL can group by them.
- [x] `streamlit` added to the dependencies in `pyproject.toml` (its stated need); the shared image is rebuilt once (`docker compose up -d --build`). A new `ui` service in `docker-compose.yml`: the same image, `streamlit run /app/ui/app.py`, port `127.0.0.1:8501`, `./src` and `./ui` bind-mounted so edits reload, the same environment block as `dagster-code`, started after Postgres and Qdrant.
- [x] `ui/app.py` (Streamlit entry with two pages), `ui/dashboard.py`, and `ui/style.py` (the shared theme, below). The UI reads its addresses from the environment, like the search CLI.
- [x] Look and feel (applies to both pages, since "cool" is a requirement):
  - A dark theme set in `ui/.streamlit/config.toml`, with one accent colour, and a small block of custom CSS in `style.py`: a gradient page title, rounded cards, a quiet background.
  - Headline cards at the top of the dashboard (best model, best strategy, best nDCG@5, fastest search p50), each with the value large and the experiment name beneath it.
  - Charts in Altair (it comes with Streamlit; no plotting library is added), with one palette chosen for the three models and used on every chart, a single-hue scale for the heatmap grid, readable labels, and tooltips. The `dataviz` skill is used for the palette and chart rules when this is built.
  - Text and table hits in the query page are cards with a coloured badge for the modality and a similarity bar; the agreement marking below uses a second badge colour.
- [x] Dashboard contents, all read from `rag_metrics`:
  - A grid with strategy as rows and model as columns, filled with a metric you pick (nDCG@5, recall@5, MRR, search p50, embed chunks per second, and so on). This is the main view.
  - Bar charts by experiment for quality, search latency (p50 and p95) and ingestion throughput.
  - The raw `experiment_summary` table, sortable.
  - A per-query drill-down: for two chosen experiments, the first relevant rank of each query side by side, to see where one wins.
  - A query-set version selector, with a warning when the experiments shown were benchmarked on different versions.

### 7.4 Query page

- [x] `ui/query.py`: a query box, top-k, an optional modality filter, and a choice of experiments (all 15, or filter by model or strategy). Each experiment gets a column of ranked hits with rank, similarity, source file and page, and text. Hits whose page also appears in other experiments' results are marked, so agreement and disagreement are visible at a glance. Each hit also shows its `{source_file, page}` as a ready-to-paste line for `eval/queries.yaml`.
- [x] `search()` takes an optional precomputed query vector, so the page embeds the query once per distinct embedding setup (three times for the matrix, not fifteen).
- [x] Nothing is written to `search_log` from this page.

### 7.5 The comparison run

- [x] A slice first: one model and two strategies (for example `q3-0-6b` with `hybrid` and `fixed`), checked end to end through the dashboard and the query page.
- [x] Then the full 15 experiments. Read the dashboard and note what you find (best model, best strategy, whether the larger models are worth their latency and memory) in this section.

Done when: `--dry-run` lists 15 experiments, the full run leaves each with a finished benchmark on the same query-set version, the UI at `localhost:8501` shows a quality metric for every model and strategy pair in the dashboard grid, and the query page shows the same query's results from several experiments side by side. Checked once, on the real stack.

**Status: done (2026-10-02).** Checked on the real stack with `aiayn.pdf` and a 30-query set (20 text, 10 table):
- `--dry-run` lists the 15 experiments with their config hashes. The full run, started from the code container, finished all 15 in 11.7 minutes (a first slice of two experiments had been run and checked before). Every experiment has a finished benchmark on the same query-set version (`1ec485bcb7c0`).
- A coverage check confirmed that all 30 expected items are findable in the chunks of every one of the 15 experiments, so no experiment is penalised by where its chunker happened to cut.
- The UI is at `http://localhost:8501`. The dashboard grid shows nDCG@5 for every model and strategy pair; the query page shows one question across all 15 experiments, with one query embedding per model. Both pages were rendered in a real browser and checked for exceptions with Streamlit's app tester.

Results (30 queries on one paper, so read them as a guide, not a verdict):

| Model | Vector size | Mean nDCG@5 | Mean recall@5 | Search p50 | Embed tokens/s | Seconds per document |
|---|---|---|---|---|---|---|
| `qwen3-embedding:0.6b` | 1024 | 0.896 | 0.967 | 29 ms | 5436 | 36 |
| `qwen3-embedding:4b` | 2560 | 0.924 | 0.993 | 42 ms | 1870 | 41 |
| `qwen3-embedding:8b` | 4096 | 0.942 | 0.993 | 57 ms | 1199 | 45 |

| Strategy | Mean nDCG@5 | Mean MRR |
|---|---|---|
| `semantic` | 0.928 | 0.908 |
| `recursive` | 0.927 | 0.909 |
| `fixed` | 0.920 | 0.898 |
| `hierarchical` | 0.918 | 0.896 |
| `hybrid` | 0.910 | 0.896 |

- A larger model helps more than a different chunker: each step up in model size adds about 0.02 to 0.03 nDCG@5, at roughly 1.4 times the search latency and 0.3 to 0.5 times the embedding speed. The strategies span only 0.02, which is within what one or two queries can move (one query is 0.033 of recall@5), so no strategy can be declared best on this data.
- Best single experiments (nDCG@5 0.955): `q3-4b-hierarchical`, and `q3-8b` with `fixed`, `recursive` or `semantic`. `hierarchical` is the least stable: best at 4b, worst of the 8b row (0.893). It keeps formula placeholders as chunks (the `<!-- formula-not-decoded -->` chunk ranks first for the optimizer question at 0.6b).
- Recall@5 is 0.93 to 1.00 in every experiment, so the right chunk is nearly always in the top five. Most of the difference is in how high it ranks (MRR 0.855 to 0.940), not whether it is found.

Changes from the plan, and notes:
- Expected items use `{source_file, contains}` only, without `page`: Docling assigns a chunk's page differently depending on the chunker (Table 1 is on page 5 in the hybrid chunks, page 6 in the PDF), so a page match would have penalised some experiments unfairly. The 30 expected snippets were drafted from the parsed paper and checked against the chunks; they still need your review.
- The `hostenv.py` step was dropped: the UI runs in a container that already has the connection settings in its environment.
- Ollama is on this machine here (`OLLAMA_BASE_URL=http://host.docker.internal:11434`) with all three models pulled, so step 7.1's Mac check did not apply. Memory was not a problem for the 8b model.
- The runner skips ingestion when the experiment's Qdrant collection already holds the document (not when metrics rows exist), and skips an experiment whose config hash already has a finished benchmark on the current query set. This is needed because `q3-0-6b-hybrid` has the same settings, hence the same config hash, as the earlier `docs` experiment.
- That shared hash exposed a bug, now fixed: the experiments row kept the old config JSON (with the old name) when a second name claimed the hash, so a benchmark of `q3-0-6b-hybrid` silently ran against the `docs` collection. The upsert now updates the config with the name. The row was renamed to `q3-0-6b-hybrid`, the experiment was rerun, and the mislabelled benchmark run was deleted. The `docs` Qdrant collection and `data/artifacts/docs/` are now unused and can be deleted. The notebook examples use `q3-0-6b-hybrid` and `q3-0-6b-fixed`.
- `streamlit` was added to the shared image, so all Dagster services restarted once. After editing a module other than a page script (for example `ui/style.py`), restart the UI with `docker compose restart ui`.
- Migration `0003` recreates the `experiment_summary` view with `model`, `strategy` and `vector_dimension` columns.

Decided (2026-10-02): the document is `aiayn.pdf`; I draft the query set and you review it; the UI is one Streamlit app with two pages, run as a Docker service; the design should look polished (see 7.3).

## Phase 8 — Ingestion on LlamaIndex

Goal: do document ingestion (read, chunk, index) with LlamaIndex components instead of our own code, with the same outputs: the same four Dagster stages and on-disk intermediates, the same Qdrant collection layout and payload, the same metrics in `rag_metrics`, and the same search, benchmark and UI. Nothing downstream of ingestion changes.

The plan was written after reading the code and the current packages (`llama-index-core` 0.14.25, `llama-index-readers-docling` 0.5.0, `llama-index-node-parser-docling` 0.5.0, `llama-index-vector-stores-qdrant` 0.10.3). Where a step says "verify", the package source did not settle it.

### What LlamaIndex replaces, and what stays

| Stage | Today | With LlamaIndex |
|---|---|---|
| Read the PDF | `build_converter()` + `converter.convert()` in `parsing/parse.py` | `DoclingReader(doc_converter=build_converter(cfg), export_type=JSON)`. Docling stays the parser: LlamaIndex has no PDF reader with table structure of its own (its default `PDFReader` is plain pypdf and drops tables; LlamaParse is a paid cloud service). |
| `hybrid`, `hierarchical` | Docling's chunkers called directly (`chunking/native.py`) | `DoclingNodeParser(chunker=<the same chunker with the Markdown table serialiser>)` |
| `fixed` | own token windows (`fixed.py`) | `TokenTextSplitter` |
| `recursive` | own separator recursion (`recursive.py`) | `SentenceSplitter` (paragraph, then sentence, then word) |
| `semantic` | own (`semantic.py`) | `SemanticSplitterNodeParser`, then `SentenceSplitter` on nodes still over the limit |
| Embed | `OllamaEmbedder` | **unchanged**; seen by LlamaIndex through a small `BaseEmbedding` adapter that only `SemanticSplitterNodeParser` uses |
| Index | `QdrantStore.upsert` | `QdrantVectorStore.add` on a collection that `QdrantStore.ensure_collection` created |
| Segmenting, tables, page and heading bookkeeping | `chunking/segment.py`, `sectioned.py` | **unchanged**: LlamaIndex splitters work on plain text, so tables, headings, pages and bounding boxes still come from our Docling-aware segmenter |

### Decisions

| Topic | Decision | Reason |
|---|---|---|
| Replace or add | Add beside the current code first. `ChunkConfig.engine` (`llamaindex`, or `native`) picks the chunker. It was `native` by default at first; it is `llamaindex` since 2026-10-03 (see 8.5). Parse, embed and index move to LlamaIndex outright, because their output is the same. Retire the native chunkers only after the comparison run (8.4). | The chunkers are the only part whose results change. Keeping both lets the benchmark say whether the move costs quality, and the 15 existing experiments stay as the baseline. |
| `engine` and the hash | `config_hash()` leaves `chunk.engine` out when it is `native`, the same way it leaves out unused strategy settings. | Existing hashes and rows stay valid; a `llamaindex` experiment gets a different hash. One added assertion in the config hashing test. |
| `IngestionPipeline` | Not used. Each Dagster asset calls its LlamaIndex components itself. | A single pipeline run would merge the four stages into one and drop the per-stage files and metrics that the design rules require. Its docstore dedup is an idea (below). |
| Embedding | Keep our `OllamaEmbedder`. Do not add `llama-index-embeddings-ollama`. | The stock class does not report Ollama's `total_duration`, `load_duration` and `prompt_eval_count` (the embed metrics), does no Matryoshka truncation or re-normalisation, and joins the query instruction with a space instead of `Instruct: ...\nQuery: ...`. |
| Embedded text | The headings go in each Document's metadata as the one key visible to the embedder, with `metadata_template="{value}"`, `metadata_separator="\n"` and `text_template="{metadata_str}\n{content}"`. The chunk text that is stored and embedded is `node.get_content(MetadataMode.EMBED)`. `include_headings_in_text=false` puts the key in `excluded_embed_metadata_keys`. | LlamaIndex's splitters reserve room for the embed-visible metadata, which reproduces "headings use part of `max_tokens`". The text comes out as today (headings, newline, body). |
| Token counting | Pass the Qwen3 Hugging Face tokenizer (`encode(text, add_special_tokens=False)`) as the splitters' `tokenizer`. | `max_tokens` keeps meaning what the embedder sees. LlamaIndex's default tokenizer is tiktoken, which would count differently. |
| Page and bounding box | Found by searching for each node's body in its section text (from the previous chunk's start, so overlapping chunks work), then `Section.locate()`, as today. For the Docling chunkers, from the node metadata's `doc_items` (`label`, `prov`). | LlamaIndex splitters copy the section's metadata to every child, so per-chunk page needs the character offset. Its own `start_char_idx` is relative to the intermediate node after the semantic strategy's second pass, so one search is used for all strategies. |
| Qdrant payload | Node metadata carries the flat fields (`chunk_id`, `modality`, `strategy`, `page`, `headings`, `bbox`, `token_count`, `experiment`, `config_hash`, `source_file`, `ingested_at`) plus `text`, with `text` excluded from embed and LLM views. The Document id is our `doc_id`; node ids are `to_point_id(chunk_id)`. | `search()` reads `text` and the other fields straight from the payload, and the payload indexes and `delete_document` filter on `doc_id`. LlamaIndex writes `doc_id` from the source Document's id, so setting it keeps both working. |
| Dependencies | `llama-index-core`, `llama-index-readers-docling`, `llama-index-node-parser-docling`, `llama-index-vector-stores-qdrant`. | The stated need: one package per replaced stage. They need `numpy>=2` and `qdrant-client>=1.16`; check that `uv` resolves them next to Docling and the `sqlalchemy<2.1` pin (8.0). |

### 8.0 Groundwork

- [x] Add the four packages to `pyproject.toml`; rebuild the shared image once (`docker compose up -d --build`).
- [x] Set `NLTK_DATA=/root/.cache/nltk_data` on the code and UI services: LlamaIndex's sentence splitter downloads NLTK's `punkt` data on first use, and `/root/.cache` is the `model_cache` volume, so it survives rebuilds. The first run needs internet.
- [x] `config.py`: `ChunkConfig.engine`, the hash rule above, and the matching assertion in the hashing test.

### 8.1 Read (`parsed_document`)

- [x] `parsing/parse.py` calls `DoclingReader` with our converter (`do_ocr` off and every `ParseConfig` option still applied), with `id_func` returning the document's content-hash id so the Document id is our `doc_id`. The Docling JSON, Markdown and `.meta.json` files are written as before, from the Document's JSON text.
- [x] Loses: `status` `partial_success` and the `errors` list. The reader calls `convert()` with Docling's default of raising on error, so a failed parse fails the asset with Docling's message. Pages, tables, characters and the scanned-PDF check come from the loaded Docling document. `model_load_seconds` is still timed (`initialize_pipeline` runs first), so `parse_seconds` stays comparable with the existing rows.

### 8.2 Chunk (`chunks`)

- [x] `chunking/llamaindex.py`: one function, `chunk_llamaindex(ctx)`, which `chunk_document` calls when `engine = llamaindex` (no registry entry per strategy). It returns the same `Chunk` objects, so `write_chunks`, `summarise` and everything after `chunks` are untouched.
- [x] `hybrid`, `hierarchical`: wrap the saved Docling JSON in a Document and run `DoclingNodeParser` with the chunker built as in `native.py`. Node text is the chunk body without headings, so `Chunk.text` adds the headings (joined by newline) when `include_headings_in_text` is on, which is what Docling's `contextualize()` does. Modality (`table` if any `doc_item` is a table), page and bbox come from the node metadata; `table_handling` `skip` and the `row-wise` warning behave as today.
- [x] `fixed`, `recursive`, `semantic`: `segment()` yields sections and tables as now. Each section becomes a Document (headings metadata as above, page and bbox kept in excluded keys); tables skip the splitter and follow `table_handling` as today. The splitter is built once per document with `chunk_size = max_tokens`. Pass `chunk_overlap = 0` explicitly for `recursive` and `semantic` (`SentenceSplitter`'s default is 200).
- [x] `semantic`: `SemanticSplitterNodeParser` with our sentence pattern as its `sentence_splitter`, then `SentenceSplitter` for oversize nodes. The embedding adapter (`BaseEmbedding` over `OllamaEmbedder`, document mode) wraps `embed_cached`, so the on-disk sentence-group cache still applies, and it counts texts, cache hits and milliseconds for the chunk metrics.
- [x] The chunk metrics row and asset metadata are unchanged in shape; `details` gains `engine`.

What does not carry over, so the `llamaindex` results are comparable in quality, not chunk-for-chunk identical:

| Setting | With `engine = llamaindex` |
|---|---|
| `recursive.separators` | Not used; `SentenceSplitter` has its own order. A non-default value raises an error rather than being ignored. |
| `semantic.breakpoint_type` | Only `percentile` (as an integer). `stddev` and `absolute` raise an error. |
| `semantic.min_tokens` | Not applied (no merging of small chunks); a warning is logged. |
| Semantic threshold | Taken per section (LlamaIndex computes it per Document), not over the whole document. A section with one or two sentences never splits. |
| Semantic stats | `threshold` and `breakpoints` are not reported; sentence and embedding counts come from the adapter. |
| Heading budget | Reserved by LlamaIndex with a few tokens of format overhead; the cap at half of `max_tokens` is gone, so a very long heading raises an error instead of being clipped. |
| `fixed` | `TokenTextSplitter` breaks on word boundaries and merges words up to the limit, so windows are not exactly `max_tokens` wide. |

### 8.3 Embed and index

- [x] `embeddings`: no change.
- [x] `qdrant_index`: build one `TextNode` per chunk (id, text, metadata above, `embedding` from the `.npy` file) and write with `QdrantVectorStore.add`, on a store that takes `QdrantStore.client` and `batch_size=256`. `ensure_collection` and `delete_document` still run first. The store notices the existing collection's unnamed vector and uses it, so `query_points` in `search()` works as it does now (verify against a real collection before relying on it). Each point also gets a `_node_content` payload field with a copy of the node as JSON, so the chunk text is in the payload three times (`text`, the metadata copy inside `_node_content`, and `_node_content`'s own `text`); that does not matter at lab scale. `QdrantStore.upsert` had no other user and is gone.
- [x] Hybrid (dense plus BM25) search, the optional item in Phase 5, becomes cheaper through this store's `enable_hybrid`, but it is not part of this phase.

### 8.4 Comparison run

- [x] A slice first: `experiments/matrix-llamaindex.yaml`, a copy of `matrix.yaml` with `shared.chunk.engine: llamaindex` and the model labels prefixed `li-` (`li-q3-0-6b`), so the experiments are named `li-q3-0-6b-fixed` and so on. Run the 0.6b model only (`--only 'li-q3-0-6b-*'`), five experiments.
- [x] Then the other two models, 15 in all.
- [x] Checks, once: every experiment has a finished benchmark on the same query-set version; the coverage check from Phase 7 (all 30 expected snippets findable in each experiment's chunks); the Phase 7 dashboard and query page show the `li-` experiments next to the native ones with no change; the search CLI and `ask()` find them.
- [x] Record the per-strategy nDCG@5, recall@5 and MRR next to the native rows in this section, with the chunk counts and token statistics, and note what differs and why.

Done when: `parsed_document` through `qdrant_index` run on LlamaIndex for all five strategies and three models, the comparison is recorded here, and the LlamaIndex experiments score within what the Phase 7 results call noise (about 0.02 to 0.03 nDCG@5, one query is 0.033 of recall@5) of their native counterparts, or the gap is explained by an item in the table above. One real run, no new tests beyond the hashing assertion.

**Status: done (2026-10-03), except 8.5, which waits for your decision.** Checked on the real stack with `aiayn.pdf` and the 30-query set:
- **Image:** the four packages resolve next to Docling (`llama-index-core` 0.14.25, readers-docling 0.5.0, node-parser-docling 0.5.0, vector-stores-qdrant 0.10.3) and the image rebuilt once. `docker-compose.yml` sets `NLTK_DATA` on all Dagster services and the UI through the shared env block; LlamaIndex downloaded its sentence data into `/root/.cache/nltk_data` on first use.
- **Read:** the Docling JSON written through `DoclingReader` is byte-identical to the one the old converter call wrote (same MD5 for `aiayn.pdf`), and the parse metrics rows have the same shape. The `status` and `errors` fields are gone from them.
- **Chunk:** on the same parsed paper, `hybrid` (41 chunks) and `hierarchical` (88) give identical text, page, bounding box, modality and headings in both engines, and `recursive` (44) gives identical chunk texts. `fixed` shares 37 of 45 chunk texts (word-boundary windows, 249.7 tokens on average against 250.0). `semantic` gives 82 chunks against 55 (mean 135 tokens against 198), because small chunks are not merged and the threshold is per section.
- **Index:** a `li-` collection has an unnamed 1024-dimension cosine vector, the stored vectors equal the `.npy` rows, the payload holds every native field plus LlamaIndex's own (`_node_content`, `_node_type`, `ref_doc_id`, `document_id`), `has_document` and the search CLI work on it without a change, and the matrix runner and the dashboard and query pages (rendered with Streamlit's app tester) run with no exceptions. The Phase 7 coverage check passes: all 30 expected snippets are in the chunks of all 15 `li-` experiments.
- **Comparison:** the full `matrix-llamaindex.yaml` run finished 15 of 15 in 10.0 minutes (the 0.6b slice, 5 experiments, took 3.8). Every experiment has a finished benchmark on query-set version `1ec485bcb7c0`.

| Strategy | nDCG@5 native | nDCG@5 llamaindex | MRR native | MRR llamaindex |
|---|---|---|---|---|
| `fixed` | 0.920 | 0.924 | 0.898 | 0.903 |
| `hierarchical` | 0.918 | 0.914 | 0.896 | 0.890 |
| `hybrid` | 0.910 | 0.906 | 0.896 | 0.896 |
| `recursive` | 0.927 | 0.932 | 0.909 | 0.916 |
| `semantic` | 0.928 | 0.890 | 0.908 | 0.853 |
| all 15 | 0.921 | 0.913 | 0.901 | 0.892 |

Mean recall@5 is 0.985 for both engines; mean search p50 is 43 ms against 45 ms and mean seconds per document 41 against 43.

- Four of five strategies are within about 0.005 nDCG@5 of their native counterparts, which is inside the noise (below). The one real gap is `semantic`: 0.038 lower nDCG@5 and 0.055 lower MRR, from the 82 small chunks that follow from the two missing features in the 8.2 table (`min_tokens` merging and the document-wide threshold). It is largest at 0.6b (0.853 against 0.906) and smaller at 8b (0.919 against 0.955).
- **Noise between identical runs:** `hybrid` has identical chunks and an identical parse in both engines, yet `q3-0-6b-hybrid` and `li-q3-0-6b-hybrid` differ by one query (recall@5 0.933 against 0.900, nDCG@5 0.882 against 0.869). The stored embeddings of the same texts differ by up to 0.005 per component (0.027 for one `fixed` run) between two runs, which is Ollama's own numerical variation. So a gap of one query, and some of the gaps in the 4b and 8b rows (for example `q3-8b-hierarchical` 0.893 against 0.880 with identical chunks), is not the engine.

Changes from the plan, and notes:
- The page search above replaced `start_char_idx`; `chunk_llamaindex` is one function, not registry entries; the `parse_pdf` result lost `status` and `errors` as planned.
- `OllamaLabEmbedding` (in `embedding/llamaindex.py`) is used only by the semantic splitter, and routes through `embed_cached`, so a repeat semantic run reuses cached sentence vectors. Its chunk stats are `sentences`, `group_texts`, `embed_cache_hits`, `embedded` and `embed_ms`; `threshold` and `breakpoints` are not reported by LlamaIndex.
- `sectioned.py` gained `prefix_and_budget()` and a public `table_texts()` (moved out of `chunk_sections`), and `native.py` gained `build_chunker()` and `warn_row_wise()`, so both engines share the heading budget, the table handling and the Docling chunker setup. The native experiments were not re-run after this refactor; in the comparison script the refactored native `hybrid`, `hierarchical` and `recursive` gave the same chunks as LlamaIndex, and the stored `q3-0-6b-hybrid` chunks match them too.
- `experiments/matrix-llamaindex.yaml` is `matrix.yaml` with `shared.chunk.engine: llamaindex` and `li-` model labels, so no code change was needed in `matrix.py`.
- `CLAUDE.md` is updated for the engine, the dependencies and the layout.
- Dashboard fix: the grids and charts are keyed by model and strategy, so a `li-` experiment drew over its native twin. Migration `0004_experiment_summary_engine.sql` adds an `engine` column to `experiment_summary`, and the dashboard has a "Chunk engine" selector in the sidebar that filters the Quality, Speed and Ingestion tabs and the headline cards to one engine. The Queries tab and the Data tab still show every experiment, so native and `li-` can be compared there.
- The `li-` experiments appear in the query page and the Queries tab next to the native ones. Their Qdrant collections and `data/artifacts/li-*` can be deleted without touching the native ones.

### 8.5 Retire the native chunkers (only after you decide)

Decided (2026-10-03): the native chunkers stay, and `llamaindex` is the default engine. A config that does not set `engine` is now `llamaindex`; `experiments/matrix.yaml` sets `engine: native` so its experiments keep their hashes (otherwise they would share a hash with the `li-` ones). The deletion steps below are not done. The comparison above says `llamaindex` matches `native` for `hybrid`, `hierarchical`, `recursive` and `fixed`, and loses 0.038 nDCG@5 on `semantic`. Retiring `native` means accepting that, or restoring the missing semantic features on the LlamaIndex side.

- [ ] If the results hold: make `llamaindex` the default engine and delete `fixed.py`, `recursive.py`, `semantic.py` and the Docling calls in `native.py`. Keep `segment.py`, `sectioned.py`, `tokens.py` and `embedding/cache.py`, which the LlamaIndex path uses. `engine` can then go, with no migration needed (old hashes stay valid because `native` was never in the hash).
- [ ] If they do not: keep both. `engine` is a stage choice and stays a config value, in line with the lab rule.
- [ ] Update `CLAUDE.md` (status line, chunking facts, dependencies, layout) when this lands, not before.

Not included: the stock `OllamaEmbedding`, `IngestionPipeline` and its docstore, LlamaParse, LlamaIndex retrievers and query engines (search stays on `QdrantStore.query`), and any change to the metrics schema, Dagster partitions, benchmark or UI.

## Phase 9 — Upload, chunk preview and embed (Streamlit)

Goal: a new page in the Streamlit app where you upload a PDF, pick a chunking strategy and its parameters, press **Chunk** to see the whole document cut into chunks with visible separators, and press **Embed** to put the chunked document into Qdrant under an experiment name, so it can be searched on "Try a query".

Decided (2026-10-03): the new "tab" is a third page in the app's navigation, beside "Dashboard" and "Try a query".

### How it works

```
upload PDF -> parse once (Docling, default settings, cached by document id)
           -> choose engine, strategy, parameters -> Chunk -> preview with separators
           -> choose embedding model, experiment name -> Embed -> chunks, vectors and points,
              as if ingest_job had produced them
```

### Decisions

| Topic | Decision | Reason |
|---|---|---|
| Where the work runs | In the Streamlit process of the `ui` container, with the same plain-Python functions the Dagster assets use. No Dagster run is started. | Matches the rule that stages are plain Python and the UI has no Dagster client. The cost is that these runs are not listed in the Dagster UI; the metrics rows in `rag_metrics` are still written. |
| Where uploads go | `data/uploads/<doc_id>.pdf` (the content-hash id, so the same file twice is one document). Not `data/raw`. | The sensor would register everything in `data/raw` as a partition. Uploading to try a chunker must not add a document to the lab's document set. Moving a file to `data/raw` later gives it the same id. |
| Parsing | Once per uploaded document, with default `ParseConfig`, cached on disk under `data/artifacts/_uploads/<doc_id>/`. Changing chunk settings never re-parses. | Parsing takes 20 to 30 s on CPU; chunking takes under a second (semantic: seconds, with the sentence cache). The parse options are not on the page (see Ideas). |
| Chunk settings | The page builds a `ChunkConfig` and calls `chunk_document()`, the same call the `chunks` asset makes, so the preview is what `chunks` would produce. | One code path; no preview-only chunker to drift from the real one. |
| What the preview shows | Two views of the same chunks. **In the document** (the main one): the parsed document's text, read in order, with a cut mark where each chunk starts and chunks told apart by alternating backgrounds. Where two chunks share text (`fixed` with overlap), that text has its own highlight and a label naming both chunks. Text that is in no chunk (a table under `skip`, heading lines when headings are not embedded) is dimmed. **As embedded**: each chunk as a block (number, modality, page, tokens, headings, then the stored text with its headings prefix dimmed), because the embedded text is the chunk text with its headings added, which the document view cannot show. | You asked to see the original text with cut marks and overlaps. The second view is kept because it is exactly what will be embedded. |
| The "original" text and where chunks sit in it | The reference text is the document as the segmenter reads it (`chunking/segment.py`): heading lines where the heading path changes, section text with paragraphs separated by blank lines, tables as Markdown. Each `Chunk` gets a `span` (start and end character offsets into that text), recorded by the chunker itself, not found afterwards by searching. | A trial of the search approach on `aiayn.pdf` found only 23 of 41 `hybrid` chunks and 78 of 88 `hierarchical` chunks (Docling's chunk text differs from the paragraph text: formula placeholders, list and caption serialisation), and missed the table chunks of the other strategies. Offsets from the chunker are exact where the chunker knows them. |
| How spans are known | `fixed`, `recursive`, `semantic`, both engines: the chunk's offset inside its section plus the section's offset (the native engine already has the span; the LlamaIndex engine finds the body in the section text, which is exact because its nodes are substrings). `hybrid`, `hierarchical`: the chunk's Docling `doc_items` are matched by `self_ref` to the paragraphs and tables the segmenter recorded; the span runs from the first to the last of them, narrowed to the matching text when one long paragraph was split across several chunks. A table is one region: its chunks (one, or several when split by rows) all point at the whole table, and the page lists them by number. A chunk with no placeable item (a formula Docling did not decode) gets an empty span where the previous chunk ends, shown as a cut mark with a "no source text" note. | Keeps the position logic in the chunkers, next to the code that already decides the cuts. The one remaining approximation (tables, split paragraphs under `hybrid`) is labelled on the page. |
| Embed target | Writes `parse`, `chunk`, `embed` and `index` files under `data/artifacts/<experiment>/`, the experiment row, the four stage metric rows, and points in the collection named `<experiment>`. | An embedded upload is then indistinguishable from an ingested document: the "Try a query" page lists it, and a later Dagster run on that experiment finds its inputs. |
| Experiment name rules | The name must match the existing pattern. A name that exists is refused unless it has the same config hash and no finished benchmark. A config hash that already exists under another name is refused, with that name in the message. | The `experiments` row is keyed by hash and keeps the last name, so a second name for the same settings would rename a matrix experiment in the dashboard (the quirk in `CLAUDE.md`). A benchmarked experiment is a fixed reference point; adding a document would change what its numbers mean. |
| Embedding models | The Qwen3 embedding models that Ollama reports (`/api/tags`), plus an optional vector size. | The tokenizer used for chunk sizes is the Qwen3 one; other model families are an idea, not part of this phase. |
| Settings that do not apply | Controls are shown only when the chosen engine and strategy use them: `overlap` for `fixed`; `recursive.separators` and `semantic.min_tokens` for `native`; `stddev`/`absolute` breakpoints for `native`; `merge_peers` for `hybrid`. | The `llamaindex` engine raises an error for several settings. The page should not offer a combination that fails. |
| Dashboard side effect | The Ingestion tab shows only experiments with a finished benchmark on the selected query set, like the other tabs. | An uploaded experiment has no benchmark but the same model, strategy and engine as a matrix experiment; in the Ingestion charts it would draw over it, the same bug as before the engine filter. |

### 9.0 Groundwork

- [x] `docker-compose.yml`, `ui` service: mount `./data:/app/data` (uploads, artifacts, the sentence-embedding cache) and `model_cache:/root/.cache` (Docling models and the Qwen3 tokenizer; without it they would download again into the container). Recreate the `ui` container once.
- [x] `src/rag_lab/ingest.py`: move the bodies of the `embeddings` and `qdrant_index` assets, unchanged, into plain functions (`embed_chunks`, `index_chunks`), and the metric-row and experiment-row writing of `parsed_document` and `chunks` into `record_parse` and `record_chunk`. The assets keep the Dagster parts (partition key to document id, `Failure`, `MaterializeResult`, `dagster_run_id`). Missing inputs raise a plain `IngestError`, which the assets turn into `Failure`. No behaviour change.
- [x] `config.py` or a small helper: the name and hash checks above as one function, `check_experiment_name(store, name, config_hash)`.
- [x] Spans (so the preview can show the original text): `Chunk` gets an optional `span` field (a pair of offsets; old `.chunks.jsonl` files without it still load). `segment()` gives each block its offset in the reference text and each paragraph the `self_ref` of its Docling item, and `reference_text(blocks)` returns the text. Then the four chunk code paths record spans as described in "How spans are known": `sectioned.chunk_sections` (native `fixed`, `recursive`, `semantic`), `native._convert` (`hybrid`, `hierarchical`), and `llamaindex._section_chunks` and `_docling_chunks`. The Qdrant payload and the metrics are unchanged; spans are only in the chunks file.

### 9.1 Upload and parse

- [x] `ui/upload.py` (the page) and a third entry in `ui/app.py`'s navigation ("Upload", `:material/upload_file:`). `st.file_uploader` accepts PDF only. The upload is hashed, saved, and parsed with `parse_pdf` inside `st.status`; a document already parsed is not parsed again. The scanned-PDF warning (the existing characters-per-page check) is shown on the page. A parsed document stays in `st.session_state`, so a script rerun does not parse again.
- [x] Shows pages, tables, and parse seconds for the document, as small cards.

### 9.2 Chunk settings and Chunk

- [x] A form: embedding model (it fixes the tokenizer and, for `semantic`, the sentence embeddings), engine, strategy, `max_tokens`, `table_handling`, `include_headings_in_text`, and the strategy's own settings per the "Settings that do not apply" row. Values are the `ChunkConfig` defaults; the form builds a `ChunkConfig`, so a typo or a bad combination fails in the same validation the assets use, shown with `st.error`.
- [x] **Chunk** runs `chunk_document` (`semantic` needs Ollama, so a failed call is shown, not raised). The result and its config hash are kept in `st.session_state`; changing a setting afterwards marks the preview as out of date until Chunk is pressed again, so the Embed button never embeds chunks that differ from the settings on screen.
- [x] Above the preview: chunk count, count by modality, token minimum, mean and maximum, and the engine's warnings (for example "overlap is only used by fixed").

### 9.3 Preview

- [x] **In the document** (default view). The reference text is cut at every chunk start and end; each stretch is covered by no chunk, one chunk, or two or more. One chunk: alternating background colours, with a cut mark at its start (`#12 · 340 tokens · p. 4`). Two or more (overlap): a separate highlight colour and a label with the chunk numbers. No chunk: dimmed. Table regions get a header naming their chunk numbers. Text is HTML-escaped and kept as `white-space: pre-wrap`, built like the hit cards on the "Try a query" page and using the page's palette.
- [x] A short key above the document (the colours and what a cut mark is), and a count of chunks whose span is approximate (a table, or a paragraph split by `hybrid`).
- [x] **As embedded**: one block per chunk (header bar, then the stored text with its headings prefix dimmed). Table chunks show their Markdown source in a monospace block.
- [x] Both views are paged, 100 chunks per page with a selector above them; the document view shows the reference text from the first chunk of the page to the first chunk of the next page. Overlap that crosses a page boundary is highlighted on both pages. Nothing else is limited.

### 9.4 Embed

- [x] A section under the preview: experiment name (default `up-<file name>-<strategy>`, lower-cased and cleaned to the name pattern), and the check above. **Embed** is enabled only when the preview is current and the name passes.
- [x] **Embed** runs inside `st.status`: copy the cached parse files and write the chunks file under `data/artifacts/<experiment>/`, `record_parse` and `record_chunk` (from the saved parse metadata and the chunk run), `embed_chunks`, `index_chunks`. On success it shows chunks embedded, tokens per second, points in the collection, and the experiment name to pick on "Try a query"; on failure it shows which step failed and leaves earlier files in place (a rerun overwrites them, as `qdrant_index` already deletes the document's points first).
- [x] Not included here: a benchmark button. An uploaded document has no entries in `eval/queries.yaml`, so quality metrics would be empty; `search_benchmark` from Dagster still works on the experiment for latency.

### 9.5 Dashboard

- [x] `ui/dashboard.py`: the Ingestion tab uses only the experiments that have a finished benchmark on the selected query set (the names that `data.benchmark_metrics()` returns), as the other tabs already do. The Data tab keeps showing every experiment.

**Status: done (2026-10-03).** Checked on the real stack with `aiayn.pdf`: the page was driven end to end with Streamlit's app tester (it supports uploads), and the extracted asset code was run through Dagster's CLI.
- **Spans** (every chunk's span against the reference text, all five strategies in both engines):
  - `fixed`, `recursive`, `semantic`: the text inside every non-table span equals the chunk's body, ignoring whitespace. `fixed` 40 exact, `recursive` 39, `semantic` 50 (native) and 77 (llamaindex); in each the other 5 are table pieces (tables 2 to 4 are split by rows), marked approximate. No wrong span. `fixed` with overlap 64 has 6 overlapping neighbour pairs, and every span start is non-decreasing.
  - `hybrid` (41 chunks): 25 equal, 14 contained, 2 approximate (the split tables), 0 empty, 0 wrong. `hierarchical` (88): 75 equal, 7 contained, 6 with no source text (undecoded formulas), 0 wrong. Both engines give the same numbers. "Contained" is a weaker test than "equal": for text chunks it checks the first and last six words are inside the span, for table chunks only that the span is a table region of at least half the chunk's length; the chunk text differs from the paragraph text in list markers (`[1]`), formula placeholders and the table Markdown's formatting.
  - The native engine's chunks for all five strategies are identical to the stored `q3-0-6b-*` ones (text, page, bounding box, modality, headings, ids, token counts) after the span changes.
- **The page:** uploading `aiayn.pdf` parsed it in 33 s and showed 15 pages, 5 tables. `fixed` with overlap 64 gave 45 chunks and a document view with 48 marks (40 chunk starts, 2 approximate table regions, 6 overlap labels); changing a setting marked the preview out of date and disabled Embed; `hybrid` gave 45 marks including 1 approximate region; a malformed name, and the name of an existing experiment, were refused. Embedding `hybrid` with max tokens 480 as `p9-test` gave 41 chunks and 41 points, the four metric rows (parse, chunk with `engine: llamaindex`, embed, index), the `parse`, `chunk` and `embed` files under `data/artifacts/p9-test/`, and the experiment showed up on "Try a query" (the CLI search returned the multi-head attention section first, similarity 0.733).
- **Dagster:** `parsed_document` through `qdrant_index` ran for a new experiment through `dagster asset materialize` with the code now in `ingest.py` (44 points, four metric rows).
- **Dashboard:** with the two test experiments in the database, the filter the Ingestion tab now uses (benchmarked names only) leaves the 15 experiments of an engine and none of the uploads (computed with the same expression, not read off the rendered charts), and the dashboard and query pages render with no exceptions. The test experiments, their collections, files and metric rows were removed afterwards.

Changes from the plan, and notes:
- **Settings that equal a benchmarked experiment cannot be embedded.** The config hash does not include the document, so the page's defaults (and any settings that match a matrix experiment) have the same hash as, for example, `li-q3-0-6b-hybrid`, and the name rule refuses them: the message tells you to change a setting, for example max tokens. That follows the plan's rule, but it means the first upload with default settings cannot be embedded as is. Lifting it needs the `experiments` table to be keyed by name instead of hash (or the hash to include the document); neither was done in this phase. Phase 10 plans a fix that keeps the schema.
- **Not seen in a browser.** The document and chunk views were checked by their HTML (mark counts, labels, no exceptions), not by looking at them, so the colours and layout in `style.py` are unverified visually.
- `Chunk` has two new fields, `span` and `span_approx`; the span of a one-chunk table is exact, and of a split table approximate. `sectioned.table_chunks()` builds table chunks for both engines (it replaces two copies), `chunking/spans.py` places Docling's chunks, and `segment.reference_text()` defines the text.
- The extracted functions in `rag_lab/ingest.py` are `register_experiment`, `record_parse`, `record_chunk`, `embed_chunks`, `index_chunks` and `check_experiment_name`; the assets are wrappers around them. `parsed_document` now registers the experiment before parsing and the parse row after, as before.
- The name check also refuses an existing name only when its settings differ or it has been benchmarked, as planned; the message for a hash that already belongs to a benchmarked experiment says to change a setting.
- The `ui` service mounts `./data` and the `model_cache` volume. Uploads are saved as `data/uploads/<doc_id>/<file name>` and parsed into `data/artifacts/_uploads/<doc_id>/`.

Not included: parse options on the page, choosing an existing document from `data/raw`, editing chunks by hand, a benchmark from the page, and deleting an uploaded experiment (delete the Qdrant collection and `data/artifacts/<name>` by hand).

Done when: after `docker compose up -d`, a PDF uploaded on the new page is parsed, chunked with two different strategies (the document view changes and shows the cut marks; with `fixed` and an overlap the shared text is highlighted), embedded under a new name, and the same question on "Try a query" returns chunks from it; the same name is refused a second time with other settings; the dashboard grids and charts are unchanged. One real run through the browser or Streamlit's app tester, one `ingest_job` run to confirm the extracted asset code still works, and one throwaway script over `aiayn.pdf` for all five strategies in both engines that checks every span: for `fixed`, `recursive` and `semantic`, the reference text inside the span equals the chunk's body (ignoring whitespace); for `hybrid` and `hierarchical`, it contains it, except chunks with no source text; the numbers are recorded here. Spans only drive a display, so no test file. No new tests: nothing in this phase is retrieval-metric, truncation or hashing logic.

## Phase 10 — Name your own experiment, or add to an existing one (Upload page)

Goal: on the Upload page's Embed step, choose between two things. **A new experiment** under any unused name you type, even when its settings equal another experiment's. **An existing experiment** whose settings equal the ones on the page, to add the document to it. This replaces the Phase 9 rule that refused a name or settings that belonged to another experiment.

### Why it is refused today, and the change

The `experiments` row is keyed by the config hash, and the hash covers the settings only. Two experiments with the same settings are therefore one row, and the metrics tables (`ingestion_stage_metrics`, `benchmark_runs`, `search_log`) are tied to that hash too. A second name would rename the first experiment and mix the numbers of two collections. The fix keeps the schema and gives an experiment its own identity when it needs one:

- `ExperimentConfig` gets an optional `tag`. When it is set it is part of `config_hash()`; when it is empty it is left out, as `engine` `native` was, so every existing hash stays the same.
- An experiment created on the Upload page gets `tag = its name`, so its hash is its own. Matrix experiments have no tag and keep their hashes.
- A second hash, `settings_hash()`, ignores the name and the tag. "The settings are the same" means the same `settings_hash()`.

### Decisions

| Topic | Decision | Reason |
|---|---|---|
| New experiment | Any name that matches the name pattern and is not used: no `experiments` row has it and no Qdrant collection has it. The name field starts with a suggestion you can overwrite. | A name only has to be unique. Refusing an existing collection (for example the unused `docs`) avoids writing into something that was not made by the page. |
| Existing experiment | A list of the experiments whose `settings_hash()` equals the page's, shown with their document count, point count and whether they have been benchmarked. Hidden when none match. The document goes into that experiment's collection under its name and hash. | Only experiments with the same settings are offered, so the vectors and chunks are comparable. |
| A benchmarked experiment | Allowed, behind a checkbox you must tick, with the consequences stated next to it: its benchmark numbers were measured without this document and are not redone; its collection and its ingestion numbers in the dashboard (pages, chunks, seconds per document) now cover two documents. | You asked for it. The earlier refusal protected reference results; now it is a choice you confirm. |
| Hash of an existing row | The page takes the stored `config_hash` of the selected experiment as it is, and checks that the page's settings produce it (plus its tag). It never rebuilds the hash from the stored config JSON. | Rows made before `engine` existed have no `engine` key, and a rebuilt config would default to `llamaindex` and give a different hash. |
| Dagster re-runs of a tagged experiment | `ExperimentResource` gets the same optional `tag`, so a run config for an upload experiment can include it; the search CLI and the benchmark need nothing, because they read the stored settings by name. | Without the tag in the run config an asset would compute the untagged hash and write its rows under the wrong experiment. This is stated in `CLAUDE.md`. |

### 10.0 Config

- [x] `config.py`: `ExperimentConfig.tag`, `config_hash()` including it only when set, and `settings_hash()` ignoring name and tag (one shared helper). `ExperimentResource.tag`, passed to `config()`. One added test: an empty tag leaves the hash as it was, a tag changes it, and `settings_hash()` ignores it. This is config hashing, which the project tests.

### 10.1 Store and ingest functions

- [x] `MetricsStore.list_experiments()`: name, config hash, config, and per experiment the number of documents (distinct `doc_id` in the stage metrics), whether a benchmark has finished, and the newest creation time.
- [x] `ingest.py`: `check_new_experiment_name(store, qdrant, name)` replaces `check_experiment_name`. `experiments_with_settings(store, settings)` returns the rows to offer: an untagged row matches when its stored hash equals the page's `settings_hash()`; a tagged row (made by this page, so its config has `engine`) matches when the hash recomputed without its tag does. The Qdrant point count of each is read for the display only.

### 10.2 The Embed step on the page

- [x] A radio, "Embed into": "A new experiment" (default) or "An existing experiment with the same settings" (disabled when the list is empty). New: the name field and the checks above; the experiment is created with `tag = name`. Existing: a selector over the list, the benchmark notice and checkbox when it applies, and no name field.
- [x] The embed steps are the ones from Phase 9. In the existing case the config is the selected experiment's name and tag with the page's settings, and the step stops with an error if its hash is not the stored one. Parse and chunk files and the stage rows for this document go under that experiment; its collection gets the new points; adding the same document again overwrites its points, as `index_chunks` already does.
- [x] The success message names the experiment and the collection's new point count.

### 10.3 Dashboard

- [x] An experiment that has been benchmarked and shares model, strategy and engine with another one would still draw over it in the grids. A caption above the grid lists such pairs by name when it finds them. No change to the charts.

### 10.4 Docs

- [x] `CLAUDE.md`: replace the line that says settings equal to a matrix experiment cannot be embedded, and the quirk about two names sharing a hash (it now applies only to untagged experiments); note `tag` and `settings_hash()`.

Done when, checked once on the real stack through Streamlit's app tester: the default settings on `aiayn.pdf` (equal to `li-q3-0-6b-hybrid`) can be embedded as a new experiment with a name you type, and afterwards both experiments exist with their own collections and metric rows and the matrix experiment's numbers are unchanged; a second, small PDF (generated for the check) is added to that new experiment through "An existing experiment" and its collection holds both documents; the benchmarked matrix experiment appears in the list with its notice and checkbox and Embed stays disabled until the box is ticked (without embedding into it); a name that is taken is refused; the throwaway experiments are removed afterwards. No new tests beyond the hash test.

**Status: done (2026-10-03).** Checked once on the real stack through Streamlit's app tester, with `aiayn.pdf` and a one-page PDF written by hand for the check:
- **Hashes:** `matrix.yaml`'s experiments keep their hashes (`q3-0-6b-hybrid` is still `5bd37b6548ff04be`); the new hash test passes (7 tests).
- **New experiment with the same settings as a matrix one:** `aiayn.pdf` with the default settings (equal to `li-q3-0-6b-hybrid`) was embedded as `p10-new`. The taken name `li-q3-0-6b-hybrid` was refused; `p10-new` was accepted. `p10-new` got hash `afade546c2fa214a` against `41d2696cdcf4b66a` for the matrix experiment, its own collection (41 points) and its own metric rows, and `li-q3-0-6b-hybrid` stayed at 41 points and 4 stage rows.
- **Adding to an existing experiment:** the second document (a different PDF, same settings) was offered the list `li-q3-0-6b-hybrid · 1 document(s) · 41 points · benchmarked` and `p10-new · 1 document(s) · 41 points`. Choosing the benchmarked one showed the notice and kept Embed disabled until the box was ticked (not embedded). Embedding into `p10-new` took its collection to 42 points; `experiment_summary` showed 2 documents and 42 chunks for it, and searching it found the lantern paragraph (similarity 0.540) and the attention section (0.733) from the same collection.
- **Dashboard:** after `search_benchmark` was run on `p10-new`, the `llamaindex` view showed the warning that it and `li-q3-0-6b-hybrid` share model, strategy and engine; the `native` view showed none.
- All of this was removed afterwards (experiment row, metric and benchmark rows, collection, artifacts and uploads).

Changes from the plan, and notes:
- The overlap notice on the dashboard is a warning box above the tabs, not a caption.
- In "An existing experiment" the experiment's row is left as it is; only files, stage rows and points are added.
- The selector shows each experiment's document count (distinct documents in its stage metrics) and point count; an experiment whose collection does not exist yet reads "no collection yet".
- Experiments from before this phase are matched through their stored hash, not a rebuilt config, so native experiments (no `engine` key in their stored config) are offered correctly.
- The Phase 9 limitation (settings equal to a benchmarked experiment cannot be embedded) is gone.

Not included: renaming or deleting experiments, keying the `experiments` table by name, and moving a document between experiments.

## Phase 11 — From a benchmark lab to a RAG platform

Goal: the app's main use becomes uploading documents and searching them. The comparison of models and chunkers moves to one dedicated page that takes **one uploaded document**, runs every chosen embedding model × chunking strategy on it, and produces a report you can reopen later and download as a PDF. The old dashboard and its machinery are removed.

Parts of Phases 6, 6a, 7 and 8 describe what is removed here. Those sections stay as the history of how the lab was built; this phase and `CLAUDE.md` describe what is current.

### Decisions (taken 2026-10-03)

| Topic | Decision | Reason |
|---|---|---|
| What the report measures | Always: speed and behaviour. Optionally: retrieval quality, when you give test queries. | An uploaded document has no known correct answers. Speed and behaviour need none; quality needs a snippet per query that must appear in the right chunk, as the old query set did. |
| Clean-out | Everything goes: the code, the benchmark tables and view, and the 30 matrix experiments with their collections, files and rows. | You chose a clean slate. |
| The report's experiments | Kept and queryable, named `bench-<document>-<run id>-<model>-<strategy>`. Each run has a button that deletes its experiments. | They then show up on "Try a query". A full run is 15 small collections. |
| Run mode | A background thread in the UI process. Progress and results are saved in Postgres, so closing the tab loses nothing and old reports can be reopened. One run at a time. | A full run takes about ten minutes. |
| Navigation | Upload (the default page), Try a query, Benchmark. | Upload and search are the platform; Benchmark is a tool. |
| Try a query | Pick one experiment by name, set top k, search. No model or strategy choices. The text/table content filter stays. | As asked; the content filter was not mentioned, so it is kept and can be dropped. |
| Report format | The report is built once as data (settings, tables, bar series, notes) and drawn twice: on screen with Streamlit and the existing Altair charts, and as a PDF with `reportlab`. | One new dependency, no browser or image rendering. The stated need is the PDF download. |

### What the benchmark does

For each chosen model (in turn, so each model loads once) and strategy, it runs the same steps the Upload page's Embed button runs, on one parsed document:

1. chunk, embed and index into a new experiment (tagged with its name, so its hash is its own), recording the usual stage rows;
2. run the **probe queries** against it: about 20 sentences sampled evenly from the document (8 to 40 words, no table or markup lines), the same for every experiment, searched once to warm up and then three times for timing. They give search latency and similarity scores, never quality, and the report says they are sentences from the document, so their scores show how scores are spread, not how well the right chunk is found;
3. if test queries were given, search them once at the chosen top k and score them with the existing `metrics/retrieval.py` (hit rate, recall, precision, MRR, MAP, nDCG at 1, 3, 5, 10 up to top k), judging a hit by whether its text contains the query's snippet.

| Group | Per experiment |
|---|---|
| Chunking | chunk count, text and table counts, token minimum, mean and maximum, chunking seconds, warnings, and for `semantic` the sentence and embedding counts |
| Embedding | seconds, chunks per second, tokens per second, model load time, vector size |
| Index | seconds and points |
| Search (probe queries) | embed, Qdrant and total time (p50, p95, mean), queries per second, cold first query, top-1 similarity (mean, min, max), gap between rank 1 and rank k |
| Quality (test queries only) | the metrics above, the first relevant rank of each query, and **snippet coverage**: how many snippets appear whole in some chunk of this experiment (a snippet cut by a chunk boundary caps what any model can score, as the Phase 7 coverage check found) |

Similarity scores are never compared across models, as before: the report ranks models on quality and speed only.

### 11.0 Clear out

- [x] Remove code and files: `ui/dashboard.py`; `query_grid` in `ui/charts.py` and the SQL functions in `ui/data.py` (the chart helpers and the model and strategy label helpers stay, the new page uses them); `src/rag_lab/benchmark/`; `src/rag_lab/assets/benchmark.py` (`search_benchmark`, `experiment_summary`) and its entries in `definitions.py`; `src/rag_lab/experiments/` and `experiments/` (both matrices); `eval/queries.yaml`; `src/rag_lab/metrics/summary.py`; `summary()` in `search/quick.py` and its notebook cells (the notebook's examples take an experiment name you set); `BenchmarkConfig`; the `./eval` and `./experiments` mounts in `docker-compose.yml`. Kept for the new benchmark: `metrics/retrieval.py`, `stats.py`, `timing.py`, `tests/test_retrieval.py`, `search()`.
- [x] Migration `0005_drop_benchmark.sql`: drop `experiment_summary`, `benchmark_metrics`, `query_results`, `benchmark_runs` and `eval_queries`. `experiments`, `ingestion_stage_metrics` and `search_log` stay.
- [x] Remove the data, once, by hand and not as code: the 15 `q3-*` and 15 `li-*` Qdrant collections and the unused `docs` one; `data/artifacts/q3-*`, `li-*` and `docs`; the rows in `experiments`, `ingestion_stage_metrics` and `search_log` for those experiments; `data/matrix*.log`. `data/artifacts/_cache` (the sentence-embedding cache), `data/raw` and the document files stay. Before this, a few of the 30 queries from `eval/queries.yaml` are copied to the scratch folder to test the new page.
- [x] Check once: the stack starts, Dagster loads its definitions (parse to index and `ingest_job` remain), the tests pass, and the app starts with only the pages that still exist.

### 11.1 Try a query

- [x] `ui/query.py` is rewritten: a selector over the experiments that have a Qdrant collection (each shown as `name · model · strategy · documents · points`), a top-k slider (default 5), the content filter, and a Search button. Results are one list of hit cards (rank, similarity bar, modality, page, headings, snippet, full text) with the embed and search times; the cross-experiment columns, agreement badges and expected-entry lines go. The query is embedded with the selected experiment's own settings through `search()`. Nothing is written to `search_log`. With no experiment the page says to upload a document first.

### 11.2 Shared pieces

- [x] `ingest.py`: `ensure_parsed(path, doc_id)` (the cached Docling parse that the Upload page does today) and `ingest_document(...)`, the Embed button's steps (copy the parse files, write the chunks, register the experiment, record parse and chunk, embed, index). The Upload page calls both, so it keeps its behaviour, and the benchmark runner calls them too.

### 11.3 Benchmark storage

- [x] Migration `0006_benchmark_reports.sql`:
  - `benchmark_reports`: `report_id`, `document_id`, `document_name`, `status` (`running`, `done`, `failed`, `stopped`, `interrupted`), `settings` (models, strategies, engine, shared chunk settings, top k, repeats, probe count), `test_queries` (null or a list of query and snippet), `probes`, `document_info` (pages, tables, parse seconds, characters per page), `total_experiments`, `stop_requested`, `error`, `created_at`, `updated_at`, `finished_at`.
  - `benchmark_results`: `report_id` (cascading delete), `experiment`, `model`, `strategy`, `status` (`done` or `failed`), `metrics` (jsonb, the table above), `per_query` (jsonb, null without test queries), `error`, `created_at`; primary key `(report_id, experiment)`.
  Results are read whole into one report, so one jsonb per experiment replaces the old one-row-per-metric layout.
- [x] `MetricsStore` methods for them: create a report, save a result, update status and `updated_at`, request a stop, list and load reports.

### 11.4 Benchmark runner

- [x] `src/rag_lab/benchmark/` again, plain Python, no Dagster or Streamlit: `probe_queries(reference_text, n)`; `snippet_coverage(chunks, snippets)`; `run_report(report_id, ...)` with the steps above. It saves each experiment's result as soon as it is done, updates `updated_at` at every step, checks `stop_requested` between experiments, records a failed experiment with its error and goes on, and ends with `done`, `stopped` or `failed`.
- [x] Reuses `search()`, `latency_summary`, `timed` and `metrics/retrieval.py`. Parsing happens once (`ensure_parsed`), not per experiment.

### 11.5 The Benchmark page

- [x] `ui/benchmark.py`, third in the navigation. A document: upload a PDF, or use the one already uploaded on the Upload page. Settings: models (the Qwen3 models Ollama has, all by default), strategies (all five), engine (`llamaindex` by default), `max_tokens`, overlap (for `fixed`), table handling, headings, top k and probe count. Test queries in `st.data_editor` (a query column and a "text that must be in the right chunk" column; empty rows ignored), with a warning for any snippet that is not in the document at all (the silent-zero trap from Phase 7.1). The number of experiments is shown before the Start button.
- [x] **Start** begins the run in a background thread kept in a module-level registry (so a rerun of the script does not lose it) and refuses to start when one is running. While it runs the page refreshes itself every few seconds with `st.fragment`: experiment `n` of `N`, the current step, a table of the finished ones, and a **Stop after this experiment** button. A report still `running` whose `updated_at` is older than 15 minutes is shown as `interrupted` (the UI container restarted).
- [x] A list of previous reports (document, date, status, experiment count) to reopen. Each report has **Delete this run's experiments** (collections, `data/artifacts/<name>`, experiments and stage rows); the report itself stays.

### 11.6 The report and the PDF

- [x] `rag_lab/benchmark/report.py` builds the report as plain data from the saved rows: the document and settings, a summary table (one row per model and strategy: chunks, mean tokens, embedding tokens per second, search p50, and with test queries recall@5, MRR and nDCG@5 plus snippet coverage), series for the charts, the per-query table, failed experiments, and the notes ("scores are not comparable across models", "probe queries are sentences from the document", the snippet-coverage caveat).
- [x] On screen: the summary table and the existing grouped-bar and heatmap charts (`ui/charts.py`) for speed, chunk counts and, with test queries, quality; the per-query table; the notes.
- [x] `rag_lab/benchmark/pdf.py` draws the same data with `reportlab` (a title block, the settings, the summary table with the best value of each column marked, bar charts from `reportlab.graphics`, the per-query table, the notes) and returns bytes; `st.download_button` names the file `benchmark-<document>-<date>.pdf`. `reportlab` is added to `pyproject.toml` and the image rebuilt once. Its standard fonts only cover Latin-1, so other characters in a document name are replaced with `?` in the PDF.

### 11.7 Navigation and docs

- [x] `ui/app.py`: Upload (default), Try a query, Benchmark. The Upload page's heading now describes the platform: upload, chunk, embed, then search.
- [x] `CLAUDE.md` is rewritten for what the project now is (a RAG platform with a benchmark tool), without the matrix, query-set, summary-view and `li-` comparison facts; this section of `PLAN.md` gets the status and the numbers.

Done when, checked once on the real stack (Streamlit's app tester for the pages, Docker for the rest): the clean-out check above passes; "Try a query" lists an experiment made on the Upload page and searches only it; a benchmark of `aiayn.pdf` with the 0.6b model and all five strategies, five test queries from the old set and the default probes runs in the background, keeps running when the page object is discarded and recreated, and ends as `done` with a report that shows the speed and quality sections; the PDF downloads, is rendered to images with `pypdfium2` and looked at; stopping a run after one experiment ends it as `stopped`; deleting the run's experiments removes their collections; and the whole run's experiments and rows are cleaned up afterwards. No new test files: nothing here is retrieval-metric, truncation or hashing logic, and the metric formulas keep their existing test.

**Status: done (2026-10-03).** Checked on the real stack with `aiayn.pdf`; the pages were driven with Streamlit's app tester, and the runs, the PDF and the clean-out were checked against Postgres, Qdrant and the rendered PDF.
- **Clean-out:** the code and files in 11.0 are gone, migrations `0005` and `0006` are applied, and 31 Qdrant collections (the 15 `q3-*`, the 15 `li-*` and `docs`), their artifact folders, 30 `experiments` rows (their stage rows went with them) and the matrix logs were removed. One older test experiment, `full-ingest-test` (a collection and four stage rows from an early phase, not one of the 30), was left. The stack starts, Dagster loads its definitions (the four assets, `ingest_job` and the sensor), the 7 tests pass, and the app has the three pages.
- **Try a query:** it lists the experiments that have a collection with `name · model · strategy · documents · points`; searching a benchmark experiment returned 5 chunks (embed 208 ms, search 4 ms).
- **Benchmark runs:** (1) `aiayn.pdf`, the 0.6b model and all five strategies, started from the page, ran in the background thread and finished in 36 s with the page object discarded after Start (the thread kept going and the report ended as `done`). (2) The 0.6b and 4b models with `hybrid`, `fixed` and `recursive`, 7 test queries (5 text, 2 table, from the old set) and the default probes: 6 of 6 done in 52 s.

  | Model | Strategy | Chunks | Embed tokens/s | Search p50 | Snippet coverage | MRR | nDCG@5 |
  |---|---|---|---|---|---|---|---|
  | 0.6b | hybrid | 41 | 6587 | 28 ms | 1.00 | 0.929 | 0.947 |
  | 0.6b | fixed | 45 | 7107 | 32 ms | 1.00 | 0.929 | 0.947 |
  | 0.6b | recursive | 44 | 7033 | 30 ms | 1.00 | 0.929 | 0.947 |
  | 4b | hybrid | 41 | 1438 | 45 ms | 1.00 | 0.810 | 0.857 |
  | 4b | fixed | 45 | 2355 | 44 ms | 1.00 | 0.857 | 0.895 |
  | 4b | recursive | 44 | 2330 | 47 ms | 1.00 | 0.857 | 0.895 |

  Seven queries is a small sample (one query moves MRR by about 0.1 here), so this table checks that the report works, not which model is better.
- **While running and after:** the page loaded during a run shows the progress panel and a Stop button without errors and, once the run has ended, shows no progress panel. A run stopped after its first experiment ended as `stopped` with 2 of 3 results (one more experiment was already under way when the stop was requested; a stop takes effect between experiments). "Delete this run's experiments" removed all five collections and left the report, which still opened.
- **The PDF:** four pages (landscape A4), rendered to images with `pypdfium2` and looked at: document facts, settings, a summary table with the best values marked green, five bar charts (embedding speed, search latency, chunks, nDCG@5, MRR; value labels, one colour per model), the test-query rank table and the notes. The first render showed the chart legend overlapping the axis labels, tables centred instead of left-aligned and the "Charts" heading alone at the foot of a page; all three were fixed and the second render was checked.
- **Everything the checks made was removed**: the reports, their experiments, collections, artifacts, uploads and the files written for the checks.

Changes from the plan, and notes:
- **Not verified:** how the pages look in a browser (the colours and layout in `style.py` and the charts on the Benchmark page), the every-3-seconds refresh of the progress panel (the app tester runs a fragment once), and the "interrupted after 15 minutes" rule (no container was restarted mid-run). The app tester counted no chart elements on the report view although the page rendered without an exception.
- The app tester resets `st.data_editor` after the run in which its state was set, so the first benchmark above ran without test queries; the second set the state in the run that clicked Start. This is a test-tool limit, not a page fault.
- `ui/data.py` keeps only `frame`, the model and strategy labels and ordering, and `embedding_models()` (now shared by Upload and Benchmark). `ingest.py` gained `save_upload`, `ensure_parsed` and `ingest_document`, which the Upload page now calls too; its Embed behaviour is unchanged (an experiment in a benchmark report now shows "in a benchmark report" in its selector and the notice says its report is not redone).
- `Services` bundles the metrics store, embedder and Qdrant store for the runner thread; `CURRENT_STEP` (this process only) holds what each running report is doing for the progress line.
- The report row's `stop_requested` and heartbeat are in Postgres, so a second browser tab sees the same state; the thread registry itself is per process.
- The PDF uses Latin-1 fonts only (other characters in names become `?`), as planned and left open.

Not included: running the benchmark on several documents at once, generating test queries with an LLM, benchmarks of other parameters than model and strategy (the shared settings are fixed per run), sharing or exporting reports other than the PDF, and filtering "Try a query" by document.

## Phase 12 — An Experiments page: see what is in each experiment and delete it

Goal: a new page in the Streamlit app, above Upload, that lists every experiment with the PDFs inside it, and lets you delete a single PDF from an experiment or a whole experiment, so that uploads and benchmark runs do not pile up for good.

### What "in an experiment" means

An experiment is a Qdrant collection (the searchable state), its row in `experiments`, its stage metric rows and its folder `data/artifacts/<name>/`. A PDF is "in" an experiment when the collection has points with that document id. The page reads that from Qdrant, because it is what search sees; names come from the points' `source_file`. (Qdrant's facet call on the indexed `doc_id` field gives the points per document in one request; checked on this server.)

### Decisions

| Topic | Decision | Reason |
|---|---|---|
| Page | "Experiments", first in the navigation, above Upload (Upload is still the page that opens): Experiments, Upload, Try a query, Benchmark. One expander per experiment. | Placed above Upload as asked; one expander per experiment matches the other pages. |
| What is listed | Every experiment row, plus every Qdrant collection that has no experiment row (shown as "no experiment row", deletable as a collection). An experiment whose collection is missing is listed with "no collection". | Stale leftovers (a collection after a failed run, a row after a manual delete) are exactly what this page is for. |
| An experiment shows | Name, embedding model, strategy, engine, number of documents, number of points, created time, and a badge when it belongs to a benchmark report. | Enough to recognise it and to know what a delete costs. |
| A document shows | File name, document id, number of points (chunks), and when it was ingested (`ingested_at` of its points). | The same facts as the Upload page's selector. |
| Delete a PDF from an experiment | Removes that document's points from the collection, its files (`parse`, `chunk` and `embed` under `data/artifacts/<name>/` named by its document id) and its stage metric rows for that experiment. The experiment, its other documents and its settings stay. It does not delete the uploaded file in `data/uploads/`, a PDF in `data/raw/`, the parse cache in `data/artifacts/_uploads/`, or anything in Dagster. | Those files are shared with other experiments and with Dagster; "remove from this experiment" is the safe meaning. Leaving an experiment with no documents is allowed (it can take new ones from the Upload page). |
| Delete an experiment | Removes its collection, its folder `data/artifacts/<name>/`, its `experiments` row (its stage rows and search log go with it). Benchmark reports that used it stay and show it as deleted, as they do today. | The same as the Benchmark page's "delete this run's experiments", for one experiment. |
| Confirmation | Delete opens an inline "Delete X? This cannot be undone." with Delete and Cancel. For an experiment that belongs to a benchmark report, the line adds that the report's numbers were measured with it. | Deletion is permanent, and inline confirmation also works with the app tester (a dialog may not). |
| Order of the steps | Qdrant first, then the files, then the database rows. If the Qdrant call fails, nothing else is touched; every step is safe to repeat. | A half-finished delete can be finished by pressing Delete again. |
| Protected names | Folders starting with `_` (`_cache`, `_uploads`) are never touched. A collection with no experiment row is only deleted as a collection; its folder is removed only when the name matches the experiment name pattern and the folder is inside `data/artifacts`. | A name from Qdrant must never turn into a path outside the artifacts folder. |
| Where the code lives | Plain functions in `src/rag_lab/library.py` (no Streamlit, no Dagster): list the experiments with their documents, delete a document, delete an experiment. The Benchmark page's "delete this run's experiments" calls the experiment function. | Same rule as the other stages, and one copy of the delete logic. |

### 12.0 Library functions

- [x] `MetricsStore`: `delete_document_rows(config_hash, doc_id)` (the stage rows of one document in one experiment). `list_experiments()` also returns `created_at`.
- [x] `QdrantStore`: `documents(collection)` returning, per document id, the points, the file name and the newest `ingested_at` (the facet call for the counts, one point per document for the rest); `delete_collection(name)`.
- [x] `library.py`: `list_library(metrics, qdrant)` (experiments, collections without a row, experiments without a collection), `delete_document(name, doc_id, ...)`, `delete_experiment(name, ...)`, following the order and the protected names above. `benchmark.runner.delete_report_experiments` calls `delete_experiment` instead of its own copy.

### 12.1 The page

- [x] `ui/experiments.py` and an entry in `ui/app.py` above Upload. A summary line (experiments, documents, points), then one expander per experiment (newest first) with its facts, a table of its documents and a Delete button per document and one for the experiment. The collections without a row come last under their own heading. A Refresh button, and the experiments and query caches are cleared after any delete, so "Try a query" does not offer a deleted experiment.
- [x] Deleting shows what was removed (points, files, rows) and redraws the page.

### 12.2 Docs

- [x] `CLAUDE.md`: the new page, `library.py`, and what a delete does and does not remove.

Done when, checked once on the real stack (Streamlit's app tester for the page, Postgres, Qdrant and the file system for the results): two throwaway experiments each holding two documents (`aiayn.pdf` and a small generated PDF) are listed with the right document names and point counts; deleting one document from one experiment removes exactly its points, its three files and its stage rows, leaves the other document searchable on "Try a query", and leaves `data/uploads/` and `data/raw/` untouched; deleting an experiment that belongs to a benchmark report removes its collection, folder and rows and the report still opens; a collection created directly in Qdrant shows as "no experiment row" and deletes as a collection; pressing Delete twice does no harm; the throwaway data is gone afterwards. No new test files: nothing here is retrieval-metric, truncation or hashing logic.

**Status: done (2026-10-03).** Checked once on the real stack: two throwaway experiments, `ex12-a` and `ex12-b`, each holding `aiayn.pdf` and a one-page generated PDF, an experiment `ex12-b` that a (made-up) benchmark report points at, and a collection `ex12-orphan` created directly in Qdrant. The page was driven with Streamlit's app tester; the results were read from Qdrant, Postgres and the file system.
- **Listing:** the page listed every experiment with the right file names and point counts (`aiayn.pdf` 57 points, `lantern.pdf` 1 in each), the badge "in a benchmark report" on `ex12-b`, and `ex12-orphan` as "no experiment row"; the experiments already in the database (including the benchmark reports' experiments made by someone using the app meanwhile) were listed too and left alone.
- **Delete a PDF:** the inline confirmation named the file and its point count; confirming removed exactly that document from `ex12-a`: points 58 to 57 (lantern 0, aiayn 57), five files (`parse` json, md and meta, `chunk` jsonl, `embed` npy for its id), four stage rows. The other document's files and four stage rows stayed, `search` on `ex12-a` still returned `aiayn.pdf`, and `data/uploads/` and `data/raw/` were unchanged. Running the same delete again returned zeros and no error.
- **Delete an experiment:** the confirmation for `ex12-b` mentioned the benchmark report; confirming removed its collection, folder and row, and the report stayed (`done`). The orphan collection was deleted as a collection. Deleting `ex12-b` a second time returned "nothing to remove" without an error.
- The throwaway experiments, report and the generated PDF's files were removed afterwards.

Changes from the plan, and notes:
- `QdrantStore.documents()` falls back to scrolling the points when the collection has no index on `doc_id`: the orphan collection made directly in Qdrant had none, so the facet call failed. Collections made by the app always have the index.
- The delete confirmation names what will go; the success message states what did (points, files, metric rows; collection, folder, record).
- `benchmark.runner.delete_report_experiments` now calls `library.delete_experiment`.
- Found while checking, not part of this phase: `ui/charts.py` now calls `style.palette()`, which `ui/style.py` does not define, so the Benchmark page's report view (its charts) raises an `AttributeError` until the two files agree. Both were edited outside this phase; nothing here touches them.
- Also found: while this phase was being built, files in `data/uploads/` and `data/artifacts/_uploads/` (the upload copies and parse cache of `aiayn.pdf`) were removed by the cleanup of earlier checks, which assumed they were all test data. Experiments and benchmark reports keep their own copies of what they need; a document uploaded on the Upload page has to be uploaded again to be parsed again.

Not included: deleting the uploaded files, the parse cache or PDFs in `data/raw/`, deleting Dagster partitions or run history (they belong to Dagster), renaming an experiment, moving a document between experiments, and bulk or "delete everything" actions.

## Phase 13 — Search strategies in the benchmark

Goal: the benchmark compares, besides embedding model and chunking strategy, how the vectors are searched: **dense** (today), **hybrid** (dense plus BM25, fused), and either of the two followed by a **reranker**. A report row becomes model × chunking × search strategy, so you can see whether hybrid or reranking is worth its cost on a given document.

### Naming

"Strategy" already means the chunking strategy everywhere (`benchmark_results.strategy`, the charts, `STRATEGIES`). The new axis is called **search strategy** on the page and in the report, and `method` in code (`method="hybrid+rerank"`), so the two do not collide.

### What the four methods are

| Method | How it searches | Needs at index time |
|---|---|---|
| `dense` | Today's search: query embedded with the instruction prefix, cosine nearest neighbours. | nothing new |
| `hybrid` | Two Qdrant prefetches, dense and BM25 sparse (`candidates` hits each), fused with reciprocal rank fusion (`FusionQuery(RRF)`), cut to top k. | a BM25 sparse vector on every point |
| `dense+rerank` | Dense search for `candidates` hits, the reranker scores each (query, chunk) pair, the best top k are returned. | nothing new |
| `hybrid+rerank` | The `hybrid` search for `candidates` hits, then the same reranking. | a BM25 sparse vector on every point |

### Decisions

| Topic | Decision | Reason |
|---|---|---|
| Matrix | The search strategy is **not** a new experiment axis. A report still builds one experiment (collection) per model × chunking strategy, and every chosen search strategy searches that same collection. `total_experiments` and the progress bar still count experiments. | All four methods only differ at query time (the sparse vector is added once). Re-embedding the document per method would cost the most expensive step for nothing. |
| Sparse vector | Our own BM25 in `embedding/sparse.py`, no new dependency: lower-cased `\w+` tokens, term hashed to a stable 31-bit index, weight `tf·(k1+1) / (tf + k1·(1 − b + b·len/avg_len))` with `k1=1.2`, `b=0.75`, `avg_len` = mean token count of the chunks being indexed. The collection gets a named sparse vector `bm25` with `Modifier.IDF`, so Qdrant does the IDF at query time. The query side is each unique token with weight 1. | Keeps FastEmbed out of the project (CLAUDE.md: "keep dependencies few"; FastEmbed is only ruled out for dense). BM25 is simple enough to own. |
| Not LlamaIndex's `enable_hybrid` | The sparse vector is written by a second step after `QdrantVectorStore.add`: `client.update_vectors` on the same deterministic point ids. | `enable_hybrid` renames the dense vector (`text-dense`) and wants a FastEmbed sparse model, which breaks the "unnamed vector" layout every existing collection has. |
| `IndexConfig.sparse` | New `sparse: bool = False`. Left out of the config hash while false (the way `engine` is when `native`), so no existing hash changes. The benchmark sets it true for every experiment when a hybrid method is chosen. Dagster runs get it from `resources.experiment.config.index.sparse`; the assets need no change. | Hybrid needs the sparse vector at index time, and an experiment's hash must say whether it has it. |
| Search settings are not experiment settings | New `SearchConfig` in `config.py` (`method`, `candidates=20`, `reranker`, `rerank_instruction`) passed to `search()`; it is **not** part of `ExperimentConfig` or its hash. | Changing how you search must not make a new experiment. |
| Existing collections | `ensure_collection(sparse=True)` adds the sparse vector config only when it creates the collection. A collection that exists without it and is asked for `sparse=True` raises, as a wrong vector size does. | Benchmark collections are always new. Adding a sparse vector to a live collection is not needed yet. |
| Reranker | Qwen3-Reranker through Ollama `/api/generate`, one call per (query, chunk): the model's own yes/no prompt sent with `raw: true`, `num_predict: 1`, `logprobs` on; the score is `P(yes) / (P(yes) + P(no))` from the first token's log-probabilities. One reranker per report (a setting), not an axis. The page lists installed models whose name contains `reranker`; with none installed the rerank methods are disabled and the hint says what to pull. | Same family as the embedding models, no new service or dependency. Ollama has no rerank endpoint. |
| What each search strategy reports | `search`: latency (query embedding, Qdrant, rerank, total; p50 etc.). Only `dense` also reports `top1_similarity` and `gap_top1_topk`. `quality` (with test queries): the same metrics as now, plus per-query first relevant rank. | RRF scores and reranker probabilities are not cosine similarities, so "similarity spread" means something only for `dense`. |
| Reranking cost | Rerank latency is measured on the first `rerank_probe_count` probe queries (default 5), one timed pass after the warm-up. Quality uses every test query. The page shows the number of reranker calls before Start (experiments × rerank methods × queries × `candidates`). | 20 candidates × 20 queries × 4 passes × 2 methods × 15 experiments would be about 48,000 CPU generate calls. |
| Stored result shape | No migration. Each `benchmark_results.metrics` gets `searches: {method: {search, quality, per_query}}`; the experiment-level parts (`chunking`, `embedding`, `index`, `snippet_coverage`) stay where they are. A saved report without `searches` is read as one `dense` entry made from its old `search`, `quality` and `per_query`. | Old reports keep opening, and the results table keeps one row per experiment (it is also what "delete this run's experiments" uses). |
| Report | One row per (model, chunking, search strategy). Chunks, mean tokens and embed speed are experiment-level, so they repeat on the experiment's rows and are marked best over experiments, not rows. New columns: `Search`, and `Rerank p50 (ms)` when a rerank method ran. A segmented control picks the search strategy shown in the speed and quality charts and the per-query rank table (default `dense`); the PDF draws those on one page per strategy. One new chart, "Search strategies compared" (nDCG@cutoff, and median latency, per model, mean over the chunking strategies), answers the main question at a glance. | Charts and the rank table are already as wide as they can be with model × chunking. |
| Dense query embedding | Not shared between the methods of one experiment. Each method calls `search()` as it is. | The cost is the same for every method, and `search()` stays simple. |
| Try a query page | A "Search strategy" select above the form (the experiment select moved out of the form with it, because which strategies are offered depends on the experiment): `hybrid` only for an experiment whose stored config has `index.sparse`, the rerank strategies only when a reranker is installed, plus a reranker select and `candidates` when one is chosen. The result line names the strategy, the rerank time and what the score means. | Asked for in the request; the benchmark's experiments are kept and searchable, so their hybrid collections can be tried by hand. |
| Tests | One case in the existing hash test (`sparse=False` leaves every hash unchanged, `sparse=True` changes it). One small test each for the sparse vector builder and for the yes/no score. | The rules allow tests for hashing and vector construction; a silent bug in these two changes every ranking without an error. Drop them if you disagree. |

### 13.0 Check the assumptions on the real stack (throwaway scripts, not in the repo)

- [x] Qdrant: one collection with the unnamed 1024-dimension dense vector and a named sparse vector `bm25` (`Modifier.IDF`); `QdrantVectorStore.add` still writes the unnamed vector when the collection also has a sparse config; `update_vectors` writes `bm25` for the same point ids; `query_points` with two `Prefetch` entries and `FusionQuery(RRF)` returns fused hits.
- [x] Ollama: this Ollama's version supports `logprobs` on `/api/generate`; pull a Qwen3-Reranker (a community GGUF, for example `dengcao/Qwen3-Reranker-0.6B`; confirm the tag, and which of 0.6b, 4b and 8b is worth the time on this hardware); a relevant chunk scores clearly above an irrelevant one for the same query; time one call.
- [x] If `logprobs` is missing, stop and decide with the user before going on (a hard yes/no score would rerank poorly). Write what was found, and the chosen reranker tag, into the decisions table above.

### 13.1 Config

- [x] `IndexConfig.sparse` and its hash exception; `SearchConfig`. One extra case in `tests/test_config_and_vectors.py`.

### 13.2 Sparse vectors

- [x] `embedding/sparse.py`: tokeniser, `document_vectors(texts)` and `query_vector(text)` returning indices and values; a small test.
- [x] `QdrantStore`: `ensure_collection(..., sparse)`; `add_sparse(collection, ids, vectors)`; `query(..., method)` runs the hybrid query when the method is hybrid.
- [x] `ingest.index_chunks`: when `config.index.sparse`, build the vectors after the dense write, and add `sparse_ms` to the `index` stage details.

### 13.3 Reranker

- [x] `reranking/ollama.py`: `OllamaReranker(base_url)` with `score(query, texts) -> list[float]` (the same retry and timeout habits as `OllamaEmbedder`); the pure score function from log-probabilities, with a small test. `resources/` is not touched: it is only used by the benchmark and the UI for now.

### 13.4 Search

- [x] `search()` takes `method`, `candidates` and a reranker, and returns hits ranked by the method; `SearchResult` gains `method` and `rerank_ms`. The default is `dense`, so the CLI, `ask()` and Try a query behave as today.

### 13.5 Benchmark runner

- [x] `BenchmarkSettings`: `search_methods` (default `["dense"]`), `reranker`, `candidates`, `rerank_probe_count`. `experiment_config` sets `index.sparse` when a hybrid method is chosen. `Services` gets the reranker.
- [x] `run_experiment`: after the experiment is built, loop over the methods (all non-rerank ones first, so the embedding model is not swapped out between them), `_search_metrics` and `_quality` take the method, and the result is stored in `searches` as above. The progress step reads "searching: hybrid+rerank".
- [x] `start_report` stores the new settings with the report.

### 13.6 Report, PDF and page

- [x] `report.py`: read old and new result shapes, one row per method, the new columns, the experiment-level "best", the "Search strategies compared" chart, notes on score scales and on reranking cost. The charts' `strategy` order stops assuming it is a chunking strategy (`strategy_order` is given the order).
- [x] `pdf.py`: the same rows, one chart page per search strategy, the new chart.
- [x] `ui/benchmark.py`: a "Search strategies" multiselect (default: `dense`), the reranker select, `candidates` and `rerank_probe_count` inputs shown when a rerank method is chosen, the call-count caption, and the segmented control on the report view. `data.py`: `reranker_models()` beside `embedding_models()`.

### 13.7 Docs

- [x] `CLAUDE.md`: the four methods, `IndexConfig.sparse` and its hash rule, the reranker, the new result shape and the "similarity is only cosine for `dense`" fact, the new files in the layout, and the status line.

Done when, checked once on the real stack: a benchmark of `aiayn.pdf` (uploaded again on the Benchmark page if `data/uploads/` no longer has it) with `qwen3-embedding:0.6b`, the `hybrid` and `recursive` chunking strategies, all four search strategies and about eight test queries (some that quote a rare term from the paper, some that paraphrase it) finishes and gives 8 rows; the experiments' collections have a `bm25` vector on every point; the `hybrid` rows rank at least one query differently from `dense`; the rerank rows show a rerank time and a changed order; a report made before this phase still opens and reads as `dense`; the PDF downloads and has the new rows and chart; the config hash of an experiment made before this phase is unchanged. The throwaway experiments are deleted afterwards. No other test files are added.

### Not in this phase

The Upload page choosing whether to build the BM25 vector (a normal experiment has none, so hybrid is only offered on Try a query for experiments made with `index.sparse`, which the benchmark does); DBSF fusion or learned sparse models such as SPLADE; the reranker as its own axis; caching reranker scores; exact search and `hnsw_ef`; Thai or other languages without spaces between words (the BM25 tokeniser needs a word segmenter for those, and a table of English academic text is what it is checked on).

### Risks

| Risk | Handling |
|---|---|
| `QdrantVectorStore` misreads a collection that also has a sparse vector config and writes the dense vector under a different name | Checked first in 13.0; if it does, write the points ourselves for sparse experiments. |
| Ollama has no `logprobs`, or the reranker model's answer is not a clean yes or no | 13.0 stops and asks before anything is built. |
| Reranking is slow, and the embedding model and the reranker may not both stay loaded in Ollama (the 8b embedding model plus an 8b reranker) | Candidate and query counts are small by default and shown before Start; the 0.6b reranker is the first to try. Slow runs are an accepted cost, not a bug. |
| BM25 over Markdown tables and numbers behaves oddly (a `|` row of figures is a handful of tokens) | Tokens are `\w+`, so separators vanish and numbers are kept; the test queries that quote a number show whether it helps. |
| Per-document `avg_len` makes scores differ between two documents in one experiment | Benchmarks hold one document. Revisit if hybrid is offered on the Upload page. |

**Status: done (2026-10-03).** Checked once on the real stack: benchmark report `ae0d37` on `aiayn.pdf` with `qwen3-embedding:0.6b`, the `hybrid` and `recursive` chunking strategies, all four search strategies, six test queries (some quoting the paper, some paraphrasing it), top k 10, 6 probe queries, 10 candidates and the first 2 probes for rerank timing. The page code was driven with Streamlit's app tester; the numbers were read from Postgres, Qdrant and the built report.
- **13.0:** Qdrant 1.16.3 takes the unnamed 4-dimension dense vector and a named sparse `bm25` (IDF modifier) in one collection; `QdrantVectorStore.add` still wrote the unnamed vector (the client shows it as `''`); `update_vectors` set `bm25` on the same ids; a `query_points` with two `Prefetch`es and `FusionQuery(RRF)` returned the point that had the rare term first (Qdrant's RRF scores were 0.75, 0.5, 0.33). Ollama 0.35.0 supports `logprobs`. The reranker was the real finding: `dengcao/Qwen3-Reranker-0.6B:Q8_0` answered `,` for every prompt with every token at `ln(1/vocab)` (a broken build), `0.6B:F16` gave `yes` at `-8` (flat, weak), `pdurugyan/qwen3-reranker-0.6b-q8_0` is an embedding build ("does not support generate"), and **`dengcao/Qwen3-Reranker-4B:Q4_K_M`** gave `yes` at `-0.14` for a relevant chunk and `No`/`no` at about `-1` for an irrelevant one. That is the default reranker. My first scoring script took the last of several spellings of "yes" and looked wrong; `yes_probability` sums `yes`, `Yes` and ` yes`. The two 0.6B builds I pulled were removed again.
- **Results (6 test queries, so one query moves recall by 0.17):** hybrid raised recall@5 from 0.83 to 1.00 and MRR from 0.73 to 0.81 (chunking `hybrid`) and from 0.71 to 0.76 (`recursive`); the paraphrase query that dense ranked 9th (or not at all) became 2nd and 3rd. Reranking did not help with every chunker: `dense+rerank` on `recursive` left MRR at 0.71, `hybrid+rerank` on `recursive` gave the best MRR (0.88) and nDCG@5 (0.91). Search p50: dense 29 to 31 ms, hybrid 33 to 34 ms, reranked 0.9 to 1.4 s for 10 candidates (about 0.1 s a pair; with the 4B model loaded the first call took 8 s). Every point of both collections had a `bm25` vector (41 and 44 points).
- **Report and pages:** 8 rows (2 experiments × 4 strategies), 16 charts (embedding speed and chunks once; the two search-strategy comparison charts; speed, nDCG and MRR for each strategy), per-strategy rank tables, a 17.8 kB PDF. The two reports from before the phase (`e1204e`, `f41495`) built and drew as `dense` with their PDFs. All 23 stored experiments recomputed to the same `config_hash`. Try a query offered `dense`, `hybrid`, `dense + rerank` and `hybrid + rerank` for the new experiment, and only `dense` and `dense + rerank` for an older one; the hybrid and the hybrid + rerank searches ran and named the score they return. On the Benchmark page, picking a rerank strategy showed the reranker, candidates and rerank-probe inputs and a call estimate (about 1,800 calls for 15 experiments).
- The report's experiments were deleted afterwards; the report row `ae0d37` stays (the Benchmark page shows it as a report whose experiments are deleted). 11 tests pass (4 new: the hash with `sparse`, the sparse vector builder, the yes/no score).

Changes from the plan, and notes:
- `QdrantStore.query` takes `sparse` and `branch_limit` instead of a `method`; `search()` builds the sparse query vector and, for the rerank methods, fetches `max(candidates, top k)` hits first. A hybrid search of a collection with no `bm25` raises a plain message instead of Qdrant's.
- The page offers only rerankers that Ollama reports with the `completion` capability, which leaves out the embedding-style 0.6B build.
- A failing search strategy is recorded in its own entry (`searches.<method>.error`) and shown under "Failed experiments"; the experiment itself still counts as done.
- The per-query rank table's cells are strings now: a column that mixed numbers and `–` made pyarrow log a conversion error (an old problem the new tables made more likely).
- The CLI and `ask()` stay dense-only.
- Not verified: how the new charts and the segmented control look in a browser (the app tester does not draw them), and a full benchmark with the 8b embedding model next to the 4B reranker in Ollama's memory.

## Ideas that later phases overtook

These were in the Ideas list. Phase 11 removed the matrix, the dashboard and the query-set file they refer to, and the benchmark now parses a document once.

- Reuse one parse across experiments instead of re-parsing for each (copy or point to the parse output, and copy its metrics row).
- A fixed-dimension axis in the matrix (all models truncated to the same size by Matryoshka) to separate model quality from vector size.
- Preset experiment configs in `experiments/*.yaml`, selectable in the launchpad.
- A dashboard in Grafana or Metabase pointed at Postgres, in place of or beside Streamlit.
- Marking hits as expected directly in the query page and writing them to `eval/queries.yaml`.

## Risks (as written during the build)

| Risk | Mitigation |
|---|---|
| Containers cannot reach the Mac over Tailscale | Tested in Phase 0; Tailscale sidecar as fallback. |
| The Mac sleeps or leaves the tailnet mid-run | Retries with backoff in `OllamaResource`; a health check at job start that fails fast. |
| Docling is slow on CPU, especially with accurate table mode | OCR is not offered, `num_threads` configurable, cached model volume, per-document partitions, `document_timeout`. |
| A scanned PDF is dropped in and parses to almost no text | Characters-per-page metric makes it visible; no OCR fallback for now. |
| Semantic chunking makes the chunk stage depend on the Mac and adds one embedding per sentence | Batched calls, on-disk sentence embedding cache, cost recorded in chunk metrics. The other four strategies stay offline. |
| Sentence splitting is language-dependent, which affects `semantic` and `recursive` | The sentence splitter is a config value; check it against the language of the actual PDFs before trusting results. |
| Ollama unloads the model between batches | Set `keep_alive` on requests; record `load_duration` so cold starts are visible in metrics. |
| Changing embedding dimension in an existing collection | Dimension is part of the config hash; a mismatch against the collection raises before any insert. |
| The 8b embedding model does not fit the Mac's memory, or Ollama swaps models and distorts latency | Check memory in Phase 7.1; the matrix runs model by model so each is loaded once; `model_load_ms` is recorded per experiment. |
| Models are ranked by similarity scores, which differ in scale between models | The dashboard compares models on recall, MRR and nDCG only; the plan requires a filled-in query set before the comparison run. |
| Qdrant data corruption from a Windows bind mount | Qdrant storage uses a Docker named volume only. |
| LlamaIndex dependencies do not resolve next to Docling, or its API moves | Check with one image rebuild in 8.0; the packages' own ranges are `llama-index-core<0.15`, `numpy>=2`, `qdrant-client>=1.16`, so pin the four packages to the versions that were checked. |
| Docling in the Streamlit container uses a lot of memory and blocks that user's session while it parses | Parse once per document and cache it; show progress in `st.status`; the model cache volume is shared with `dagster-code`. If Docker Desktop's memory limit is hit, parsing for the page can move to a Dagster run (Ideas). |
| A benchmark thread dies with the UI container, or competes with a person using the app for CPU and Ollama | Results are saved per experiment, a stale `running` report shows as `interrupted`, one run at a time, models run in turn so each loads once, and a Stop button ends a run between experiments. |
| Probe queries are sentences of the document, so their similarity scores look better than real questions would | They are used for latency and score spread only; the report says so, and quality comes only from the test queries. |
| A test snippet is cut by a chunk boundary, so a chunker scores low for a reason unrelated to retrieval | Snippet coverage is reported per experiment next to the quality numbers, and snippets absent from the document are flagged before the run. |
| A chunk's span is wrong, so cut marks sit in the wrong place without any error | The one-off check over `aiayn.pdf` in Phase 9 compares the text inside every span with the chunk's body; the page labels the approximate cases (tables, split paragraphs) instead of drawing them as exact. |
| An upload's experiment name or settings collide with a matrix experiment | The name and hash checks in Phase 9 refuse them before anything is written. |
| LlamaIndex's sentence splitter needs NLTK data downloaded at first use | `NLTK_DATA` points into the `model_cache` volume (8.0); the first run needs internet. |
| LlamaIndex splitters differ from our own, so `llamaindex` experiments are not chunk-for-chunk identical | The comparison in 8.4 decides; the differences are listed in 8.2 and `native` stays until you retire it. |
| Wiping the Postgres volume loses metrics history along with Dagster's run history | Both live in one volume; `docker compose down -v` is called out in `CLAUDE.md`. Add a `pg_dump` script if the history starts to matter. |
