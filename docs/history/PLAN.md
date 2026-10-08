# Plan

Phase 14 and the RAG chatbot (Phases 15 to 18) are built and wait for a look at the Chatbot page in a browser before they move to `COMPLETED_PLAN.md`. Chat history and chat metrics (Phases 19 and 20) are built too, with the same by-eye check open. Text-to-SQL over imported CSV files is built through Phase 27 (the by-eye checks of its pages are still open) and its evaluation, Phase 28, is planned. OCR (`glm-ocr`) with an option on the Upload page is built (Phase 29; the by-eye check of the page is open), `embeddinggemma-2` is a second embedding model for text (Phase 30), pictures are chunks embedded from the image (Phase 31), and the chatbot's model is shown the pictures it retrieves (Phase 32); the by-eye checks of these pages are open. Automatic ingestion is built (Phase 33): a PDF put into `data/raw` starts the Dagster pipeline with the settings of `config/ingest.yaml`. Options for putting the config in YAML are listed below, waiting for a decision. Phases 0 to 13 are done: the project is a RAG platform with an Upload page (chunk preview, embed into a new or existing experiment), a Try a query page, an Experiments page (list and delete what is in each experiment) and a Benchmark page that compares embedding models, chunking strategies and search strategies (dense, hybrid, reranked) and produces a PDF report. What each phase planned and found is in `COMPLETED_PLAN.md`; the current state of the repo is in `CLAUDE.md`.

A new phase is written here as `## Phase 14 — ...`: a goal, a table of decisions, numbered steps with checkboxes, and a "Done when" line that is checked once on the real stack. When it is done, its section moves to `COMPLETED_PLAN.md`.

## Phase 14 — BM25 option on the Upload page

**Goal:** an experiment made on the Upload page can be searched with `hybrid` and `hybrid+rerank` on Try a query. Today the page never sets `index.sparse`, so every uploaded experiment is dense-only, and a collection made without the BM25 vector cannot get it later.

| Decision | Choice |
|---|---|
| Control | One checkbox on the Upload page, "Add a BM25 keyword vector (needed for hybrid search)", off by default, so uploads behave as before. |
| Where | Next to the other settings in `ui/upload.py`; `build_config` passes it as `index=IndexConfig(sparse=sparse)`. |
| Backend | No change. `IndexConfig.sparse`, `ensure_collection`, `index_chunks` and Try a query's `has_sparse` check already handle it. |
| Config identity | `sparse` is in `settings_hash()` only while true, so a sparse experiment never matches a dense-only one. The "existing experiment with the same settings" list then offers only sparse experiments, and a new experiment is the way to add BM25 to a document already uploaded. |
| Preview | The hash covers `index`, so ticking the box marks the chunk preview stale and the page asks for Chunk again. Accepted; no change to that logic. |
| Existing experiments | Left as they are. They stay dense-only; delete them on the Experiments page if they are no longer wanted. |
| Tests | None. The change is UI wiring; hashing of `sparse` is already covered by its existing behaviour. |

### Steps

- [x] 1. `ui/upload.py`: import `IndexConfig`; add the checkbox with a `help` text (keyword search over the same chunks; costs a little extra index time; cannot be added to an experiment later); pass `index=IndexConfig(sparse=sparse)` in `build_config`.
- [x] 2. `CLAUDE.md`: in the Upload description and the "BM25 sparse vectors" fact, note that the Upload page has the option.
- [x] 3. Restart the UI: `docker compose restart ui` (`ui/upload.py` is a page script, so Streamlit may pick it up without it).

**Done when:** a PDF is uploaded with the box ticked into a new experiment, and on Try a query that experiment offers `hybrid` and `hybrid+rerank`, and a `hybrid` search returns hits. Checked once, on the real stack.

## RAG chatbot (Phases 15 to 18)

**Goal:** a Chatbot page where you ask questions about the documents in an experiment and get an answer written by `gemma4:e4b-mlx` (served by Ollama on the Mac) from retrieved chunks, with everything the system does shown: which step it is in, what the model is thinking, which chunks it found and how they scored, and how long each step took. The agent is built with LangGraph (the graph) and `langchain-ollama` (the chat model). Main retrieval is `hybrid+rerank`.

Four parts, each shippable on its own: 15 the agent without a UI, 16 the event stream and trace that make it transparent, 17 the page, 18 smarter behaviour (retry, follow-ups, abstaining). Phase 14 (the BM25 checkbox) is a prerequisite.

### What was checked while drafting

- `gemma4:e4b-mlx` on the Mac's Ollama (0.35.0) reports capabilities `completion, vision, audio, tools, thinking`, 8.1B parameters, `nvfp4`, 131072 context. So tool calling and a separate thinking stream are available; vision and audio are not used (Scope).
- Ollama's default `num_ctx` is small and silently cuts a long prompt, so the agent always sets `num_ctx` itself and the page shows prompt tokens against it.
- **The installed reranker is `dengcao/Qwen3-Reranker-0.6B:Q8_0`, which scores every chunk 0.0** (see the Try a query finding). `hybrid+rerank` as the main strategy needs `dengcao/Qwen3-Reranker-4B:Q4_K_M` pulled first. This blocks Phase 15's "Done when".
- `hybrid` needs an experiment made with `index.sparse` (Phase 14). The chatbot offers only such experiments.
- One Mac will hold the chat model, the embedding model and the 4B reranker. Ollama may unload and reload them between steps; the first answer after a pause can be slow. Measure it in Phase 15 and set `keep_alive` / `OLLAMA_MAX_LOADED_MODELS` on the Mac only if it hurts.

### Architecture

A fixed graph, not a free-roaming tool-calling agent. The 8B model decides only what a small model does reliably (is this a follow-up, is the evidence enough, what to rewrite the query as, how to word the answer); retrieval itself is deterministic code, always `hybrid+rerank`. A tool-calling variant is under Ideas.

```
                 +----------------------------- retry, at most max_rewrites times (Phase 18) ---+
                 |                                                                              |
question -> condense -> retrieve -> grade --enough--> generate -> answer + citations            |
 (+ history)  (standalone  (hybrid + rerank,   |                                                |
               query)       top k chunks)      +--not enough--> rewrite --------------------------+
                                               |                  (still not enough after the retries)
                                               +------------------> abstain: "not in the documents", chunks still shown
```

| Node | Does | Model call | Thinking |
|---|---|---|---|
| `condense` | Turns the question and the last few turns into one standalone query ("and the second one?" becomes a full question). Skipped on the first turn. | yes, short | off |
| `retrieve` | `rag_lab.search.search(..., options=SearchConfig(method="hybrid+rerank"), reranker=...)`. Not a LangChain retriever: our own search keeps the vectors, filters and scores under our rules. | none | |
| `grade` | Phase 15: passes through. Phase 18: the reranker's best score against a threshold first (free), then a yes/no from the model when the score is in between. | Phase 18 | off |
| `rewrite` | Phase 18: a different query when the evidence is not enough. | yes, short | off |
| `generate` | Writes the answer from the numbered chunks only, citing `[1]`, `[2]`; says so when the chunks do not contain the answer. Streams tokens. | yes | on (setting) |
| `abstain` | Phase 18: a fixed "I could not find this in the documents" plus what was searched. | none | |

State (`TypedDict`): `question`, `history`, `query`, `hits`, `rewrites`, `grade`, `answer`, `events`. Settings go in `config.py` as `AgentConfig` (`model`, `temperature`, `num_ctx`, `think`, `top_k`, `candidates`, `history_turns`, `max_rewrites`, `grade_threshold`), not part of `ExperimentConfig` or its hash, like `SearchConfig`.

Code lives in `src/rag_lab/agent/` as plain Python with no Streamlit or Dagster imports: `graph.py` (state, nodes, `build_graph`), `prompts.py`, `events.py`, `__main__.py` (CLI). `ui/chat.py` is a thin caller. Retrieval reuses `search()`, `OllamaEmbedder`, `QdrantStore`, `OllamaReranker` unchanged. New dependencies: `langgraph` and `langchain-ollama` (which brings `langchain-core`); needed because the graph, the streaming of node updates and the Ollama chat model with a thinking stream are what they provide. Needs `docker compose up -d --build`.

### Phase 15 — Agent core, no UI

**Goal:** `condense -> retrieve -> generate` runs end to end from a command line, answering with citations from `hybrid+rerank` hits.

| Decision | Choice |
|---|---|
| Graph | The linear graph above. There is no `grade` node yet; Phase 18 adds it with its edges, so no empty node is carried. `condense` is one node that returns the question unchanged when there is no history. |
| Chat model | `ChatOllama(model, base_url=OLLAMA_BASE_URL, num_ctx, temperature, reasoning=think)`; the exact thinking parameter and where the thinking text arrives are checked against the installed `langchain-ollama` before writing `generate`. |
| Structured output | Not needed in this phase (`condense` returns plain text). When Phase 18 needs it (`grade`), thinking is off for that call, because JSON-schema output and thinking can clash. |
| Prompt | Chunks are numbered with source, page and headings; the model answers only from them and cites by number. Chunk text is data, not instructions. |
| Memory | The page and the CLI keep the message list and pass `history` in; no checkpointer yet. |
| Tests | None. |

Steps:

- [x] 1. A working reranker on the Mac and an experiment with a BM25 vector. Done with `dengcao/Qwen3-Reranker-4B:Q8_0` (already pulled; P(yes) 0.999 relevant, 0.0 irrelevant, so the `Q4_K_M` pull was not needed) and the existing `full-ingest-dense-sparse`.
- [x] 2. Add `langgraph` and `langchain-ollama` to `pyproject.toml`; rebuild.
- [x] 3. `AgentConfig` in `config.py`.
- [x] 4. `agent/prompts.py`, `agent/graph.py` (state, `condense`, `retrieve`, `generate`, `build_graph(embedder, store, metrics, reranker, llm)`).
- [x] 5. `agent/__main__.py`: `python -m rag_lab.agent "question" --experiment <name>` prints the answer and the numbered sources.
- [x] 6. `CLAUDE.md`: the `agent/` package, `AgentConfig`, the dependencies.

Built differently from the draft: `AgentConfig` nests a `SearchConfig` (method `hybrid+rerank`, reranker `Q8_0` 4B) instead of repeating `candidates` and the reranker; `grade_threshold` and `max_rewrites` are left to Phase 18; `SearchConfig`'s own default reranker is untouched.

**Done when:** on the real stack, a question whose answer is in the document gets a correct answer citing the right chunk, and the hits show nonzero reranker scores. Checked once.

**Result (checked once):** `python -m rag_lab.agent "what to do when there is a door opening mid flight" --experiment full-ingest-dense-sparse` answered correctly from the chunk "Door Opening In-Flight" (page 17), cited `[1]`, and the reranker scored it 0.999 (the next hits 0.18 to 0.08). 1682 tokens in, 645 out. Search took 12.4 s, 11.1 s of it reranking 20 candidates (the first call also loads the 4B model, so a warm run is expected to be faster). The follow-up path (`condense` with history) is wired but was not exercised; Phase 18 checks it.

### Phase 16 — Transparency: events and trace

**Goal:** every step the graph takes is a typed event, streamed while it happens and saved afterwards, so a page (or a CLI) can show it without knowing the graph.

| Decision | Choice |
|---|---|
| Events | Plain dataclasses in `agent/events.py`: `StepStarted(node)`, `StepFinished(node, ms)`, `Thinking(text)` (token chunks), `Query(text, kind)` (original or rewritten), `Retrieved(hits, method, candidates, embed_ms, search_ms, rerank_ms)`, `AnswerToken(text)`, `ModelState(...)`, `Done(answer, citations, timings)`. |
| How they are produced | Every node reports through LangGraph's custom stream (`get_stream_writer()`), including the thinking and answer tokens, which `generate` reads with `llm.stream`. That replaces the `updates` and `messages` modes the first draft listed: one channel, and the node already knows what it is doing. `run(graph, experiment, question, history, ...)` streams `custom` and `values` and yields the events, then `Done`. |
| Model state | Per model call: which model, thinking on or off, prompt tokens against `num_ctx` (a warning event when the prompt reached it), output tokens, tokens per second (from Ollama's eval counts), and whether the model was already loaded (`/api/ps` before the call). |
| Citations | The answer's `[n]` markers are resolved to hits; a number with no hit is flagged. |
| Persistence | `run()` does it, given `metrics` and a `session_id`, so the CLI and the page share it. A numbered migration adds `chat_turns` to `rag_metrics` (session id, experiment, question, standalone query, model, hits as JSON, answer, timings, created time). The retrieval also goes to the existing search log. Postgres stays the source of truth. A session id is a random id per browser session. |
| Tests | None. |

Steps:

- [x] 1. `agent/events.py`; `agent/graph.py` emits them and exposes `run(...)`.
- [x] 2. `ModelState` collection around each model call (including the `/api/ps` check).
- [x] 3. Migration `rag_lab/metrics/migrations/NNN_chat_turns.sql` and a `MetricsStore` method to write a turn.
- [x] 4. `agent/__main__.py` prints the events live (steps, thinking dimmed, retrieved chunks, timings).
- [x] 5. `CLAUDE.md`: events, `chat_turns`.

**Done when:** one CLI run prints every step with its time, the thinking text, the retrieved chunks with scores, the model state line, and writes one `chat_turns` row. Checked once.

Built differently from the draft: `Query(text, rewritten)` carries a flag instead of a `kind`; the `usage` entry is gone from the graph state (the `ModelState` event carries the counts); the CLI applies the migrations itself on start, as the UI does.

**Result (checked once):** `python -m rag_lab.agent` printed each step with its time, the standalone query, the five chunks with reranker scores (rerank 2.2 s warm), the dimmed thinking, the streamed answer, the model state (`1593 of 8192 context in, 236 out, 39.3 tok/s`, model was loaded) and the citations, and wrote one `chat_turns` row and one `search_log` row per turn (migration `0007` applied by the CLI). A two-question chat rewrote "and what causes that kind of accident?" to "what causes an open door accident in flight?" and answered it from the same chunk. A bug found on the way: the token counts are on the chunk that ends the generation, not the last chunk, so the first run showed 0 in and 0 out until that was fixed. Checked the thinking path and the saved thinking text on the first run (before that fix); the later runs had thinking off.

### Phase 17 — Chatbot page

**Goal:** a Chatbot page in the Streamlit UI that shows the answer and, with it, how it was produced.

| Decision | Choice |
|---|---|
| Page | `ui/chat.py`, added to `ui/app.py` next to Try a query. A thin caller of `agent.run`. |
| Settings | In the sidebar: experiment (only ones with a BM25 vector), embedding model is the experiment's, chat model (installed models that report `completion` and `tools`, default `gemma4:e4b-mlx`), reranker (as on Try a query), top k, candidates, thinking on/off. `hybrid+rerank` is fixed and said so. |
| Conversation | `st.chat_input` and `st.chat_message`; history in `st.session_state`; a "New chat" button. |
| While answering | An `st.status` panel shows the current state ("Rewriting the question", "Searching: hybrid, 20 candidates", "Reranking", "Writing the answer") and fills in live: the standalone query, then the thinking text, then the answer streaming below. |
| After answering | Under each answer an expander "How this was answered" with: the steps and times as a timeline, the thinking text, the model state line, and the retrieved chunks as the same cards Try a query uses (rank, reranker score, source, page, headings, modality, text), with the cited ones marked. Clicking a citation `[n]` is not needed; the numbers match the cards. |
| Reuse | `hit_card` and the score colours from `ui/query.py` move to a shared module (`ui/style.py` or `ui/hits.py`) rather than being copied. |
| Failure | Ollama unreachable, model missing or reranker returning nothing shows as an error in the panel with the step it failed in, never a blank answer. |
| Tests | None; the app tester can drive the page but not streaming meaningfully. |

Steps:

- [x] 1. Move the hit card out of `ui/query.py` into a shared module; Try a query imports it.
- [x] 2. `ui/chat.py` with the sidebar, the chat loop and the live status panel.
- [x] 3. The "How this was answered" expander.
- [x] 4. Add the page to `ui/app.py`; `docker compose restart ui`.
- [x] 5. `CLAUDE.md`: the fifth page, the layout list.

**Done when:** in the browser, a question gets a streamed answer, the status panel showed each step while it ran, and the expander lists the thinking, the timeline and the chunks with nonzero scores. Checked once.

Built differently from the draft: the tabs inside the expander are "Steps and model", "Thinking" and "Retrieved chunks" (Streamlit has no nested expanders, so the thinking is a tab, not a second expander); `data.chat_models()` was added for the model list; the page starts a new conversation when the experiment changes; a failed turn is kept in the display with its error but not in the history the agent sees.

**Result:** driven with Streamlit's app tester inside the `ui` container against the real stack: the page loads without an exception, lists `full-ingest-dense-sparse`, `gemma4:e4b-mlx` (thinking on) and the 4B reranker, answers a question with an expander holding the three tabs and five chunks, and answers a follow-up (the history carries both turns). **Not checked: how the live status panel looks in a browser.** The Chrome tab froze on two screenshot attempts (the server was healthy: health check 200, no errors in the log), so the layout and the live streaming have not been seen by eye; look at the page once before moving Phase 17 to `COMPLETED_PLAN.md`.

### Phase 18 — Better behaviour: retry, follow-ups, abstaining

**Goal:** the agent notices when retrieval did not find the answer, tries once more, and otherwise says so instead of inventing one.

| Decision | Choice |
|---|---|
| `grade` | The best reranker score against two scores, `enough_score` and `missing_score` (one threshold cannot have a band): at or above the first is enough, far below is not, the band between asks the model for a yes/no on whether the chunks answer the question. The threshold is chosen from real scores on a few answerable and unanswerable questions, not guessed. |
| `rewrite` | One retry by default (`max_rewrites = 1`) with a different query; the event stream shows both queries and both hit lists. |
| `abstain` | After the retries, a fixed message plus what was searched. The chunks are still shown. |
| Follow-ups | `condense` already handles them; this phase checks it on a real multi-turn chat and trims `history` to `history_turns`. |
| Grounding check | Not here (Ideas). |
| Tests | None. |

Steps:

- [x] 1. `grade`, `rewrite`, `abstain` nodes and the conditional edges.
- [x] 2. Pick `grade_threshold` from a handful of real queries; record the numbers here.
- [x] 3. The page shows retries and the abstain case in the timeline.
- [x] 4. `CLAUDE.md`.

**Done when:** a question that is not in the document ends in the abstain message with the chunks it looked at, a question needing a reworded query is answered after one retry, and a follow-up question ("and the second one?") is answered correctly. Checked once.

Built differently from the draft:

- Two scores instead of `grade_threshold`: `enough_score` 0.5 and `missing_score` 0.1, with the model asked only in between. `max_rewrites` is 1.
- `rewrite` is given the section titles of the closest chunks. Its first version, without them, turned "the stick does nothing when I pull it" into "joystick unresponsive when pulled" (score 0.047); with the titles it wrote "Flight Control Malfunction" (0.932).
- The answer is written for the standalone question, not for the rewrite; the retry's hits replace the first search's (the latest search is what the answer and the notice use).
- New events `Graded` and `Rewrote`; `Done.abstained`; step times are summed over a retry. The page lists every search with its score and verdict, shows the abstain message as a notice, and keeps the abstain message in the history like any answer.

**Thresholds, from the real document (FAA handbook chapter 18, `full-ingest-dense-sparse`), best chunk's reranker score:**

| Kind | Questions | Scores |
|---|---|---|
| Answered by the document | door opening, engine failure after takeoff, in-flight fire, landing gear, flaps, elevator, ditching and others (about 20) | 0.74 to 1.00, except three below: "the stick does nothing when I pull it" 0.357, "what if a wheel will not come down" 0.107, "the undercarriage is stuck up and will not lower" 0.258 |
| Not in the document | capital of France, bread, football, tyre, Boeing 747 weight, pilot licence cost, VNE of a Cessna 172, how pilots are paid, carburetor icing (never mentioned), spiral dive (mentioned once, not explained) | 0.000 to 0.051 |
| Not covered but near the topic | "what causes icing" (mentioned three times, causes not explained) | 0.006 |

So 0.5 and 0.1 leave a band of 0.1 to 0.5 that held three answerable questions, and the model judged all three answerable. It also judged "what to do if the aircraft shakes violently and the nose drops" (a stall; best score 0.385, about a ditching) answerable and the answer was off-topic. The grading model without thinking is lenient; raising `missing_score` would trade that against losing the three real ones. Not tuned further on one document.

**Result (checked on the real stack):**

- Not in the document: "what is the capital of France" and "what causes icing" each searched twice (the second query rewritten), both scored under 0.01, and ended in the abstain message listing both queries, with the chunks kept under "How this was answered". About 23 to 34 s, almost all of it two rerank passes.
- Needs a rewrite: retrieval was too robust on this document to give a natural weak first search (every rewording scored 0.74 or more). It was checked by raising both scores to 0.9 for one run: "the stick does nothing when I pull it" (0.357) was rewritten to "Flight Control Malfunction" (0.932), graded enough and answered correctly from the elevator-cable passage, citing `[2]`.
- Follow-up through the new graph: "and what causes that kind of accident?" after the door question became "what causes an open door accident in flight" (0.997) and was answered correctly.
- The Streamlit page, driven with the app tester: the off-topic question shows the abstain notice and both searches with scores; a normal question still answers.
- Not seen by eye: the page in a browser (the Chrome tab froze in Phase 17, and was not retried).
- Slow on the Mac: a rerank pass took 14 to 18 s on the off-topic questions against 2 to 4 s in other runs, and one grade call took 10 s for one output token. It looks like Ollama swapping the 4B reranker and gemma in and out of memory; not investigated.

### Open choices (defaults used above unless you say otherwise)

- Fixed graph rather than a tool-calling agent (above), because of the small model.
- Chunks in the answer prompt are the top k after reranking (default 5); the answer cites by number.
- Chat history is kept in the browser session only; no checkpointer, no chat history page.
- The page works on one experiment at a time.

## Chat history and chat metrics (Phases 19 and 20)

**Goal:** the Chatbot page keeps its conversations: a refresh does not lose the chat, and an earlier chat can be reopened with everything the page showed for it (the steps, the thinking, the chunks). And what the agent did and how long it took is in the database in full, including retries and failures.

### What is stored today

Checked in the live `rag_metrics` database (18 turns in 13 sessions, most of them test runs).

| Stored | Where |
|---|---|
| The question, the last query searched, the model, whether it thought, the answer, the thinking text | `chat_turns` |
| The final search's hits with their scores (`hits`) | `chat_turns` |
| Milliseconds per step (`timings`: condense, retrieve, grade, rewrite, generate; a retry adds up) | `chat_turns` |
| One record per model call (`model_states`): prompt and output tokens, tokens per second, whether the model was already loaded, `num_ctx`, `context_full` | `chat_turns` |
| Embed, search and total milliseconds of each search, top k, the top score | `search_log` (`total_ms` includes the rerank; the rerank alone is not stored) |

| Not stored | Why it matters |
|---|---|
| The retries: the queries tried, each grade verdict and score, each attempt's hits | The page shows them live, but only the last query is kept, so a past answer cannot be shown as it was. |
| Whether the turn abstained, how many tokens the answer cites, the turn's total time | Cannot be counted or averaged without parsing text. |
| The settings of the turn: reranker, candidates, top k, thresholds, `num_ctx` | A slow turn cannot be compared with another without knowing what was different. |
| The rerank time on its own (it is only in the live event) | The biggest cost in a turn (2 to 18 s) cannot be queried. |
| Failed turns | Nothing is written when a step fails, so errors are invisible. |

Session history: none. The page keeps its messages in `st.session_state`, so a refresh or a new tab starts empty. `chat_turns` already has the content of each turn and a `session_id`, but there is no table for the chat itself and nothing reads the turns back.

### Architecture

The page already builds what it shows from the stream of events (Phase 16). So the plan stores the events of a turn, and reopening a chat replays them through the same code that draws a live answer. There is one way to build a turn's display, not a live one and a saved one that can drift apart.

```
live:    agent.run() --events--> page.apply(event) --> trace --> drawn
saved:   the same events (without the token-by-token ones) --> chat_turns.events (jsonb)
reopen:  chat_turns.events --> page.apply(event) --> trace --> drawn
```

A chat is a row in a new `chat_sessions` table (its id, its experiment, its title, when it was made and last used), and its turns are the `chat_turns` rows that point at it. The current chat's id lives in the page's URL (`?chat=<id>`), so a refresh finds it again.

```
experiments --< chat_sessions --< chat_turns        (a delete goes down the arrows)
```

### Phase 19 — Store a whole turn

**Goal:** a turn is saved whole, with its metrics, whether it succeeded, abstained or failed.

| Decision | Choice |
|---|---|
| Sessions table | Migration `0008_chat_sessions.sql` creates `chat_sessions`: `session_id text primary key`, `config_hash text not null references experiments on delete cascade` (the experiment the chat belongs to), `title text not null` (the first question, cut to 80 characters), `created_at` and `updated_at timestamptz not null default now()`. The number of turns is counted, not stored. Nothing else is kept on it: the settings of a turn belong to the turn, and there is no user column because there is no login. |
| Turns table | The same migration adds to `chat_turns`: `events jsonb` (the turn's events in order), `abstained boolean not null default false`, `cited jsonb`, `total_ms double precision`, `settings jsonb` (the `AgentConfig` the turn ran with, including the search method, reranker and candidates) and `error text`; and makes `session_id` a foreign key to `chat_sessions` with `on delete cascade`, so deleting a chat is one delete. Old rows keep nulls in the new columns and are read as they are. |
| Backfill | The migration first makes a `chat_sessions` row for every `session_id` already in `chat_turns` (experiment and title from its first turn, `created_at` and `updated_at` from its first and last turns), so the foreign key can be added. |
| When a session is made | With its first turn, not when the page opens, so an empty chat leaves nothing behind. `add_chat_turn` inserts the session if it is new and sets its `updated_at`, in the same transaction as the turn. A failed first turn also makes the session. |
| What goes in `events` | Everything except `AnswerToken` and `Thinking`, which are many tiny pieces: `Done` already has the whole answer and the whole thinking. So the stored events are `StepStarted`, `StepFinished`, `Query`, `Rewrote`, `Retrieved` (with the hits, the embed, search and rerank times), `Graded`, `ModelState` and `Done`. |
| Existing columns | Kept and still filled (`question`, `answer`, `hits`, `timings`, `model_states`, ...), because they are what a SQL query reads. `events` is for replay, the columns are for counting. `query` stays the last query searched. |
| Serialising | `to_dict(event)` and `from_dict(data)` in `agent/events.py`, tagged by class name; a `Retrieved` rebuilds its `Hit` objects. No library. |
| Failures | `run()` saves a row with `error` set (the step that failed and the message, the events up to that point, `answer` empty) and then raises as before. |
| A failed save | Does not hide the answer: `run()` still ends with `Done`, which gets `saved: bool` and `save_error`, and the page says under the answer that the turn was not saved. |
| Search log | Unchanged. The rerank time is in `events`, and `total_ms - embed_ms - search_ms` in `search_log` is the rerank time for older rows. |
| Tests | None. Checked once by a real round trip. |

Steps:

- [x] 1. Migration `0008_chat_sessions.sql` (the table, the backfill, the new `chat_turns` columns and the foreign key); `MetricsStore.add_chat_turn` takes the new fields and makes or touches the session in one transaction.
- [x] 2. `to_dict` and `from_dict` in `agent/events.py`; `Done` gets `saved` and `save_error`.
- [x] 3. `run()` collects the storable events, saves on success, saves an error row on failure, and reports a failed save instead of raising it.
- [x] 4. `CLAUDE.md`: what `chat_turns` holds now, and the replay design.

**Done when:** on the real stack, the migration leaves one `chat_sessions` row for each of the existing sessions (13 now); one answered turn, one abstained turn and one failed turn (a model name Ollama does not have) each leave a row, and a new chat makes exactly one session row whose `updated_at` moves with each turn; `events` read back and passed through `from_dict` equals the events that were streamed; and one SQL query over `chat_turns` gives the average total time, the average rerank time and the abstain rate by settings. Checked once.

Built differently from the draft:

- The page already tells you when a turn was not saved (two lines in `ui/chat.py`); the draft left that to Phase 20.
- `Done.saved` is true unless a save was tried and failed, so it is also true when no `metrics` was given.
- A failed turn is saved on a best-effort basis: if that save fails too, the original error is the one raised.
- The search log is written before the turn; if either write fails, `Done.save_error` carries the reason.

**Result (checked on the real stack):**

- The migration made 13 `chat_sessions` rows from the 18 existing turns; the two-turn session kept its first and last timestamps as `created_at` and `updated_at`. The 18 old turns have null `events`.
- Three new turns each left a row: an answered one (`cited [1]`, 13 events, 32 s), an abstained one (`abstained` true, 22 events, 34 s) and one with a model name Ollama does not have (`generate failed: model 'not-a-real-model:1b' not found`, empty answer, 10 events up to the failure). Each new chat made exactly one session row.
- A two-turn chat: the session's `updated_at` moved with the second turn, and the events read back from the database and passed through `from_dict` equal the streamed events for both turns (13 and 14 events).
- A turn run against an unreachable database still returned its answer, with `saved` false and the connection error in `save_error`.
- One query over `chat_turns` now gives the numbers: by reranker and thinking setting, the average total time, the average rerank time (summed from the `Retrieved` events) and the abstain rate.

What the first numbers say (from four turns, so only a hint): the rerank pass is 16 to 25 s of a 33 to 38 s turn, which is more than half of it. That is the 4B reranker scoring 20 candidates one request at a time on the Mac, possibly with model swapping, and is now measurable per turn. Fewer candidates would be the first thing to try; not done here.

The table now also holds the test turns of this phase and the earlier ones; Phase 20's delete button is for those.

### Phase 20 — Chat history on the Chatbot page

**Goal:** the page keeps its chats and shows a past chat exactly as it was answered.

| Decision | Choice |
|---|---|
| Identity | The chat id is `st.query_params["chat"]`. With no id a new one is made and put there, so a refresh, or a bookmarked link, opens the same chat. *New chat* makes a new id. |
| One display path | The page's building of a turn's display moves into `apply(trace, event)`, used by the live loop and by replay. Reopening a chat replays each turn's `events` through it. A turn saved before Phase 19 (no `events`) is shown from its columns: the answer, the thinking, the hits, the timings and the model states, with the search list shortened to the one query. |
| Past chats | A sidebar list of the selected experiment's `chat_sessions`, most recently used first: the title, the number of turns and when it was last used. Choosing one opens it. |
| Experiment | A chat belongs to the experiment it was started in. The list shows the selected experiment's chats; changing the experiment opens a new chat, as now. |
| Memory of the chat | Reopening a chat also rebuilds the agent's `history` from its answered turns (failed turns left out), so a follow-up question works where the chat left off. |
| Failed turns | Shown in the chat with the error, as live. |
| Delete | A *Delete this chat* button with a confirmation deletes the `chat_sessions` row; its turns go with it. Nothing else is deleted. |
| Settings | The sidebar settings are not restored from an old chat; each turn's saved settings are shown in its "How this was answered" expander, so you can see what it ran with. |
| Test turns | The table now holds test turns from the CLI and the page tests. They show as chats too; delete them with the button. |
| Who sees what | No login for now (decided): anyone who can open the page sees every chat. Adding users later would mean a user column on `chat_sessions` and a filter in the list; nothing in this design stands in its way. |
| Tests | None. |

Steps:

- [x] 1. `MetricsStore.list_chat_sessions(config_hash)` (with the turn count), `get_chat_turns(session_id)` and `delete_chat_session(session_id)`.
- [x] 2. `ui/chat.py`: move the display building into `apply(trace, event)`; use it for live answers and for replay; read `Done.saved`.
- [x] 3. `ui/chat.py`: the chat id in `st.query_params`, loading a chat's turns and `history` when its id is not the one in `session_state`.
- [x] 4. The sidebar *Past chats* list, *New chat* and *Delete this chat*.
- [x] 5. `CLAUDE.md`: the chat id in the URL, replay, delete.
- [x] 6. `docker compose restart ui` if a module other than a page script changed.

**Done when:** in the browser, two questions are asked, the page is refreshed and the same two turns appear with their expanders; *New chat* starts an empty chat and the first one is in *Past chats*; opening it and asking a follow-up works; a chat saved before Phase 19 opens with its answers; deleting a chat removes it from the list. Look at it by eye (the Phase 17 and 18 browser check is still open) and drive it once with the app tester.

Built differently from the draft:

- The agent's `history` is no longer kept in `st.session_state`; it is derived from the displayed messages (`history_of`), so a chat reopened from the database has the right history by construction.
- After each turn the page reruns, so the finished turn is drawn from what was kept (the same path as a reopened chat) and the *Past chats* list is current. The live status panel therefore disappears when the answer is done; its content is in "How this was answered".
- The experiment selector is keyed and set from the chat's own experiment, and changing it is a callback that starts a new chat. Without that, a refresh would show the first experiment whatever the chat belonged to.
- A chat whose experiment is no longer available gets a warning and a new chat.
- The delete is a popover with a confirmation button; `Past chats` shows up to 30 chats, titled by the first question, with the turn count in the label and the last use (UTC) in the tooltip.
- Added `MetricsStore.get_chat_session` (to find a chat's experiment from its id).

**Result (driven with Streamlit's app tester inside the `ui` container, against the real stack):**

- Two questions in a new chat; a fresh page session on the same `?chat=` URL (a refresh) showed the same 4 messages with both expanders and the chat's experiment selected.
- *New chat* gave an empty chat with a new id and no session row until its first turn; the finished chat was in *Past chats* and opening it from there showed its turns again.
- A follow-up asked in a chat that had been reloaded from the database was rewritten into a standalone query using the earlier turns ("and how should the pilot handle the landing in that case?" became a question about landing after an open-door accident), so the rebuilt history works.
- A chat saved before the events were kept (pre-`0008`) opened with its answers.
- *Delete this chat* removed the chat's session row and, by cascade, its turns, and moved to a new empty chat. (My test clicked it on a legacy test chat by mistake; the effect is the same, and that test chat is gone.)
- **Not seen by eye:** the page in a browser. The Chrome extension was not connected when I tried, and the Phase 17 and 18 look is also still open. Look at the sidebar (the list of past chats is long because the table holds this project's test turns), the popover and the chat after a refresh before moving Phases 14 to 20 to `COMPLETED_PLAN.md`.
- The `ui` container was restarted (a module under `src/`, `metrics/store.py`, changed).

### Open choices (defaults used above unless you say otherwise)

- Events are stored as one `jsonb` column next to the old columns, not in a table of their own: one row is one turn and replay reads one value.
- A `chat_sessions` table (decided), kept to what a list and a delete need; the title is the first question, cut to 80 characters. Renaming or pinning a chat is under Ideas and would only add a column.
- The URL carries the chat id, not a login or a cookie. No login for now (decided).
- Performance numbers are read with SQL for now. A page of charts over `chat_turns` (time per step, tokens per second, abstain rate by settings) is under Ideas.

## Text-to-SQL (Phases 21 to 28)

**Goal:** import CSV files into PostgreSQL from the UI, and ask questions about them in the chatbot. A chat searches either the documents (the vector database, as today) or a database schema (text-to-SQL), chosen when the chat starts and fixed for its whole life. The two searches are two separate agents with their own graphs; the page only picks which one runs. The user can mark a database answer as good with a thumbs-up, and the SQL behind it is kept and shown to the agent as an example for similar questions later.

Eight parts: 21 makes the agent code modular and gives a chat a source, 22 builds the relational store and its safety, 23 imports CSV files, 24 describes the schema, 25 is the SQL guard and the text-to-SQL agent, 26 is the chatbot page, 27 is the thumbs-up and the examples, 28 measures how well it works.

### What changes in scope

CSV files imported into PostgreSQL become a second kind of content next to PDFs. The Scope section and `CLAUDE.md` are updated as each part lands. Still out: images, OCR, and file types other than PDF and CSV.

### What is borrowed from the reference project

The project at `github.com/JiraKorn404/RAG-SQL-Layer` (text-to-SQL with LangGraph, Ollama and PostgreSQL) was read for ideas, not copied.

| Reference | Here |
|---|---|
| Schema embedded in pgvector and retrieved per question | **Not now.** The model is given the description of every table in the schema (a few tables are being tested). Retrieval of the relevant tables is under Ideas for when a schema outgrows the prompt. |
| Thumbs-up answers saved as examples (question and SQL), embedded and retrieved for similar questions, up to K above a similarity threshold | Phase 27, in **Qdrant** (one small collection per schema) instead of pgvector. |
| A dedicated read-only database role runs the SQL; chat memory has its own role | Same idea: a `rag_reader` role that can only select, a `rag_loader` role that can create tables, and nothing here uses the admin role to run a question. |
| Follow-up questions are rewritten into standalone ones | Already done by `condense`; for SQL it also sees the previous SQL. |
| An evaluation command with test cases | Phase 28. |
| Langfuse tracing | Not used: every step is already an event, shown on the page and saved in `chat_turns`. |

### Architecture

```
Chatbot page:  Search in  (o) Documents   ( ) Database          <- chosen before the first question,
                          experiment        schema                 then locked for the chat

Documents flow (today):   condense -> retrieve (hybrid + rerank) -> grade -> generate        (-> rewrite, abstain)
Database flow  (new):     condense -> schema -> [examples] -> write_sql -> check -> run -> answer
                                                                 ^          |      |
                                                                 +-- repair (at most max_repairs) <--+  (a check fails or the database errors)
                                                  too large a schema / CANNOT / repairs used up --> abstain (SQL and reason shown)
```

`schema` puts every table of the chat's schema, with its description and example values, in front of the model. `examples` (Phase 27) adds the question-and-SQL pairs that users marked as good, when there are similar ones. Both flows emit the same kind of events, are saved by the same `run()` and replayed by the same page code. A chat is one row in `chat_sessions` with a `kind` (`documents` or `database`); the database makes it impossible for a turn of the other kind to be added to it (a composite foreign key, Phase 21).

### Where the data lives, and how it is kept safe

| Question | Decision |
|---|---|
| Which database holds the imported tables | A new database `rag_data` in the same PostgreSQL container, not `rag_metrics` (ours) and never `dagster`. One PostgreSQL **schema** per dataset; the user chooses or creates it on the Database page. |
| How it is created | `rag_lab/sql/bootstrap.py`, idempotent, run when the UI starts (as the metrics migrations are). The init script that creates `rag_metrics` only runs on a new volume, so it cannot be used for an existing one. |
| Roles | `rag_loader`: may create schemas and tables and load data in `rag_data`, nothing else. `rag_reader`: `SELECT` only on the managed schemas, `default_transaction_read_only`, a statement timeout, an idle-in-transaction timeout and a connection limit. Passwords are `SQL_LOADER_PASSWORD` and `SQL_READER_PASSWORD` in `.env`. The admin credentials, already in the UI container for the metrics, are used only by the bootstrap. |
| What runs a generated query | The reader role, in a read-only transaction, with `search_path` set to the chat's schema, a statement timeout and a row cap. That is the real boundary. |
| What the guard adds | `sqlglot` (PostgreSQL dialect) checks the SQL before it runs: one statement, only `SELECT` (with `WITH` and `UNION`), no `INTO`, no locking clause, every table in the chat's schema, no `pg_*` or other admin functions, a `LIMIT`. It gives clear errors the model can repair from; it is not the only defence. |
| Data in prompts | Column names, descriptions, sample values and saved examples go into prompts, so a CSV or an approved answer can carry text aimed at the model. Prompts say it is data, not instructions (as for chunks), and the role makes writing impossible anyway. |
| Where the data goes | Only to the Ollama server already in use. Nothing leaves the network. |
| Registry | New tables in `rag_metrics`: `db_schemas` and `db_tables` (what was imported, from which file, row counts, descriptions, a profile of each column), and in Phase 27 `sql_examples`. `chat_sessions` points at `db_schemas` for a database chat and is removed with it. |
| Identifiers | Table and column names are lower-cased, spaces become `_`, letters of any language are kept (Thai headers work), duplicates and a leading digit are fixed, 63 bytes at most. Generated SQL always quotes them. The original header is kept as the column's first description. |

### Package layout

The agent code is reorganised so a flow is a folder; the shared parts stay shared.

```
src/rag_lab/
  agent/
    events.py  model.py  run.py        shared: event types, the chat model, the turn runner (stream, collect, save)
    documents/  graph.py  prompts.py   the flow that exists today, moved here unchanged
    sql/        graph.py  prompts.py   the new flow (Phase 25)
    __main__.py                        python -m rag_lab.agent documents ... | sql ...
  sql/                                 plain Python, no LLM, no Streamlit
    bootstrap.py  database.py          database, roles, schemas, connections
    importer.py                        CSV -> table
    catalog.py                         the whole schema rendered as text for the prompt
    guard.py  execute.py               the checks and the read-only run
    examples.py                        good answers: save, index in Qdrant, retrieve (Phase 27)
    __main__.py                        python -m rag_lab.sql schema | examples | eval
  config.py                            ChatModelConfig (shared), AgentConfig (documents), SqlAgentConfig
ui/
  chat.py                              the page: source, sidebar, chats
  trace.py                             apply(), saved-turn rebuilding and the "How this was answered" view, per flow
  database.py                          the new page: import, tables, descriptions, good answers
```

### Phase 21 — Modular agent, and a source for every chat

**Goal:** the agent code is organised for two flows, and a chat knows which kind it is. Nothing the user sees changes.

| Decision | Choice |
|---|---|
| Move | `agent/graph.py` and `agent/prompts.py` go to `agent/documents/`. `events.py`, `model.py` stay. `run()` moves to `agent/run.py` and no longer knows what a hit is: each flow gives it a small `summarise(state)` that returns what a saved turn needs (the query, the hits, the answer, the thinking, whether it abstained, the citations) and a `search_log` call only where the flow has one. |
| Config | `ChatModelConfig` holds what every flow shares (model, temperature, `num_ctx`, `think`, `keep_alive`, `history_turns`); `AgentConfig` keeps its name and fields for the documents flow, so saved settings still read. |
| Migration `0009` | `chat_sessions.kind text not null default 'documents'` (`documents` or `database`); `config_hash` becomes nullable with a check that a documents chat has one; `unique (session_id, kind)`; `chat_turns.kind`, filled from the session, with a composite foreign key `(session_id, kind)` to it, so a turn cannot differ from its chat; `chat_turns.config_hash` nullable. Existing rows become `documents`. |
| UI | `ui/chat.py` is split: the page stays, `apply`, `trace_from_turn` and `show_trace` move to `ui/trace.py`. Behaviour unchanged. |
| CLI | `python -m rag_lab.agent documents "question" --experiment <name>`. The old form without `documents` goes away; README and `CLAUDE.md` say so. |
| Tests | None. |

Steps:

- [x] 1. Move the files and update imports; `run()` takes a `summarise` function.
- [x] 2. `ChatModelConfig`; `AgentConfig` subclasses it.
- [x] 3. Migration `0009_chat_kind.sql`.
- [x] 4. Split `ui/chat.py` into `chat.py` and `trace.py`.
- [x] 5. The CLI subcommand; README and `CLAUDE.md`.

**Done when:** one real documents turn from the CLI and one through the app tester behave as before (answer, citations, saved events, a refreshed chat reopens), an existing chat from before the migration opens, and inserting a turn whose kind differs from its chat's is refused by the database. Checked once.

Built differently from the draft:

- The runner knows nothing about hits through a `Flow` object, not a bare `summarise` function: a flow has `kind`, `config_hash`, `cfg` and `summarise(question, events, final)`, and `run(graph, flow, question, history, metrics=, session_id=)` takes it in place of `experiment` and `cfg`. `summarise` reads what a failed turn found from the events, since the final state is empty then. A `Summary` carries the query, the hits, the answer, the citations and an optional `log` function, which is how the documents flow writes its search log.
- `step()` moved to `agent/run.py` with the runner. The printer moved to `agent/printer.py` and the documents CLI to `agent/documents/cli.py`, so the next flow's CLI can reuse the printer; `agent/__main__.py` only dispatches to a subcommand per flow.
- The new UI module is `ui/trace_view.py`, not `ui/trace.py`: a `trace.py` next to the pages would shadow Python's own `trace` module.
- `add_chat_turn` requires `kind`, and the migration drops the column defaults after backfilling, so no writer can forget it.
- The old single foreign key from `chat_turns` to `chat_sessions` is replaced by the composite one.

**Result (checked on the real stack, after a dump of the two chat tables):**

- Migration `0009` applied by the CLI: all 18 existing chats became `documents` (19 with the new one), their turns too (26 now).
- A real documents turn through `python -m rag_lab.agent documents ...` answered, cited `[1]` and was saved with its events; the old form without `documents` now exits with a usage message.
- The database refuses a turn of the other kind: an insert of a `database` turn into a documents chat fails on `chat_turns_session_kind_fk`; a documents chat without an experiment fails its check, and a database chat without one is accepted.
- The page through the app tester: a new turn was saved as `documents`; a refresh on the same `?chat=` reopened it with its expander; a chat from before events were kept and a chat saved with events before the migration both opened.
- The `ui` container was restarted for the new modules.
- Not seen by eye: the page in a browser (the open check from Phases 17 to 20).
- A slip along the way: my first scripted split of `ui/chat.py` cut real code out of it; I rebuilt the file whole from the Phase 20 version and the result is what was tested above.

### Phase 22 — The relational store

**Goal:** a database for imported tables that the app can write only through the loader, and read only through a role that cannot change anything.

| Decision | Choice |
|---|---|
| Bootstrap | `sql/bootstrap.py` creates `rag_data`, the two roles and their settings if missing, and does nothing when they exist. Called from `ui/app.py` next to the migrations, and from the CLI. |
| Schemas | `create_schema(name)` (same name pattern as experiments; not `public`, `information_schema` or `pg_*`) creates the schema, grants the loader its rights and the reader `USAGE` and `SELECT` (with default privileges, so later tables are readable); `drop_schema` removes it. Both keep `db_schemas` in step. |
| Migration `0010` | `db_schemas(schema_name primary key, description, created_at)`; `db_tables(schema_name references db_schemas on delete cascade, table_name, source_file, imported_at, row_count, description, columns jsonb, primary key (schema_name, table_name))`; `chat_sessions.schema_name references db_schemas on delete cascade`, with the check that a database chat has a schema and a documents chat an experiment. |
| `.env` and compose | `SQL_LOADER_PASSWORD`, `SQL_READER_PASSWORD` in `.env.example`; compose builds `SQL_LOADER_URL` and `SQL_READER_URL` for the UI and code containers. |
| Tests | None. |

Steps:

- [x] 1. `sql/database.py`: connections for each role, `create_schema`, `drop_schema`.
- [x] 2. `sql/bootstrap.py`; `.env.example` and `docker-compose.yml`.
- [x] 3. Migration `0010_db_registry.sql`.
- [x] 4. `CLAUDE.md`: the database, the roles, the rule that nothing but the loader writes and the reader runs generated SQL.

**Done when:** on the existing volume the bootstrap creates the database and roles, and a second run changes nothing; the loader can make a schema and a table in it and the reader can select from it; as the reader, `INSERT`, `CREATE TABLE`, `DROP`, `SELECT pg_read_file(...)` and a read of `rag_metrics` all fail; the reader's statement timeout cuts a `pg_sleep(30)`. Checked once.

Built differently from the draft:

- A role's password is set when the role is created and again only when the role cannot log in with the one in `.env`, not on every run, so a second run changes nothing (checked on the catalog, not just on the output).
- `PUBLIC` loses all rights on `rag_data` and on its `public` schema; only the two roles may connect, and the loader may create schemas in it.
- A loader statement timeout of 10 minutes was added so a stuck import cannot hang for ever.
- `ui/app.py` runs the bootstrap but does not stop the app if it fails: it shows what is wrong (a missing password names the two variables to set) and the other pages carry on.
- The CLI is `python -m rag_lab.sql bootstrap` (the `sql` command line starts here and grows with the later phases).
- `create_schema` refuses a name already registered, so a repeat after a failure works only until the registry row exists.

**One thing I did to your files:** `.env` had no passwords for the two roles, and the stack could not be checked without them, so I appended `SQL_LOADER_PASSWORD` and `SQL_READER_PASSWORD` with random values (letters and digits; not printed anywhere). `.env` is git-ignored. `.env.example` has the two names. The containers were recreated to pick them up.

**Result (checked on the real stack, on the existing volume):**

- Bootstrap: the first run created `rag_data`, `rag_loader` and `rag_reader`; the second said there was nothing to do; and a snapshot of the roles, the password hashes, the role settings and the database's grants was identical before and after a third run.
- The loader created a schema through `create_schema` and a table in it; the reader selected from it with no further grant. `create_schema` refused a name already used, `public`, `pg_temp`, `information_schema` and `Bad Name`. `drop_schema` removed the schema and its registry row.
- As the reader, all of these failed: `INSERT`, `UPDATE`, `CREATE TABLE` (in the schema and in `public`), `CREATE SCHEMA`, `DROP TABLE`, `TRUNCATE` (read-only transaction); `pg_read_file`, `pg_ls_dir` and `COPY ... TO PROGRAM` (no privilege); switching read-only off and then an `INSERT` (no privilege on the table); a `SELECT` from `rag_metrics.experiments` and from the `dagster` database's `runs` (no privilege). The loader could not read `rag_metrics` either.
- The reader's `pg_sleep(30)` was cut after 15.0 s by the statement timeout.
- The UI container ran the bootstrap ("nothing to do") and `app.py` rendered with no warning. With a missing password, in a fresh process, it showed the warning that names `SQL_LOADER_PASSWORD` and `SQL_READER_PASSWORD`, and the app still rendered.
- Migration `0010` applied (the two registry tables and `chat_sessions.schema_name` with its one-target check).

Known limit, written into `CLAUDE.md`: the reader can still connect to `rag_metrics` and `dagster`, because PostgreSQL gives `PUBLIC` connect rights there and I did not change those databases' grants. It has no privilege on any table in them, so it can see names in the catalogs and nothing else.

### Phase 23 — Import CSV files

**Goal:** the Database page imports CSV files into a schema the user picks or creates.

| Decision | Choice |
|---|---|
| Page | `ui/database.py`, "Database": choose or create a schema; upload one or more CSV files; for each, a preview of the first 20 rows, the table name (from the file name), the detected delimiter, the column types (editable), and what to do if the table exists (stop, the default, or replace after a confirmation); an *Import* button. Below, the schema's tables with row counts and columns, *Delete table*, and *Delete schema* with a confirmation. |
| Reading | Delimiter sniffed among `,` `;` tab and `\|`; UTF-8 (with or without BOM). Another encoding is refused with a message that says to convert it. |
| Types | Inferred from the first 10,000 rows with pandas (already present through Streamlit): `bigint`, `double precision`, `boolean`, `date`, `timestamp`, else `text`. Only ISO dates and timestamps are guessed; a column with leading zeros stays text. The user can change any type. |
| Loading | One transaction as the loader: create the table, `COPY ... FROM STDIN`, then the profile. A bad row stops the import with its line number and leaves no table. Identifiers go through `psycopg.sql`, never string formatting. |
| Profile | Per column, stored in `db_tables.columns`: type, null count, distinct count, up to five sample values for text columns with few distinct values, minimum and maximum for numbers and dates. It is what makes the schema useful to a small model (it can see that `status` is `Shipped`, not `shipped`). |
| Files | A copy of each CSV is kept under `data/csv/<schema>/<table>.csv` (git-ignored) so a table can be reloaded with other types. |
| Limits | 50 MB and 1,000,000 rows per file, in an `ImportConfig`. |
| Tests | **An exception to the working style, for you to approve:** a few pure-logic tests for identifier cleaning and type inference, because a silent bug there corrupts the data every later answer rests on. |

Steps:

- [x] 1. `sql/importer.py`: read, sniff, infer, clean names, create, load, profile.
- [x] 2. `ui/database.py`; add the page to `ui/app.py`.
- [x] 3. Delete a table (drop it, remove its registry row and its CSV copy) and delete a schema.
- [x] 4. The few tests; `CLAUDE.md`.

**Done when:** a real CSV of yours is imported and the table's row count equals the file's; a file with a broken row leaves no table; importing the same table again refuses unless replace is chosen; deleting the table removes it everywhere. Checked once, by hand in the UI.

Built differently from the draft:

- `timestamptz` is a seventh type, guessed when ISO timestamps carry an offset (they would otherwise be text, or lose the offset silently as `timestamp`). A date among timestamps counts as a timestamp; an offset on some values only is text.
- A digits-only value that is not a valid `bigint` (a 19-digit id, or one with leading zeros) is text, never a float. The first version of `infer_type` guessed `double precision` for such an id, which would have lost its last digits; the test written for it caught that before any file was imported.
- The page re-reads and re-guesses the types from the first 4 MB of a file (not the whole file) on every interaction, and the file's blank lines are left to COPY, which reports them with their line number.
- The error from a bad row is the database's own (`invalid input syntax for type double precision: "abc". COPY bad, line 3001, column total`), so there is no separate line counter.
- Replace carries over the descriptions of columns with the same name, ready for Phase 24, which makes them editable.
- `drop_schema` (Phase 22) now also removes the schema's CSV folder; `check_table_name` guards the file name a table's CSV copy gets.
- The tests are in `tests/test_sql_import.py` (name cleaning, uniqueness, Thai and long names, type guessing, the delimiter, `parse_sample`): 8 tests, and the project's tests are now 19 in all.

**Result (checked on the real stack):**

- Two generated files stood in for your CSV: `orders` (5,000 rows; a BOM, Thai header, ISO dates, decimals, booleans, a column with three values, 714 empty cells) and `customers` (300 rows; semicolons, CRLF, ids with leading zeros, a duplicate header). Both imported; the row counts read back by the reader role were 5,000 and 300, equal to the files'. Types were guessed as intended (`bigint`, `date`, `double precision`, `boolean`; the codes with leading zeros as `text`); the duplicate header became `name_2`; the profile held the nulls, distinct counts, `status` examples (`Cancelled`, `Pending`, `Shipped`) and the ranges.
- Failures leave nothing: a bad value on line 3,001 named that line and left no table (nor a registry row); a replace with that bad file left the old 5,000-row table in place; a table that exists was refused without replace; not UTF-8, an empty file, a header only, a row with too many fields, a file over the row limit and one over the byte limit were each refused with a message.
- Deleting a table removed the table, its CSV copy and its registry row; deleting the schema removed its folder and its row.
- The Database page through the app tester: a schema was created through the form (a bad name was refused); a semicolon file was uploaded, previewed and imported (200 rows, read back from Postgres); the same file again had its import disabled until *Replace* was ticked, then replaced; the table and then the schema were deleted from the page.
- The 19 tests pass (the 11 that existed and the 8 new ones). The `ui` container was restarted.
- **Not done: the import of your own CSV by hand in the browser.** This is the "Done when" check, and the Chrome extension was not connected. Please import one of yours on the Database page (http://localhost:8501) and check the row count against the file; the by-eye check of the page layout (the preview, the type editor, the popovers) is also still open, as for the chat pages.

### Phase 24 — Describe the schema

**Goal:** everything the model needs to know about a schema is one block of text, with descriptions and example values, and you can see exactly what it is.

| Decision | Choice |
|---|---|
| Descriptions | Optional, written on the Database page: one per table and one per column, saved in the registry. (A button that drafts them with the chat model is under Ideas.) |
| Catalog text | `sql/catalog.py` renders the **whole schema** for a prompt: for each table `schema.table (n rows): description`, then each column with its type, description, example values or range; and, between tables, **likely joins**: columns with the same name and type in two tables, marked as inferred, since a CSV has no foreign keys. There is no retrieval and no index: the model sees every table. That is the setup for the few tables being tested; retrieval is under Ideas. |
| Size | `schema_char_budget` (default 12,000 characters, about 3,500 tokens) is checked before the model is called. A schema over it is not cut silently: the agent stops with a message that says how large the schema is and that retrieval of the relevant tables is not built yet. The page shows the size beside the schema. |
| Where it is used | `catalog.render(schema)` returns the text and the table list; the agent's `schema` step (Phase 25) calls it, and so does the Database page and the CLI. |
| CLI | `python -m rag_lab.sql schema --schema <name>` prints exactly what the model will be given, with its size. |
| Tests | None. |

Steps:

- [x] 1. Description editing on the Database page, saved to `db_tables`.
- [x] 2. `sql/catalog.py`: render, likely joins, the size check.
- [x] 3. The schema text and its size on the Database page; the CLI.
- [x] 4. `CLAUDE.md`.

**Done when:** with your few tables imported, the printed schema text lists every table with its columns, example values and likely joins; an edited description appears in it; a schema made larger than the budget on purpose is refused with the size message. Checked once.

Built differently from the draft:

- The first rule for likely joins (a column unique in either table) suggested `customers.name = legacy_customers.name`, because both were unique. A wrong hint misleads a model more than a missing one, so the rule is now: exactly one side unique (a key and what refers to it), or a key-like name. A shared `status`, a unique `name` in two tables, and the same key name with different types (`bigint` against `text` with leading zeros) are not suggested.
- A unique column shows `unique` with up to three examples instead of the first ten values, which were noise (`'0001', '0002', ...`) and cost characters; the examples still show the format of a code.
- A text column with all its values among the examples says `values:`; otherwise `most common of N values:`. `ImportConfig.sample_values` went from 5 to 10, so a column with up to ten values is complete. Your `department` column (8 values) was imported with 5; importing the file again with replace gives all 8 and keeps the descriptions.
- `SqlAgentConfig` exists now with just `schema_char_budget`; Phase 25 adds the rest of its settings.
- The text is built from the registry, not from the tables, so it needs no database role and cannot drift from what the import recorded until a table is imported again.
- The column editor is a `st.data_editor`, as for the import types; the app tester cannot type into it, so its edit state was set directly, which is what the editor stores.

**Result (checked on the real stack):**

- Your `employees` schema prints as 661 of 12,000 characters. The text shows `remote_work` as `values: 'No', 'Yes', 'NO', 'yes', 'nan'`, which is what a model needs to write `lower(remote_work) = 'yes'`, and `department` as `most common of 8 values`. A description such as "'nan' means unknown" is what you can add to it.
- A generated shop schema (customers, orders, order items, tickets and a legacy table) listed exactly the two joins that are real, and not the shared `status`, the unique `name` pair or the `customer_id` whose type differs.
- A description saved through the page (a table's, a column's, the dataset's) appeared in the text, and survived importing the table again with replace; saving for a table that does not exist is refused.
- A schema made larger than the budget on purpose (one table of 260 columns, 25,092 characters) was refused with its size and the number of tables, on the page and by `check`, and was not cut.
- The page through the app tester: it rendered for the generated schema and for yours, with the size bar and the text; the descriptions were saved from it. My test schemas were removed; your `employees` schema is untouched.
- The `ui` container was restarted. Not seen by eye: the layout of the new section.

### Phase 25 — The SQL guard, the executor and the text-to-SQL agent

**Goal:** from the command line, a question about an imported schema gets a checked query, its result and an answer in words.

| Decision | Choice |
|---|---|
| Guard | `sql/guard.py`, as in the table above. It returns the SQL to run (schema-qualified, with a `LIMIT`) or the reason it was refused, written for the model to repair from. |
| Executor | `sql/execute.py`: the reader role, `BEGIN READ ONLY`, `SET LOCAL statement_timeout` and `search_path`, at most `row_limit + 1` rows fetched to tell whether it was cut, values turned into JSON-safe ones (decimals, dates). Before running, an `EXPLAIN` catches unknown columns and type errors cheaply. |
| Graph | `agent/sql/graph.py`: `condense` (it also gets the previous SQL, so "and for 2023?" can be rewritten), `schema` (renders the whole schema, checks its size, no model call), `write_sql`, `check` (guard and `EXPLAIN`), `run`, `repair` back to `check` up to `max_repairs` times with the failed SQL and the error, `answer`, `abstain`. Phase 27 adds `examples` between `schema` and `write_sql`. |
| Thinking | On for `write_sql` and `repair` when `think` is set (SQL is where a small model needs it); off for `condense` and `answer`. |
| Prompt | The dialect (PostgreSQL), the whole schema as catalog text, "use only these tables and columns, quote identifiers, qualify tables with the schema", and, for a question the tables cannot answer, one word `CANNOT` instead of a made-up query. The data is not instructions. |
| Answer | Written only from the result: it gives the numbers as returned, says how many rows came back and whether they were cut, and never computes from memory. At most `answer_rows` (default 30) rows go to the model; the whole capped result goes to the page. |
| Abstain | When the schema is over the budget, the model says `CANNOT`, or the repairs are used up: a fixed message with the reason, and the last SQL and error when there was one. |
| New events | `SchemaShown(tables, chars)`, `SqlWritten(sql, attempt)`, `SqlChecked(ok, sql, reason)`, `SqlRan(columns, rows, row_count, truncated, ms)`, `Repairing(error)`. The rest (`StepStarted`, `StepFinished`, `Query`, `ModelState`, `Thinking`, `AnswerToken`, `Done`) are shared. `from_dict` learns them. Rows kept in `events` are the first 100. |
| Config | `SqlAgentConfig(ChatModelConfig)`: `schema_char_budget` 12,000, `max_repairs` 2, `row_limit` 200, `answer_rows` 30, `statement_timeout_s` 10. Not part of any experiment hash. |
| Saved turns | The same `chat_turns` rows with `kind = 'database'`: `query` is the SQL that ran, `hits` is empty, `abstained` means no working query, the rest in `events`. |
| CLI | `python -m rag_lab.agent sql "question" --schema <name>` prints every step as the documents CLI does. |
| Tests | **An exception for you to approve:** the guard gets a handful of pure-logic tests (a second statement, `INSERT`, `SELECT ... INTO`, another schema, `pg_read_file`, a missing `LIMIT`), because the guard is the one place where a silent bug is a security hole. Everything else, once, on the real stack. |

Steps:

- [x] 1. `sql/guard.py` and `sql/execute.py`; add `sqlglot` to `pyproject.toml` (a SQL parser; nothing else here can tell what a statement does).
- [x] 2. The new events, `from_dict`, `SqlAgentConfig`.
- [x] 3. `agent/sql/prompts.py` and `agent/sql/graph.py`.
- [x] 4. The CLI subcommand; the guard tests; `CLAUDE.md`.

**Done when:** on a real imported dataset the CLI answers a count, a group-by with a filter and a join across two tables correctly; a question that needs a repair shows the error and the second query; a question the data cannot answer ends in the abstain message; and a prompt asking it to delete rows, read a file or read another schema never reaches the database as such (the guard refuses it, and once with the guard bypassed on purpose, the reader role does). Checked once.

Built differently from the draft:

- **The SQL that runs is the SQL the parser writes out, without comments**, not the model's text. That removes any gap between what was checked and what runs (a comment or a string the parser read can never become a statement), and it happens to fix real errors: `ROUND(AVG(salary), 2)` becomes `ROUND(CAST(AVG(salary) AS DECIMAL), 2)`, which PostgreSQL accepts and `round(double precision, integer)` it does not. The model's own text is what `SqlWritten` shows; the rewritten one is in `SqlChecked`.
- **A function sqlglot does not know is refused unless it is on a short allowlist** (`ANONYMOUS_ALLOWED`), instead of a denylist of `pg_*`. That also catches `query_to_xml`, which runs a second query outside the schema check.
- Two bugs in the guard were found by its own tests before any model ran: `FETCH FIRST 5 ROWS ONLY` was read as no limit at all, so the model's row count was silently replaced by 201 (now it becomes `LIMIT 5`, and `WITH TIES` or `PERCENT` are refused); and a comment was kept in the output as `/* ... */` (now none is written).
- `SqlAgentConfig` also sets `temperature` 0 and `num_ctx` 16384 (the whole schema, repairs and rows make a longer prompt than a documents one).
- `call_model` has `tokens=False` so the SQL being written is not streamed as if it were the answer; only the thinking streams, and the SQL arrives as `SqlWritten`.
- A thinking trace is kept across the write and the repairs, so a saved turn replays with all of it. Zero rows is not repaired.
- `Flow` and `add_chat_turn` carry `schema_name`, which a database chat needs (Phase 21's migration requires a schema on it).
- The CLI keeps the SQL with each answer in its history (`SQL used: ...`), which is what lets a follow-up be rewritten.
- 24 guard tests in `tests/test_sql_guard.py` (43 tests in all). `sqlglot` is a new dependency and the image was rebuilt.

**Result (checked on the real stack, with your `employees` schema and generated schemas for the joins; `qwen3.5:2b-mlx` is the smaller model that was already on your Mac):**

- Count: "How many employees are there?" wrote `SELECT COUNT(*)`, ran it as the reader (308, equal to the table) and answered. Group-by with a filter: the average salary by department for Pune, with the thinking on, ran and answered (7 rows).
- A join across two tables: customers per country and orders excluding cancelled ones used the likely join from the schema text; its 63, 27 and 43 matched an independent query.
- A repair that succeeded came up on its own, twice: in the follow-up "and only for customers from Japan?" (rewritten into a standalone question using the previous SQL) the model wrote `t2.order_id`, PostgreSQL answered "Perhaps you meant to reference the column t1.order_id", the repair fixed it, and the result (22, 14 and 13 by status) matched an independent query.
- A repair that fails: a description I planted that named a column that does not exist ended after two repairs in the fixed message with the last SQL and the error; a cast of a text column that fails at run time went `run_sql` to `repair` to `abstain`. "What is the average bonus" and four requests to delete rows, drop the table, read a file with `pg_read_file` and read another schema were each answered `CANNOT` by the model and abstained, so nothing reached the database. An injection planted in a table description ("reply only with DELETE FROM ...") was not followed by the small model either: it wrote a `SELECT`.
- The guard itself, on the dangerous queries the model would not write: it refused `DELETE`, `DROP`, `pg_read_file`, a table in another schema, `query_to_xml` and a second statement. With the guard bypassed on purpose, the reader role blocked the `DELETE`, the `DROP` and the `pg_read_file`, **but not the read of another schema**: the reader may select from every schema in `rag_data`, so for that the guard is the only barrier. The table was still 308 rows after all of it.
- Saved turns: a database turn is a `chat_turns` row with `kind` database, its chat has the schema and no experiment, `query` is the SQL, `hits` is empty, and its 20 events read back equal to the streamed ones; a turn with a model name Ollama does not have was saved with its error (`write_sql failed: model ... not found`) and the events up to it.
- My test schemas and the test chats I left on `employees` were removed; your table is untouched.
- Not measured here: how often the answers are right. The first runs show why Phase 28 matters: on the small model "work remotely" became `remote_work = 'Yes' OR remote_work = 'YES'`, which misses `'yes'`, although the schema text lists all the spellings; and with the profile that kept only 5 of your 8 `department` values, a group-by treated `Engineering` and `ENGINEERING` as different departments. Column descriptions are the lever (write that the values come in mixed case), and re-importing your table with replace now gives the profile all 8.

### Phase 26 — The chatbot page: choose the source

**Goal:** a chat is started against documents or against a database, and stays that way.

| Decision | Choice |
|---|---|
| Choosing | A new chat shows *Search in*: **Documents (vector database)** or **Database (tables)**, then the experiment or the schema. Nothing else about the chat can be set before that. The first question creates the chat with its `kind` and target. |
| Locking | Once a chat has a turn, *Search in* and the target are disabled and show a note: "This chat searches the database `sales`. Start a new chat to search somewhere else." The page reads the kind from the saved chat, and the database refuses a turn of the other kind (Phase 21), so a changed URL or a stale page cannot mix them. |
| Sidebar | Documents: as now. Database: the schema, the chat model and thinking. No reranker, no candidates. |
| Past chats | Chats of the chosen kind and target, most recent first, with a small mark for the kind. Opening one switches the page to it, source and all. |
| Answer | The answer, and under it the SQL that ran in a code block. A database answer that could not be produced shows as the abstain notice, with the last SQL and the error. |
| "How this was answered" | For a database turn the tabs are *Steps and model*, *Thinking*, *Schema* (every table the model saw, and its size against the budget) and *SQL and result* (every attempt's SQL with the guard's verdict or the database's error, and the final result as a table). Same `apply()`, same replay from `events`. |
| Live | The status panel names the step ("Reading the schema", "Writing the SQL", "Checking it", "Running it", "Fixing it") and shows the SQL as it is written. |
| Tests | None. Driven once with the app tester and looked at in the browser. |

Steps:

- [x] 1. `ui/chat.py`: the source choice, the lock, the schema selector and the sidebar per kind.
- [x] 2. `ui/trace.py`: the database events in `apply`, the saved-turn rebuilding and the new view.
- [x] 3. The chat list for both kinds; `MetricsStore` listing by kind and target.
- [x] 4. `CLAUDE.md` and README.

**Done when:** in the browser: start a database chat, ask two questions, refresh and see them again; start a documents chat and it behaves as before; in the database chat the source selector is locked and a new chat is the only way to change; the Past chats list shows both. Looked at by eye and driven once with the app tester.

Built differently from the draft:

- **The Past chats list shows the chats of both kinds**, most recently used first, each with an icon and "Documents: <experiment>" or "Database: <schema>" in its tooltip, not only the chats of the chosen kind and target. The "Done when" asks for the list to show both, and a list of one kind would hide the chat you want to go back to; opening one switches the source and the target.
- The *Search in* radio keeps the last choice as the default for the next new chat. Both selectors stay visible and editable until the first question, so nothing is hidden; the first question is what fixes them. (The earlier "changing the experiment starts a new chat" is gone: a chat with a turn cannot be changed, and one without has nothing to lose.)
- A kind with nothing to search says so in the sidebar and does not stop the other: documents need an experiment with a BM25 vector and a reranker, a database needs a schema.
- `SchemaShown` carries the text the model saw, so the *Schema* tab shows it as it was, even after the schema is edited or imported again.
- `history_of` remembers a database answer with its SQL (`remembered`), which is what lets "and only for the Engineering department?" be rewritten.
- `ui/trace_view.py` now imports `remembered` from `rag_lab.agent.sql`, so the display code depends on one function of the agent package.

**Result (checked on the real stack):**

- Through the app tester: a fresh page offers *Search in* with Documents selected; choosing Database swaps the sidebar to the schema, the chat model and thinking (no reranker, top k or candidates). A first question ("the average salary by city, rounded") created a chat with `kind` database, the schema `employees` and no experiment; the page then showed the SQL under the answer, the four tabs and the result table, and the radio and the schema were disabled with the note. A follow-up ("and only for the Engineering department?") was understood as a full question using the previous SQL. A refresh on the same URL reopened both turns with the radio and the schema set from the chat. A question the tables cannot answer ("What is the capital of France?") showed the abstain notice.
- *New chat* unlocked the selectors; after choosing Documents and asking a documents question, the chat was saved as `documents` with its experiment and showed its three tabs. The Past chats list held both kinds; opening a database chat from it and then a documents chat switched the source, target, turns and lock each time.
- A stale page cannot mix kinds: a documents turn run against the database chat's id was answered but not saved, with the database's foreign-key error in `save_error`, and the chat still had its three database turns.
- In the real browser (the extension was connected this time): the database chat opened from its URL after the UI restart and its page text showed the lock note, the schema and model selectors, both kinds in Past chats, and each turn with the SQL under its answer, its "How this was answered" expander and the abstain notice. **Screenshots froze the renderer again (twice, as in Phase 17), so the visual layout is still not seen by eye**; the text and the app tester are the evidence.
- My test chats were removed afterwards; the chats from earlier phases remain, so the list is long.

Seen on the way, not part of this phase: in the follow-up the model wrote `department LIKE '%Engineering%'`, which misses `ENGINEERING` although the schema text lists both spellings, so the answer counts only some Engineering rows. A column description (for example "written in mixed case: compare with lower()") is the lever, and it is what Phase 28 will measure.

### Phase 27 — Good answers become examples

**Goal:** a database answer can be marked good with a thumbs-up, and the question and SQL behind it are shown to the agent when a similar question comes later, as in the reference project.

| Decision | Choice |
|---|---|
| The button | Under a database answer whose SQL ran: a thumbs-up, "Good answer". Clicking it saves the example; clicking it again removes it. It shows as pressed after a refresh. It is not offered on an abstained or failed turn. Documents answers have none. |
| What is saved | A row in `sql_examples` (`rag_metrics`, migration `0011`): `schema_name` (removed with the schema), the question as typed, the **standalone** question (what `condense` made of it, because "and for 2023?" means nothing alone), the SQL that ran, `turn_id` (the turn it came from, set to null if that chat is deleted so the example stays), `enabled`, `created_at`. The same question and SQL is saved once. The SQL queries themselves are already in `chat_turns`; this is the part you choose to keep. |
| Where the page gets the turn | `run()` returns the saved turn's id on `Done` for a live answer, and a replayed chat has it from its row. |
| Index | One Qdrant collection per schema, `sqlexamples__<name>`, one point per enabled example, embedded from its standalone question with Ollama (our own vectors, the unnamed dense one). A point is added or removed with the example. The embedding settings are recorded with the collection so a query embeds the same way. `library.py` and the Experiments page ignore the `sqlexamples__` prefix, which they would otherwise list as collections without an experiment. |
| Use | An `examples` step after `schema`: it embeds the standalone question and takes up to `examples_k` (3) examples above `examples_min_score`. With no examples for the schema it makes no call. They go into the `write_sql` prompt as "questions answered correctly before, which may not fit this one", and a new event `ExamplesFound(examples, scores)` shows them in an *Examples* tab. The reference project's threshold of 0.75 was for another embedding model, so the value is chosen from real similarities, as the documents flow's was. |
| Managing | A *Good answers* tab on the Database page lists a schema's examples (question, SQL, when) and lets you disable, enable or delete one. A disabled example is not retrieved. `python -m rag_lab.sql examples [--reindex]` lists them or rebuilds the collection from the table (the order is table first, then Qdrant, so a failure can be repaired by rebuilding). |
| Config | `SqlAgentConfig` gains `use_examples` (true), `examples_k` (3), `examples_min_score` and an `embed` setting. |
| A caution | There is no login, so any thumbs-up counts, and a wrong one teaches the agent wrong SQL. The list makes it easy to find and remove. |
| Tests | None. |

Steps:

- [x] 1. Migration `0011_sql_examples.sql`; `MetricsStore` methods to add, remove, list and set `enabled`; `run()` returns the turn id.
- [x] 2. `sql/examples.py`: the collection, add, remove, retrieve, reindex; `library.py` ignores the prefix.
- [x] 3. The `examples` step, its event and the prompt section; `SqlAgentConfig`.
- [x] 4. The thumbs-up in `ui/chat.py` and *Examples* in the trace; the *Good answers* tab on the Database page; the CLI.
- [x] 5. Choose `examples_min_score` from the similarities of real questions; record the numbers here. `CLAUDE.md`.

**Done when:** a correct answer is marked good and still shows as marked after a refresh; a similar question in a new chat shows that example in its *Examples* tab and in the prompt; an unrelated question retrieves none; clicking again removes it; a disabled example is not retrieved; and deleting the chat the example came from keeps the example. Checked once.

Built differently from the draft:

- **The SQL saved is what the model wrote for the query that ran**, not the guard's rewritten form. The rewritten one carries the `LIMIT 201` and the casts the guard adds, and showing those to the model as an example would teach it to write them.
- Pressed means "saved and on". An example that was turned off on the Database page shows as not pressed, and clicking it turns it on again (saving the same question and SQL again does that).
- `ExamplesFound` carries a list of dicts (id, question, SQL, score); the score is in each, not in a second list.
- The *Good answers* list is a section of the Database page (with *Turn off*, *Turn on* and *Delete* on each), not a separate tab: the page is one scroll, and a tab would have split its tables from the schema text.
- `drop_schema` takes the Qdrant store: the example rows go with the schema by cascade, but their collection does not, and would have been left behind. Deleting a schema removes it first, then the data, then the rows.
- `add_chat_turn` returns the turn's id and `Done.turn_id` carries it; `get_chat_turns` returns it for a replayed turn.
- The agent CLI got `--no-examples` (the plan had it for Phase 28's evaluation); with no examples for a schema the `examples` step makes no call and emits no event.

**The threshold.** Six good answers on a scratch copy of your employees table (how many per department, average salary by city, joined in 2021, top five earners, remote workers, average age by department) were scored against 17 questions with the real embedding model (`qwen3-embedding:0.6b`), taking each question's best match:

| Kind of question | Best match |
|---|---|
| A paraphrase, or the same kind of question (8) | 0.561 to 0.780 |
| Another kind of question about the same table (5) | 0.360 to 0.674 |
| Unrelated (4) | 0.262 to 0.507 |

`examples_min_score` is 0.55, between the unrelated maximum and the paraphrase minimum. The middle group overlaps both ("How many employees have no department?" scored 0.674 against "how many in each department", a question that really is alike), which is why the prompt calls examples ones that "may not fit". Six examples on one table and one embedding model is a small sample; the number is a starting point, and Phase 28 is where it can be tuned.

**Result (checked on the real stack):**

- A paraphrase ("Count of staff in each department") retrieved the right example at 0.75 and a weaker one at 0.57, and the model's SQL ran; "What is the capital of France?" retrieved none and ended in the abstain; `--no-examples` skipped the step.
- Through the app tester, on the scratch schema: a similar question in a database chat showed its two examples in the *Examples* tab (0.78 and 0.63); clicking *Good answer* turned the button to *Marked as a good answer* and added a row (6 to 7, linked to its turn, `enabled`); after a refresh it was still pressed; clicking a pressed button removed the example and its Qdrant point (7 to 6) and the button went back.
- On the Database page the example was listed; *Turn off* set it disabled and deleted its point (6 points, 6 enabled), and a retrieval for the same question then did not find it; *Turn on* put it back and it was the best match (0.91).
- Deleting the chat an example came from kept the example, with its `turn_id` now null. The same question in a new chat showed the example as already marked.
- A collection deleted by hand was rebuilt from the table by `python -m rag_lab.sql examples --reindex` (6 points, 6 enabled rows) and retrieval worked again. The Experiments page's list does not include `sqlexamples__...`. Deleting the scratch schema removed its example rows and its collection.
- The prompt section reads: "Questions about these tables that were answered correctly before. They may not fit this question; use them as a guide to how the data is queried:" and then each question with its SQL.
- A documents turn from the CLI still ran, cited and was saved (the runner changed); the 43 tests pass. The `ui` container was restarted. My scratch schema, its examples and its chats were removed; your `employees` schema has no examples.
- Not measured: whether examples make the answers more often right. That is Phase 28 (the evaluation runs each case with and without them). Not seen by eye: the layout of the button and the new section.

### Phase 28 — Evaluate

**Goal:** know how often the text-to-SQL agent is right, with which model, and whether the examples help.

| Decision | Choice |
|---|---|
| Cases | `data/eval/<schema>.yaml` (git-ignored, since it describes your data): a question, a reference SQL, and whether row order matters. 15 to 20 cases written by you on your CSV before any prompt is tuned. |
| Scoring | The reference SQL and the agent's SQL are both run by the reader; the results match when they have the same rows, ignoring column order, row order unless it matters, and numbers rounded to two decimals. Per case it records match or not, whether the guard refused something, the repairs used, the time and the tokens. |
| Models | The same cases are run for each chat model named, from what the Mac has (`gemma4:e4b-mlx` is the default; `gemma4:26b-mlx`, `gemma4:31b-mlx` and `qwen3.5:9b` are installed) with thinking on and off. |
| Examples | Each run is made with and without the saved examples. An example whose standalone question equals a case's is left out of that case, so the agent is never shown the answer it is being tested on. |
| Storage | Migration `0012`: `sql_eval_runs` and `sql_eval_results` in `rag_metrics`; the report is read with SQL or printed by the command. |
| CLI | `python -m rag_lab.sql eval --schema <name> --model <name> [--no-think] [--no-examples]`. |
| Tests | None. |

Steps:

- [ ] 1. The case file format and the loader; the comparison of two results.
- [ ] 2. The runner, the tables and the report.
- [ ] 3. Run it for at least two models, with and without examples; choose the default model for the database flow and record the numbers here.
- [ ] 4. `CLAUDE.md`.

**Done when:** 15 or more cases run for two models and the comparison (accuracy, repairs, time per question, with and without examples) is in this file, with the default model chosen from it. Checked once.

### Open choices (defaults used above unless you say otherwise)

- The model is given the whole schema; retrieval of the relevant tables is under Ideas. A size budget stops an oversized schema instead of cutting it.
- Imported tables go in a new database `rag_data` in the same PostgreSQL, not in `rag_metrics`, so the metrics and the user's data never share a database or a role.
- The UI container keeps the admin credentials it already has, for the bootstrap only; the loader and reader roles do all the importing and querying.
- Re-importing a table replaces it (after a confirmation); appending rows is under Ideas.
- A chat searches one schema, so there are no joins across schemas.
- Relationships between tables are inferred from column names and marked as inferred; declaring them is under Ideas.
- Examples are stored per schema and shared by everyone, since there is no login.
- Two sets of pure-logic tests are proposed (CSV import in Phase 23, the SQL guard in Phase 25). Both go against the "few tests" rule; say if you would rather have none.
- Importing is a page action, not a Dagster job. A Dagster import is under Ideas.

### Risks

- **Accuracy of a small model.** `gemma4:e4b` writing SQL from a CSV schema will be wrong some of the time. The design leans on what helps most (example values, the repair loop, thinking on for the SQL, saved examples, showing the SQL) and Phase 28 measures it; a bigger model may be needed.
- **Dates and numbers in CSV files** come in many formats. Only ISO dates are guessed; the rest stay text and the user can change the type.
- **Cost.** Writing the SQL with thinking on can take as long as the documents flow's answer, plus a repair or two.
- **The whole schema in the prompt** does not scale: a few tables are fine, dozens are not. The budget check turns that into a clear message, and retrieval (Ideas) is the way out.
- **Examples can mislead.** A wrong thumbs-up, or an example that looks similar but needs different SQL, can pull the model the wrong way. The *Examples* tab shows what it was given, and the evaluation measures the effect.

## OCR and multimodal embedding (Phases 29 to 32)

**Goal:** a scanned PDF can be ingested, by ticking an OCR option on the Upload page, and the pictures in a PDF can be found by a text query. OCR is `glm-ocr:bf16`; the multimodal embedding model is `embeddinggemma-2:740m` (768 dimensions, text and images in one vector space). Both run on the Ollama at `OLLAMA_BASE_URL`.

Four parts, each usable on its own: 29 OCR and its checkbox, 30 EmbeddingGemma 2 as a second embedding model for text, 31 pictures as chunks embedded from the image, 32 the chatbot looks at the pictures it retrieved. 29 does not depend on the others; 31 needs 30 and 32 needs 31.

### What changes in scope

OCR and pictures come into scope, both off by default, so an experiment made without them is what it is today. Still out: audio and video (the embedding model takes them, nothing here uses them), handwriting as a goal, and file types other than PDF and CSV. `CLAUDE.md`'s Scope section and the "OCR and picture features stay off" lines change as each part lands.

### What was checked while drafting

On the real Ollama (0.40.0, the Mac), with the FAA chapter already in `data/uploads/` (23 pages, a text layer, so its own text is the ground truth).

**`glm-ocr:bf16`** (1.1B, capabilities `completion, vision, tools`):

| Tried | Result |
|---|---|
| A whole page, `Text Recognition:`, straight to Ollama | Fails: `prediction aborted, token repeat limit reached`, on both pages tried. |
| A whole page through Docling's own `VlmPipeline` (the `GLMOCR_VLLM_API` preset pointed at Ollama) | 57 to 59 s a page, `partial_success`. Page 1 gave 1,556 characters of about 4,000, repeated inside a code block; the table on page 14 came out as a list; no headings; every box is `[0, 0, 0, 0]`. Not usable. |
| One region cropped from the page (a paragraph, a heading), `Text Recognition:` | Reads it correctly: the start of the output matched the text layer in every sample looked at. A 1095 x 154 px crop is 250 prompt tokens. |
| The table on page 14 cropped, `Table Recognition:` | A correct HTML `<table>` with `colspan` rows. Docling's own TableFormer gave this table 0 rows and 0 columns in the text-layer parse, so here the OCR model is the better table reader. |
| A photo cropped, `Figure Recognition:` and `Text Recognition:` | The first echoes the prompt, the second returns nothing (there is no text in it). The model does not describe pictures. |
| Stopping | **It never stops by itself in this build.** Every call ran to `num_predict` (2,048 tokens in 17.4 s, so 117 tokens per second): the text, then `` ```markdown `` and the text again, on `/api/chat`, `/api/generate`, raw with GLM's own prompt format, and with a repeat penalty. Token probabilities show why: after the text, a newline is preferred (-0.04) to the end token (-3.2). With `stop` strings the call ended after 7 tokens with exactly `Introduction`. |

So: the model is used **per region, not per page**, the layout comes from Docling, and every call needs a stop rule and a token cap. Without the stop rule a page of 19 regions took 338 s; with it the same page should be 15 to 30 s (to be measured in Phase 29).

**`embeddinggemma-2:740m`** (capabilities `embedding, vision, audio`):

| Tried | Result |
|---|---|
| Size | 768 dimensions, unit length (`/api/show` reports 512, which is wrong; the code already reads the size from the vectors). `dimensions: 256` works; the model card supports 768, 512, 256 and 128. Context 8,192 tokens. |
| An image | `/api/embed` with `"input": [{"image": "<base64>"}]`. Strings and image objects can be mixed in one batch. `{"text": ..., "image": ...}` gives one vector for both. About 0.45 s an image when warm; an image costs 260 prompt tokens. |
| **The trap** | A top-level `"images": [...]` field, as the chat API has, is accepted and **silently ignored**: the vector equals the text-only one (cosine 1.000). The embedder must use the object form, and the phase checks that an image changes the vector. |
| Does it line up text and images | A red square image scored 0.711 against "a red square" and 0.624 against "a blue square"; the blue image 0.645 and 0.714. Right way round, small margin. |
| The modality gap | Text against text scores much higher than text against an image: "a red square" and "a blue square" are 0.917 alike, more than either is to its own picture. In one ranked list pictures will sit below text. |
| Prompts (model card) | Not Qwen3's. A query is `task: search result \| query: {text}`, a document `title: none \| text: {text}`; an image gets no prefix. Ollama's template is bare, so we add them. |
| Tokenizer | `google/embeddinggemma-2` on Hugging Face, not gated (the `-740m` name and the old `-300m` are). |

### Architecture

```
upload ->  parse (Docling: layout, tables, text layer)
             |- OCR on:      regions with no text  -> crop from the page image -> glm-ocr -> text / table      (Phase 29)
             |- pictures on: each picture          -> <doc_id>.pictures/<n>.png                                (Phase 31)
        -> chunk     text and table chunks as today; one chunk per picture (modality `picture`, its caption as text)
        -> embed     text: the model's document prompt (Qwen3 or EmbeddingGemma 2)                             (Phase 30)
                     picture: the image, with its caption, as one input (EmbeddingGemma 2 only)                (Phase 31)
        -> index     one collection per experiment, as today; a picture is a point with `modality = picture`
        -> search    a text query; pictures come back as hits and are drawn on the hit card
```

Everything after parsing sees an ordinary Docling document, so the five chunking strategies, BM25, the rerankers and the benchmark do not change for OCR. New code: `parsing/ocr.py` (the region loop and the Ollama call), `chunking/pictures.py` (picture chunks, added after any strategy), and image inputs in `embedding/ollama.py`. No new dependency: `pypdfium2`, Pillow and `httpx` are in the image. Docling's own OCR engines stay off (`do_ocr=False`), so there is one OCR path.

### Phase 29 — OCR with glm-ocr, and the Upload checkbox

**Goal:** a PDF with no text layer, uploaded with OCR ticked, becomes the same kind of parsed document as a PDF with one (headings, paragraphs and tables with their pages and boxes), and is chunked, embedded and searched like any other.

| Decision | Choice |
|---|---|
| Where the regions come from | Docling's standard pipeline as today (`do_ocr=False`), with the layout option `keep_empty_clusters` and `generate_page_images`. The layout model works on the page image, so a scanned page still gets its regions, with boxes and labels, but empty text. |
| What OCR fills | Every text-like item with no text and every table with no cells, cropped from the page image at `ocr_scale`. Label decides the prompt: `Text Recognition:`, `Table Recognition:` for a table, `Formula Recognition:` for a formula. Pictures are left alone (Phase 31). |
| What "OCR on" means | **OCR where there is no text layer.** A page that has one keeps it; a scanned page inside an otherwise normal PDF is read. Ignoring a text layer that exists (a broken one) is under Ideas. |
| The call | `POST /api/chat` with the crop as the message's image, temperature 0, `stop` strings and `num_predict = ocr_max_tokens`. The stop rule is settled in step 1 on a whole page; a region that still hits the cap is counted and reported, not hidden. |
| Tables | The returned HTML table becomes Docling `TableData`, so a table chunk is serialised as it is today. Docling's own HTML table reader is used if it can be called on a fragment; otherwise a small one of ours (`html.parser`, `colspan` and `rowspan`). |
| Config | `ParseConfig` gains `ocr: bool = False`, `ocr_model = "glm-ocr:bf16"`, `ocr_scale = 2.0`, `ocr_max_tokens`, `ocr_keep_alive`. All left out of the config hash while `ocr` is false, so every existing hash is unchanged. |
| Who calls Ollama | `parse_pdf` takes the Ollama address when `cfg.ocr`; without it, parsing needs no Ollama, as now. The `parsed_document` asset passes the Ollama resource. |
| Upload page | A checkbox above the uploader's result, "Read scanned pages with OCR (glm-ocr)", off by default; disabled with a reason when Ollama does not list the model. The parse cache gets a second variant, `data/artifacts/_uploads/<doc_id>/ocr/`, so ticking and unticking does not parse twice. The experiment is built with `parse=ParseConfig(ocr=True)`, so it is recorded and "an existing experiment with the same settings" offers only OCR experiments. The scanned-PDF warning now says to tick the box. Progress is shown per page. |
| Metrics | The parse meta and the `parse` stage row gain `ocr_regions`, `ocr_seconds`, `ocr_output_tokens`, `ocr_cut_regions` (hit the cap) and `ocr_pages`. `chars_per_page` counts the OCR text. |
| Tests | Config hashing gets the new fields (a tested area). **For you to approve:** a few pure-logic tests for the HTML table to `TableData` step if we write our own, since a silent bug there corrupts table content. Nothing else. |

Steps:

- [x] 1. Settle the stop rule: OCR every region of page 1 and the table of page 14 of the FAA chapter with stop strings and a cap; all regions must end by `stop`. If some do not, try in order: cutting the output where it starts to repeat, the `glm-ocr:q8_0` or `latest` tag, a newer Ollama. Record the seconds per page here.
- [x] 2. Check that `keep_empty_clusters` gives items with boxes and empty text on an image-only PDF (the FAA chapter rasterised with pypdfium2 and Pillow). If it does not, the fallback is a Docling OCR engine of our own in its `layout_regions` mode.
- [x] 3. `ParseConfig` fields and the hash rule; `parsing/ocr.py` (crop, call, clean, fill, table); `parse_pdf` and `build_converter` use it when `cfg.ocr`.
- [x] 4. `ingest.ensure_parsed(path, doc_id, parse_cfg)` with the cache variant; `record_parse` details; the `parsed_document` asset.
- [x] 5. `ui/upload.py`: the checkbox, the per-page status, the warning text, `parse=` in `build_config`.
- [x] 6. `CLAUDE.md` (Scope, the parse facts, the Upload description) and README.

**Done when:** the FAA chapter rasterised into an image-only PDF (so `chars_per_page` is near 0 without OCR) is uploaded with the box ticked, and (a) its OCR text is compared with the original's text layer page by page, with the character similarity and the seconds per page recorded here; (b) the page 14 table is a table chunk with its rows; (c) an experiment made from it answers "what to do when a door opens in flight" on Try a query from the same passage the text-layer experiment does. The same PDF without the box still shows the scanned warning. Checked once.

Built differently from the draft:

- **No stop string works for text, so the reply is read as a stream and cut by a rule of ours.** A `` ``` `` stop ended only 6 of 19 regions on page 1: a paragraph is followed by a newline and the same paragraph again, with no fence. Stopping at the first newline ended all of them but lost the second line of a two-line heading. So `first_reading` reads the lines as they arrive and the connection is closed (which stops the generation) at the first line that is a code fence, starts like the first line, has Chinese characters when the first line has none, has no letter or digit, or was seen before. The last three came from the first full run, which was killed after it spent minutes on page 1: two regions were followed by `境` on every line up to the token limit, and three regions kept a junk line such as `醒 读：`. A `stop` string is used only for tables (`</table>`). The other tags and a newer Ollama were not needed.
- The crop has 1 point of padding. The 6 points of the drafting checks read the neighbouring paragraphs' lines into a region, which looked like an OCR error and was not.
- A region is one piece of running text: its lines are joined with a space, as Docling joins a paragraph's lines. Code and formulas keep their lines and use only the first two end rules. A list item's leading `-` is dropped, because Docling writes the marker itself.
- The HTML table goes through `docling.utils.deepseekocr_utils._parse_table_html`. It does exactly this on a fragment, so there is no reader of ours and no tests for one; it is a private function of Docling, which an upgrade could move.
- `ensure_parsed` takes `ocr: bool`, not a whole `ParseConfig`: the Upload page parses with the defaults either way.
- A table is read when none of its cells has text, not only when it has no cells, so a table that TableFormer left empty on a text-layer page is read too.
- A text region where nothing is read is removed from the document. `OcrStats` also counts these (`empty_regions`).
- `ocr_max_tokens` is 4096 (the 25-row table took 688 tokens). There is also a retry of a failed connection, three times, as the embedder has.
- `Formula Recognition:` is wired for formula regions but was not run on a real formula: the test document has none.

**Result (checked on the real stack; the scan is the FAA chapter rendered at 144 dpi into an image-only PDF, 23 pages, 6.4 MB):**

- Layout on a scan: with `keep_empty_clusters` Docling gave 252 text regions with boxes and no text (section headers, paragraphs, list items, captions), 2 tables and 13 pictures; without it, no text items at all. The text-layer parse of the original has 240 text items, so the regions are close to the real ones.
- (a) Against the original's text layer: 0.998 of its words are found again in the OCR text, page by page (the lowest page 0.991). Character similarity is 0.978 to 1.000 on 21 pages. Pages 14 and 16 score 0.81 and 0.40 only because the OCR text has more than the reference: those are the two tables, which the text-layer parse left empty and the OCR read (24 rows and 17 rows, 3 columns each).
- Time: layout 135 s and OCR 284 s in the first measured run, so about 6 s and 12 s a page, 18 s together; no region reached the token limit. Through the Upload page the whole parse took 384 s (241 s of it OCR, 254 regions, 17,922 output tokens).
- (b) The page 14 table is a table chunk (489 tokens, its rows as a Markdown table under its caption); the page 16 table is two.
- (c) "what to do when a door opens in flight" returns the same three passages from the same pages as the text-layer experiment `full-ingest-dense-sparse`, with "Door Opening In-Flight" (page 17) first at 0.763 against 0.768.
- The Upload page through the app tester: without the box the scan shows "this looks like a scanned PDF. Tick the OCR box above to read it."; ticking it parses with OCR and shows "OCR read 254 regions on 23 pages in 241 s."; Chunk gave 51 chunks (48 text, 3 table); Embed and index wrote the experiment `up-afh-ch18-scanned-ocr`, whose stored settings have `parse.ocr` and whose `parse` stage row has the OCR numbers. A second session with the box ticked before the upload took the parse from the cache (25 s). The parse file is 423 kB: the page images are not kept.
- The 44 tests pass (the 43 and one for the hash: the `ocr*` settings change it only when `ocr` is on).
- What is left over: a full-width bracket or a Chinese `一` for a dash inside English text, in 2 of 254 regions. Only English, and only a clean render, were looked at: a real scan (skewed, noisy) will be worse.
- Not seen by eye: the checkbox and the per-page status in a browser. The app tester ends the embed step with a page-link error that only exists in the tester (it runs `upload.py` without the navigation); the embed itself had finished.
- Kept for you to look at: the experiment `up-afh-ch18-scanned-ocr` and its upload `afh_ch18_scanned.pdf`. Delete the experiment on the Experiments page if you do not want it.

### Phase 30 — EmbeddingGemma 2 as an embedding model (text)

**Goal:** an experiment can be embedded with `embeddinggemma-2:740m` and used on Upload, Try a query, the Chatbot and the Benchmark, text and tables only.

| Decision | Choice |
|---|---|
| Prompts | `EmbedConfig` gains `query_template` (default Qwen3's `Instruct: {instruction}\nQuery: {text}`) and `document_template` (default `{text}`). `OllamaEmbedder.embed` formats with them instead of the hard-coded Qwen3 prefix. Both are left out of the hash while they are the defaults. |
| Model families | One small table in `config.py`, keyed by the start of the model name: tokenizer, the two templates, native size, the sizes it can be cut to, whether it takes images. `EmbedConfig.for_model(name)` fills a config from it; the pages call that. The experiment still records every value, so the table is only a source of defaults. |
| EmbeddingGemma 2's row | tokenizer `google/embeddinggemma-2`; query `task: search result \| query: {text}`; document `title: none \| text: {text}` (headings stay inside the chunk text, as for Qwen3); native 768; sizes 768, 512, 256, 128; images yes. |
| Vector size | Still our own truncation and re-normalisation, one path for every model. Ollama's `dimensions` option is not used. |
| The pages | `data.embedding_models()` lists the installed models whose family is in the table (today it is `startswith("qwen3-embedding")`). The Upload page's *Vector size* list follows the chosen model. `model_label` shows the family when it is not Qwen3, so `740m` is not read as a Qwen3 size. |
| Semantic chunking | It embeds sentences in document mode, so it follows the templates; the embedding cache key must include the document template (to check in `embedding/cache.py`). |
| Left alone | The good-answers index of text-to-SQL keeps its own `EmbedConfig` (Qwen3). Existing experiments are not re-embedded. |
| Tests | Config hashing: a fixed list of today's configs keeps its hashes; a config with the new templates gets a different one. |

Steps:

- [x] 1. The templates in `EmbedConfig`, the hash rule, the family table and `for_model`; `OllamaEmbedder.embed` uses the templates.
- [x] 2. `ui/data.py`, `ui/upload.py`, `ui/benchmark.py`, the model labels in `ui/data.py` and `benchmark/report.py`.
- [x] 3. Check on one long chunk that Ollama's `prompt_eval_count` is our token count plus the prefix, so nothing is cut short of `max_tokens`.
- [x] 4. The hashing tests; `CLAUDE.md` (the "Key technical facts" that say Qwen3 only).

**Done when:** the FAA chapter is embedded into a new experiment with `embeddinggemma-2:740m` (768, BM25 on); the door question returns the "Door Opening In-Flight" chunk first on Try a query and is answered in the Chatbot; and one benchmark report with `qwen3-embedding:0.6b` and `embeddinggemma-2:740m` on that document finishes. The hashing tests pass. Checked once.

Built differently from the draft:

- The family table has no native size and no "takes images" flag. The native size differs inside the Qwen3 family (1024, 2560, 4096) and the code already reads the size from the vectors; the images flag comes with Phase 31, which is the first thing to need it.
- A label is the size for a Qwen3 model (`4b`, as before) and the whole name for any other (`embeddinggemma-2:740m`), in one function, `config.embed_model_label`, that the pages and the report both use. A benchmark experiment of another family is named with it (`bench-<doc>-<id>-embeddinggemma-2-740m-<strategy>`). The charts order models by size, reading `740m` as 0.74.
- The embedding cache key is the text as it is sent (the document template applied), not the template as a separate part. With Qwen3's template that is the text itself, so every file already cached keeps its key.
- `ui/benchmark.py` needed no change: the runner builds the config with `for_model`.
- The hash test pins four hashes computed with the config module as committed before Phases 29 and 30, so it also guards the OCR settings.
- `EmbedConfig(model=...)` alone still means Qwen3's tokenizer and templates. A Dagster run config for an EmbeddingGemma 2 experiment therefore has to set `embed.tokenizer` and both templates; the pages and the benchmark do it through `for_model`.

**Result (checked on the real stack):**

- Token counts: for chunks of 128, 512, 2,000 and 6,000 tokens Ollama counted our count of the sent text plus 2 for EmbeddingGemma 2 (the document prefix adds 6 tokens to a chunk) and our count plus 1 for Qwen3 (512 to 6,000). Nothing is cut short. Vectors are 768 wide.
- The Upload page through the app tester: the model list is `embeddinggemma-2:740m` and the three Qwen3 models, with `qwen3-embedding:0.6b` still the default; *Vector size* offers native, 256, 512, 768, 1024 for Qwen3 and native, 128, 256, 512, 768 for EmbeddingGemma 2; with BM25 ticked the FAA chapter was chunked (51 chunks, counted with the new tokenizer) and embedded into `up-afh-ch18-gemma2` (51 points, 3,359 tokens per second).
- "what to do when a door opens in flight" on that experiment returns "Door Opening In-Flight" (page 17) first at 0.839, then Cabin Fire 0.729 and In-Flight Fire 0.720. The scale is the model's own: the same question scores 0.768 with Qwen3.
- The chatbot agent (`python -m rag_lab.agent documents`, the code the Chatbot page calls) answered the door question from that chunk, citing `[1]`, in 43 s (retrieve 20 s, generate 23 s with a cold model).
- A benchmark report (`34e8ff`: two models, `hybrid` and `recursive` chunking, `dense` and `hybrid` search, 7 test queries) finished in 142 s with all four experiments done, and its report and PDF build with both labels. Embedding speed 13.3 and 13.6 chunks per second for EmbeddingGemma 2 against 7.4 and 7.9 for `qwen3-embedding:0.6b`. Dense MRR 0.89 and 0.87 against 0.86 and 0.86; hybrid MRR 0.90 against 0.93; recall@5 is 1.0 everywhere except one EmbeddingGemma 2 dense run (0.86). Seven queries on one document cannot rank the two; what it shows is that the prompts and the tokenizer are right.
- The 46 tests pass (two new ones for the hash).
- Not seen by eye: the model list and the size list in a browser, and the Chatbot page itself (the agent was run from the command line).
- Kept for you to look at: the experiment `up-afh-ch18-gemma2` and the report `34e8ff` with its four `bench-19_afh_ch18-34e8ff-...` experiments. The Experiments and Benchmark pages delete them.

### Phase 31 — Pictures as chunks, embedded from the image

**Goal:** a figure in a PDF is a point in the experiment's collection, embedded from its image, and a text query finds it.

| Decision | Choice |
|---|---|
| Parse | `ParseConfig.pictures: bool = False` switches on Docling's `generate_picture_images`; each picture is written to `<doc_id>.pictures/<n>.png` beside the parse files (also in the `_uploads` cache). Pictures whose shorter side is under `picture_min_side` pixels (rules, logos, icons) are dropped. Left out of the hash while false. |
| Chunk | A picture chunk has `modality = "picture"`, `image` (the file, relative to the parse folder), page, box, headings, and as `text` its caption, or "Figure on page N" when it has none, so the text is never empty. `chunking/pictures.py` adds them after any strategy, so the five strategies do not change. They have no source text, so their span is empty. |
| What is embedded | `EmbedConfig.picture_input`: `image+caption` (the default: one input with both), `image`, or `caption` (text only, which also works with Qwen3 and is the baseline that says whether the image adds anything). Left out of the hash unless the experiment has pictures on. |
| Request | `{"image": "<base64>"}` or `{"text": ..., "image": ...}` inside `input`, never the top-level `images` field. The embedder refuses images for a family the table says cannot take them, before any call. |
| Config check | `pictures` with `picture_input` other than `caption` needs a model that takes images; an invalid combination fails when the config is built, with a message naming the two settings. |
| Index | The same collection and the same unnamed vector. The payload gains `image`. `modality` already has a payload index. BM25, when on, is made from the caption. |
| Search | Nothing new in `search()`. *Content* on Try a query gains `picture`. Because of the modality gap pictures will rank under text in a mixed list; this phase measures by how much and does not correct it (a reserved number of picture hits is under Ideas). |
| Rerank | The reranker reads text, so it judges a picture by its caption. Known limit: a picture whose caption does not say what it shows is dropped by the `+rerank` methods. |
| Showing a hit | `Hit` gains `image`; `ui/hits.py` draws the picture on the card (Try a query and the Chatbot's chunk cards). The Upload preview lists picture chunks with their thumbnails. |
| Chatbot | A picture hit reaches the answer model as its caption, page and headings, like any chunk, and can be cited. The image itself is Phase 32. |
| Delete | `library.py` removes a document's `.pictures` folder with its parse files. |
| Benchmark | Unchanged and text only: test queries are judged by a snippet of text, which a picture does not have. |
| Upload page | A checkbox "Index the pictures (needs a multimodal embedding model)", enabled when the chosen model takes images. |
| Tests | Config hashing for the new fields. Nothing else. |

Steps:

- [x] 1. `ParseConfig.pictures`, `picture_min_side`; the parser writes the picture files; the hash rule.
- [x] 2. `Chunk.image` and the `picture` modality; `chunking/pictures.py`; the chunk summary and the Upload preview.
- [x] 3. `OllamaEmbedder` takes image inputs; `embed_chunks` builds them from `picture_input`; a check that an image input's vector differs from the text-only one.
- [x] 4. `index_chunks` payload, `Hit.image`, `ui/hits.py`, the *Content* filter, the CLI's `--modality picture`.
- [x] 5. `library.py`; `CLAUDE.md` (Scope, modalities, the embed request shape and its trap).

**Done when:** the FAA chapter (13 pictures) is ingested with pictures on and `embeddinggemma-2:740m`; for five questions about what a figure shows, written by you, the right picture is in the top 5 with *Content* set to `picture`; its rank in an unfiltered search and the scores of the best text and best picture hit are recorded here; the same five are run with `picture_input = caption` for comparison; and deleting the document removes its picture files. Checked once.

Built differently from the draft:

- **The pictures are cropped by us, not by Docling.** Docling already records every picture and its box in the parsed document, so `parsing/pictures.py` renders the page with pypdfium2 and crops it. A document that is already parsed (the Upload page's cache, parsed once with the defaults) gets its pictures in about a second instead of going through Docling again, there is no third and fourth cached parse variant, and the parse file does not grow. `generate_picture_images` stays off.
- `picture_min_side` is in PDF points (50, about 18 mm), not pixels, so it does not depend on `picture_scale` (a new setting, 2.0).
- Which pictures are kept is one rule, `kept_pictures(doc, parse_cfg)`, used by the parse stage (to save the files) and by the chunk stage (to make the chunks), so the chunker needs no listing of the folder.
- A picture chunk's text is its headings and its caption when headings are added to chunk text (the default), like every other chunk. Its empty span sits at the end of the last text or table before the picture, and the Upload preview marks it there as a picture.
- `Hit.image` is relative to `data/artifacts` (`<experiment>/parse/<doc_id>.pictures/<n>.png`), so a card, or a saved chat turn, finds the file without knowing the experiment; the payload keeps the path relative to the parse folder. The card inlines the picture as a JPEG at most 560 pixels wide (23 to 83 kB each; as PNG five cards were 1.5 MB of HTML) and shows nothing when the file is gone.
- The parse meta has no picture count: the chunk summary's `by_modality` and the `embed` stage row (`pictures`, `picture_input`) say how many there were.
- I wrote the five questions (the plan says you would), after looking at the pictures, so that each describes what is seen and not the caption. Replace them with your own if you want a fairer test.
- All three ways of embedding were run, not only `caption`.

**Result (checked on the real stack; `embeddinggemma-2:740m`, native 768, BM25 on):**

- The Upload page through the app tester: the pictures box is disabled with a Qwen3 model and enabled with EmbeddingGemma 2; Chunk gave 64 chunks (51 text, 13 picture) and wrote 13 picture files (115 kB to 1.2 MB); the preview drew the 13 pictures; Embed and index wrote `up-afh-ch18-pictures` (64 points). Four of the 13 pictures have no caption in the parse, so their text is only "Figure on page N".
- The trap: for the same picture, the vector made from image and caption has cosine 0.894 with the vector made from the caption alone, and the image-only vector 0.697, so the image does reach the model. The 51 text chunks are identical in all three (1.0000).
- The five questions, and where the right picture came:

| Question (the picture) | image+caption: among pictures / in all chunks | image: pictures / all | caption: pictures / all |
|---|---|---|---|
| an airplane flying low over farm fields (page 5, no caption) | 1 / 1 | 1 / 1 | 7 / 43 |
| chart of stopping distance at 50 mph and 100 mph (Figure 18-2) | 1 / 1 | 1 / 2 | 1 / 1 |
| an airplane in a steep spiral descending turn (page 9, no caption) | 1 / 1 | 1 / 6 | 3 / 6 |
| what the airspeed indicator, altimeter and vertical speed indicator show when the pitot or static source is blocked (page 15, no caption) | 1 / 1 | 1 / 1 | 1 / 4 |
| an airplane resting in tall grass after landing (Figure 18-1) | 1 / 1 | 1 / 1 | 5 / 31 |

- So with the image and the caption together the right picture is first of the 13 for all five, and first of all 64 chunks too; the image alone is as good among pictures and a little lower among all chunks; the caption alone finds a picture only when its caption says what it shows. Scores of the right picture, image+caption: 0.74 to 0.84. **The modality gap did not show here**: a question that describes a picture ranked that picture above every text chunk, so no correction was needed. A question about the text is not disturbed: the door question returns five text chunks, "Door Opening In-Flight" first.
- Embedding the 64 chunks took 13 s with the images against 4 s with captions only.
- Try a query through the app tester: *Content* offers `picture`; with it the five cards each draw their picture.
- Deleting the document from an experiment removed its 13 picture files and the folder with its other files; the two comparison experiments were then deleted.
- The 48 tests pass (two new: the picture settings are in the hash only with pictures on; pictures as images are refused for a model that takes no images, and allowed as captions).
- Not run: pictures through Dagster (`parse_pdf` with `parse.pictures` calls the same `save_pictures` the page calls, but no Dagster run was made), and pictures on a scanned PDF together with OCR. Not seen by eye: the cards and the preview in a browser.
- Small sample: five questions, 13 pictures, one document, all photographs or diagrams with little text in them. The pitot-static picture is really a table drawn as an image, and it was found from its content, which is promising for scanned tables but is one case.
- Kept for you to look at: the experiment `up-afh-ch18-pictures`.

### Phase 32 — The chatbot looks at the pictures

**Goal:** when a retrieved chunk is a picture, the answer model sees the image, not only its caption. `gemma4:e4b-mlx` reports the `vision` capability.

| Decision | Choice |
|---|---|
| Where | `generate` only: the images of the picture hits among the top k are attached to the answer message, numbered like their chunks. `grade` and `rewrite` stay on text. |
| Setting | `AgentConfig.show_pictures: bool = True`, and at most `max_pictures` (2) images a turn, because each costs context; the `ModelState` event already shows prompt tokens against `num_ctx`. |
| A model without vision | The toggle is off for it, as the Thinking toggle is for a model that cannot think. |
| Saved turns | Unchanged: the hits carry the image path, and the image is read again on replay. |
| Tests | None. |

Steps:

- [x] 1. `agent/documents/graph.py` attaches the images; `AgentConfig`; the page's toggle.
- [x] 2. `CLAUDE.md`.

**Done when:** a question that only a figure answers (for example what a diagram's labels say) is answered from the picture with its `[n]` cited, and the same question with the toggle off is not. Checked once.

Built differently from the draft:

- **The passage number is drawn onto the picture.** The model read the pictures correctly from the first try, but would not cite one by its passage number: it wrote `[Table in the image]` or `[Table]`, and once "which is not a numbered passage". Five wordings were tried (the picture named in its passage, a reminder before the question, after the question, a rule in the system prompt, a short "the attached image is passage [4]"), with thinking on and off, and none changed it; with the rule in the system prompt and thinking on it even refused to use the picture. So each picture is sent with a white strip above it that says `Passage [n]` (`graph.py: labelled`). With that it cited `[4]` and `[1]` in every run, and with two pictures attached it cited the right one.
- The page marks the cards of the pictures the model was given ("image given to the model"), worked out from the turn's saved settings by the same rule the agent uses (the first `max_pictures` picture hits), so a saved turn needs no new event. The settings line under an answer says "up to 2 pictures shown to the model".
- `data.chat_models()` returns each model's capabilities instead of only whether it can think, since the page now asks about `vision` too.
- The CLI got `--no-pictures`.

**Result (checked on the real stack; `up-afh-ch18-pictures`, `gemma4:e4b-mlx`):**

- "In the figure about turning back to the runway after engine failure, what distances and angles are labelled?" The figure (Figure 18-5) was the first hit (reranker 1.000, from its caption). With pictures on the answer was "300 feet AGL, 4,480 feet, 1,016 feet, 180°, and 225° [1]": all five labels, cited to the picture. The 4,480 feet is nowhere in the document's text. With pictures off: "The documents do not contain information about what distances and angles are labelled in the figure".
- "What do the airspeed indicator, altimeter and vertical speed indicator show when both static sources are blocked?" The answer is a row of a table that exists only as a picture with no caption (page 15). With pictures on the model read the row correctly (airspeed decreases with altitude gain and increases with altitude loss; altitude and vertical speed do not change) and cited `[4]`; with pictures off it answered from a text passage about a partly restricted static system, which is a different case.
- Cost: about 600 prompt tokens a picture (2,276 against 1,056 with two pictures), no slower to answer.
- The Chatbot page through the app tester: the *Show pictures to the model* toggle is on for `gemma4:e4b-mlx`; the turn-back question was answered from the figure with `[1]` cited; of the three picture cards two carry "image given to the model"; a refresh on the same chat shows the same; with the toggle off the answer says it has no figure and no card carries the mark.
- **The limit that remains is retrieval, not reading.** The pitot-static picture reached the model only because it came 4th with a reranker score of 0.029: the reranker judges a picture by its text, which for a picture without a caption is "Figure on page 15". With more competing text it would have been dropped before the model could see it. Reading the text inside a picture at ingest (Ideas) would fix that.
- Not seen by eye: the toggle and the marked cards in a browser. Not tried: a chat model without vision (the toggle is disabled from the capability Ollama reports, which was not exercised).
- Two questions, one document, one chat model. Kept for you to look at: the chat with the turn-back question (pictures on), on `up-afh-ch18-pictures`.

### Decided

- OCR reads only the regions that have no text layer; a page with one keeps its text.
- OCR is glm-ocr only; Docling's CPU engines (RapidOCR is in the image) stay off, so there is no fallback when Ollama is unreachable: the parse fails with a message.
- A picture is embedded as its image and caption together, as one input. `image` and `caption` alone stay as settings, for the comparison in Phase 31's "Done when".
- The chatbot sees the picture itself, not only its caption: Phase 32 is part of the plan.

### Open choices (defaults used above unless you say otherwise)

- Pictures are off in the Benchmark.
- One set of pure-logic tests is proposed (the HTML table step, only if we write our own reader).

### Risks

- **glm-ocr not stopping.** If no stop rule is reliable, OCR costs 17 s a region instead of about 1 s, which is not usable. Step 1 of Phase 29 settles this before anything is built on it.
- **Layout on scanned pages.** The quality of the regions decides the quality of the text: a merged or missed region is text read in the wrong order or not at all. The "Done when" comparison with the text layer measures it on one clean document; a real scan (skewed, noisy) will be worse.
- **Languages.** Only English was looked at. Thai and others need their own look, and BM25 already needs a word segmenter for them.
- **The modality gap** may keep pictures out of an unfiltered top k altogether. Phase 31 measures it; the fix is under Ideas.
- **Small evidence for the embedding model.** Two coloured squares show the direction, not the quality. The five figure questions are the first real evidence.
- **Memory on the Mac.** OCR, the embedding model, the reranker and the chat model are now four models on one Ollama; the swapping seen in Phase 18 gets more likely.

## Automatic ingestion from data/raw (Phase 33)

**Goal:** a PDF put into `data/raw/` is ingested with no click: the sensor sees it, starts `ingest_job` for it, and the job runs parse, chunk, embed and index into Qdrant with the settings written in one YAML file.

### What exists today, and what was checked

- `new_pdf_sensor` (every 30 s, on by default) registers each PDF in `data/raw` as a Dagster partition, keyed by the hash of its bytes. **It starts no runs.**
- `ingest_job` runs the four assets for a partition, but only when it is launched by hand with `resources.experiment.config` typed into the launchpad.
- Runs execute in the `dagster-code` container (Dagster 1.13.25, `DefaultRunLauncher`), which already mounts `./data`, so a run started by the sensor can read the PDF and write the artifacts. The run coordinator is already the queued one, but with its default limit, so several PDFs dropped together would run at the same time.
- A sensor result can add a partition and request a run for it in the same tick (`SensorResult(run_requests=..., dynamic_partitions_requests=...)`), and a `RunRequest` carries its own `run_config`, `run_key` and `partition_key`.
- PyYAML 6.0.3 is in the image (other packages bring it) but is not declared in `pyproject.toml`.
- `data/raw` holds 5 PDFs now (the FAA chapter, `handbook.pdf`, `sample-report.pdf`, `sample-scanned.pdf`, `The Myth of Sisyphus`), and 7 partitions are registered (two for files no longer there).
- `dagster.yaml` is copied into the image, not mounted, so a change to it needs `docker compose up -d --build`.

### Architecture

```
config/ingest.yaml  --read at every tick-->  new_pdf_sensor  --RunRequest(partition, run config, run key)-->  ingest_job
     (the experiment:                              |                                                              |
      name, parse, chunk,                  data/raw/*.pdf                                  parsed_document -> chunks -> embeddings -> qdrant_index
      embed, index)                     (complete files only)                              (one run at a time; the collection is the YAML's `name`)
```

The YAML is one experiment: the same keys as the launchpad's `resources.experiment.config`. The sensor turns it into the run config of each run it starts, written out in full, so a run in the Dagster UI shows exactly what it ran with and a later edit of the file does not change it.

```yaml
# config/ingest.yaml: what a PDF put into data/raw is ingested with. A key left out keeps its default;
# an unknown key is an error. Changing a setting means changing `name` too (a new experiment).
name: auto-ingest
parse:
  ocr: false            # true reads scanned pages with glm-ocr
  pictures: false       # true needs an embedding model that takes images
chunk:
  strategy: hybrid
  max_tokens: 512
embed:
  model: qwen3-embedding:0.6b
index:
  sparse: true          # a BM25 vector, so the Chatbot and hybrid search can use the experiment
```

| Decision | Choice |
|---|---|
| The file | `config/ingest.yaml`, committed, mounted read-only into `dagster-code` at `/app/config`; its path is `INGEST_CONFIG` (compose sets it). One file, one experiment. |
| Loading | `load_experiment_file(path)` in `config.py`: `yaml.safe_load`, then `ExperimentConfig.model_validate`, which already refuses unknown keys and bad values and names them. Plain Python, so the same function can be used outside Dagster. |
| The embedding family | When `embed` names a model and does not set the tokenizer or the templates, they are filled from the model's family (`EmbedConfig.for_model`). So `model: embeddinggemma-2:740m` alone is right, which the launchpad form is not. |
| The tag | `tag` defaults to `name`, as the Upload page and the benchmark do, so the experiment has a hash of its own. |
| When it is read | At every sensor tick, so an edit takes effect for the next PDF with no restart. A file that does not load fails the tick: the error shows on the sensor in the Dagster UI and nothing runs until it is fixed. |
| **Which PDFs run** | **Only PDFs that arrive after this is switched on** (decided). New means not yet a partition, which is what the sensor has always meant by it: the 5 PDFs already in the folder are partitions, so they start nothing, and neither does the same file under another name. A changed `name` in the YAML applies to the PDFs that arrive after it; nothing already ingested is done again. An older PDF is ingested by launching `ingest_job` for its partition by hand. |
| **Settings changed, name not** | Refused. Before it requests anything the sensor compares the YAML's `settings_hash()` with the experiment already recorded under that name; if they differ it starts no run and the tick says to change `name`. Without this, chunks made with different settings would share a collection, or the index step would fail on a different vector size. |
| A file still being copied | A PDF is taken only when it is complete: its last kilobyte holds `%%EOF`. A file that is not there yet is looked at again at the next tick. Its id is hashed only then, so a half-copied file never becomes a partition. Modification time and size are not used: Windows keeps the source's time on a copy and can reserve the full size at the start (step 1 checks what a copy in progress looks like through the Docker mount). |
| One run at a time | `dagster.yaml` limits the run queue to one run, so PDFs dropped together are ingested one after another: each run loads Docling's models, and there is one Ollama. This also applies to runs launched by hand. |
| What a run is labelled with | Tags `experiment` and `source_file`, so the runs list reads as "which file into which experiment". |
| A run that fails | Stays failed; its PDF is a partition by then, so the sensor does not start it again. *Re-execute* on the run in the Dagster UI repeats it with the same config. (A scanned PDF with `ocr: false` fails at `chunks` with "No chunks produced", as now.) |
| Removing a PDF from `data/raw` | Does nothing: its points stay in the experiment. The Experiments page deletes a document. |
| The Upload page | Unchanged. It still writes to `data/uploads`, which the sensor does not watch. |
| Code layout | The sensor moves to `assets/sensors.py` (it now needs `ingest_job`, and `jobs.py` imports the partitions from `partitions.py`). The assets do not change. |
| Dependency | `pyyaml` is declared in `pyproject.toml`; it is already installed. |
| Tests | **For you to approve:** one small test of `load_experiment_file` (a key left out keeps its default, an unknown key is refused, the family fill for `embeddinggemma-2`, `tag` defaulting to `name`), because a silent mistake there ingests every document with the wrong tokenizer or prompts. Nothing else. |

### Steps

- [x] 1. Check on this machine what a large PDF looks like to the container while Windows is still copying it into `data/raw` (size, last bytes), and that the `%%EOF` rule waits for it.
- [x] 2. `load_experiment_file` in `config.py`; `config/ingest.yaml` with today's defaults; `pyyaml` in `pyproject.toml`.
- [x] 3. `assets/sensors.py`: read the YAML, the settings guard, complete files only, partitions and run requests with run key, run config and tags. `definitions.py` and `partitions.py` follow.
- [x] 4. `docker-compose.yml`: mount `./config` read-only into `dagster-code` and set `INGEST_CONFIG`. `dagster.yaml`: one run at a time. `docker compose up -d --build`.
- [x] 5. `CLAUDE.md` (the sensor, the YAML, the run key rule, the commands) and README ("Batch ingestion with Dagster").

**Done when,** on the real stack (rewritten for the decision that only new arrivals run): (a) the PDFs already in `data/raw` start nothing; (b) a PDF copied into `data/raw` gets a run within a minute, the run's config in the Dagster UI equals the YAML, and a search of `auto-ingest` finds its text; (c) two PDFs copied together run one after the other; (d) with a chunk setting changed in the YAML and `name` left alone, a new PDF starts no run and the sensor's tick says why, and it runs once the file is put right; (e) with `name` changed (and `embed.model: embeddinggemma-2:740m`, `parse.pictures: true`, which also runs pictures through Dagster for the first time), a new PDF goes into the new experiment, nothing older is ingested again, and a picture search finds a figure; (f) a misspelled key stops the tick with a message that names it. Checked once.

Built differently from the draft:

- **Only new arrivals run** (your decision), so there is no backfill of the folder and a changed `name` does not re-ingest it. The rule is the sensor's own old rule, "not yet a partition", so nothing new had to be remembered. The run key (`<name>:<doc_id>`) is still set, as a second guard against a run starting twice.
- **A run is one process.** The first automatic run took 204 s for a two-page PDF, of which the four steps took 31 s: with Dagster's default executor every step starts a process of its own, and each imports Docling again (about 45 s). `ingest_job` now uses the in-process executor, and the same kind of PDF takes 29 to 48 s. This also applies to runs launched by hand.
- **`dagster.yaml` is mounted, not rebuilt.** The plan said the change needs `docker compose up -d --build`. Declaring `pyyaml` changes `pyproject.toml`, and a build would then reinstall every dependency (none are pinned), so the three Dagster services mount `docker/dagster/dagster.yaml` over the copy in the image instead: `docker compose up -d` was enough, and a later edit needs a restart, not a build. The next real build will still reinstall the dependencies, as any change to `pyproject.toml` does.
- **A file Windows is still copying cannot be opened at all.** Step 1: during a 1.5 GB copy into the data folder the container got "permission denied" for the whole 2.6 s, then the complete file. So the sensor treats a file it cannot read as not complete; the `%%EOF` check stays for a program that writes a file bit by bit.
- The YAML is read only at a tick that finds a new PDF, not at every tick: a broken file is harmless until something arrives, and then fails that tick.
- A YAML that does not load, or that names an experiment with other settings, fails the tick (red on the sensor, with the reason) instead of skipping it quietly; the PDF is not registered, so it runs at the first tick after the file is put right.
- `load_experiment_file` refuses a model of no known family (a misspelled model name would otherwise be found only when Ollama is called), unless the file sets the tokenizer and both templates itself.
- The run config leaves out the settings that are `None` (`embed.dimension`, `index.hnsw_ef`); a run config that names every other value is what the Dagster UI shows.
- Two tests for the loader, in the config tests (50 tests in all).

**Result (checked on the real stack, with six small PDFs cut from the FAA chapter, each with its own content hash):**

- (a) After the change the sensor's ticks say "No new PDFs." and no run started for the 5 PDFs in the folder.
- (b) A PDF copied in at 15:59:41 had its run started at 16:00:09. The run's config equals the YAML, its tags are the experiment and the file name, and "what to do when a door opens in flight" on `auto-ingest` returns its "Door Opening In-Flight" chunk first (0.768).
- (c) Two PDFs copied in together: one run 16:07:42 to 16:08:30, the other 16:09:05 to 16:09:34, so one after the other (the queue is limited to one run).
- (d) With `chunk.strategy` changed to `fixed` and the name left as `auto-ingest`, a new PDF started nothing and the tick failed with "An experiment named 'auto-ingest' already exists with other settings. Settings cannot change under the same name: choose a new `name` (a new experiment), or put the settings back. Not started: t33-guard.pdf." With the file put back it ran at the next tick.
- (e) With `name: auto-ingest-pictures`, `embed.model: embeddinggemma-2:740m` and `parse.pictures: true`, a new four-page PDF was ingested into the new experiment in 59 s (12 points, pictures among them) and nothing older was ingested again. "a diagram of an airplane turning back to the runway" with pictures only returns Figure 18-5 first (0.777). This is also the first run of pictures through Dagster, and it shows the family fill: the file names only the model.
- (f) With `chunk.stratgy`, the tick failed with "/app/config/ingest.yaml: chunk.stratgy: Extra inputs are not permitted", and the PDF ran once the key was corrected.
- Time: a two-page PDF takes 29 to 48 s as a run; from the copy to the start of the run is 30 to 70 s (the sensor's 30 s interval, then the queue and the start of the run's process).
- Cleaned up afterwards: the six test PDFs, their partitions and the two test experiments (`auto-ingest`, `auto-ingest-pictures`) are removed, and `config/ingest.yaml` is the default again, so your first PDF makes `auto-ingest`. The six runs stay in Dagster's run list.
- Not seen by eye: the runs and the sensor's ticks in the Dagster UI (read through its GraphQL API). Not tried: OCR through the sensor (`parse.ocr: true`), and a run that fails.

### Found in use: a long document was cut, and the run said success

The first real document through the sensor, `Homer-Odyssey.pdf` (444 pages, a text layer of 792,142 characters), gave 40 chunks in a run marked as a success.

- **Cause.** Docling stops at `parse.document_timeout` (600 s) and returns what it has, with the status `partial_success`; LlamaIndex's `DoclingReader` hands on only the document, so the status was lost and `ParsedDocument.status` said "success" whatever happened. The parse had reached page 38 of 444 (63,533 characters, 8% of the book). This PDF takes 3 to 7 s a page on this CPU (20 pages: 68 to 135 s, with the table stage on, fast or off, so it is not the tables), and 10 to 15 s a page while the UI container was also busy; the FAA chapter takes about 1 s a page.
- **Fix 1: a parse is whole or it fails.** `parse.py: RecordingConverter` keeps the conversion result, and `parse_pdf` raises when the status is not `success`, saying how many pages have content, the limit, and which settings to raise. It writes nothing. Checked by forcing a 25 s limit on the book: "Docling did not finish Homer-Odyssey.pdf (… document timeout exceeded): it has content for 0 of 444 pages after 40 s, with a limit of 25 s. Nothing was kept. Raise `parse.page_timeout` …", and no files.
- **Fix 2: the limit follows the size of the document.** New `ParseConfig.page_timeout` (20 s): the limit is `document_timeout` or `page_timeout` for each page, whichever is longer, so 444 pages get 8,880 s. It is left out of the config hash while it is the default, so no experiment's hash changed (the pinned-hash test passes; 50 tests).
- **Result.** The run was re-executed in Dagster with the same config. `parsed_document` now took 1,471 s (24.5 minutes, 3.3 s a page) and has all 444 pages (1,700 characters a page); `chunks` gave 477 chunks (460 text, 16 table, 1 picture). **`embeddings` then failed: "Ollama embed failed after 3 attempts".** The Mac that runs Ollama went offline on Tailscale during the parse (`tailscale status`: offline, last seen 18 minutes before; no reply to a ping). Nothing is wrong with the chunks; the step could not reach the model.
- **What is left.** The collection still holds the old 40 points of this document. When the Mac is back, *Re-execute from failure* on run `3c81827f` in the Dagster UI runs only `embeddings` and `qdrant_index` from the files on disk (a minute or two, no second parse), and replaces the 40 points with the 477.

### Open choices (defaults used above unless you say otherwise)

- One YAML, one experiment. Several experiments from one drop (a list in the file) is under Ideas.
- One run at a time, for all Dagster runs.
- `sparse: true` in the committed file, so the Chatbot can use what is ingested. The code default stays false.
- One test is proposed for the loader; say if you would rather have none.
- This is one YAML file for one purpose. It does not move the code's defaults into YAML: that larger question stays open in the next section, and this file fits its option B3 (a file is an experiment) if you choose it.

### Risks

- **A long run holds the queue.** With `ocr: true` a 23-page scan takes about 7 minutes, and everything behind it waits.
- **An older PDF is never picked up by itself.** That is the decision; it means the 5 PDFs in the folder, a PDF whose run failed, and the PDFs of an experiment deleted on the Experiments page all need a launch by hand, with the config typed into the launchpad. Prefilling it from the YAML is under Ideas.
- **The copy check is a heuristic.** A PDF that does not end in `%%EOF` within its last kilobyte is never taken. It would sit in the folder with nothing said, so the sensor's tick lists the files it is waiting for.

## Config in YAML (options; to be decided)

**What was asked:** the parameters of the system in YAML, not in Python. This section lists the ways to do it, with one file and with several, and what each costs. Nothing here is built; the choice is yours, and a phase is written from it (see "After you choose").

### Where configuration lives today

| Where | What is in it | Edited by |
|---|---|---|
| `src/rag_lab/config.py` (pydantic classes, also Dagster run config) | `ParseConfig`, `ChunkConfig` (with `HybridSettings`, `RecursiveSettings`, `SemanticSettings`), `EmbedConfig`, `IndexConfig`, `SearchConfig`, `ChatModelConfig` and `AgentConfig` (documents chatbot), `SqlAgentConfig` (database chatbot), `ImportConfig` (CSV import), `ExperimentConfig` (name, tag and the four ingest sections). About 90 values, each with a default and a comment. | Code |
| `src/rag_lab/benchmark/runner.py` | `BenchmarkSettings`, a dataclass of its own (models, strategies, `top_k`, `repeats`, `probe_count`, rerank probes). Not in `config.py`. | Code, and the Benchmark page |
| Constants in modules | BM25 `K1` and `B` (`embedding/sparse.py`), `SCANNED_PDF_CHARS_PER_PAGE` (`ingest.py`), `ROWS_KEPT` (`agent/events.py`), the catalog's `KEY_SUFFIXES` and `VALUE_LIMIT`, the guard's `ANONYMOUS_ALLOWED`, the reader role's limits (`sql/bootstrap.py`), `QUALITY_KS`, the prompts (`agent/*/prompts.py`). | Code |
| `.env`, through `docker-compose.yml` | Secrets and addresses: `OLLAMA_BASE_URL`, `POSTGRES_*`, `SQL_*_PASSWORD`; compose adds `QDRANT_URL`, the database URLs and `DATA_DIR`. | You, by hand |
| `ui/.streamlit/config.toml`, `ui/style.py` | The theme and the chart colours. | Code |
| The page widgets | The Upload, Try a query, Chatbot and Benchmark pages start from `ChunkConfig()`, `SearchConfig()`, `AgentConfig()` and so on, and let you change a value for one run. | You, per run |
| `rag_metrics.experiments.config`, `benchmark_reports.settings`, `chat_turns.settings` (jsonb) | What an experiment, a report or a turn actually ran with, written out in full. | The system |

Two things this shows. First, the defaults are already one source for the pages, the CLIs and Dagster, so a YAML layer would sit on top of classes that exist; it would not replace them. Second, the same choice is sometimes made in more than one place: the default reranker is `dengcao/Qwen3-Reranker-0.6B:Q4_K_M` in `SearchConfig` (which scored every chunk 0.0 when it was tried in Phase 15), `Qwen3-Reranker-4B:Q8_0` inside `AgentConfig`, and `SearchConfig().reranker` again in `BenchmarkSettings` and the Try a query page. One file where it is written once is the cleanest fix for that.

### Rules any option has to keep

From `CLAUDE.md` and how the system works:

- **A stage choice is a config value, and a new experiment is a new config, not a code edit.** YAML should make that easier, not add a second way.
- **The experiment's settings are recorded in full.** `experiments.config` holds every value, so an old experiment, a benchmark report or a saved chat turn does not change when a YAML default does. Anything that changes what is *stored* (the chunker, the embedding model and its dimension, `index.sparse`, and the BM25 `K1` and `B` if they were made settings) must be part of the experiment, so it is in its hash; only what is read *at question time* (the search method, the agent's model and thresholds, the import limits) is safe as a global default.
- **Secrets and addresses stay in `.env`.** A YAML file is committed and read by every container.
- **Stage code has no Dagster or Streamlit imports**, and the config classes subclass `dagster.Config` so they are also the launchpad form. Dagster's `Config` is a pydantic `BaseModel`, not a `BaseSettings`.
- **A typo fails loudly.** The classes already forbid unknown keys (`extra="forbid"`); a YAML layer should keep that and name the file and the key.
- **Tests run on the code defaults.** The few tests (config hashing among them) must not depend on a file someone edited.
- **The containers see `./src` and `./ui` through bind mounts, so no rebuild is needed for code.** The YAML would be mounted the same way, read-only.

### Questions that are the same for every layout

| Question | Choices | Notes |
|---|---|---|
| What goes in the file? | **Defaults only** (a missing key keeps the code default) · defaults **and experiment presets** (a whole ingest setup under a name) · defaults and presets and **agent presets** | Defaults only is the smallest change. Presets are what "a new experiment is a new config" asks for. |
| Does the file replace the Python defaults? | **No: the class keeps its default as the fallback and the file overrides it** · yes: the file is the only place defaults are written and the classes have none | The first works with no file at all (tests, a fresh clone) and keeps a default readable next to its class. The second removes the duplicate but breaks without the file. |
| Order of precedence | code default < YAML < `local.yaml` (git-ignored) < environment variable < a page widget or a run config | A widget or a Dagster run config is always the last word, as now. |
| Machine-specific values | A git-ignored `local.yaml` · an environment variable per value · edit the committed file | The chat and reranker model names depend on the Ollama server, and yours are a Mac's. |
| When is it read? | At import (a restart picks up a change) · on every page run (the pages pick it up at once; Dagster and the CLIs at their next start) | Reading per run is friendlier for the UI and adds a file read per interaction. |
| Environment variables inside the file (`${OLLAMA_BASE_URL}`) | No: secrets and addresses stay in `.env` · yes: interpolation | No is simpler and safer. |
| Do the prompts move too? | Stay in code · to `config/prompts/*.md` or `*.yaml` | The prompts are text with rules the code depends on (`CANNOT`, `[n]`); moving them lets you edit them without code, and lets a bad edit break the agents. A separate decision. |
| Can a page write the file? | No: YAML is edited by hand · yes | No. Writing YAML from a page loses comments and makes two writers. The pages keep their per-run widgets. |

### Option A — one file

A single `config/config.yaml`. Its sections mirror the classes, so a key is the class's field, and a section may be left out.

```yaml
# Overrides of the defaults in src/rag_lab/config.py. A missing key keeps its default; an unknown key is an error.
embed:
  model: qwen3-embedding:0.6b
  dimension: null              # native size
  keep_alive: 30m
chunk:
  strategy: hybrid
  max_tokens: 512
index:
  sparse: true                 # a new experiment gets a BM25 vector, so hybrid search is possible
search:                        # how the documents are searched, unless a page or a run says otherwise
  method: hybrid+rerank
  candidates: 20
  reranker: dengcao/Qwen3-Reranker-4B:Q8_0     # the one place the reranker is named
chatbot_documents:
  model: gemma4:e4b-mlx
  think: true
  top_k: 5
  enough_score: 0.5
  missing_score: 0.1
  max_rewrites: 1
chatbot_database:
  model: gemma4:e4b-mlx
  num_ctx: 16384
  schema_char_budget: 12000
  max_repairs: 2
  row_limit: 200
  examples_min_score: 0.55
csv_import:
  max_bytes: 52428800
  max_rows: 1000000
benchmark:
  repeats: 3
  probe_count: 20
```

- **A1, overrides only.** Exactly the above: the file holds the values you want to differ from the code. Smallest change; the file can be absent.
- **A2, plus presets.** The same file gains a `presets:` block of named experiment setups (`full-ingest`, `tables-256`) and agent profiles (`fast`, `careful`), which the Upload and Chatbot pages offer as a first choice and Dagster takes as `preset: full-ingest`.

| For | Against |
|---|---|
| One place to look, search and read; the whole system on one screen | Grows to a long file once presets are in it (about 90 defaults, and each preset repeats part of them) |
| One file to mount, one loader, one error message format | Everyone edits the same file, so diffs and merges touch the same lines |
| The reranker, the models and the thresholds are written once | A mistake in one section stops the load of all of them, unless sections are loaded and checked one by one |
| Easy to review as a whole: you see every knob together | Mixes things that change for different reasons (what an ingest stores, what a chat asks, what an import allows) |

### Option B — several files

All under `config/`. Each file has the same rules as Option A (overrides, unknown keys refused), but they are split. Four ways to split; B4 can be combined with any of the others.

**B1, by concern.** One file per kind of setting:

```
config/
  ingest.yaml      parse, chunk, embed, index          what a new experiment stores
  search.yaml      method, candidates, reranker, rerank prompt      how the documents are searched
  chatbot.yaml     the documents agent and the database agent       models, context, thresholds, budgets
  database.yaml    CSV import limits, schema budget, good answers   the tables side
  benchmark.yaml   repeats, probe counts, quality cut-offs
```

**B2, by flow.** One file per thing a user does:

```
config/
  shared.yaml      Ollama models used by every flow (embedding, reranker, chat), keep-alive times
  documents.yaml   ingest + search + the documents chatbot
  database.yaml    CSV import + schema + the database chatbot + good answers
  benchmark.yaml
```

**B3, defaults and a folder of experiments.** The "a new experiment is a new config" reading of the project:

```
config/
  defaults.yaml            what a new experiment starts from, and the search and agent settings
  experiments/
    full-ingest.yaml       name, tag, parse, chunk, embed, index for one experiment
    tables-hybrid-256.yaml
  agents.yaml              the two chatbots
  benchmark/
    paper-comparison.yaml  a benchmark's models, strategies and search methods
```

A preset file is only a starting point: when an experiment is first used its settings are written to `experiments.config` in full, so the file can change or be deleted later without touching what ran. The Upload page would offer "load a preset", the CLI `python -m rag_lab.ingest --preset <file>`, and a Dagster run would name the file instead of repeating the launchpad form.

**B4, layered.** A committed file and a git-ignored override next to it (`config.yaml` and `config.local.yaml`, or `search.yaml` and `search.local.yaml`). The override is merged over the committed file. It holds this machine's values (model names, a smaller `num_ctx`) without a branch.

| For | Against |
|---|---|
| A file is short and has one reason to change; a diff says what kind of change it was | More files to find; a value can be in the wrong one |
| Presets (B3) and per-flow files (B2) fit how you work: one file per experiment, one per flow | Shared settings (the embedding model, the reranker) must live in exactly one file or drift, which is the problem to be solved |
| A mistake in one file stops only what reads it | A loader, a merge order and a cross-file check are more code (about the size of a small module, not a framework) |
| Different people or runs can own different files | A search across files to answer "where is X set?" |
| Presets are natural: a file is a preset | More mounting and documentation to explain |

### Side by side

| | A1 one file, overrides | A2 one file, presets | B1 by concern | B2 by flow | B3 defaults + experiments | B4 local override (add-on) |
|---|---|---|---|---|---|---|
| Files to read to see the whole config | 1 | 1 (long) | 5 | 4 | 3 and a folder | +1 per file |
| Fits "a new experiment is a new config" | a little | yes | a little | a little | **best** | n/a |
| The reranker or a model named once | yes | yes | needs a `search.yaml` rule | needs `shared.yaml` | `defaults.yaml` | n/a |
| Size of the code change | smallest | small | medium | medium | largest (presets in the pages and Dagster) | small |
| Merge conflicts | most likely | most likely | rare | rare | rare | none (not committed) |
| Works with no file | yes | yes | yes | yes | yes | yes |
| Easiest to document | **yes** | yes | each file | each file | each file | one note |

### How the code could read it

1. **Overrides merged onto the class defaults, at import (smallest).** `config.py` loads the YAML once; each field's default becomes "the YAML value if there is one, else the value written in the class" (a small helper, so the fallback stays readable next to the field). The classes, their hashes, the pages and Dagster's launchpad all keep working, because they only see defaults. A change needs a restart of the container. No new dependency: PyYAML is already installed (Dagster, LangChain and LlamaIndex all require it).
2. **`pydantic-settings` with a YAML source.** Standard and tidy, but it needs the classes to be `BaseSettings`, and ours must stay `dagster.Config`; it would be a second layer of classes, and `pydantic-settings` is installed only by accident (it is not a declared dependency).
3. **An explicit `load_config()` that returns one object passed to whoever needs it.** Cleanest to test and no import-time effect, but about twenty call sites that write `AgentConfig()` today would change, and so would Dagster's resource definitions.

Reading per page run (a `load()` the pages call) instead of at import is possible with 1 or 3, if you want an edit to show in the UI without a restart.

### What stays out of YAML

- Secrets, passwords and addresses: `.env`.
- Everything that is the security boundary of the tables: the guard's allowlist and forbidden statements, the roles' limits, the reader's grants. A change to these should be a reviewed code change, not an edit to a data file.
- Parameters that change what is stored but are not experiment settings yet (BM25 `K1` and `B`): if they are to be tunable they belong in `IndexConfig`, so they are in the experiment's hash, not in a global file.
- The theme and colours (`config.toml`, `style.py`).
- The prompts, unless you decide otherwise (see the table above).

### Gotchas to design for

- **YAML 1.1 reads `yes`, `no`, `on`, `off` as booleans** (the `remote_work` column in your data is a reminder of how quietly that goes wrong). Keys and values would be written as `true` and `false`, and the loader refuses a non-boolean where a boolean is expected, since the classes are typed.
- **Hashes.** A new experiment made after a YAML default changed has different settings, so a different `config_hash`; that is correct. Old experiments are unaffected because their settings are stored in full. A check before and after the change: the hash of a fixed list of configs is identical with no file.
- **Tests** use the code defaults: either an environment variable that points the loader at nothing, or a conftest that sets it.
- **Dagster's code location** reads the file when it starts (about 50 s), so a change needs `docker compose restart dagster-code`; the UI and the CLIs are quicker.
- **A page shows its own defaults** from the same classes, so a YAML override shows there with no page change.
- **Windows and the containers** both read the file; paths are relative to `config/`, which is mounted at `/app/config`.

### After you choose

The phase that follows is short for A1 and larger for B3. Its shape:

- [ ] 1. The loader (`rag_lab/settings.py`): find the files (`RAG_LAB_CONFIG_DIR`, default `config/`), parse with `yaml.safe_load`, merge in the chosen order, and refuse unknown keys naming the file and the key.
- [ ] 2. The classes take their defaults from it (reading way 1 above, unless another is chosen); the benchmark's `BenchmarkSettings` and the stray constants you choose move into the config.
- [ ] 3. Compose mounts `./config` read-only into the containers; `.env.example`, README and `CLAUDE.md` say where each kind of setting lives.
- [ ] 4. With presets (A2, B3): the Upload page, the CLI and the Dagster run take a preset.
- [ ] 5. A first `config/` written from today's defaults, with every key commented out or equal to its default, so nothing changes until a value is edited.

**Done when:** with the new `config/` in place and nothing edited, the hash of a fixed list of configs is identical to today's and the 43 tests pass; editing one value (for example `search.reranker`) changes the Chatbot's and the Try a query page's default after a restart with no code edit; a misspelled key stops the start with a message that names the file and the key; and an experiment made before the change still shows its original settings. Checked once.

### What I need from you

1. **One file or several?** (A1, A2, B1, B2, B3, or a mix.) If you want a pick: A1 is the smallest change and enough to put the defaults in YAML; B3 is the one that matches "a new experiment is a new config".
2. **Defaults only, or also presets** of experiments and agents?
3. **A git-ignored `local.yaml` override** for machine-specific values (B4), yes or no?
4. **Read at start, or on every page run?**
5. **Do the prompts move to files you can edit?** If yes, `.md` per prompt or a YAML file.
6. **Which of the stray constants** (BM25 `K1` and `B`, `QUALITY_KS`, `ROWS_KEPT`, the scanned-PDF threshold, the benchmark settings) come into the config, and, for BM25, as experiment settings.

## Working approach

Keep the project minimal. Testing is deliberately light:

- **Tests:** only for pure logic where a silent bug would corrupt results (retrieval metric formulas, vector truncation and re-normalisation, config hashing). Nothing else gets a test file.
- **Verification:** each phase's "Done when" is checked once, by a real run against the real stack. No mocks, no fake embedders, no integration test suite.
- **After small edits:** do not rerun tests, rebuild containers or re-materialise assets. Rerun only when the logic of a tested area changed.
- **Ideas and deferred items** are built only when you ask for them.

## Scope

- **Documents:** PDF with a text layer.
- **Tables:** CSV files imported into PostgreSQL, one schema per dataset (Phases 21 to 28, planned).
- **Content:** text and tables. OCR for scanned PDFs (Phase 29) and pictures as searchable chunks (Phase 31) are built, both off by default; the chatbot's model is shown the pictures it retrieves (Phase 32).
- **Hardware:** no NVIDIA GPU on this machine. Docling runs on CPU; Ollama runs where `OLLAMA_BASE_URL` points.

## Ideas

Not planned and not in any phase. Each is built only when you ask for it.

- Text-to-SQL: per-schema reader roles, so the database and not only the guard keeps one dataset from reading another (today the reader may select from every schema in `rag_data`; one role per schema, made by the bootstrap since the loader cannot create roles, and the executor switching to it with `SET LOCAL ROLE`); retrieval of the relevant tables, for when a schema is too large to show the model whole (a Qdrant collection `schema__<name>` with a point per table, embedded from its catalog text; `tables_k` tables retrieved per question, all of them when the schema is small; `library.py` ignoring that prefix); a button that drafts table and column descriptions with the chat model; declared relationships between tables; appending rows on re-import; importing through a Dagster job; a result download button; a chart of a result.
- Chatbot: a "Performance" page or tab over `chat_turns` (time per step, tokens per second, abstain rate and score by settings, slowest turns), and renaming or pinning a chat.
- Chatbot: a tool-calling variant where the model chooses between searching, searching one document (`source_file` filter) and answering directly; a groundedness check that verifies each cited claim against its chunk; a Postgres checkpointer and a page of past chats; restricting a chat to chosen documents.
- Automatic ingestion (after Phase 33): several experiments from one drop (a list in `config/ingest.yaml`); removing a document's points when its PDF leaves `data/raw`; the launchpad of `ingest_job` prefilled from the YAML for a run started by hand; retrying a failed run automatically.
- A "Process with Dagster" option on the Upload page: copy the PDF to `data/raw/`, register its partition, launch `ingest_job` with the page's settings (and `tag`) as run config, and link to the run. The run would then show in the Dagster UI, at the cost of the page's chunk preview.
- Benchmark: several documents in one report; test queries generated by a local LLM from the document's own text; a Unicode font in the PDF so names with other characters are not replaced with `?`.
- Try a query: restrict the search to one document of an experiment (`source_file` already has a payload index).
- Experiments page: list the uploaded files and the parse cache (`data/uploads/`, `data/artifacts/_uploads/`) with the experiments that use each, and delete the ones nothing uses.
- On the upload page: parse options (`table_mode`, formulas), and picking a document already in `data/raw`.
- Parsing for the upload page as a Dagster run (a partition and `parsed_document`) instead of inside the Streamlit process, so it shows in the Dagster UI and does not use the UI container's memory.
- One `IngestionPipeline` with a docstore, to skip unchanged documents and dedupe chunks (see Phase 8 in `COMPLETED_PLAN.md`; it conflicts with per-stage files, so it would be a separate fast path).
- OCR and pictures (after Phases 29 to 32): OCR of a page that has a text layer, for a PDF whose text layer is broken; several OCR calls at once; the text inside a picture (a chart's labels) read with `Text Recognition:` and added to the picture chunk's text, which would help BM25 and the reranker; a reserved number of picture hits in a mixed search (a second `Prefetch` filtered to `modality = picture`), if the modality gap keeps pictures out of the top k; whole pages embedded as images; pictures in the Benchmark, with test queries that name a picture.
- Embedding model families beyond Qwen3 and EmbeddingGemma 2 (a new row in the family table of Phase 30).
- Quantisation settings in Qdrant.
- `--exact` search to measure what HNSW gives up. See Phase 5 in `COMPLETED_PLAN.md`.
- `docling-serve` as a separate container, compared against in-process parsing.

## Deferred

- **Image and figure ingestion** and **OCR** are no longer deferred: see Phases 29 to 32. What stays deferred from them: describing a picture in words with a vision model (pictures are embedded from the image instead), and Docling's own CPU OCR engines.
- **Other document types** (DOCX, PPTX, HTML). Docling handles them; the sensor and `ParseConfig` would need format-specific options.
