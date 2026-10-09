# Plan: the documents chatbot's flow

This plan puts `rag-flow-spec.md` into this repo. It covers the chat over the vector database only (`serve/chat_documents.py`). The text-to-SQL chat (`serve/chat_database.py`) is not touched; what it could reuse later is in the last section.

A phase is a goal, numbered steps with checkboxes, and a "Done when" line that is checked once on the real stack. Each phase ends with one run of the eval, and its numbers go into the table in [Results](#results). If a phase makes the numbers worse, work stops there and the numbers are reported.

**Contents**

1. [What the code does today](#1-what-the-code-does-today)
2. [The target, in this repo's names](#2-the-target-in-this-repos-names)
3. [Decisions](#3-decisions)
4. [Phases](#4-phases)
5. [Results](#results)
6. [Later: text-to-SQL](#later-text-to-sql)

---

## 1. What the code does today

The spec asks for this report before anything changes. The graph is built in `serve/chat_documents.py: build_graph`, its state is `AgentState` in the same file, a turn is run by `serve/run.py: run`, the search is `serve/search.py: search`, and the vector database is **Qdrant** (`core/qdrant.py`).

```
condense -> retrieve -> grade -> generate
               ^          |----> abstain
               |          v
               +------ rewrite
```

The spec describes an older state of this flow. Four of its five problems are already solved in part:

| The spec says | The repo today | Left to do |
|---|---|---|
| Every query goes through retrieval | True. `condense` always goes to `retrieve` | `analyze` with a route, and `direct_answer` |
| Retrieval is dense-only | **Not true.** `retrieve` is always `hybrid+rerank`: a dense and a BM25 prefetch fused with RRF inside Qdrant. The collection `doc` has the `bm25` vector, and a collection without it is not offered to the chatbot | No collection migration. Only several queries in one search |
| Grading is done by the chat model | Half true. `grade` uses the best reranker score; the model is asked only between `missing_score` (0.1) and `enough_score` (0.5) | Remove the model call; add a floor per chunk |
| The rewrite loop has no cap and repeats its strategy | The cap exists (`max_rewrites: 1`). `tried` is shown to the model but a repeated query is not refused, and there is one strategy | Refuse repeats; a strategy per attempt; cap of 2 |
| Nothing verifies the answer | True | `check_grounding` |
| No reranker yet | There is one: `OllamaReranker`, a Qwen3-Reranker model on Ollama, one call per chunk, scored `P(yes)/(P(yes)+P(no))` | Make it a node of its own; score against the question, not the rewritten query |

Checked on the running stack on 2026-10-09:

- LangGraph 1.2.14, langchain-ollama 1.1.0, qdrant-client 1.19.1, Ollama 0.40.1. Nothing in the spec is blocked by a version.
- **Structured output works.** `/api/chat` with a JSON schema in `format` returned valid `{"question", "route"}` JSON from both `gemma4:e4b` and `gemma4:e4b-mlx`, in about 1.0 to 1.5 s once the model was loaded (10 s cold). "thanks, that helps!" came back `direct`, a knowledge question `retrieve`.
- A documents question could not be asked: `config/llm.yaml` named the reranker `dengcao/Qwen3-Reranker-4B:Q4_K_M`, and Ollama has only the `Q8_0` build, so every question was refused with 409. Fixed with decision 2.
- The only collection was `doc`: one PDF (`db-19c-architecture.pdf`, English), 42 chunks, `qwen3-embedding:0.6b`, BM25 on. The work below uses a new one, `docs-gemma2` (decision 1).
- `chat_turns` is empty, so there are no real questions to build an eval set from.
- The benchmark of the old lab was removed in the October refactor (`docs/history/REFACTORPLAN.md`). The eval below is new and much smaller: one file of cases and one command.

## 2. The target, in this repo's names

The spec says to keep the names the code has. So `grade` stays `grade`, `retrieve` stays `retrieve`, and the state keeps `standalone`, `tried` and `rewrites`.

```
analyze --direct--> direct_answer ---------------------------------------> END
   | retrieve
   v
retrieve -> rerank -> grade --enough--> generate -> check_grounding --supported--> END
   ^                    | weak                            | unsupported
   |                    +-- rewrites left --> rewrite     v
   +---- new queries ----------------------------+     abstain -> END
                    (rewrite with nothing new, or no rewrites left -> abstain)
```

`retrieve` runs once after `analyze`, and again only after a `rewrite` that raised `rewrites`. `rewrite` runs only while `rewrites < max_rewrites`. So no path searches more than `max_rewrites + 1` times.

Model calls on the way to an answer: `analyze`, `generate`, `check_grounding`. The reranker is a model too, but not the chat model: one token per chunk and a score, no judgment in words.

### Nodes

| Spec | Here | Model call | What it does |
|---|---|---|---|
| `analyze` | `analyze` (new; replaces `condense` in this graph only) | yes, structured | The standalone question and the route, in one reply. A reply that cannot be read means `retrieve` with the question as typed |
| `direct_answer` | `direct_answer` (new) | yes | Answers from the conversation only. Its prompt forbids facts about the documents' subject |
| `hybrid_retrieve` | `retrieve` (kept) | no | Every query of `queries` as a dense and a BM25 search, fused by RRF in **one** Qdrant call, which returns each chunk once |
| `rerank` | `rerank` (new; today inside `search()`) | reranker | Scores the candidates against `standalone` and keeps the best `top_k` |
| `sufficiency_gate` | `grade` (kept) and the edge after it | no | Best score at or above `enough_score`: answer, from the chunks at or above `min_doc_score` |
| `rewrite` | `rewrite` (kept) | yes, structured | Attempt 1 rephrases; attempt 2 splits into 2 or 3 narrower questions, or broadens. A query already in `tried` is dropped |
| `generate` | `generate` (kept) | yes | As today, plus: say so when the passages answer only part |
| `grounding_check` | `check_grounding` (new) | yes, structured | `{"supported": bool, "claim": str}`. Off with `grounding_check: false` |
| `abstain` | `abstain` (kept) | no | The fixed message, worded by reason: `retrieval` or `grounding` |

### State (`AgentState`)

| Spec | Here | |
|---|---|---|
| `messages` | `question`, `history` | kept |
| `question` | `standalone` | kept |
| `route` | `route` | new |
| `search_queries` | `queries: list[str]` | replaces `query` |
| `candidates` | not kept in state: they go from `retrieve` to `rerank` in `retrieval` | |
| `context` | `context: list[Hit]` | new: the chunks the answer is written from. `retrieval.hits` stays the best `top_k`, so an abstained turn still shows its closest passages |
| `attempts` | `rewrites` | kept |
| `tried_queries` | `tried` | kept |
| `answer` | `answer` | kept |
| `citations` | `[n]` in the answer, read by `core/events.py: citations` | kept. The spec says "chunk id"; here a claim cites its passage number, which is the rank of a hit, and the hit carries the `chunk_id` |
| `grounded` | `grounded` | new |
| `outcome` | `outcome` | new: `answered`, `direct` or `abstained` |
| `abstain_reason` | `abstain_reason` | new: `retrieval` or `grounding` |
| | `scores: dict[chunk_id, float]` | new, see below |

Because every attempt is now scored against the same question, a chunk's score does not change between attempts. `scores` keeps them, so a retry only sends the reranker the chunks it has not seen, and `rerank` keeps the best `top_k` of everything found in the turn. This matters: the reranker is one Ollama call per chunk, and after a rewrite most candidates are the same chunks.

### Settings

All in one place, as the spec asks: `documents_chat` in `config/llm.yaml` (`AgentConfig` in `core/config.py`).

| Spec | Key | Today | After |
|---|---|---|---|
| `RETRIEVE_TOP_K` | `candidates` | 20 | 20 (the spec starts at 30; see the baseline in Phase 0) |
| `RERANK_KEEP` | `top_k` | 5 | 5 |
| `RERANK_THRESHOLD` | `enough_score` | 0.5 | set in Phase 2 from the eval |
| `RERANK_MIN_DOC_SCORE` | `min_doc_score` | | new, set in Phase 2 |
| | `missing_score` | 0.1 | removed, with the model's vote |
| `MAX_REWRITES` | `max_rewrites` | 1 | 2 (in Phase 3) |
| `ENABLE_GROUNDING_CHECK` | `grounding_check` | | new, `true` |

An unknown key stops the settings from loading, so `missing_score` leaves `llm.yaml` in the same commit as it leaves `AgentConfig`.

### Events (the contract with a front end)

The way a turn is asked and what the stream looks like do not change. These are additions, each with a default, so a saved turn still replays and a client that ignores what it does not know keeps working:

| Event | Change |
|---|---|
| `StepStarted` / `StepFinished` | New node names: `analyze` (in place of `condense` for a documents turn), `rerank`, `direct_answer`, `check_grounding` |
| `Query` | `route: str = "retrieve"` |
| `Rewrote` | `queries: list[str]` (all of them; `query` stays the first), `strategy: str` |
| `Retrieved` | Sent by `rerank`, once per attempt as today, with the kept hits |
| `Graded` | Same fields; `by` is always `score` |
| `Grounded` | New: `supported`, `claim` |
| `Done` | `route`, `rewrites`, `top_score`, `outcome`, `abstain_reason`. `abstained` stays |

`Done` is the per-request log the spec asks for. It is saved as the last of a turn's events, next to the `timings` column (milliseconds per node), so no migration is needed: `SELECT events->-1->>'outcome' FROM chat_turns`.

The spec puts UI changes out of scope. Two small ones cannot be avoided, because the page looks up every node name in `STEP_NAMES` and fails on one it does not know: the four new labels in `ui/trace_view.py`, and `apply` must not list a `direct` question under "Searched for".

## 3. Decisions

Confirmed on 2026-10-09, except 6, which is still the plan's own choice.

| # | Decision | Choice |
|---|---|---|
| 1 | **Which models** | `embeddinggemma-2:740m` for embedding and `gemma4:e4b-mlx` for chat, as the spec names them. A collection holds the vectors of one embedding model, so this is a new experiment: `docs-gemma2` in `config/pipeline.yaml`. `doc` (made with `qwen3-embedding:0.6b`) stays as it is and is not used here |
| 2 | **The reranker** | `dengcao/Qwen3-Reranker-4B:Q8_0` on Ollama, through `OllamaReranker`: multilingual, local, already behind one class with one method (`score`), no new dependency. Phase 0 measures what it costs per turn. The thresholds of Phase 2 belong to this build: the VM must use the same one, or the calibration is run again there |
| 3 | **The corpus of the eval** | A few more PDFs are added before Phase 0. (With the one PDF of 42 chunks, 30 candidates were 71% of the collection, so hybrid could not be told from dense and the thresholds would rest on one document) |
| 4 | **The language** | English, documents and questions. BM25 here splits on spaces (`core/embed.py: tokens`), which fits |
| 5 | **The answer and the grounding check** | With `grounding_check` on, the answer is held until it is checked and then sent whole (the thinking still streams), so a reader never sees an answer the check rejects. With it off, tokens stream as today. `call_model` already has `tokens=False` for this |
| 6 | **A check whose reply cannot be read** | Counts as supported, and the eval counts how often it happens. With a schema in `format` it should be never. *Not confirmed yet* |
| 7 | **The eval cases** | Written together once the PDFs are in: a draft of about 40 from the parsed documents, each marked `"reviewed": false`, corrected before the baseline counts |
| 8 | **Tests** | One small test file for the pure decisions: the gate (answer, rewrite, abstain, with the rewrites used up), the per-chunk floor, and the dropping of repeated queries. No graph run, no fakes. (`CLAUDE.md` says no tests unless asked; the spec asks that "tests cover the exhausted path") |

Also seen: Ollama lists `tev1:0.8b` and `tev1:4b` with the capability `decision`. I did not find out what they are, and the plan does not use them.

## 4. Phases

After a change under `src/`: `docker compose restart api`. After `ui/trace_view.py`: `docker compose restart ui`. `CLAUDE.md` and `INTEGRATE.md` (section 11: the nodes and events of a documents turn) are brought up to date in the phase that changes what they describe.

### Phase 0: the eval and the baseline

**Goal:** numbers for the flow as it is, from a command that is run again after every phase.

| Decision | Choice |
|---|---|
| Cases | `data/eval/<experiment>.jsonl`, git-ignored like the PDFs it is about. One case per line: `id`, `question`, `history` (pairs, for a follow-up), `expect` (`answer`, `abstain` or `direct`), `facts` (short strings the answer must contain), `source` (`file`, `pages`, `snippets`), `tags`, `reviewed`, and an optional `note` for the reader |
| The mix | The first set is 20 cases on the two PDFs of `docs-gemma2`: 15 answerable (7 on each document, 5 of them tagged `exact-term`: process names, parameter names, codes; and 1 follow-up that needs the history), 3 the documents cannot answer, 2 conversational. It grows towards the spec's 30 to 50 as documents are added |
| The source of a case | A snippet: a few words copied from the passage that holds the answer. A search found the source when a kept hit of that file contains one of the snippets (compared with whitespace collapsed). This names one chunk, where a page can hold four, and it survives another chunking. `pages` is for the person who reviews the case |
| Command | `python -m rag_lab eval --experiment <name> --label <label> [--against <label>]`. The logic is `serve/evaluate.py`; the command in `cli.py` is a thin caller |
| How a case runs | Through `run()` with the case's history, the settings of `llm.yaml`, and no `session_id`, so it is the real graph and no chat is saved |
| Answer correctness | Every string of `facts` is in the answer, ignoring case. No model judges: the number must mean the same in every phase. The answers are written to the results file to be read |
| Output | `data/eval/results/<experiment>-<label>.json` (every case with its events' summary) and a printed table; `--against` prints the change |

What the command reports:

- Retrieval hit rate: a kept hit contains one of the source's snippets (`expect: answer` cases).
- Answer correctness, as above.
- Abstaining: of the turns that abstained, how many should have (precision), and of those that should have, how many did (recall); by reason once there is one.
- Route accuracy, and separately the count of knowledge questions sent to `direct`. That count is the dangerous one: such a question is answered from the model's memory.
- Latency: median and p95 of the total, the median per node, model calls per turn, and searches per turn with the maximum.
- The best first-attempt reranker score of every case next to its `expect`. Phase 2 sets the thresholds from this.
- Answers that were given but say the documents do not hold the answer. They count as answered; the list shows how common they are.

Steps

- [x] 1. `config/llm.yaml`: `embeddinggemma-2:740m`, `gemma4:e4b-mlx` and the reranker build `Q8_0`. `config/pipeline.yaml`: `name: docs-gemma2`. The settings load on both sides and Ollama has the three models.
- [x] 2. The PDFs are in `docs-gemma2`: `db-19c-architecture.pdf` (42 chunks) and `flying_handbook.pdf` (51 chunks, 23 pages), text only.
- [x] 3. The cases: `data/eval/docs-gemma2.jsonl`, 20 cases, written from the chunks and checked against them (every snippet is in a chunk on the pages named, every fact is in that chunk, and the three unanswerable topics are in no chunk). Reviewed on 2026-10-09.
- [x] 4. `serve/evaluate.py` and the `eval` command.
- [x] 5. Run it with `--label phase0`. Fill in [Results](#results).

**Done when:** the baseline row is in the table, with how long one run of the eval takes. *Done: one run takes about 11 minutes.*

What the baseline shows (`gemma4:e4b-mlx` with thinking, `Qwen3-Reranker-4B:Q8_0`, 20 candidates, top 5, `max_rewrites: 1`):

- **The answerable cases are at the ceiling**: 15 of 15 sources found and 15 of 15 answers correct, the follow-up and the five exact-term cases among them. So the later phases cannot show a gain here, only a loss. Their gains can show in abstaining, in the route and in latency. Harder cases (an answer spread over two passages, a question worded far from the text) are worth adding when the set grows.
- **The score separates cleanly on the first search**: 0.975 to 1.000 for the fifteen answerable cases, 0.000 to 0.032 for the three unanswerable ones and the greeting. A gate by score alone (Phase 2) has room.
- **Every wrong turn came from the rewrite, because the second search is scored against the rewritten query and not the question.** "Hi there!" was rewritten to a general query, a passage scored 0.514 against that, and the model was asked to answer a greeting from it (43 s, two searches). The carburetor question scored 0.010, was rewritten, and its second search scored 1.000. In both the model then said the documents do not contain it, so nothing false was shown, but the turn counts as answered. Phase 2 (score against the question) and Phase 4 (the route) are aimed at exactly these.
- **The outcome changes from run to run.** Asked again, the carburetor question got another rewrite ("Carburetor icing emergency procedures"), scored 0.149, and was abstained on correctly. The temperature is 0.2 and there are three unanswerable cases, so a difference of one case between two runs is not a result.
- **Where the time goes**: a median turn is 31 s: about 16 s in `retrieve` and 14 s in `generate`. Embedding the query (0.1 s) and Qdrant (0.05 s) are nothing; the 16 s is the reranker's 20 calls. Ollama keeps all three models loaded, so it is not loading. A search whose question and chunks were scored shortly before takes about 2 s, because Ollama still has those prompts evaluated (seen in Phase 1: the same case asked six times took 15.5 s, then 3.9 s, then 2.1 s each). So about 16 s is what a new question costs. The eval records `rerank_ms` for each case.
- **`candidates` should stay 20**, not go to the spec's starting value of 30: at these times 30 would add about 8 s to every search, and at 20 the source is found 15 times of 15.

### Phase 1: the loop is bounded, and a turn says how it ended

**Goal:** no turn searches more than `max_rewrites + 1` times, a repeated query is never searched again, and every turn records its route, rewrites, best score, outcome and reason.

- [x] 1. `core/events.py`: the new fields of `Done`. `serve/run.py`: `Summary` carries them; `SqlFlow` leaves them empty.
- [x] 2. `chat_documents.py`: `outcome`, `abstain_reason` and `top_score` in the state (`abstained` is now read from `outcome`). `abstain` sets the reason `retrieval`, and its message suggests how to ask again. The wording for `grounding` comes with Phase 5, when that reason exists.
- [x] 3. `rewrite`: a query that is already in `tried` (compared without case and extra spaces), or an empty one, spends the attempt and sends no `Rewrote`. A new edge after `rewrite`: to `retrieve` with a new query, else to `rewrite` or `abstain` by the same rule as after `grade`.
- [x] 4. `after_grade`, `after_rewrite` and `is_new` are functions at module level, with `tests/test_documents_gate.py` (decision 8): five tests, one of which walks the two decisions for a turn that never finds enough and counts its searches.
- [x] 5. `cli.py`: the printer shows the outcome.
- [x] 6. Run the eval (`--label phase1 --against phase0`).

**Done when:** the eval's "searches per turn, maximum" is at most `max_rewrites + 1`, and the test of the used-up path passes. *Done: the maximum is 2 with `max_rewrites: 1`, and the tests pass.*

What the run shows:

- **No rewrite repeated a query in this run**, so the new edge was exercised by the tests only. Nothing Phase 1 changed could move an answer: the differences from the baseline are the model's own variation.
- **`f07` is a coin toss, and it is why "correct" went from 15/15 to 14/15.** The passages and scores were the same as in the baseline (best 0.975), and the answer was "the documents do not contain it". Asked six more times with the same passages, it was right three times and wrong three. The passage says "the untrained instrument pilot" and "no more than 10° bank angle"; the question says "a VFR pilot caught in IMC", and the model does not always connect the two.
- **The abstaining numbers (3/3, were 2/3) are the same variation in the other direction**: the carburetor question got another rewrite this time.
- **So one run cannot tell a change of one case from chance.** The answer is written at `temperature: 0.2`. Proposed before Phase 2: `documents_chat.temperature: 0` (the database chat already has it), so that a case goes the same way every time and a difference between two runs means something.

### Phase 2: a rerank node and a gate by score alone

**Goal:** `grade` makes no model call, and its two thresholds are in `llm.yaml` with the eval's evidence written above them.

- [x] 1. `serve/search.py`: the reranking leaves `search()` and becomes `rerank_hits(question, hits, reranker, cfg, scored)`. The methods `dense+rerank` and `hybrid+rerank` go with it (the agent was their only caller), and `SearchConfig` no longer carries the reranker. `Retrieved.method` still says `hybrid+rerank`.
- [x] 2. `chat_documents.py`: the node `rerank`, scoring against `standalone`. The state keeps `scored`, every chunk the turn has scored, best first; `retrieval.hits` is its best `top_k`. It sends `Retrieved`.
- [x] 3. `grade` by score alone; `context` is the hits at or above `min_doc_score` (`above`, with a test). The kept hits are in score order, so the dropped ones are the last, and passage numbers stay the ranks the page shows. `generate` and the citation count read `context`.
- [x] 4. Remove `GRADE_SYSTEM`, `grade_prompt` and `missing_score`; add `min_doc_score`.
- [x] 5. `ui/trace_view.py`: the label for `rerank`. Left as it is, since the UI is out of scope: the "Retrieved chunks" tab shows all the kept chunks and does not mark one that was below `min_doc_score` and so was not given to the model.
- [x] 6. Run the eval with the thresholds as they are (`--label phase2`, `min_doc_score: 0`).
- [x] 7. The values, and the numbers behind them, are above the keys in `core/config.py` and `llm.yaml`. `min_doc_score` changed, so the eval ran once more (`--label phase2-floor`).

**Done when:** no `ModelState` with `node: grade` appears in the eval, and correctness and both abstaining numbers are not below Phase 1. *Done: `grade` takes 0 ms and the unanswerable cases make one model call (the rewrite); 15/15 correct, 3/3 unanswerable cases abstained.*

The thresholds (Qwen3-Reranker-4B:Q8_0, 20 cases, scores against the question):

| Key | Value | Evidence |
|---|---|---|
| `enough_score` | 0.5 (as before) | The best chunk scored 0.975 to 1.000 for the 15 answerable questions and 0.000 to 0.032 for the 3 unanswerable ones, also after their rewrite. Every value between separates them; 0.5 is the middle |
| `min_doc_score` | 0.1 (new) | The 18 chunks that hold an answer scored 0.512 or more, 17 of them 0.975 or more. Of the 57 other kept chunks of those questions, 37 scored below 0.1 and 12 scored 0.5 or more. Any value up to 0.512 keeps every source; it is set low so that only a chunk the reranker is sure about is left out, because the eval has no question whose answer needs a second, weaker passage |

What the runs show:

- **Scoring against the question is what fixed the wrong turns of the baseline.** "Hi there!" scored 0.003, was rewritten ("What is the status of the system?"), and its best chunk was still 0.007, so it now abstains instead of being answered from a passage that matched the rewrite. The eval counts that as wrong (it expects `direct`, which is Phase 4), and it is why abstain precision reads 3/4.
- **A second search costs less**: the unanswerable cases took 20 to 26 s, where Phase 1 took 31 to 35 s, because the rewrite's search brings mostly chunks that are scored already.
- **The floor changed no outcome and no timing** (the two runs have the same numbers; the median `generate` was 10.7 s in both). On these cases it keeps about two of every three passages that do not hold the answer away from the model.
- **`f07` was right in both runs**, but it was a coin toss in Phase 1 and the temperature is still 0.2, so that is not a result.

### Phase 3: several queries, and a strategy per rewrite

**Goal:** a rewrite can search for more than one query, each attempt tries something different, and there are two attempts.

- [ ] 1. `serve/run.py`: one helper for a structured reply: a JSON schema in Ollama's `format`, the reply read with pydantic, `None` when it cannot be read. `ModelState` is reported as for any call.
- [ ] 2. `core/qdrant.py: query` and `search()` take several queries: one `/api/embed` call for all of them, one Qdrant call with a dense and a BM25 prefetch per query, fused by RRF. One dense query with no BM25 stays the plain search, so `python -m rag_lab search` still gets a cosine similarity.
- [ ] 3. `rewrite`: the prompt of attempt 1 asks for one rephrasing (synonyms, abbreviations written out); attempt 2 for two or three narrower questions, or a broader one. `Rewrote` carries `queries` and `strategy`.
- [ ] 4. `max_rewrites: 2`.
- [ ] 5. The spec's evidence for hybrid search, which is already on: the eval's `exact-term` cases searched once with `dense` and once with `hybrid`, hit within `top_k` before reranking. One small option of the eval command (`--search-only <method>`); worth reading only with the larger corpus of decision 3.
- [ ] 6. Run the eval.

**Done when:** the eval shows turns answered after a rewrite that Phase 2 abstained on, or shows there are none to win; the maximum of searches per turn is 3; and the p95 latency of abstained turns is reported, since two attempts make "not found" slower.

### Phase 4: `analyze` and the direct route

**Goal:** a greeting or a question about the conversation is answered without a search.

- [ ] 1. `analyze`: the standalone question and the route in one structured reply, from the last `history_turns` pairs. When unsure, `retrieve`; a reply that cannot be read is `retrieve` with the question as typed. The SQL graph keeps `condense` and `run.py: standalone_question`.
- [ ] 2. `direct_answer`: streamed, without thinking, from the history only. `outcome` is `direct`.
- [ ] 3. `DocumentsFlow.summarise`: a direct turn has no hits and writes no search log.
- [ ] 4. `Query.route`. `ui/trace_view.py`: the labels for `analyze` and `direct_answer`; a direct question makes no "Searched for" line. `cli.py`: the printer.
- [ ] 5. Run the eval.

**Done when:** every conversational case goes `direct` with no `retrieve` step, route accuracy is reported, and **no knowledge question was sent to `direct`**. Also reported: what the first question of a chat now costs, since today it makes no model call before the search and `analyze` makes one (about 1 to 1.5 s in the probe above).

### Phase 5: the grounding check

**Goal:** an answer that the passages do not support is not shown; the reader gets the abstain message with the reason `grounding`.

- [ ] 1. `check_grounding`: given the passages exactly as `generate` saw them (the pictures too, when they were given) and the answer; a structured `{"supported", "claim"}`, without thinking. It sends `Grounded`.
- [ ] 2. With `grounding_check` on, `generate` holds its tokens and `check_grounding` sends the answer whole when it is supported (decision 5). With it off, the node is not in the graph and tokens stream as today.
- [ ] 3. `ui/trace_view.py`: the label, and the verdict in "Steps and model". `cli.py`: the printer.
- [ ] 4. Run the eval twice, with the flag on and off.

**Done when:** the eval reports, for the flag on against off: correct answers the check threw away, wrong answers it caught, and the latency it adds (median and p95). The small model is a weak judge, so the first number decides whether the default stays `true`.

## Results

One row per run of the eval. Filled in as the phases are done.

| Run | Cases | Hit rate | Correct | Abstain precision / recall | Route accuracy | Knowledge sent to `direct` | Median / p95 s | Model calls, median | Searches, max |
|---|---|---|---|---|---|---|---|---|---|
| phase0 | 20 | 15/15 | 15/15 | 2/2 / 2/3 | 18/20 | 0 | 30.9 / 40.6 | 1 | 2 |
| phase1 | 20 | 15/15 | 14/15 | 3/3 / 3/3 | 18/20 | 0 | 27.3 / 34.9 | 1 | 2 |
| phase2 (`min_doc_score: 0`) | 20 | 15/15 | 15/15 | 3/4 / 3/3 | 18/20 | 0 | 26.3 / 31.0 | 1 | 2 |
| phase2-floor (`min_doc_score: 0.1`) | 20 | 15/15 | 15/15 | 3/4 / 3/3 | 18/20 | 0 | 26.1 / 31.1 | 1 | 2 |
| phase3 | | | | | | | | | |
| phase4 | | | | | | | | | |
| phase5, check on | | | | | | | | | |
| phase5, check off | | | | | | | | | |

## Later: text-to-SQL

Not planned here. What this work leaves ready for it:

- The structured-reply helper of Phase 3, for `write_sql`'s `CANNOT` and for a route.
- `outcome` and `abstain_reason` on `Done`: `SqlFlow` can fill them (`schema too large`, `cannot`, `repairs used up`).
- The eval command, with a cases file per schema and a reference SQL in place of `facts` and `source` (this was Phase 28 of `docs/history/PLAN_depricated.md`).
- A direct route, so that "thanks" does not write a query. Choosing between documents and tables in one chat is a larger change: a chat has one `kind` for its whole life, and the database enforces it.
