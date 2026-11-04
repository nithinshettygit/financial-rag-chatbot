# Financial RAG Application — Technical Report

## 1. Purpose and scope

This report explains the working application in `app/`: its runtime
architecture, core request path, primary functions, retrieval and grounding
strategies, configuration, and operational behavior. It is intended as a
technical handoff and as source material for project demonstrations and
presentations.

The application is a financial-filings question-answering service. It accepts
chat requests, resolves intent and conversational context, searches indexed
filing chunks, ranks evidence, generates a cited answer, applies safeguards,
and returns both the answer and source chunks. The core backend is a FastAPI
service. A Streamlit chat UI is included as a client of that API.

### Scope boundary

- `app/main.py` exposes the HTTP API and manages its startup/readiness lifecycle.
- `app/pipeline.py` coordinates query planning, cache, retrieval, reranking,
  context assembly, answer generation, and response construction.
- `app/utils/` contains the specialized retrieval, routing, caching, guardrail,
  PII, model-client, token, and logging components.
- `app/guardrails_config/` configures NeMo Guardrails flows and custom actions.
- `app/app_ui.py` is the Streamlit client.
- Document parsing and Qdrant seeding are performed by the separate `admin/`
  ingestion workflow; this report focuses on serving and answering questions.

## 2. Executive overview

The system is a hybrid retrieval-augmented generation (RAG) application with
three important design characteristics:

1. **Queries are planned before retrieval.** A semantic intent classifier
   provides confident fast paths; a Groq-backed router then interprets
   financial intent, companies, metrics, years, follow-ups, and search wording.
   Deterministic company/year logic supplements the model's output.
2. **Evidence is found and selected by multiple stages.** Dense Qdrant
   retrieval and local BM25 lexical retrieval run concurrently. Reciprocal
   Rank Fusion (RRF) combines results. FlashRank reranks them, with an
   entity-balanced strategy for cross-company questions. A token-limited
   context builder adds source tags and citation metadata.
3. **Answers are constrained and verified.** Input PII masking and policy
   checks precede normal RAG processing. The answer prompt requires grounding
   and citations. Output rails check prompt leakage, PII, and numerical
   grounding; a failed numerical check can trigger one stricter generation
   retry.

```mermaid
flowchart TD
    UI[Streamlit client<br/>app/app_ui.py] --> API[FastAPI<br/>POST /v1/chat]
    API --> VALID[ChatRequest validation<br/>app/schemas.py]
    VALID --> PIPE[run_rag_pipeline_async<br/>app/pipeline.py]
    PIPE --> FAST[Semantic intent fast path]
    FAST -->|smalltalk / out of scope| FASTRESP[Fast response / refusal]
    FAST -->|normal query| IN[Input guardrails<br/>PII + policy]
    IN --> PLAN[Query planning<br/>intent, follow-up, entities,<br/>years, metrics, state]
    PLAN --> SPECIAL{Intent path}
    SPECIAL -->|chat summary / general chat| CHAT[LLM response + free-text output rails]
    SPECIAL -->|financial RAG| CACHE[Scoped answer cache<br/>Redis L1 exact + L2 semantic]
    CACHE -->|hit| CACHED[Cached answer and sources]
    CACHE -->|miss| FILTER[Catalog resolution<br/>document/company/year filters]
    FILTER --> SEARCH[Parallel dense and BM25 search]
    SEARCH --> FUSE[Entity-balanced RRF fusion]
    FUSE --> RERANK[FlashRank reranking]
    RERANK --> CONTEXT[Token-budgeted evidence context<br/>with [S1] citations]
    CONTEXT --> GEN[Groq answer generation]
    GEN --> CHECK[Output rails<br/>numeric grounding, prompt leak, PII]
    CHECK -->|numeric failure| RETRY[One strict-prompt retry]
    RETRY --> CHECK
    CHECK -->|allowed| SAVE[Build response and cache if complete]
    CHECK -->|blocked| BLOCK[Refusal / verification message]
    FASTRESP --> API
    CHAT --> API
    CACHED --> API
    SAVE --> API
    BLOCK --> API
    API --> UI
```

## 3. Application architecture

### 3.1 API and lifecycle

[`main.py`](./main.py) owns the HTTP boundary:

- `FastAPI` is configured with a lifespan manager.
- Startup sets readiness false, warms service/model dependencies, initializes
  the financial cache, then sets readiness true.
- `GET /health` reports process health.
- `GET /health/ready` reports whether startup warmup completed.
- `POST /v1/chat` validates a `ChatRequest`, invokes the RAG pipeline, and
  applies the configured request deadline. It returns a `ChatResponse`, a 504
  on timeout, and a 500 for unexpected processing failures.
- FastAPI and HTTPX calls are instrumented with Logfire.

`lifespan()` calls `warmup_pipeline()` and `init_financial_cache()`. Warmup
preloads the vector retriever, BM25 index, reranker, Redis/semantic cache,
guardrails, PII analyzer, intent router, Groq clients, and document catalog.
Startup exceptions are logged and re-raised: the API does not become ready if
required initialization fails.

### 3.2 Request and response contract

[`schemas.py`](./schemas.py) defines the external data shapes:

- `ChatRequest`: required `session_id`, 1–2000 character `query`, optional
  conversation `history`, and optional `SessionState`.
- `ChatTurn`: a `role` and `content` for a prior turn.
- `SessionState`: company, fiscal year, target metric, and document ID carried
  between requests.
- `ContextChunk`: the source chunk plus its document, file, page, type, headings,
  and ranking scores.
- `ChatResponse`: session ID, answer, intent, normalized search query, updated
  state, retrieved chunks, and citation source map.

The response keeps the user-facing answer and machine-readable evidence
together. For financial answers, the `source_map` maps tags such as `[S1]` to
file/page/section descriptions, while `retrieved_chunks` lets a client inspect
the actual retrieved text.

### 3.3 Core coordinator

[`pipeline.py`](./pipeline.py) is the main orchestration module. It uses
lazy-initialized, process-local singletons for the Qdrant retriever, BM25
retriever, and FlashRank reranker. Initialization is guarded by a thread lock.
Blocking work (model calls, vector search, cache operations) is moved off the
async event loop using `asyncio.to_thread`; independent retrievals are awaited
concurrently.

Core public/internal entry points:

| Function | Responsibility |
| --- | --- |
| `warmup_pipeline()` | Concurrently initializes expensive models/services and refreshes the catalog before serving requests. |
| `get_catalog(force=False)` | Caches the indexed document catalog for `CATALOG_TTL_S`; falls back to manifest-derived BM25 documents if Qdrant returns no documents. |
| `resolve_document_ids(entity, document_id, catalog)` | Converts a company/entity into one or more indexed document IDs when possible. |
| `_search_target(text, filters, top_k)` | Runs dense and BM25 retrieval concurrently under a search semaphore; can retry with relaxed company/year filters when strict retrieval yields too little. |
| `hybrid_search_balanced(sub_queries, entity_filters, total)` | Fuses lexical/vector rankings with RRF and distributes candidates by entity. |
| `build_context(results, max_tokens)` | Packs selected chunks into a bounded, citation-tagged context and returns the source map and included chunks. |
| `run_rag_pipeline_async(request)` | Executes the end-to-end request path and builds the response. |

## 4. End-to-end request lifecycle

The following sequence describes a normal financial-filings question.

### Stage 1 — Request validation and readiness

1. The client sends `POST /v1/chat`.
2. Pydantic validates the request shape and query length.
3. If startup warmup is incomplete, the endpoint returns HTTP 503.
4. `chat_endpoint()` starts a Logfire span and awaits
   `run_rag_pipeline_async()` with the configured `REQUEST_TIMEOUT_S` (default
   40 seconds).

### Stage 2 — Semantic fast path

The pipeline calls `get_intent_router()` and, when available, embeds the query
against labelled examples in `app/data/intent_examples.yml`.

- A confident `smalltalk` query uses `_answer_greeting()`: it checks the exact
  greeting cache, generates a short answer if needed, and stores the response
  with a longer greeting TTL.
- A confident `out_of_scope` query returns the configured scope refusal without
  invoking the main RAG path.
- The fast path is not used when PII masking would alter the input.
- If the semantic router is unavailable, an anchored greeting regex can
  provide a degraded greeting fast path. Other uncertain queries continue to
  the LLM router.

The classifier is conservative by design: a result must clear score and
runner-up margin thresholds, and some labels must be short messages. A
low-confidence result incurs extra routing work rather than forcing an
incorrect classification.

### Stage 3 — Input guardrails and privacy

For non-fast-path requests, `check_input()` runs configured NeMo input rails:

1. mask PII/secrets;
2. detect prompt-injection / invalid-length heuristics;
3. ask a policy model whether the query is in scope.

The output of this step is the safe query used from then on. Rejected requests
return a refusal, and the raw request is not echoed as the search query in the
response. Prior turns are passed through `_sanitize_history()` before use in
query planning.

If NeMo fails or times out, the configured fallback path uses deterministic
injection checks and PII masking when `GUARDRAILS_FAIL_OPEN` is true. When it is
false, the request is blocked with a temporary-unavailability message.

### Stage 4 — Catalog lookup and query planning

`get_catalog()` refreshes its document list when forced or when its TTL expires.
The preferred catalog comes from Qdrant. If the vector collection has no
documents, the application derives a catalog from BM25's loaded manifests.
Company names can be inferred from known indexed metadata or document/file
names for the current corpora.

`plan_query()` combines:

- the latest safe query;
- sanitized recent chat history;
- session state;
- indexed company names;
- the optional semantic-router label; and
- a Groq JSON-mode router result.

The plan contains intent, follow-up type, entities, years, financial metrics,
standalone search wording, per-entity subqueries, and updated session state.
Deterministic logic normalizes company names, detects indexed companies
mentioned explicitly, extracts years from the query, and carries state where
appropriate. Follow-ups are classified as:

- `none`: self-contained query;
- `reference`: a new question that depends on omitted conversational details;
- `elaboration`: requests for additional angles on an already discussed topic.

For an elaboration, the router requests new aspects (for example, segment
breakdown, drivers, or year-over-year change) so retrieval is not limited to
the same evidence used in the previous answer.

If the router LLM fails, its helper returns an empty result and deterministic
fallback logic still constructs a plan. The resulting intent defaults to
financial RAG rather than failing the whole request.

### Stage 5 — Intent-specific paths

Before RAG retrieval, the pipeline handles:

- `out_of_scope`: returns a scope refusal;
- `general_chat`: generates a short capability/greeting response and runs
  free-text output rails;
- `chat_summary`: summarizes the supplied conversation history and runs
  free-text output rails.

These paths do not use filing retrieval. Summaries and the router's
general-chat path are not stored in the RAG answer cache.

### Stage 6 — Scoped cache lookup

RAG cache keys are built from a `CacheScope` containing:

- intent;
- normalized entity set;
- requested year set;
- a corpus version derived from catalog document IDs and chunk counts; and
- for elaborations, a hash of the previous assistant answer.

The question text depends on follow-up type: reference questions use the
standalone rewritten question, while normal and elaboration queries use the
current query. Elaboration cache lookup is exact-only because the answer
depends on prior conversation context.

[`cache.py`](./utils/cache.py) implements:

- **L1 exact cache:** Redis string lookup keyed by scope and normalized text.
- **L2 semantic cache:** RedisVL vector search using BGE-small embeddings,
  filtered by intent/entity/year/corpus scope.
- **Lexical verification:** semantic candidates must satisfy a distance
  threshold and sufficient content-term overlap; number sets must match.

L2 hits are promoted into L1. Cache read/write failures are logged and return a
miss or continue processing rather than failing the user request. Complete RAG
answers are stored asynchronously; incomplete multi-entity answers are not
cached.

### Stage 7 — Document resolution and hybrid retrieval

For each planned entity, the coordinator resolves document IDs from the
catalog, then builds filters for years, company, and document IDs. An unknown
requested company is reported rather than silently broadening to all
documents.

`hybrid_search_balanced()` creates subqueries for each entity/metric
combination and searches each one:

1. **Dense retrieval:** `QdrantVectorRetriever.retrieve()` embeds the query with
   `BAAI/bge-small-en-v1.5` and queries the configured named vector in
   `financial_rag_analytics`.
2. **Lexical retrieval:** `BM25Retriever.retrieve()` tokenizes query and
   searchable chunk text, preserving financial numbers, years, percentages,
   hyphenated terms, and metadata terms.
3. **Parallel execution:** vector and BM25 results run concurrently for each
   target; a semaphore caps concurrent search targets.
4. **Soft filter retry:** if strict year/company filtering returns fewer than
   `MIN_STRICT_YIELD` distinct items, the pipeline retries without those
   filters while retaining document-ID restrictions.
5. **RRF fusion:** each ranking contributes `1 / (RRF_K + rank)` to the
   candidate's score; duplicate candidate IDs within a ranking are counted
   once.
6. **Entity balance:** per-entity quotas prevent one company's results from
   dominating a multi-company comparison; remaining capacity is filled from
   the best leftover scores.

The dense path supports metadata filters in Qdrant. BM25 loads manifests from
the local data directory and applies equivalent company/year/document filters
in memory. Because these stores have different sources, successful admin
indexing should update both the Qdrant collection and the local manifest
directory used by BM25.

### Stage 8 — Reranking and evidence packing

`FlashRankReranker.rerank_balanced()` reranks candidate groups per entity for
multi-company comparisons and interleaves the best results round-robin. For a
single entity it behaves like ordinary reranking. If FlashRank errors, the
reranker logs the failure and returns the initial candidate order with a
fallback score.

`build_context()` then:

- creates source tags `[S1]`, `[S2]`, and so on;
- adds filename, page, section headings, chunk type, and content;
- checks each candidate against the configured context token budget
  (`MAX_CONTEXT_TOKENS`, default 4096);
- excludes chunks that would exceed the budget; and
- returns the final source map and exact set of included chunks.

This source map is the basis for answer citations and the returned source
details.

### Stage 9 — Answer generation and output checks

`_system_prompt()` instructs the model to rely only on retrieved context, cite
sources, preserve monetary scale/units, and state when evidence is absent.
For elaboration it asks for new detail without repeating prior figures. A
strict second-attempt prompt requires every figure to be copied from the
evidence or calculated transparently from it.

`_generate_answer()` sends the system message and user question to the Groq
model. Recent conversation turns are included only for elaboration follow-ups.
`chat_with_retry()` rotates across configured Groq keys on rate limits where
possible and respects retry hints within a bounded wait budget.

`check_output()` invokes configured output rails:

- detect system-prompt leakage;
- mask configured PII in generated text;
- check answer figures against retrieved evidence.

If the output is blocked specifically for a numeric-grounding reason, the
pipeline retries once with the strict prompt. Other blocking failures are
returned without another generation attempt. If guardrails fail technically,
deterministic fallback checks are used when configured to fail open.

### Stage 10 — Completeness, response, and cache store

The pipeline checks whether evidence covers every requested entity. If an
entity lacks documents or retrieved evidence, it adds a note and does not cache
the incomplete result. It builds response `ContextChunk` objects with scores
and provenance, returns updated session state, and asynchronously stores
complete answers.

## 5. Core functionality by source file

### `app/pipeline.py` — request orchestration

This is the central file for explaining application behavior. Key functions:

- `run_rag_pipeline_async`: complete branching request flow.
- `_answer_greeting`: L1 greeting cache plus short chat generation.
- `_guard_free_text`: applies output rails to non-RAG generated answers.
- `get_catalog` / `resolve_document_ids`: map company language to indexed
  document scope.
- `_search_target` / `hybrid_search_balanced`: combine retrieval backends and
  maintain entity coverage.
- `build_context`: citation-aware evidence construction.
- `_system_prompt` / `_generate_answer` / `_llm`: answer policy and model call.
- `_reply`: consistent `ChatResponse` construction.
- `corpus_version`: makes cached results depend on a catalog-derived corpus
  fingerprint.

### `app/utils/router.py` — language-to-search plan

- `RoutePlan` is the structured intermediate plan.
- `plan_query()` combines LLM routing and deterministic correction/fallback.
- `detect_known_entities()` matches indexed companies deterministically.
- `company_key()` normalizes common legal suffixes for matching.
- `classify_followup()` determines whether history is required to interpret a
  query.
- `build_filters()` constructs typed retrieval filters.

This separation lets user language (including pronouns and conversational
follow-ups) be converted into explicit search targets and constraints.

### `app/utils/vector.py` — dense Qdrant search

- `QdrantVectorRetriever.__init__()` initializes the BGE-small embedding model
  and chooses remote Qdrant when `QDRANT_URL` is set, otherwise local storage.
- `list_documents()` scans collection payloads and groups points by document
  and source file to provide the planning catalog.
- `retrieve()` embeds a query, applies supported metadata filters, requests
  top-k named-vector results, and normalizes payload data for downstream use.

### `app/utils/bm25.py` — local lexical search

- `load_index()` reads local manifest JSON and builds an in-memory BM25 index.
- `searchable_text()` enriches chunk text with company, year, metric, headings,
  columns, and row labels.
- `retrieve()` scores lexical matches and returns ranked source chunks after
  optional filter checks.

BM25 is complementary to dense retrieval: it can strongly match exact numeric,
metric, and filing terminology that semantic similarity alone may rank lower.

### `app/utils/flashrank_reranker.py` — candidate relevance and fairness

- `rerank()` enriches each passage with company/source/section and financial
  table metadata before scoring.
- `rerank_balanced()` reranks per entity and interleaves results to retain
  coverage in cross-company questions.
- A reranker failure falls back to the initial ranking; it does not discard all
  retrieved evidence.

### `app/utils/cache.py` — answer reuse with scope safety

- `CacheScope` and `build_scope()` prevent unrelated companies, years, intents,
  corpus versions, or elaboration contexts from sharing cache entries.
- `cache_lookup()` checks exact Redis L1 before optional semantic L2.
- `terms_compatible()` guards semantic hits against lexical and numeric
  mismatches.
- `cache_store()` serializes response chunks and writes L1/L2 entries with
  configured TTLs.

### `app/utils/guardrails.py`, `guard_rules.py`, and `pii.py`

- `check_input()` and `check_output()` wrap NeMo input/output rails with
  timeouts and deterministic fallback behavior.
- `guard_rules.py` supplies injection heuristics, prompt-leak checks, and
  numeric-grounding calculations.
- `pii.py` uses Presidio when available and falls back to targeted regexes for
  common sensitive patterns/secrets.
- `guardrails_config/config.yml`, `prompts.yml`, `rails.co`, and
  `guardrails_config/actions.py` define policy flows and custom actions.

The numeric check is a guardrail, not a formal proof of semantic correctness:
it tests whether material figures appear in the evidence (with configured
scale tolerance), not whether the answer's arithmetic or interpretation is
otherwise correct.

### `app/utils/intent_router.py` and `app/utils/llm.py`

- `IntentRouter.decide()` scores labelled examples using normalized embeddings
  and confidence/margin thresholds.
- `get_intent_router()` lazily builds a singleton and marks it unavailable
  after initialization failure so the main LLM router can take over.
- `get_groq_clients()` caches clients for primary/fallback credentials.
- `chat_with_retry()` handles retryable Groq failures with key rotation,
  rate-limit hints, and bounded delay.

### `app/config.py`, `schemas.py`, `main.py`, and `app_ui.py`

These files provide configuration, API data structures, HTTP lifecycle, and
the sample Streamlit interaction. The Streamlit UI checks `/health/ready`,
sends the chat history and current state to `/v1/chat`, updates state from the
response, and exposes retrieved chunks for inspection.

## 6. Key configuration and defaults

Settings are loaded through Pydantic Settings and `.env`; environment values
can override defaults. Important settings include:

| Area | Settings / defaults | Effect |
| --- | --- | --- |
| LLM | `GROQ_API_KEY` (required), optional `FALLBACK_GROQ_API_KEY`, `LLM_MODEL=openai/gpt-oss-20b` | Routing and answer generation. |
| Vector search | `EMBEDDING_MODEL=BAAI/bge-small-en-v1.5`, `VECTOR_NAME=fastembed-BAAI/bge-small-en-v1.5`, `COLLECTION_NAME=financial_rag_analytics` | Must match the ingestion collection/vector configuration. |
| Qdrant | `QDRANT_URL` and optional API key, else `QDRANT_STORAGE_PATH=data/qdrant_storage` | Selects remote or local vector storage. |
| Retrieval | `RETRIEVAL_K=30`, `RRF_K=60`, `FINAL_TOP_K=5`, pipeline candidate defaults are environment-driven | Search depth and rank fusion. |
| Context/history | `MAX_CONTEXT_TOKENS=4096`, `MAX_HISTORY_TURNS=6` | Controls prompt context and retained conversation. |
| Cache | Redis at `redis://localhost:6379`; answer TTL 24 hours, greeting TTL 7 days | Exact and semantic answer caching. |
| Guardrails | Enabled by default; 8-second timeout; fail-open deterministic fallback enabled; numeric grounding enabled | Request policy, PII, and numeric checks. |
| Intent routing | Enabled; min score 0.85 and margin 0.03 | Confidence threshold for fast paths. |
| API timeout | `REQUEST_TIMEOUT_S=40` | Maximum backend chat execution time at HTTP boundary. |

Some pipeline controls are read directly from environment variables rather than
`Settings`, including rerank candidate count, final top-k, catalog TTL,
concurrency, and LLM retry budgets. Deployment documentation should therefore
include both settings-based and direct environment configuration.

## 7. Runtime dependencies and data flow

### Main external systems and libraries

- **Groq:** query planner, guardrail policy checks, and answer generation.
- **Qdrant:** dense vector collection and document catalog.
- **FastEmbed / BGE-small:** query and cache embeddings.
- **BM25 / rank-bm25:** local lexical index built from manifest chunks.
- **FlashRank:** candidate reranking.
- **Redis / RedisVL:** exact and semantic answer cache.
- **NeMo Guardrails / LangChain Groq:** policy rails.
- **Presidio / spaCy:** PII recognition, with regex fallback.
- **Logfire:** request traces, spans, and operational events.
- **FastAPI / Pydantic:** service boundary and request/response validation.

### Retrieval and answer data path

```text
ChatRequest
  -> safe query + sanitized history
  -> RoutePlan (intent, entities, years, metrics, follow-up, subqueries)
  -> cache scope + lookup
  -> document filters
  -> Qdrant dense results + local BM25 results
  -> RRF candidate list (entity-aware)
  -> FlashRank final chunks (entity-balanced)
  -> token-limited context + citation source map
  -> Groq answer
  -> output guardrails / optional strict retry
  -> ChatResponse(answer, state, retrieved_chunks, source_map)
```

## 8. Error handling and graceful degradation

| Failure or condition | Current behavior |
| --- | --- |
| API not warmed up | `/health/ready` returns 503; chat endpoint also rejects while not ready. |
| Request deadline exceeded | API returns HTTP 504. |
| Unexpected pipeline exception | Logged with Logfire and returned as HTTP 500. |
| Semantic intent router unavailable | Falls through to LLM routing; regex greeting fallback is available. |
| Query router LLM fails | Returns an empty routing object; deterministic/default planning continues. |
| One retriever fails | `_safe_call()` logs the failure and supplies an empty ranking, allowing the other retriever to contribute. |
| Strict metadata retrieval yields too few results | Retries without year/company constraints while preserving document IDs. |
| FlashRank fails | Returns initial candidate order with fallback scores. |
| Cache read/write fails | Logs the failure; request continues without a cache hit/store. |
| NeMo rails fail or time out | Deterministic fallback or temporary refusal depends on `GUARDRAILS_FAIL_OPEN`. |
| Numeric output check blocks | One stricter generation retry; if still blocked, return a verification refusal and do not cache. |
| Incomplete multi-entity retrieval | Answer may include a missing-company note; it is not cached. |
| Startup warmup/cache initialization fails | Startup error is logged and re-raised; readiness is not set true. |

## 9. Operational notes and current limitations

1. **Readiness includes cache initialization.** Startup calls
   `init_financial_cache()` as well as model warmup. A Redis/cache setup failure
   can therefore prevent the API from becoming ready, even if vector search is
   otherwise available.
2. **Different request timeouts exist.** The API defaults to 40 seconds while
   the sample Streamlit UI sends requests with a 60-second client timeout.
   The server-side 504 can occur before the client timeout.
3. **BM25 is local-manifest based.** It builds an in-memory index at
   initialization. Qdrant remote configuration does not make BM25 remote; the
   local manifest directory must remain available to the application.
4. **Catalog cache invalidation is approximate.** The corpus fingerprint uses
   document IDs and chunk counts. Replacing content with the same IDs and
   counts may not change the fingerprint immediately; the catalog itself is
   also refreshed on a TTL.
5. **Retrieval errors can become empty evidence.** Retriever failures are
   caught and logged to allow the other search backend to work. If both fail,
   the user receives an answer indicating no supporting documents rather than
   a direct retrieval-service error.
6. **`shutdown_pipeline()` is currently a no-op.** The lifecycle hook calls it,
   but it does not explicitly close singleton clients or background resources.
7. **Grounding checks are limited.** The generation prompt and numeric rail
   reduce unsupported figures, but they are not a guarantee against every
   unsupported non-numeric claim or incorrect calculation.
8. **Config values are split.** Some important limits are in `Settings`, while
   others are direct environment reads in `pipeline.py` and `llm.py`; operators
   should audit both locations when tuning.

## 10. Presentation narrative

### Suggested 60–90 second explanation

> “The application is a FastAPI financial-filings assistant with a Streamlit
> client. Each request first passes through intent and privacy checks, then a
> query planner resolves companies, years, metrics, and conversational
> follow-ups. For financial questions, the system runs vector search in Qdrant
> and lexical BM25 search in parallel. It fuses those results, reranks them
> with FlashRank while preserving company balance for comparisons, and packs
> the best evidence into a source-tagged context. Groq generates the answer
> from that context, and output rails check for sensitive data, prompt leakage,
> and unsupported numbers before the response returns with source chunks and
> citations. A scope-aware Redis cache reduces repeated work without mixing
> answers across companies, years, or corpus versions.”

### Demo walkthrough

1. Ask a direct metric question with a company and year.
2. Show the returned answer, source tags, and retrieved chunk details.
3. Ask a reference follow-up such as “What about the prior year?” to show
   session-state resolution and query rewriting.
4. Ask for a comparison across companies to show entity-balanced retrieval and
   reranking.
5. Ask for more detail to show elaboration subqueries and new evidence angles.
6. Contrast cache and guardrail events in Logfire to explain speed and answer
   quality controls.

## 11. Source map for future maintenance

- API lifecycle/endpoints: [`main.py`](./main.py)
- Request and response models: [`schemas.py`](./schemas.py)
- Main control flow: [`pipeline.py`](./pipeline.py)
- Runtime settings: [`config.py`](./config.py)
- Query routing/state: [`utils/router.py`](./utils/router.py)
- Dense retrieval: [`utils/vector.py`](./utils/vector.py)
- Lexical retrieval: [`utils/bm25.py`](./utils/bm25.py)
- Fusion/context/generation: [`pipeline.py`](./pipeline.py)
- Reranking: [`utils/flashrank_reranker.py`](./utils/flashrank_reranker.py)
- Exact/semantic cache: [`utils/cache.py`](./utils/cache.py)
- Guardrails: [`utils/guardrails.py`](./utils/guardrails.py),
  [`utils/guard_rules.py`](./utils/guard_rules.py),
  [`guardrails_config/config.yml`](./guardrails_config/config.yml),
  [`guardrails_config/actions.py`](./guardrails_config/actions.py)
- PII masking: [`utils/pii.py`](./utils/pii.py)
- Intent classification: [`utils/intent_router.py`](./utils/intent_router.py)
- LLM retries/client setup: [`utils/llm.py`](./utils/llm.py)
- Token counting/history: [`utils/tokens.py`](./utils/tokens.py)
- Logging: [`utils/logger.py`](./utils/logger.py)
- Sample chat UI: [`app_ui.py`](./app_ui.py)
