# app/pipeline.py
from __future__ import annotations

import asyncio
import hashlib
import logging
import os
import re
import threading
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

import logfire
from groq import Groq

from app.config import settings
from app.schemas import ChatRequest, ChatResponse, ContextChunk, SessionState
from app.utils.bm25 import BM25Retriever
from app.utils.llm import chat_with_retry, get_groq_clients
from app.utils.router import QueryFilters, RoutePlan, SubQuery, build_filters, company_key, plan_query
from app.utils.tokens import count_tokens
from app.utils.vector import QdrantVectorRetriever
from app.utils.flashrank_reranker import FlashRankReranker
from app.utils.cache import CacheScope, build_scope, cache_lookup, cache_store
from app.utils.guard_rules import mask_pii
from app.utils.intent_router import get_intent_router, warmup_intent_router
from app.utils.pii import warmup_pii
from app.utils.guardrails import REFUSAL_SCOPE, check_input, check_output, warmup_guardrails
logger = logging.getLogger(__name__)
from app.utils.cache import init_financial_cache


RERANK_CANDIDATES = int(os.getenv("RERANK_CANDIDATES", "30"))
FINAL_TOP_K = int(os.getenv("FINAL_TOP_K", "5"))
MAX_FINAL_TOP_K = int(os.getenv("MAX_FINAL_TOP_K", "10"))
MIN_STRICT_YIELD = int(os.getenv("MIN_STRICT_YIELD", "5"))
CATALOG_TTL_S = int(os.getenv("CATALOG_TTL_S", "300"))
MAX_CONCURRENT_SEARCHES = int(os.getenv("MAX_CONCURRENT_SEARCHES", "8"))

_init_lock = threading.Lock()
_vector: Optional[QdrantVectorRetriever] = None
_bm25: Optional[BM25Retriever] = None
_reranker: Optional[FlashRankReranker] = None
_search_sem: Optional[asyncio.Semaphore] = None
_catalog: Dict[str, Any] = {"ts": 0.0, "docs": []}


def get_vector() -> QdrantVectorRetriever:
    global _vector
    with _init_lock:
        if _vector is None:
            _vector = QdrantVectorRetriever()
    return _vector


def get_bm25() -> BM25Retriever:
    global _bm25
    with _init_lock:
        if _bm25 is None:
            _bm25 = BM25Retriever()
    return _bm25


def get_groq() -> Tuple[Groq, ...]:
    return get_groq_clients(timeout=30.0)


def _get_reranker() -> FlashRankReranker:
    global _reranker
    with _init_lock:
        if _reranker is None:
            _reranker = FlashRankReranker()
    return _reranker


def _semaphore() -> asyncio.Semaphore:
    global _search_sem
    if _search_sem is None:
        _search_sem = asyncio.Semaphore(MAX_CONCURRENT_SEARCHES)
    return _search_sem


async def warmup_pipeline() -> None:
    t0 = time.perf_counter()
    def _warm_vector(): get_vector().retrieve("warmup query", 1, None)
    def _warm_bm25(): get_bm25().retrieve("warmup query", 1, None)
    def _warm_reranker(): _get_reranker().rerank("warmup query", [{"content": "warmup"}], 1)
    def _warm_cache(): init_financial_cache()  # pre-loads HFTextVectorizer, connects to Redis, creates L2 index
    def _warm_guard(): warmup_guardrails()      # builds NeMo LLMRails once (loads its small embedding model)
    def _warm_pii(): warmup_pii()               # loads Presidio + spaCy model once
    def _warm_intent(): warmup_intent_router()  # embeds the intent examples once

    await asyncio.gather(
        asyncio.to_thread(_warm_vector),
        asyncio.to_thread(_warm_bm25),
        asyncio.to_thread(_warm_reranker),
        asyncio.to_thread(_warm_cache),
        asyncio.to_thread(_warm_guard),
        asyncio.to_thread(_warm_pii),
        asyncio.to_thread(_warm_intent),
        asyncio.to_thread(get_groq),
    )
    await get_catalog(force=True)
    logfire.info("Pipeline warmup complete in {duration}s", duration=round(time.perf_counter() - t0, 1))


def shutdown_pipeline() -> None:
    pass


async def get_catalog(force: bool = False) -> List[Dict[str, Any]]:
    if force or time.time() - _catalog["ts"] > CATALOG_TTL_S:
        try:
            docs = await asyncio.to_thread(lambda: get_vector().list_documents())
            if not docs:
                # Fallback to local manifests if Qdrant is warming up or empty
                bm_chunks = get_bm25().valid_chunks
                manifest_docs = {}
                for c in bm_chunks:
                    doc_id = c.get("document_id") or "doc_default"
                    src = c.get("source_file") or f"{doc_id}.pdf"
                    comp = c.get("company") or (c.get("metadata") or {}).get("company") or ""
                    if not comp:
                        lower_name = f"{doc_id} {src}".lower()
                        if "nike" in lower_name:
                            comp = "Nike"
                        elif "duolingo" in lower_name:
                            comp = "Duolingo"
                    key = (doc_id, src)
                    if key not in manifest_docs:
                        manifest_docs[key] = {
                            "document_id": doc_id,
                            "source_file": src,
                            "company": comp,
                            "chunk_count": 0,
                        }
                    manifest_docs[key]["chunk_count"] += 1
                docs = list(manifest_docs.values())
            _catalog.update(ts=time.time(), docs=docs)
        except Exception:
            logfire.exception("Catalog refresh failed; using stale copy")
            _catalog["ts"] = time.time()
    return _catalog["docs"]


def resolve_document_ids(entity: Optional[str], document_id: Optional[str], catalog: List[Dict[str, Any]]) -> Optional[List[str]]:
    if document_id:
        return [document_id]
    if not entity or not catalog:
        return None
    key = company_key(entity)
    if not key:
        return None
    matches = []
    for d in catalog:
        comp = company_key(d.get("company", ""))
        indexed = company_key(f"{d.get('company', '')} {d.get('document_id', '')} {d.get('source_file', '')}")
        if key == comp or f" {key} " in f" {indexed} " or key in comp or comp in key:
            matches.append(str(d["document_id"]))
    return list(dict.fromkeys(matches))


async def _safe_call(fn: Callable, *args) -> List[Dict[str, Any]]:
    try:
        return await asyncio.to_thread(fn, *args) or []
    except Exception:
        logfire.exception("Retriever failed")
        return []


def _uid(item: Dict[str, Any]) -> str:
    return f"{item.get('document_id')}:{item.get('chunk_id')}"


async def _search_target(text: str, filters: QueryFilters, top_k: int) -> List[List[Dict[str, Any]]]:
    vec, bm = get_vector(), get_bm25()
    async with _semaphore():
        v, b = await asyncio.gather(
            _safe_call(vec.retrieve, text, top_k, filters),
            _safe_call(bm.retrieve, text, top_k, filters),
        )
        rankings = [v, b]
        distinct = len({_uid(x) for x in v + b})
        if distinct < MIN_STRICT_YIELD and (filters.years or filters.company):
            soft = QueryFilters(document_ids=filters.document_ids)
            v2, b2 = await asyncio.gather(
                _safe_call(vec.retrieve, text, top_k, soft),
                _safe_call(bm.retrieve, text, top_k, soft),
            )
            rankings += [v2, b2]
    return rankings


async def hybrid_search_balanced(
    sub_queries: List[SubQuery],
    entity_filters: Dict[Optional[str], QueryFilters],
    total: int = RERANK_CANDIDATES,
) -> List[Dict[str, Any]]:
    n_entities = max(len(entity_filters), 1)
    per_entity_k = max(total // n_entities, MIN_STRICT_YIELD)

    tasks = [_search_target(sq.text, entity_filters[sq.entity], per_entity_k) for sq in sub_queries]
    all_rankings = await asyncio.gather(*tasks)

    fused: Dict[Optional[str], Dict[str, Dict[str, Any]]] = {e: {} for e in entity_filters}
    for sq_idx, (sq, rankings) in enumerate(zip(sub_queries, all_rankings)):
        bucket = fused[sq.entity]
        for results in rankings:
            seen = set()
            for rank, item in enumerate(results, start=1):
                uid = _uid(item)
                if uid in seen:
                    continue
                seen.add(uid)
                entry = bucket.setdefault(uid, {**item, "rrf_score": 0.0, "sq_ids": set(), "entity": sq.entity})
                entry["rrf_score"] += 1.0 / (settings.RRF_K + rank)
                entry["sq_ids"].add(sq_idx)

    quota = max(total // n_entities, 1)
    selected, leftovers = [], []
    for bucket in fused.values():
        ranked = sorted(bucket.values(), key=lambda x: x["rrf_score"], reverse=True)
        selected += ranked[:quota]
        leftovers += ranked[quota:]
    spare = total - len(selected)
    if spare > 0:
        leftovers.sort(key=lambda x: x["rrf_score"], reverse=True)
        selected += leftovers[:spare]
    return selected


def build_context(
    results: List[Dict[str, Any]], max_tokens: int = settings.MAX_CONTEXT_TOKENS
) -> Tuple[str, Dict[str, str], List[Dict[str, Any]]]:
    sep = "\n\n---\n\n"
    sep_tokens = count_tokens(sep)
    parts, source_map, selected, used = [], {}, [], 0
    for item in results:
        content = (item.get("content") or "").strip()
        if not content:
            continue
        doc_id = item.get("document_id", "Doc")
        source_file = item.get("source_file") or f"{doc_id}.pdf"
        page = item.get("page_number", "N/A")
        chunk_type = item.get("chunk_type", "text")
        headings = " > ".join(item.get("headings") or []) or "Main Section"

        source_id = f"S{len(selected) + 1}"
        type_tag = f" | Type: {chunk_type.upper()}" if chunk_type != "text" else ""
        part = f"[{source_id}]{type_tag}\nFile: {source_file} | Page: {page}\nSection: {headings}\nContent:\n{content}"
        cost = count_tokens(part) + sep_tokens
        if used + cost > max_tokens:
            continue
        used += cost
        source_map[source_id] = f'Source: {source_file}, Page {page}, Section: "{headings}"'
        parts.append(part)
        selected.append(item)
    return sep.join(parts), source_map, selected



# --------------------------------------------------------------------------- #
# Request-path helpers
# --------------------------------------------------------------------------- #
_GREETING_RE = re.compile(
    r"^\s*(?:hi+|hello+|hey+|hiya|yo|good\s+(?:morning|afternoon|evening)|how\s+are\s+you(?:\s+doing)?|"
    r"what\s+can\s+you\s+do|who\s+are\s+you|thanks|thank\s+you)(?:\s+(?:there|bot|assistant))?\s*[!.?]*\s*$",
    re.I,
)
_GREETING_SYSTEM = (
    "You are a professional, friendly, and helpful Financial RAG Analytics Assistant. "
    "Respond naturally and briefly to the user's greeting or casual message without requiring document context."
)
_CHAT_SYSTEM = (
    "You are a Financial RAG Analytics Assistant that answers questions about a set of indexed company filings. "
    "Reply briefly and naturally to greetings, thanks, and questions about what you can do, and mention that you can "
    "answer questions about the filings (financial metrics, history, risks, comparisons). You cannot perform actions "
    "(payments, transfers, purchases, bookings), access accounts, or help with unrelated tasks. If asked for any of "
    "those, say so in one sentence and steer back to the filings. Never invent financial data."
)
_bg_tasks: set = set()


async def _guard_free_text(query: str, answer: str, events: list) -> str:
    """Output rails for replies that are NOT grounded in retrieved context (small talk, summaries)."""
    g = await check_output(query, answer, "")
    if not g.allowed:
        events.append(f"output_blocked:{g.rail}")
        return g.message or REFUSAL_SCOPE
    return g.text


def _fire_and_forget(fn: Callable, *args) -> None:
    """Run a blocking function in a thread without blocking the response; keep a reference so it isn't GC'd."""
    task = asyncio.create_task(asyncio.to_thread(fn, *args))
    _bg_tasks.add(task)
    task.add_done_callback(_bg_tasks.discard)


def corpus_version(catalog: List[Dict[str, Any]]) -> str:
    """Changes whenever documents are added/removed/re-indexed -> stale cache entries become unreachable."""
    sig = "|".join(sorted(f"{d.get('document_id')}:{d.get('chunk_count', '')}" for d in catalog))
    return hashlib.sha1(sig.encode()).hexdigest()[:8]


def _sanitize_history(history: List[Any]) -> List[Any]:
    """Mask PII in prior turns before they reach any LLM."""
    out = []
    for t in history:
        try:
            out.append(t.model_copy(update={"content": mask_pii(t.content or "")}))
        except AttributeError:  # pydantic v1
            out.append(t.copy(update={"content": mask_pii(t.content or "")}))
    return out


def _last_assistant(history: List[Any]) -> str:
    for t in reversed(history or []):
        if t.role == "assistant":
            return t.content or ""
    return ""


async def _llm(messages: List[Dict[str, str]], temperature: float):
    resp = await asyncio.to_thread(
        lambda: chat_with_retry(get_groq(), model=settings.LLM_MODEL, messages=messages, temperature=temperature)
    )
    return resp.choices[0].message.content.strip(), resp


def _reply(request: ChatRequest, answer: str, intent: str, search_query: str, state: SessionState,
           chunks: Optional[list] = None, source_map: Optional[dict] = None) -> ChatResponse:
    return ChatResponse(
        session_id=request.session_id, answer=answer, intent=intent, search_query=search_query,
        state=state, retrieved_chunks=chunks or [], source_map=source_map or {},
    )


def _system_prompt(context: str, followup_type: str, strict: bool = False) -> str:
    rules = [
        "1. Tabular & Multi-Year Facts: When data is given as 'Document: Metric | Column = Value | ...', extract the exact figures for each requested year.",
        "2. Multi-Year Comparisons: If the user asks for changes, growth rates, or comparisons over time, calculate and state the difference clearly based on the provided numbers.",
        "3. Units & Scale: State monetary units accurately (e.g. in millions, thousands, or percentages as specified in the context).",
        "4. Citations: Cite your sources using passage tags like [S1], [S2] where applicable.",
        "5. Grounding: Rely strictly on the context. If figures for a requested year or metric are not present, explicitly state that they are unavailable.",
    ]
    if followup_type == "elaboration":
        rules.append(
            "6. Elaboration: The user wants MORE detail on your previous answer. Provide only NEW information from the "
            "context (breakdowns, drivers, comparisons, commentary). Do not restate figures you already gave. "
            "If the context holds nothing new, say so plainly."
        )
    if strict:
        rules.append(
            "7. STRICT: every figure you state must be copied verbatim from the context (or be a simple calculation "
            "from context figures that you show). Never estimate or recall numbers from memory."
        )
    return (
        "You are a precise, professional financial QA assistant. Answer the user query accurately and comprehensively "
        "using ONLY the context provided below.\n\nGuidelines:\n" + "\n".join(rules) + f"\n\nCONTEXT:\n{context}"
    )


async def _generate_answer(system_prompt: str, user_content: str, history: List[Any], followup_type: str):
    messages = [{"role": "system", "content": system_prompt}]
    if followup_type == "elaboration":  # only elaboration depends on the previous answer
        for t in history[-4:]:
            messages.append({"role": t.role, "content": (t.content or "")[:1500]})
    messages.append({"role": "user", "content": user_content})
    return await _llm(messages, 0.0)


async def _answer_greeting(request: ChatRequest, query: str, root_span) -> ChatResponse:
    """Greetings/small talk: no router, no guardrail LLM, no history. Served from the L1 exact cache."""
    state = request.state or SessionState()
    scope = CacheScope(intent="general_chat")
    hit = await asyncio.to_thread(cache_lookup, query, scope, False)
    if hit:
        root_span.set_attribute("cache_hit", True)
        return _reply(request, hit["answer"], "general_chat", query, state)
    root_span.set_attribute("cache_hit", False)
    answer, _ = await _llm(
        [{"role": "system", "content": _CHAT_SYSTEM}, {"role": "user", "content": query}], 0.7
    )
    _fire_and_forget(
        cache_store, query, scope,
        {"answer": answer, "retrieved_chunks": [], "intent": "general_chat", "search_query": query, "document_ids": []},
        False, settings.CACHE_GREETING_TTL,
    )
    return _reply(request, answer, "general_chat", query, state)


# --------------------------------------------------------------------------- #
# Main entry point
# --------------------------------------------------------------------------- #
async def run_rag_pipeline_async(request: ChatRequest) -> ChatResponse:
    t0 = time.perf_counter()

    with logfire.span("Execute RAG Pipeline", session_id=request.session_id) as root_span:
        raw_query = request.query.strip()
        state0 = request.state or SessionState()
        events: list = []   # guardrail activity, attached to the root span

        def _mark():
            root_span.set_attribute("guardrail_events", list(events))

        # --- 0. Embedding intent router: cheap, data-driven fast paths (falls through when not confident) ---
        semantic_label, semantic_ok = None, False
        intent_router = get_intent_router()
        if intent_router is not None:
            try:
                decision = await asyncio.to_thread(intent_router.decide, raw_query)
                semantic_label, semantic_ok = decision.label, True
                root_span.set_attribute("intent_top", decision.top)
                root_span.set_attribute("intent_score", round(decision.score, 3))
                root_span.set_attribute("intent_fast_path", decision.label or "")
            except Exception:
                logfire.exception("Intent router failed; using LLM router only")

        fast_path_ok = semantic_label in ("smalltalk", "out_of_scope")
        if fast_path_ok:  # never take a fast path with PII in the text: let the full guardrail path handle it
            fast_path_ok = (await asyncio.to_thread(mask_pii, raw_query)) == raw_query
        if not semantic_ok and _GREETING_RE.match(raw_query):  # degraded mode: anchored greeting regex
            semantic_label, fast_path_ok = "smalltalk", True

        if fast_path_ok and semantic_label == "smalltalk":
            root_span.set_attribute("path", "smalltalk")
            root_span.set_attribute("query", raw_query)
            return await _answer_greeting(request, raw_query, root_span)
        if fast_path_ok and semantic_label == "out_of_scope":
            events.append("out_of_scope_fast")
            root_span.set_attribute("path", "out_of_scope")
            root_span.set_attribute("query", raw_query)
            _mark()
            return _reply(request, REFUSAL_SCOPE, "general_chat", raw_query, state0)

        # --- 1. Input rails (NeMo): PII masking, injection heuristics, LLM policy check ---
        with logfire.span("Input Guardrails") as g_span:
            guard_in = await check_input(raw_query)
            g_span.set_attribute("allowed", guard_in.allowed)
            g_span.set_attribute("rail", guard_in.rail or "")
            g_span.set_attribute("degraded", guard_in.degraded)
        if guard_in.degraded:
            events.append("guardrails_degraded")
        if not guard_in.allowed:
            safe_query = await asyncio.to_thread(mask_pii, raw_query)   # never log / echo raw PII
            events.append(f"input_blocked:{guard_in.rail}")
            root_span.set_attribute("query", safe_query)
            root_span.set_attribute("blocked_by", guard_in.rail or "")
            _mark()
            return _reply(request, guard_in.message or REFUSAL_SCOPE, "general_chat", safe_query, state0)
        query = guard_in.text                      # PII-masked from here on
        root_span.set_attribute("query", query)
        if query != raw_query:
            events.append("input_pii_masked")
        _mark()
        history = _sanitize_history(request.history or [])

        # --- 2. Catalog & query planning (follow-up resolution happens here) ---
        catalog = await get_catalog()
        known_companies = sorted({d["company"] for d in catalog if d.get("company")})
        with logfire.span("Query Planning & Intent Routing") as plan_span:
            plan = await plan_query(query, history, state0, known_companies,
                                    semantic=semantic_label, semantic_available=semantic_ok)
            plan_span.set_attribute("intent", plan.intent)
            plan_span.set_attribute("followup_type", plan.followup_type)
            plan_span.set_attribute("entities", plan.entities)
            plan_span.set_attribute("search_query", plan.search_query)

        # --- 3-0. Out of scope (allow-list): anything that isn't filings Q&A, small talk, or a summary ---
        if plan.intent == "out_of_scope":
            events.append("out_of_scope")
            _mark()
            return _reply(request, REFUSAL_SCOPE, "general_chat", plan.search_query, state0)

        # --- 3a. General chat (greetings / capability questions only): never cached ---
        if plan.intent == "general_chat":
            messages = [{"role": "system", "content": _CHAT_SYSTEM}]
            messages += [{"role": t.role, "content": t.content} for t in history[-6:]]
            messages.append({"role": "user", "content": query})
            answer, _ = await _llm(messages, 0.7)
            answer = await _guard_free_text(query, answer, events)
            _mark()
            return _reply(request, answer, plan.intent, plan.search_query, plan.state)

        # --- 3b. Chat summary: depends on the whole conversation, never cached ---
        if plan.intent == "chat_summary":
            if not history:
                answer = "We haven't discussed anything in this session yet! Feel free to ask any question about your financial documents."
            else:
                history_text = "\n".join(f"{t.role}: {t.content}" for t in history)
                answer, _ = await _llm([{"role": "user", "content": (
                    "You are the assistant participating directly in this conversation. Speak naturally to the user in a "
                    "warm, conversational, and direct tone (using \"you\" and \"we\"). Summarize the key questions the user "
                    "asked and the main points we discussed so far. Do NOT sound like an external observer.\n\n"
                    f"Conversation History:\n{history_text}\n\nNatural Conversation Summary:")}], 0.2)
                answer = await _guard_free_text(query, answer, events)
            _mark()
            return _reply(request, answer, plan.intent, plan.search_query, plan.state)

        # --- 4. Cache lookup (RAG only) ---
        # Cached answers must be a pure function of (standalone question, scope, corpus):
        #   none       -> key on the user's own wording; L1 + L2
        #   reference  -> key on the resolved standalone rewrite; L1 + L2
        #   elaboration-> depends on the previous answer: key on (wording + hash of previous answer); L1 only
        follow = plan.followup_type
        ctx = hashlib.sha1(_last_assistant(history).encode()).hexdigest()[:12] if follow == "elaboration" else ""
        scope = build_scope(plan.intent, plan.entities, plan.years, corpus_version(catalog), ctx)
        semantic = follow != "elaboration"
        cache_text = plan.search_query if follow == "reference" else query

        cached = await asyncio.to_thread(cache_lookup, cache_text, scope, semantic)
        if cached:
            root_span.set_attribute("cache_hit", True)
            root_span.set_attribute("cache_tier", cached.get("_tier", ""))
            cached_chunks = [ContextChunk(**c) if isinstance(c, dict) else c for c in cached.get("retrieved_chunks", [])]
            return _reply(request, cached["answer"], cached.get("intent", plan.intent),
                          cached.get("search_query", plan.search_query), plan.state, cached_chunks)
        root_span.set_attribute("cache_hit", False)

        # --- 5. Retrieval ---
        entity_filters = {}
        missing = []
        for entity in plan.entities or [None]:
            doc_ids = resolve_document_ids(entity, plan.state.document_id if len(plan.entities) <= 1 else None, catalog)
            if entity and doc_ids == []:
                missing.append(entity)
                continue
            entity_filters[entity] = build_filters(plan.years, entity if (entity and not doc_ids) else None, doc_ids)

        if not entity_filters:
            requested = ", ".join(missing) or "the requested company"
            return _reply(request, f"I couldn't find financial documents for {requested}. Check the company name and then try again.",
                          plan.intent, plan.search_query, plan.state)

        sub_queries = [sq for sq in plan.sub_queries if sq.entity in entity_filters]
        if not sub_queries:
            sub_queries = [SubQuery(text=plan.search_query, entity=next(iter(entity_filters)))]

        with logfire.span("Hybrid Search (Dense + BM25)", sub_queries_count=len(sub_queries)) as search_span:
            candidates = await hybrid_search_balanced(sub_queries, entity_filters, RERANK_CANDIDATES)
            search_span.set_attribute("candidates_count", len(candidates))
            search_span.set_attribute("candidate_source_files", list({c.get("source_file") for c in candidates if c.get("source_file")}))

        with logfire.span("FlashRank Reranking", candidates_in=len(candidates)) as rerank_span:
            base_k = FINAL_TOP_K + (3 if follow == "elaboration" else 0)  # elaboration needs a wider context
            n_ent = max(len(entity_filters), 1)
            # multi-company questions: at least 3 chunks per company, even if that exceeds the usual cap
            final_k = min(max(MAX_FINAL_TOP_K, 3 * n_ent), max(base_k, 3 * n_ent, 2 * len(sub_queries)))
            entity_queries = {
                e: " ".join(p for p in (e, *plan.metrics, *(str(y) for y in plan.years)) if p)
                for e in entity_filters if e
            }
            reranked = await asyncio.to_thread(
                _get_reranker().rerank_balanced, candidates, final_k, plan.search_query, entity_queries)
            rerank_span.set_attribute("final_chunks_selected", len(reranked))

        context, source_map, final_chunks = build_context(reranked)
        if not context.strip():
            return _reply(request, "I couldn't find support for your question in the available financial documents.",
                          plan.intent, plan.search_query, plan.state)

        # --- 6. Generation + output rails (one strict retry if figures are not grounded) ---
        user_content = query if query.strip() == plan.search_query.strip() else f"{query}\n(Retrieval focus: {plan.search_query})"

        answer, guard_out = "", None
        for attempt in (1, 2):
            with logfire.span("LLM Generation (Groq)", model=settings.LLM_MODEL, attempt=attempt) as llm_span:
                llm_span.set_attribute("retrieved.document_ids", [c.get("document_id") for c in final_chunks])
                llm_span.set_attribute("retrieved.chunk_ids", [c.get("chunk_id") for c in final_chunks])
                answer, response = await _generate_answer(
                    _system_prompt(context, follow, strict=attempt == 2), user_content, history, follow)
                if getattr(response, "usage", None):
                    llm_span.set_attribute("total_tokens", response.usage.total_tokens)

            with logfire.span("Output Guardrails", attempt=attempt) as o_span:
                guard_out = await check_output(query, answer, context)
                o_span.set_attribute("allowed", guard_out.allowed)
                o_span.set_attribute("rail", guard_out.rail or "")

            if guard_out.allowed or "numeric" not in (guard_out.rail or ""):
                break  # passed, or blocked for a reason a retry won't fix

        if not guard_out.allowed:
            events.append(f"output_blocked:{guard_out.rail}")
            _mark()
            root_span.set_attribute("blocked_by", guard_out.rail or "")
            return _reply(request, guard_out.message or "I couldn't produce a verified answer.",
                          plan.intent, plan.search_query, plan.state)   # not cached
        if guard_out.text != answer:
            events.append("output_pii_masked")
        answer = guard_out.text

        # Completeness: every requested company must be represented in the evidence we answered from.
        covered = {c.get("entity") for c in final_chunks}
        complete = not missing and set(entity_filters) <= covered
        if missing:
            answer = f"Note: I couldn't find documents for {', '.join(missing)}.\n\n{answer}"
        if not complete:
            logfire.warn("Incomplete evidence: entities={e} covered={c}", e=list(entity_filters), c=list(covered))

        # --- 7. Build response + store ---
        _mark()
        root_span.set_attribute("total_duration_s", round(time.perf_counter() - t0, 2))
        context_chunks = [
            ContextChunk(
                chunk_id=r["chunk_id"], content=r["content"], document_id=r["document_id"],
                source_file=r.get("source_file", f"{r['document_id']}.pdf"), page_number=r["page_number"],
                chunk_type=r["chunk_type"], headings=r.get("headings", []),
                rrf_score=r.get("rrf_score", 0.0), final_score=r.get("final_score", 0.0),
            )
            for r in final_chunks
        ]
        document_ids = list({c.get("document_id") for c in final_chunks if c.get("document_id")})

        if complete:  # never cache an answer that is missing a requested company
            _fire_and_forget(
                cache_store, cache_text, scope,
                {"answer": answer, "retrieved_chunks": context_chunks, "intent": plan.intent,
                 "search_query": plan.search_query, "document_ids": document_ids},
                semantic,
            )
        return _reply(request, answer, plan.intent, plan.search_query, plan.state, context_chunks, source_map)
