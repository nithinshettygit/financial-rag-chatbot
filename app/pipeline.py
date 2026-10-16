# app/pipeline.py
from __future__ import annotations

import asyncio
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

logger = logging.getLogger(__name__)

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

    await asyncio.gather(
        asyncio.to_thread(_warm_vector),
        asyncio.to_thread(_warm_bm25),
        asyncio.to_thread(_warm_reranker),
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


async def run_rag_pipeline_async(request: ChatRequest) -> ChatResponse:
    t0 = time.perf_counter()
    
    with logfire.span("Execute RAG Pipeline", query=request.query, session_id=request.session_id) as root_span:
        catalog = await get_catalog()
        known_companies = sorted({d["company"] for d in catalog if d.get("company")})

        with logfire.span("Query Planning & Intent Routing") as plan_span:
            plan = await plan_query(request.query, request.history or [], request.state or SessionState(), known_companies)
            plan_span.set_attribute("intent", plan.intent)
            plan_span.set_attribute("entities", plan.entities)

        # 1. Handle General Chat Intent (Dynamic Conversational Response using History)
# 1. Handle General Chat Intent (Professional Assistant Persona)
# 1. Handle General Chat Intent (Professional Assistant Persona)
        if plan.intent == "general_chat":
            with logfire.span("General Chat Generation", model=settings.LLM_MODEL):
                messages = [
                    {
                        "role": "system",
                        "content": "You are a professional, friendly, and helpful Financial RAG Analytics Assistant. Respond naturally and politely to the user's greeting, well-being inquiry, or casual message without requiring document context."
                    }
                ]
                if request.history:
                    for turn in request.history[-6:]:
                        messages.append({"role": turn.role, "content": turn.content})
                messages.append({"role": "user", "content": request.query})

                response = await asyncio.to_thread(
                    lambda: chat_with_retry(
                        get_groq(),
                        model=settings.LLM_MODEL,
                        messages=messages,
                        temperature=0.7,
                    )
                )
                answer = response.choices[0].message.content.strip()
                total_duration = round(time.perf_counter() - t0, 2)
                root_span.set_attribute("total_duration_s", total_duration)
                
                return ChatResponse(
                    session_id=request.session_id, answer=answer, intent=plan.intent,
                    search_query=plan.search_query, state=plan.state, retrieved_chunks=[]
                )

        # 2. Handle Chat Summary Intent
# 2. Handle Chat Summary Intent (Warm, Natural, Conversational Persona)
        if plan.intent == "chat_summary":
            with logfire.span("Generate Chat Summary", history_turns=len(request.history or [])):
                if not request.history:
                    answer = "We haven't discussed anything in this session yet! Feel free to ask any question about your financial documents."
                else:
                    history_text = "\n".join(f"{t.role}: {t.content}" for t in request.history)
                    summary_prompt = f"""You are the assistant participating directly in this conversation. Speak naturally to the user in a warm, conversational, and direct tone (using "you" and "we"). 

Summarize the key questions the user asked and the main points we discussed so far. Do NOT sound like an external observer, bot report, or third-person logger (e.g. do NOT say "The user asked..." or "Assistant replied...").

Conversation History:
{history_text}

Natural Conversation Summary:"""
                    summary_response = await asyncio.to_thread(
                        lambda: chat_with_retry(
                            get_groq(),
                            model=settings.LLM_MODEL,
                            messages=[{"role": "user", "content": summary_prompt}],
                            temperature=0.2,
                        )
                    )
                    answer = summary_response.choices[0].message.content.strip()

                total_duration = round(time.perf_counter() - t0, 2)
                root_span.set_attribute("total_duration_s", total_duration)

                return ChatResponse(
                    session_id=request.session_id, answer=answer, intent=plan.intent,
                    search_query=plan.search_query, state=plan.state, retrieved_chunks=[]
                )

            
        # 3. Handle Document Inventory Intent
        # if plan.intent == "document_inventory":
        #     with logfire.span("Document Inventory Lookup"):
        #         doc_list = "\n".join(f"- **{d.get('company', 'Unknown')}**: `{d.get('source_file')}` (ID: `{d.get('document_id')}`)" for d in catalog)
        #         answer = f"Here are the indexed financial documents available in the knowledge base:\n\n{doc_list if doc_list else 'No documents currently indexed.'}"
        #         total_duration = round(time.perf_counter() - t0, 2)
        #         root_span.set_attribute("total_duration_s", total_duration)

        #         return ChatResponse(
        #             session_id=request.session_id, answer=answer, intent=plan.intent,
        #             search_query=plan.search_query, state=plan.state, retrieved_chunks=[]
        #         )

        # 4. Handle Standard RAG Search Intent (Hybrid Retrieval + FlashRank Reranking + Generation)
# 3. Handle Standard RAG Search Intent
        entity_filters = {}
        missing = []
        for entity in plan.entities or [None]:
            doc_ids = resolve_document_ids(entity, plan.state.document_id if len(plan.entities) <= 1 else None, catalog)
            if entity and doc_ids == []:
                missing.append(entity)
                continue
            entity_filters[entity] = build_filters(plan.years, entity if (entity and not doc_ids) else None, doc_ids)

        if not entity_filters:
            requested_entities = ", ".join(missing) or "the requested company"
            answer = (
                f"I couldn't find financial documents for {requested_entities}. "
                "Check the company name and then try again."
            )
            total_duration = round(time.perf_counter() - t0, 2)
            root_span.set_attribute("total_duration_s", total_duration)
            return ChatResponse(
                session_id=request.session_id,
                answer=answer,
                intent=plan.intent,
                search_query=plan.search_query,
                state=plan.state,
                retrieved_chunks=[],
            )

        sub_queries = [sq for sq in plan.sub_queries if sq.entity in entity_filters]
        if not sub_queries:
            sub_queries = [SubQuery(text=plan.search_query, entity=next(iter(entity_filters)))]

        # --- Developer Observability: Track Search & Source Manifests ---
        with logfire.span("Hybrid Search (Dense + BM25)", sub_queries_count=len(sub_queries)) as search_span:
            candidates = await hybrid_search_balanced(sub_queries, entity_filters, RERANK_CANDIDATES)
            search_span.set_attribute("candidates_count", len(candidates))
            search_span.set_attribute("candidate_source_files", list({c.get("source_file") for c in candidates if c.get("source_file")}))

        with logfire.span("FlashRank Reranking", candidates_in=len(candidates)) as rerank_span:
            final_k = min(MAX_FINAL_TOP_K, max(FINAL_TOP_K, 2 * len(sub_queries)))
            reranker = _get_reranker()
            reranked = await asyncio.to_thread(reranker.rerank, plan.search_query, candidates, final_k)
            rerank_span.set_attribute("final_chunks_selected", len(reranked))

        context, source_map, final_chunks = build_context(reranked)

        if not context.strip():
            return ChatResponse(
                session_id=request.session_id, 
                answer="I couldn't find support for your question in the available financial documents.", 
                intent=plan.intent, search_query=plan.search_query, state=plan.state, retrieved_chunks=[]
            )

        system_prompt = (
            "You are a precise, professional financial QA assistant. Answer the user query accurately and comprehensively using ONLY the context provided below.\n\n"
            "Guidelines:\n"
            "1. Tabular & Multi-Year Facts: When data is given as 'Document: Metric | Column = Value | ...', extract the exact figures for each requested year.\n"
            "2. Multi-Year Comparisons: If the user asks for changes, growth rates, or comparisons over time, calculate and state the difference clearly based on the provided numbers.\n"
            "3. Units & Scale: State monetary units accurately (e.g. in millions, thousands, or percentages as specified in the context).\n"
            "4. Citations: Cite your sources using passage tags like [S1], [S2] where applicable.\n"
            "5. Grounding: Rely strictly on the context. If figures for a requested year or metric are not present, explicitly state that they are unavailable.\n\n"
            f"CONTEXT:\n{context}"
        )

        user_content = request.query if request.query.strip() == plan.search_query.strip() else f"{request.query}\n(Retrieval focus: {plan.search_query})"

        # --- Developer Observability: Attach Retrieved Chunk Manifests to LLM Span ---
        with logfire.span("LLM Generation (Groq)", model=settings.LLM_MODEL) as llm_span:
            # Attach detailed source metadata for backend debugging / Logfire dashboard filtering
            llm_span.set_attribute("retrieved.document_ids", [c.get("document_id") for c in final_chunks])
            llm_span.set_attribute("retrieved.source_files", [c.get("source_file") for c in final_chunks])
            llm_span.set_attribute("retrieved.chunk_ids", [c.get("chunk_id") for c in final_chunks])
            llm_span.set_attribute("retrieved.pages", [c.get("page_number") for c in final_chunks])
            llm_span.set_attribute("retrieved.chunk_types", [c.get("chunk_type") for c in final_chunks])

            response = await asyncio.to_thread(
                lambda: chat_with_retry(
                    get_groq(),
                    model=settings.LLM_MODEL,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_content},
                    ],
                    temperature=0.0,
                )
            )
            answer = response.choices[0].message.content.strip()
            if hasattr(response, "usage") and response.usage:
                llm_span.set_attribute("total_tokens", response.usage.total_tokens)

                
        total_duration = round(time.perf_counter() - t0, 2)
        root_span.set_attribute("total_duration_s", total_duration)

        context_chunks = [
            ContextChunk(
                chunk_id=r["chunk_id"], content=r["content"], document_id=r["document_id"],
                source_file=r.get("source_file", f"{r['document_id']}.pdf"), page_number=r["page_number"],
                chunk_type=r["chunk_type"], headings=r.get("headings", []),
                rrf_score=r.get("rrf_score", 0.0), final_score=r.get("final_score", 0.0),
            )
            for r in final_chunks
        ]
        return ChatResponse(
            session_id=request.session_id, answer=answer, intent=plan.intent,
            search_query=plan.search_query, state=plan.state,
            retrieved_chunks=context_chunks, source_map=source_map,
        )