from __future__ import annotations

import asyncio
import os
import threading
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

import logfire

from app.config import settings
from app.utils.bm25 import BM25Retriever
from app.utils.flashrank_reranker import FlashRankReranker
from app.utils.router import QueryFilters, SubQuery, company_key
from app.utils.tokens import count_tokens
from app.utils.vector import QdrantVectorRetriever


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
