# app/utils/flashrank_reranker.py
from __future__ import annotations
from typing import Any, Dict, List, Optional
from flashrank import Ranker, RerankRequest
from app.config import settings
import logfire

class FlashRankReranker:
    def __init__(self):
        logfire.info("Initializing FlashRank Ranker", model=settings.RERANKER_MODEL)
        self.ranker = Ranker(model_name=settings.RERANKER_MODEL)

    def rerank(self, query: str, candidates: List[Dict[str, Any]], top_k: int) -> List[Dict[str, Any]]:
        if not candidates or top_k <= 0:
            return []

        passages = []
        for idx, item in enumerate(candidates):
            headings = " > ".join(item.get("headings") or [])
            content = (item.get("content") or "").strip()
            metadata = item.get("metadata") or {}
            chunk_type = item.get("chunk_type", "text")
            
            text_parts = []
            company = item.get("entity") or item.get("company") or metadata.get("company")
            if company:
                text_parts.append(f"Company: {company}")
            if item.get("source_file"):
                text_parts.append(f"Source: {item['source_file']}")
            if headings:
                text_parts.append(f"Section: {headings}")

            if chunk_type in ("table", "table_fact"):
                metric = metadata.get("financial_metric") or item.get("financial_metric")
                if metric:
                    text_parts.append(f"Financial Metric: {metric}")
                
                years = metadata.get("fiscal_years") or item.get("fiscal_years") or []
                if years:
                    text_parts.append(f"Fiscal Years: {', '.join(str(y) for y in years)}")
                
                cols = metadata.get("columns") or item.get("columns") or []
                if cols:
                    text_parts.append(f"Table Columns: {', '.join(str(c) for c in cols[:10])}")

                row_labels = "; ".join(str(r) for r in (metadata.get("row_labels") or item.get("row_labels") or []))
                if row_labels and not metric:
                    text_parts.append(f"Table Rows: {row_labels}")

            text_parts.append(f"Content: {content}")
            
            passages.append({
                "id": idx,
                "text": "\n".join(text_parts),
                "meta": item
            })

        try:
            request = RerankRequest(query=query, passages=passages)
            results = self.ranker.rerank(request)

            reranked_output = []
            for res in results[:top_k]:
                original_item = dict(res["meta"])  # copy: never mutate shared candidate dicts
                score = float(res["score"])
                original_item["final_score"] = score
                original_item["rerank_score"] = score
                reranked_output.append(original_item)

            return reranked_output
        except Exception as e:
            logfire.warning(f"FlashRank reranking failed, falling back to initial ranking: {e}")
            for c in candidates[:top_k]:
                c.setdefault("final_score", c.get("rrf_score", c.get("score", 0.0)))
            return candidates[:top_k]

    def rerank_balanced(
        self,
        candidates: List[Dict[str, Any]],
        top_k: int,
        default_query: str,
        entity_queries: Optional[Dict[Optional[str], str]] = None,
    ) -> List[Dict[str, Any]]:
        """Rerank each entity's candidates separately, then interleave round-robin.

        A single global top-k lets one company's chunks crowd out the other's in comparisons.
        Round-robin guarantees every entity gets its best chunks first, and because build_context
        drops from the tail when over budget, no entity is the first to be cut.
        With 0-1 entities this is identical to rerank().
        """
        if not candidates or top_k <= 0:
            return []

        groups: Dict[Optional[str], List[Dict[str, Any]]] = {}
        for c in candidates:
            groups.setdefault(c.get("entity"), []).append(c)
        if len(groups) <= 1:
            return self.rerank(default_query, candidates, top_k)

        entity_queries = entity_queries or {}
        ranked = {
            ent: self.rerank(entity_queries.get(ent) or default_query, items, len(items))
            for ent, items in groups.items()
        }

        out: List[Dict[str, Any]] = []
        depth = 0
        while len(out) < top_k and any(depth < len(r) for r in ranked.values()):
            for r in ranked.values():
                if depth < len(r) and len(out) < top_k:
                    out.append(r[depth])
            depth += 1
        return out
