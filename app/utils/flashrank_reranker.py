# app/utils/flashrank_reranker.py
from __future__ import annotations
from typing import Any, Dict, List
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
                original_item = res["meta"]
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