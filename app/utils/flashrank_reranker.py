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
            row_labels = "; ".join(str(r) for r in metadata.get("row_labels") or [])
            
            text_parts = []
            if headings:
                text_parts.append(f"Section: {headings}")
            if item.get("chunk_type") == "table" and row_labels:
                text_parts.append(f"Table Rows: {row_labels}")
            text_parts.append(f"Content: {content}")
            
            passages.append({
                "id": idx,
                "text": "\n".join(text_parts),
                "meta": item
            })

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