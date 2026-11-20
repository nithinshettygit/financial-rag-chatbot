"""Voyage-backed passage reranking."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import logfire
import voyageai

from app.config import settings


class VoyageReranker:
    def __init__(self):
        if not settings.VOYAGE_API_KEY:
            raise RuntimeError("VOYAGE_API_KEY is required for Voyage reranking")

        logfire.info("Initializing Voyage reranker", model=settings.RERANKER_MODEL)
        self.client = voyageai.Client(
            api_key=settings.VOYAGE_API_KEY,
            timeout=settings.RERANKER_TIMEOUT_S,
            max_retries=0,
        )

    def rerank(self, query: str, candidates: List[Dict[str, Any]], top_k: int) -> List[Dict[str, Any]]:
        if not candidates or top_k <= 0:
            return []

        documents = []
        for item in candidates:
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
                    text_parts.append(f"Fiscal Years: {', '.join(str(year) for year in years)}")

                columns = metadata.get("columns") or item.get("columns") or []
                if columns:
                    text_parts.append(f"Table Columns: {', '.join(str(column) for column in columns[:10])}")

                row_labels = "; ".join(
                    str(label) for label in (metadata.get("row_labels") or item.get("row_labels") or [])
                )
                if row_labels and not metric:
                    text_parts.append(f"Table Rows: {row_labels}")

            text_parts.append(f"Content: {content}")
            documents.append("\n".join(text_parts))

        try:
            response = self.client.rerank(
                query=query,
                documents=documents,
                model=settings.RERANKER_MODEL,
                top_k=min(top_k, len(candidates)),
            )
            reranked = []
            for result in response.results:
                candidate = dict(candidates[result.index])
                score = float(result.relevance_score)
                candidate["final_score"] = score
                candidate["rerank_score"] = score
                reranked.append(candidate)
            return reranked
        except Exception as error:
            logfire.warning(
                "Voyage reranking failed; falling back to initial ranking",
                error_type=type(error).__name__,
            )
            for candidate in candidates[:top_k]:
                candidate.setdefault("final_score", candidate.get("rrf_score", candidate.get("score", 0.0)))
            return candidates[:top_k]

    def rerank_balanced(
        self,
        candidates: List[Dict[str, Any]],
        top_k: int,
        default_query: str,
        entity_queries: Optional[Dict[Optional[str], str]] = None,
    ) -> List[Dict[str, Any]]:
        """Rerank each entity separately, then interleave results round-robin."""
        if not candidates or top_k <= 0:
            return []

        groups: Dict[Optional[str], List[Dict[str, Any]]] = {}
        for candidate in candidates:
            groups.setdefault(candidate.get("entity"), []).append(candidate)
        if len(groups) <= 1:
            return self.rerank(default_query, candidates, top_k)

        entity_queries = entity_queries or {}
        ranked = {
            entity: self.rerank(entity_queries.get(entity) or default_query, items, len(items))
            for entity, items in groups.items()
        }

        output: List[Dict[str, Any]] = []
        depth = 0
        while len(output) < top_k and any(depth < len(items) for items in ranked.values()):
            for items in ranked.values():
                if depth < len(items) and len(output) < top_k:
                    output.append(items[depth])
            depth += 1
        return output
