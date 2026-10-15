import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from rank_bm25 import BM25Okapi

from app.config import settings
from app.utils.logger import logger
from app.utils.router import QueryFilters, company_key


def tokenize(text: str) -> List[str]:
    if not text:
        return []
    # Preserves percentages, years, numbers, and hyphenated terms (e.g., "10-K", "May-31", "42.9%")
    return re.findall(r"\b\d+(?:[.,]\d+)*%?|[a-zA-Z0-9\-]+\b", str(text).lower())


def searchable_text(chunk: Dict[str, Any]) -> str:
    metadata = chunk.get("metadata") or {}
    comp = str(chunk.get("company") or metadata.get("company") or "")
    if not comp:
        doc_lower = str(chunk.get("document_id", "")).lower()
        if "nike" in doc_lower:
            comp = "Nike"
        elif "duolingo" in doc_lower:
            comp = "Duolingo"

    years = " ".join(
        str(y) for y in (chunk.get("fiscal_years") or metadata.get("fiscal_years") or [])
    )
    metric = str(metadata.get("financial_metric") or chunk.get("financial_metric") or "")

    parts = [
        str(chunk.get("document_id") or ""),
        str(chunk.get("source_file") or ""),
        comp,
        years,
        metric,
        " ".join(str(v) for v in chunk.get("headings") or []),
        " ".join(str(v) for v in (metadata.get("columns") or chunk.get("columns") or [])),
        " ".join(str(v) for v in (metadata.get("row_labels") or chunk.get("row_labels") or [])),
        str(chunk.get("embedding_text") or ""),
        str(chunk.get("content") or ""),
    ]
    return " ".join(p for p in parts if p)


class BM25Retriever:
    def __init__(self, data_dir: str = getattr(settings, "DATA_DIR", "data")):
        self.data_dir = data_dir
        self.valid_chunks: List[Dict[str, Any]] = []
        self.bm25: Optional[BM25Okapi] = None
        self._doc_ids: List[str] = []
        self._years: List[set] = []
        self._companies: List[str] = []
        self.load_index()

    def load_index(self) -> None:
        data_path = Path(self.data_dir)
        manifest_files = []
        if data_path.exists():
            # Check manifests directory first, or all json manifests
            m_dir = data_path / "manifests"
            if m_dir.exists():
                manifest_files = list(m_dir.glob("*.json"))
            if not manifest_files:
                manifest_files = list(data_path.rglob("*.json"))

        fallback = getattr(settings, "BM25_MANIFEST_PATH", "")
        if not manifest_files and fallback and Path(fallback).exists():
            f_path = Path(fallback)
            manifest_files = list(f_path.glob("*.json")) if f_path.is_dir() else [f_path]

        chunks: List[Dict[str, Any]] = []
        for manifest_file in manifest_files:
            # Skip old backup manifests, temp files, or caches
            fname = manifest_file.name.lower()
            if (
                fname.startswith("old_")
                or "_old" in fname
                or fname.endswith(".bak")
                or "cache" in fname
                or "evaluation" in fname
            ):
                continue
            try:
                with open(manifest_file, encoding="utf-8") as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        chunks.extend(data)
            except Exception as e:
                logger.error(f"Error reading manifest {manifest_file}: {e}")

        self.valid_chunks = [
            c for c in chunks
            if c.get("content") and str(c["content"]).strip() not in ("", "N/A", "[Empty Content]")
        ]

        # Pre-compute filter attributes once so queries don't re-parse metadata.
        self._doc_ids = []
        self._years = []
        self._companies = []
        for c in self.valid_chunks:
            md = c.get("metadata") or {}
            self._doc_ids.append(c.get("document_id", "doc_default"))
            
            # Extract set of all covered fiscal years
            years_set = set()
            for y in (c.get("fiscal_years") or md.get("fiscal_years") or []):
                try:
                    years_set.add(int(y))
                except (TypeError, ValueError):
                    pass
            single_year = c.get("fiscal_year") or md.get("fiscal_year")
            if single_year:
                try:
                    years_set.add(int(single_year))
                except (TypeError, ValueError):
                    pass
            self._years.append(years_set)

            comp = str(c.get("company") or md.get("company") or "").lower()
            if not comp:
                doc_lower = str(c.get("document_id", "")).lower()
                if "nike" in doc_lower:
                    comp = "nike"
                elif "duolingo" in doc_lower:
                    comp = "duolingo"
            self._companies.append(comp)

        if self.valid_chunks:
            self.bm25 = BM25Okapi([tokenize(searchable_text(c)) for c in self.valid_chunks])
            logger.info(f"BM25 index initialized with {len(self.valid_chunks)} chunks.")
        else:
            self.bm25 = None
            logger.warning("No valid manifests found for BM25. Lexical search will return empty results.")

    def _allowed(self, i: int, filters: Optional[QueryFilters]) -> bool:
        if not filters:
            return True
        if filters.document_ids and self._doc_ids[i] not in filters.document_ids:
            return False
        # Chunks with missing metadata or matching any filter year are kept
        if filters.years and self._years[i]:
            if not self._years[i].intersection(filters.years):
                return False
        if filters.company and self._companies[i]:
            f_key = company_key(filters.company)
            c_key = company_key(self._companies[i])
            if f_key and c_key and (f_key not in c_key and c_key not in f_key):
                return False
        return True

    def retrieve(
        self,
        query: str,
        top_k: int = settings.RETRIEVAL_K,
        filters: Optional[QueryFilters] = None,
    ) -> List[Dict[str, Any]]:
        if not self.bm25 or not self.valid_chunks:
            return []
        tokens = tokenize(query)
        if not tokens:
            return []

        scores = np.asarray(self.bm25.get_scores(tokens), dtype=float)
        order = np.argsort(-scores)

        results: List[Dict[str, Any]] = []
        for i in order:
            i = int(i)
            if len(results) >= top_k or scores[i] <= 0:
                break
            if not self._allowed(i, filters):
                continue

            chunk = self.valid_chunks[i]
            doc_id = self._doc_ids[i]
            metadata = chunk.get("metadata") or {}
            results.append(
                {
                    "chunk_id": str(chunk.get("chunk_id", i)),
                    "content": chunk.get("content", ""),
                    "document_id": doc_id,
                    "source_file": chunk.get("source_file") or f"{doc_id}.pdf",
                    "page_number": chunk.get("page_number") or metadata.get("page_number", "N/A"),
                    "chunk_type": chunk.get("chunk_type", "text"),
                    "headings": chunk.get("headings", []),
                    "metadata": metadata,
                    "score": float(scores[i]),
                    "source": "bm25",
                }
            )
        return results