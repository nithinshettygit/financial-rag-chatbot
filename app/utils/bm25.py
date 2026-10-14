import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from rank_bm25 import BM25Okapi

from app.config import settings
from app.utils.logger import logger
from app.utils.router import QueryFilters


def tokenize(text: str) -> List[str]:
    if not text:
        return []
    return re.findall(r"\d+(?:[.,]\d+)*%?|[a-zA-Z]+", text.lower())


def searchable_text(chunk: Dict[str, Any]) -> str:
    metadata = chunk.get("metadata") or {}
    parts = [
        str(chunk.get("document_id") or ""),
        str(chunk.get("source_file") or ""),
        str(chunk.get("company") or metadata.get("company") or ""),
        " ".join(str(v) for v in chunk.get("headings") or []),
        " ".join(str(v) for v in metadata.get("columns") or []),
        " ".join(str(v) for v in metadata.get("row_labels") or []),
        str(chunk.get("content") or ""),
    ]
    return " ".join(p for p in parts if p)


class BM25Retriever:
    def __init__(self, data_dir: str = getattr(settings, "DATA_DIR", "data")):
        self.valid_chunks: List[Dict[str, Any]] = []
        self.bm25: Optional[BM25Okapi] = None

        data_path = Path(data_dir)
        manifest_files = list(data_path.rglob("*_manifest.json")) if data_path.exists() else []
        fallback = getattr(settings, "BM25_MANIFEST_PATH", "")
        if not manifest_files and fallback and Path(fallback).exists():
            manifest_files = [Path(fallback)]

        chunks: List[Dict[str, Any]] = []
        for manifest_file in manifest_files:
            try:
                with open(manifest_file, encoding="utf-8") as f:
                    chunks.extend(json.load(f))
            except Exception as e:
                logger.error(f"Error reading manifest {manifest_file}: {e}")

        self.valid_chunks = [
            c for c in chunks
            if c.get("content") and str(c["content"]).strip() not in ("", "N/A", "[Empty Content]")
        ]

        # Pre-compute filter attributes once so queries don't re-parse metadata.
        self._doc_ids: List[str] = []
        self._years: List[Optional[int]] = []
        self._companies: List[str] = []
        for c in self.valid_chunks:
            md = c.get("metadata") or {}
            self._doc_ids.append(c.get("document_id", "doc_default"))
            year = c.get("fiscal_year") or md.get("fiscal_year")
            try:
                self._years.append(int(year) if year else None)
            except (TypeError, ValueError):
                self._years.append(None)
            self._companies.append(str(c.get("company") or md.get("company") or "").lower())

        if self.valid_chunks:
            self.bm25 = BM25Okapi([tokenize(searchable_text(c)) for c in self.valid_chunks])
        else:
            logger.warning("No valid manifests found for BM25. Lexical search will return empty results.")

    def _allowed(self, i: int, filters: Optional[QueryFilters]) -> bool:
        if not filters:
            return True
        if filters.document_ids and self._doc_ids[i] not in filters.document_ids:
            return False
        # Chunks with missing metadata are kept (lenient), mirroring the vector filter.
        if filters.years and self._years[i] is not None and self._years[i] not in filters.years:
            return False
        if filters.company and self._companies[i] and filters.company.lower() not in self._companies[i]:
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