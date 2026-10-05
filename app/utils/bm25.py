import json
import re
from pathlib import Path
from rank_bm25 import BM25Okapi
from app.config import settings
from app.utils.logger import logger
from typing import List, Dict, Any

def tokenize(text: str) -> List[str]:
    if not text:
        return []
    return re.findall(r"\d+(?:[.,]\d+)*%?|[a-zA-Z]+", text.lower())

class BM25Retriever:
    def __init__(self, manifest_path: str = settings.BM25_MANIFEST_PATH):
        self.valid_chunks = []
        self.bm25 = None
        
        path = Path(manifest_path)
        if path.exists():
            with open(path, encoding="utf-8") as f:
                chunks = json.load(f)

            self.valid_chunks = [
                c for c in chunks 
                if c.get("content") and str(c.get("content")).strip() not in ("", "N/A", "[Empty Content]")
            ]
            documents = [c["content"] for c in self.valid_chunks]
            tokenized_docs = [tokenize(doc) for doc in documents]
            self.bm25 = BM25Okapi(tokenized_docs)
        else:
            logger.warning(f"BM25 Manifest file not found at {manifest_path}. BM25 search will return empty results.")

    def retrieve(self, query: str, top_k: int = settings.RETRIEVAL_K) -> List[Dict[str, Any]]:
        if not self.bm25 or not self.valid_chunks:
            return []

        query_tokens = tokenize(query)
        scores = self.bm25.get_scores(query_tokens)

        ranked_indices = sorted(
            range(len(scores)),
            key=lambda i: scores[i],
            reverse=True
        )

        results = []
        for i in ranked_indices:
            if len(results) >= top_k:
                break

            if scores[i] <= 0:
                continue

            chunk = self.valid_chunks[i]

            results.append({
                "chunk_id": str(chunk.get("chunk_id", i)),
                "content": chunk.get("content", ""),
                "document_id": chunk.get("document_id", "duolingo_10k"),
                "page_number": chunk.get("page_number") or chunk.get("metadata", {}).get("page_number", "N/A"),
                "chunk_type": chunk.get("chunk_type", "text"),
                "headings": chunk.get("headings", []),
                "metadata": chunk.get("metadata", {}),
                "score": float(scores[i]),
                "source": "bm25"
            })

        return results