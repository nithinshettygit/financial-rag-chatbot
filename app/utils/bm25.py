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
    def __init__(self, data_dir: str = getattr(settings, "DATA_DIR", "data")):
        self.valid_chunks = []
        self.bm25 = None
        
        data_path = Path(data_dir)
        manifest_files = list(data_path.rglob("*_manifest.json")) if data_path.exists() else []

        # Fallback to direct path if no wildcard manifests found
        if not manifest_files and Path(settings.BM25_MANIFEST_PATH).exists():
            manifest_files = [Path(settings.BM25_MANIFEST_PATH)]

        chunks = []
        for manifest_file in manifest_files:
            try:
                with open(manifest_file, encoding="utf-8") as f:
                    file_chunks = json.load(f)
                    chunks.extend(file_chunks)
            except Exception as e:
                logger.error(f"Error reading manifest {manifest_file}: {e}")

        self.valid_chunks = [
            c for c in chunks 
            if c.get("content") and str(c.get("content")).strip() not in ("", "N/A", "[Empty Content]")
        ]

        if self.valid_chunks:
            documents = [c["content"] for c in self.valid_chunks]
            tokenized_docs = [tokenize(doc) for doc in documents]
            self.bm25 = BM25Okapi(tokenized_docs)
        else:
            logger.warning("No valid manifests found for BM25. Lexical search will return empty results.")

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
            doc_id = chunk.get("document_id", "doc_default")

            results.append({
                "chunk_id": str(chunk.get("chunk_id", i)),
                "content": chunk.get("content", ""),
                "document_id": doc_id,
                "source_file": chunk.get("source_file") or f"{doc_id}.pdf",
                "page_number": chunk.get("page_number") or chunk.get("metadata", {}).get("page_number", "N/A"),
                "chunk_type": chunk.get("chunk_type", "text"),
                "headings": chunk.get("headings", []),
                "metadata": chunk.get("metadata", {}),
                "score": float(scores[i]),
                "source": "bm25"
            })

        return results