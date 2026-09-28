
import json
import re
from pathlib import Path

from rank_bm25 import BM25Okapi

MANIFEST_PATH = Path(
    r"D:\xfcatr\srinivasan_venkataramanan_mentoring"
    r"\project-financial-rag-chatbot\data\extracted"
    r"\chunks_manifest.json"
)


def tokenize(text):
    return re.findall(
        r"\d+(?:[.,]\d+)*%?|[a-zA-Z]+",
        text.lower()
    )


# Load chunks once
with open(MANIFEST_PATH, encoding="utf-8") as f:
    chunks = json.load(f)

documents = [
    chunk["content"] for chunk in chunks
]

tokenized_docs = [
    tokenize(doc) for doc in documents
]

bm25 = BM25Okapi(tokenized_docs)


def retrieve_bm25(query, top_k=5):

    query_tokens = tokenize(query)

    scores = bm25.get_scores(query_tokens)

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

        chunk = chunks[i]

        results.append({
            "chunk_id": str(chunk["chunk_id"]),
            "content": chunk["content"],
            "metadata": chunk.get("metadata", {}),
            "score": float(scores[i]),
            "source": "bm25"
        })

    return results