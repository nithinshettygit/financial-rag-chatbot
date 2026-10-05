# admin/seed_qdrant.py
import json
from pathlib import Path
from typing import Union
from fastembed import TextEmbedding
from qdrant_client import QdrantClient
from qdrant_client.http import models

COLLECTION_NAME = "financial_rag_analytics"
VECTOR_NAME = "fastembed-BAAI/bge-small-en-v1.5"

# Shared absolute vector storage path
DEFAULT_QDRANT_PATH = r"D:\xfcatr\srinivasan_venkataramanan_mentoring\project-financial-rag-chatbot\data\qdrant_storage"


def seed_database(
    manifest_path: Union[str, Path],
    storage_path: str = DEFAULT_QDRANT_PATH,
) -> int:
    manifest_path = Path(manifest_path)
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest missing at {manifest_path}")

    with open(manifest_path, encoding="utf-8") as f:
        chunks = json.load(f)

    valid_chunks = [c for c in chunks if c.get("content") and str(c.get("content")).strip()]
    if not valid_chunks:
        return 0

    client = QdrantClient(path=storage_path)

    # Create collection if it doesn't exist (Do not delete existing documents!)
    if not client.collection_exists(COLLECTION_NAME):
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config={
                VECTOR_NAME: models.VectorParams(
                    size=384,
                    distance=models.Distance.COSINE,
                )
            },
        )

    embedding_model = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")
    texts = [str(c.get("embedding_text") or c.get("content")).strip() for c in valid_chunks]

    # Batch size capped to 32 to prevent ONNX memory allocation issues
    embeddings = list(embedding_model.embed(texts, batch_size=32))

    points = []
    for chunk, vector in zip(valid_chunks, embeddings):
        chunk_id = chunk["chunk_id"]

        points.append(
            models.PointStruct(
                id=chunk_id,  # Uses deterministic string hash UUID from parse_pdf
                vector={VECTOR_NAME: vector.tolist()},
                payload={
                    "chunk_id": chunk_id,
                    "document_id": chunk.get("document_id", "unknown_doc"),
                    "source_file": chunk.get("source_file", "unknown.pdf"),
                    "content": chunk.get("content", ""),
                    "chunk_type": chunk.get("chunk_type", "text"),
                    "page_number": chunk.get("page_number"),
                    "headings": chunk.get("headings", []),
                    "metadata": chunk.get("metadata", {}),
                },
            )
        )

    client.upsert(collection_name=COLLECTION_NAME, points=points)
    return len(points)