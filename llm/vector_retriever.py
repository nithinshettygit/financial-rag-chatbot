
import os
from pathlib import Path

from qdrant_client import QdrantClient
from fastembed import TextEmbedding

# Configuration
DB_PATH = Path(
    r"D:\xfcatr\srinivasan_venkataramanan_mentoring"
    r"\project-financial-rag-chatbot\data\qdrant_storage"
)

COLLECTION = "financial_rag_analytics"
VECTOR_NAME = "fastembed-BAAI/bge-small-en-v1.5"

embedding_model = TextEmbedding(
    model_name="BAAI/bge-small-en-v1.5"
)

# Local Qdrant
if os.getenv("QDRANT_URL"):
    client = QdrantClient(
        url=os.getenv("QDRANT_URL"),
        api_key=os.getenv("QDRANT_API_KEY")
    )
else:
    client = QdrantClient(path=str(DB_PATH))


def retrieve_vector(query, top_k=5):

    vector = next(
        embedding_model.query_embed(query)
    ).tolist()

    result = client.query_points(
        collection_name=COLLECTION,
        query=vector,
        using=VECTOR_NAME,
        limit=top_k,
        with_payload=True
    )

    chunks = []

    for point in result.points:
        payload = point.payload or {}
        metadata = payload.get("metadata", {})

        chunks.append({
            "chunk_id": str(
                metadata.get("chunk_id", point.id)
            ),
            "content": payload.get(
                "page_content", ""
            ),
            "metadata": metadata,
            "score": point.score,
            "source": "vector"
        })

    return chunks