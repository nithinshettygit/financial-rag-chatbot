from qdrant_client import QdrantClient
from fastembed import TextEmbedding
from app.config import settings
from typing import List, Dict, Any

class QdrantVectorRetriever:
    def __init__(self):
        self.embedding_model = TextEmbedding(model_name=settings.EMBEDDING_MODEL)
        
        if settings.QDRANT_URL:
            self.client = QdrantClient(
                url=settings.QDRANT_URL,
                api_key=settings.QDRANT_API_KEY
            )
        else:
            self.client = QdrantClient(path=settings.QDRANT_STORAGE_PATH)

    def retrieve(self, query: str, top_k: int = settings.RETRIEVAL_K) -> List[Dict[str, Any]]:
        vector = next(self.embedding_model.query_embed(query)).tolist()

        result = self.client.query_points(
            collection_name=settings.COLLECTION_NAME,
            query=vector,
            using=settings.VECTOR_NAME,
            limit=top_k,
            with_payload=[
                "chunk_id",
                "content",
                "page_content",
                "document_id",
                "page_number",
                "chunk_type",
                "headings",
                "metadata",
            ]
        )

        chunks = []
        for point in result.points:
            payload = point.payload or {}
            metadata = payload.get("metadata", {})

            content = payload.get("content") or payload.get("page_content") or ""
            page_num = payload.get("page_number") or metadata.get("page_number") or "N/A"
            chunk_type = payload.get("chunk_type") or metadata.get("chunk_type") or "text"
            headings = payload.get("headings") or metadata.get("headings") or []

            chunks.append({
                "chunk_id": str(payload.get("chunk_id") or point.id),
                "content": content,
                "document_id": payload.get("document_id", "duolingo_10k"),
                "page_number": page_num,
                "chunk_type": chunk_type,
                "headings": headings,
                "metadata": metadata,
                "score": float(point.score),
                "source": "vector"
            })

        return chunks