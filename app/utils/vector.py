from qdrant_client import QdrantClient
from fastembed import TextEmbedding
from app.config import settings
from typing import List, Dict, Any

class QdrantVectorRetriever:
    def __init__(self):
        self.embedding_model = TextEmbedding(model_name=settings.EMBEDDING_MODEL)
        
        if getattr(settings, "QDRANT_URL", None):
            self.client = QdrantClient(
                url=settings.QDRANT_URL,
                api_key=settings.QDRANT_API_KEY
            )
        else:
            self.client = QdrantClient(path=settings.QDRANT_STORAGE_PATH)

    def list_documents(self) -> List[Dict[str, Any]]:
        if not self.client.collection_exists(settings.COLLECTION_NAME):
            return []

        documents = {}
        offset = None
        while True:
            points, offset = self.client.scroll(
                collection_name=settings.COLLECTION_NAME,
                limit=1000,
                offset=offset,
                with_payload=["document_id", "source_file"],
                with_vectors=False,
            )
            for point in points:
                payload = point.payload or {}
                document_id = str(payload.get("document_id") or "unknown")
                source_file = str(payload.get("source_file") or f"{document_id}.pdf")
                key = (document_id, source_file)
                if key not in documents:
                    documents[key] = {
                        "document_id": document_id,
                        "source_file": source_file,
                        "chunk_count": 0,
                    }
                documents[key]["chunk_count"] += 1

            if offset is None:
                break

        return sorted(
            documents.values(),
            key=lambda document: (document["source_file"].lower(), document["document_id"]),
        )

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
                "source_file",
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
            doc_id = payload.get("document_id") or "doc_default"
            source_file = payload.get("source_file") or metadata.get("source_file") or f"{doc_id}.pdf"

            chunks.append({
                "chunk_id": str(payload.get("chunk_id") or point.id),
                "content": content,
                "document_id": doc_id,
                "source_file": source_file,
                "page_number": page_num,
                "chunk_type": chunk_type,
                "headings": headings,
                "metadata": metadata,
                "score": float(point.score),
                "source": "vector"
            })

        return chunks