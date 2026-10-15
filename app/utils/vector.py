import os
from qdrant_client import QdrantClient
from qdrant_client.http import models
from fastembed import TextEmbedding
from app.config import settings
from app.utils.router import QueryFilters
from typing import List, Dict, Any, Optional

class QdrantVectorRetriever:
    def __init__(self):
        self.embedding_model = TextEmbedding(
            model_name=settings.EMBEDDING_MODEL,
            cache_dir=os.getenv("FASTEMBED_CACHE_DIR"),
            local_files_only=os.getenv("FASTEMBED_LOCAL_FILES_ONLY", "false").lower() == "true",
        )
        
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
                with_payload=["document_id", "source_file", "company"],
                with_vectors=False,
            )
            for point in points:
                payload = point.payload or {}
                document_id = str(payload.get("document_id") or "unknown")
                source_file = str(payload.get("source_file") or f"{document_id}.pdf")
                company = str(payload.get("company") or "").strip()
                if not company:
                    lower_name = f"{document_id} {source_file}".lower()
                    if "nike" in lower_name:
                        company = "Nike"
                    elif "duolingo" in lower_name:
                        company = "Duolingo"

                key = (document_id, source_file)
                if key not in documents:
                    documents[key] = {
                        "document_id": document_id,
                        "source_file": source_file,
                        "company": company,
                        "chunk_count": 0,
                    }
                elif not documents[key].get("company") and company:
                    documents[key]["company"] = company
                documents[key]["chunk_count"] += 1

            if offset is None:
                break

        return sorted(
            documents.values(),
            key=lambda document: (document["source_file"].lower(), document["document_id"]),
        )

    def retrieve(
        self,
        query: str,
        top_k: int = settings.RETRIEVAL_K,
        filters: Optional[QueryFilters] = None,
    ) -> List[Dict[str, Any]]:
        if not self.client.collection_exists(settings.COLLECTION_NAME):
            return []

        vector = next(self.embedding_model.query_embed(query)).tolist()
        
        must_conditions = []
        if filters:
            if filters.document_ids:
                must_conditions.append(
                    models.FieldCondition(
                        key="document_id",
                        match=models.MatchAny(any=filters.document_ids),
                    )
                )
            if filters.years:
                must_conditions.append(
                    models.Filter(
                        should=[
                            models.FieldCondition(
                                key="fiscal_year",
                                match=models.MatchAny(any=filters.years),
                            ),
                            models.FieldCondition(
                                key="fiscal_years",
                                match=models.MatchAny(any=filters.years),
                            ),
                            models.IsEmptyCondition(
                                is_empty=models.PayloadField(key="fiscal_year")
                            )
                        ]
                    )
                )
            if filters.company:
                comp_raw = str(filters.company).strip()
                comp_variants = list({
                    comp_raw,
                    comp_raw.lower(),
                    comp_raw.title(),
                    comp_raw.capitalize(),
                    comp_raw.upper(),
                })
                must_conditions.append(
                    models.Filter(
                        should=[
                            models.FieldCondition(
                                key="company",
                                match=models.MatchAny(any=comp_variants),
                            ),
                            models.IsEmptyCondition(
                                is_empty=models.PayloadField(key="company")
                            )
                        ]
                    )
                )

        query_filter = models.Filter(must=must_conditions) if must_conditions else None

        result = self.client.query_points(
            collection_name=settings.COLLECTION_NAME,
            query=vector,
            using=settings.VECTOR_NAME,
            limit=top_k,
            query_filter=query_filter,
            with_payload=True
        )

        chunks = []
        for point in result.points:
            payload = point.payload or {}
            metadata = dict(payload.get("metadata", {}))

            content = payload.get("content") or payload.get("page_content") or ""
            page_num = payload.get("page_number") or metadata.get("page_number") or "N/A"
            chunk_type = payload.get("chunk_type") or metadata.get("chunk_type") or "text"
            headings = payload.get("headings") or metadata.get("headings") or []
            doc_id = payload.get("document_id") or "doc_default"
            source_file = payload.get("source_file") or metadata.get("source_file") or f"{doc_id}.pdf"

            # Propagate top-level payload metadata into metadata dict if missing
            if payload.get("financial_metric") and not metadata.get("financial_metric"):
                metadata["financial_metric"] = payload["financial_metric"]
            if payload.get("columns") and not metadata.get("columns"):
                metadata["columns"] = payload["columns"]
            if payload.get("row_labels") and not metadata.get("row_labels"):
                metadata["row_labels"] = payload["row_labels"]
            if payload.get("fiscal_years") and not metadata.get("fiscal_years"):
                metadata["fiscal_years"] = payload["fiscal_years"]
            if payload.get("company") and not metadata.get("company"):
                metadata["company"] = payload["company"]

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