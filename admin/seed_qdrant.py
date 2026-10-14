# admin/seed_qdrant.py
import json
import uuid
from pathlib import Path
from typing import Union

from fastembed import TextEmbedding
from qdrant_client import QdrantClient
from qdrant_client.http import models

COLLECTION_NAME = "financial_rag_analytics"
VECTOR_NAME = "fastembed-BAAI/bge-small-en-v1.5"

BASE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_QDRANT_PATH = str(BASE_DIR / "data" / "qdrant_storage")


def _generate_deterministic_uuid(doc_id: str, chunk_id: str) -> str:
    """Generates a deterministic, valid UUID v5 string compatible with Local Qdrant."""
    namespace = uuid.UUID("6ba7b810-9dad-11d1-80b4-00c04fd430c8")
    return str(uuid.uuid5(namespace, f"{doc_id}:{chunk_id}"))


def seed_database(
    manifest_path: Union[str, Path],
    storage_path: str = DEFAULT_QDRANT_PATH,
    recreate_collection: bool = False,
) -> int:
    manifest_path = Path(manifest_path)
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest missing at {manifest_path}")

    with open(manifest_path, encoding="utf-8") as f:
        chunks = json.load(f)

    valid_chunks = [c for c in chunks if c.get("content") and str(c.get("content")).strip()]
    if not valid_chunks:
        return 0

    Path(storage_path).mkdir(parents=True, exist_ok=True)
    client = QdrantClient(path=storage_path)

    if recreate_collection and client.collection_exists(COLLECTION_NAME):
        client.delete_collection(COLLECTION_NAME)

    if not client.collection_exists(COLLECTION_NAME):
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config={
                VECTOR_NAME: models.VectorParams(
                    size=384, distance=models.Distance.COSINE
                )
            },
        )

    # Payload indexes for metadata filtering
    for field_name in ["document_id", "chunk_type"]:
        try:
            client.create_payload_index(
                collection_name=COLLECTION_NAME,
                field_name=field_name,
                field_schema=models.PayloadSchemaType.KEYWORD,
                wait=True,
            )
        except Exception:
            pass  # Index already exists or handle gracefully

    embedding_model = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")
    texts = [str(c.get("embedding_text") or c.get("content")).strip() for c in valid_chunks]

    embeddings = list(embedding_model.embed(texts, batch_size=32))

    points = []
    point_ids_by_document = {}
    for idx, (chunk, vector) in enumerate(zip(valid_chunks, embeddings)):
        doc_id = chunk.get("document_id", "doc_default")
        chunk_id = chunk.get("chunk_id", f"chunk_{idx}")

        point_id = _generate_deterministic_uuid(doc_id, chunk_id)
        point_ids_by_document.setdefault(doc_id, []).append(point_id)

        payload = {
            "chunk_id": chunk_id,
            "document_id": doc_id,
            "source_file": chunk.get("source_file", ""),
            "content": chunk.get("content", ""),
            "embedding_text": chunk.get("embedding_text", ""),
            "page_number": chunk.get("page_number"),
            "headings": chunk.get("headings", []),
            "chunk_type": chunk.get("chunk_type", "text"),
            "metadata": chunk.get("metadata", {}),
            "company": chunk.get("company") or chunk.get("metadata", {}).get("company"),
        }

        points.append(
            models.PointStruct(
                id=point_id,
                vector={VECTOR_NAME: vector.tolist()},
                payload=payload,
            )
        )

    client.upsert(collection_name=COLLECTION_NAME, points=points)

    # Synchronize and clean up orphan vectors for re-indexed documents
    for doc_id, point_ids in point_ids_by_document.items():
        if not point_ids:
            client.delete(
                collection_name=COLLECTION_NAME,
                points_selector=models.FilterSelector(
                    filter=models.Filter(
                        must=[
                            models.FieldCondition(
                                key="document_id",
                                match=models.MatchValue(value=doc_id),
                            )
                        ]
                    )
                ),
            )
        else:
            client.delete(
                collection_name=COLLECTION_NAME,
                points_selector=models.FilterSelector(
                    filter=models.Filter(
                        must=[
                            models.FieldCondition(
                                key="document_id",
                                match=models.MatchValue(value=doc_id),
                            )
                        ],
                        must_not=[models.HasIdCondition(has_id=point_ids)],
                    )
                ),
            )

    return len(points)