import os
from dotenv import load_dotenv
import logfire

load_dotenv()
logfire.configure(
    service_name="financial-rag-indexer",
    send_to_logfire=bool(os.environ.get("LOGFIRE_TOKEN"))
)
import argparse
import json
import re
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

@logfire.instrument("seed_database_manifest", extract_args=True)  
def seed_database(
    manifest_path: Union[str, Path],
    storage_path: str = DEFAULT_QDRANT_PATH,
    recreate_collection: bool = False,
) -> int:
    manifest_path = Path(manifest_path)
    with logfire.span("load_manifest_file", path=str(manifest_path)):
        if not manifest_path.exists():
            raise FileNotFoundError(f"Manifest missing at {manifest_path}")

        with open(manifest_path, encoding="utf-8") as f:
            chunks = json.load(f)

        valid_chunks = [c for c in chunks if c.get("content") and str(c.get("content")).strip()]
    if not valid_chunks:
        return 0

    Path(storage_path).mkdir(parents=True, exist_ok=True)
    with logfire.span("connect_qdrant_client", storage_path=storage_path):
        client = QdrantClient(path=storage_path)

        if recreate_collection and client.collection_exists(COLLECTION_NAME):
            logfire.warn("Recreating existing collection", collection=COLLECTION_NAME)
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
    index_fields = [
        ("document_id", models.PayloadSchemaType.KEYWORD),
        ("chunk_type", models.PayloadSchemaType.KEYWORD),
        ("company", models.PayloadSchemaType.KEYWORD),
        ("fiscal_year", models.PayloadSchemaType.INTEGER),
        ("fiscal_years", models.PayloadSchemaType.INTEGER),
    ]
    for field_name, schema in index_fields:
        try:
            client.create_payload_index(
                collection_name=COLLECTION_NAME,
                field_name=field_name,
                field_schema=schema,
                wait=True,
            )
        except Exception:
            pass  # Index already exists or handle gracefully
    with logfire.span("generate_embeddings", model=VECTOR_NAME, count=len(valid_chunks)):
        embedding_model = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")
        texts = [str(c.get("embedding_text") or c.get("content")).strip() for c in valid_chunks]

        embeddings = list(embedding_model.embed(texts, batch_size=32))

    points = []
    point_ids_by_document = {}
    with logfire.span("build_qdrant_points", point_count=len(valid_chunks)):
        for idx, (chunk, vector) in enumerate(zip(valid_chunks, embeddings)):
            doc_id = chunk.get("document_id", "doc_default")
            chunk_id = chunk.get("chunk_id", f"chunk_{idx}")

            point_id = _generate_deterministic_uuid(doc_id, chunk_id)
            point_ids_by_document.setdefault(doc_id, []).append(point_id)

            meta = chunk.get("metadata", {})
            
            # Company inference fallback
            company = chunk.get("company") or meta.get("company")
            if not company:
                lower_doc = doc_id.lower()
                if "nike" in lower_doc:
                    company = "Nike"
                elif "duolingo" in lower_doc:
                    company = "Duolingo"

            fiscal_year = chunk.get("fiscal_year") or meta.get("fiscal_year")
            fiscal_years = chunk.get("fiscal_years") or meta.get("fiscal_years") or []
            if not fiscal_year and not fiscal_years:
                src_str = f"{doc_id} {chunk.get('source_file', '')}"
                y_match = re.search(r"20\d{2}", src_str)
                if y_match:
                    inferred_year = int(y_match.group(0))
                    fiscal_year = inferred_year
                    fiscal_years = [inferred_year]

            payload = {
                "chunk_id": chunk_id,
                "document_id": doc_id,
                "source_file": chunk.get("source_file", ""),
                "content": chunk.get("content", ""),
                "embedding_text": chunk.get("embedding_text", ""),
                "page_number": chunk.get("page_number"),
                "fiscal_year": fiscal_year,
                "fiscal_years": fiscal_years,
                "headings": chunk.get("headings", []),
                "chunk_type": chunk.get("chunk_type", "text"),
                "columns": meta.get("columns", []),
                "row_labels": meta.get("row_labels", []),
                "financial_metric": meta.get("financial_metric") or chunk.get("financial_metric"),
                "metadata": meta,
                "company": company,
            }

            points.append(
                models.PointStruct(
                    id=point_id,
                    vector={VECTOR_NAME: vector.tolist()},
                    payload=payload,
                )
            )
    with logfire.span("upsert_points_to_qdrant", total_points=len(points)):
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

    client.close()
    logfire.info("Successfully seeded database", indexed_points=len(points))
    return len(points)


def seed_all_manifests(
    manifest_dir: Union[str, Path] = BASE_DIR / "data" / "manifests",
    storage_path: str = DEFAULT_QDRANT_PATH,
    recreate_collection: bool = False,
) -> int:
    manifest_dir = Path(manifest_dir)
    manifest_files = [
        f for f in manifest_dir.glob("*.json")
        if not f.name.startswith("old_") and "_old" not in f.name and not f.name.endswith(".bak") and "cache" not in f.name
    ]
    total_indexed = 0
    first = True
    for mf in sorted(manifest_files):
        print(f"Seeding {mf.name} into Qdrant...")
        count = seed_database(
            manifest_path=mf,
            storage_path=storage_path,
            recreate_collection=(recreate_collection and first),
        )
        first = False
        total_indexed += count
        print(f"  -> Indexed {count} vectors from {mf.name}")
    return total_indexed


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Seed Qdrant vector database from manifests")
    parser.add_argument("--manifest", help="Path to single manifest file to seed")
    parser.add_argument("--all", action="store_true", help="Seed all active manifests in data/manifests/")
    parser.add_argument("--recreate", action="store_true", help="Recreate collection before seeding")
    args = parser.parse_args()

    if args.manifest:
        cnt = seed_database(args.manifest, recreate_collection=args.recreate)
        print(f"Successfully seeded {cnt} vectors from {args.manifest}")
    elif args.all:
        cnt = seed_all_manifests(recreate_collection=args.recreate)
        print(f"Successfully seeded total {cnt} vectors across all manifests")
    else:
        # Default: seed all manifests if no arg passed
        cnt = seed_all_manifests(recreate_collection=args.recreate)
        print(f"Successfully seeded total {cnt} vectors across all manifests")