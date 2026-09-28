import json
import uuid  # Natively built-in Python module to handle standard UUID creation rules
from pathlib import Path
from typing import List, Dict, Any
from qdrant_client import QdrantClient
from qdrant_client.http import models
from fastembed import TextEmbedding

# ============================================================
# CONFIGURATION & STORAGE PATHS
# ============================================================
QDRANT_LOCAL_DB_DIR = Path(r"D:\xfcatr\srinivasan_venkataramanan_mentoring\project-financial-rag-chatbot\data\qdrant_storage")
COLLECTION_NAME = "financial_rag_analytics"
VECTOR_NAME = "fastembed-BAAI/bge-small-en-v1.5"

CHUNKS_MANIFEST_PATH = Path(r"D:\xfcatr\srinivasan_venkataramanan_mentoring\project-financial-rag-chatbot\data\extracted\chunks_manifest.json")

# ============================================================
# INITIALIZE LOCAL EMBEDDED CONTEXTS
# ============================================================
print(f"Loading BAAI/bge-small-en-v1.5 TextEmbedding  Engine...")
embedding_model = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")

print(f"Initializing Local Embedded Qdrant Storage engine at: {QDRANT_LOCAL_DB_DIR}")
QDRANT_LOCAL_DB_DIR.mkdir(parents=True, exist_ok=True)
qdrant_client = QdrantClient(path=str(QDRANT_LOCAL_DB_DIR))

# ============================================================
# SCHEMAS PROVISIONING LAYER
# ============================================================
def init_local_collection():
    """Initializes a collection optimized for CPU processing and high-speed metadata lookups."""
    if not qdrant_client.collection_exists(collection_name=COLLECTION_NAME):
        print(f"Creating local embedded collection: '{COLLECTION_NAME}'...")
        qdrant_client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config={
                VECTOR_NAME: models.VectorParams(
                    size=384,  # bge-small outputs 384 dimensions natively
                    distance=models.Distance.COSINE
                )
            }
        )
        print("Embedded index collection mapped successfully.")

# ============================================================
# LOCAL INGESTION PIPELINE
# ============================================================
def execute_local_ingestion_pipeline(batch_size: int = 32):
    """Reads processing chunks manifest, creates vectors via local ONNX drivers, and stores to database files."""
    if not CHUNKS_MANIFEST_PATH.exists():
        raise FileNotFoundError(f"Source manifest configurations missing: {CHUNKS_MANIFEST_PATH}")
        
    with open(CHUNKS_MANIFEST_PATH, "r", encoding="utf-8") as f:
        chunks_data: List[Dict[str, Any]] = json.load(f)
        
    total_chunks = len(chunks_data)
    print(f"Loaded {total_chunks} chunk nodes from data manifest. Starting local CPU inference engine...")
    
    # Establish a stable deterministic base namespace to keep your ingestion process idempotent
    # If you run the script multiple times, the generated UUIDs remain identically repeatable
    NAMESPACE_UUID = uuid.UUID("12345678-1234-5678-1234-567812345678")
    
    for i in range(0, total_chunks, batch_size):
        batch_window = chunks_data[i:i + batch_size]
        batch_texts = [node["content"] for node in batch_window]
        
        # Calculate text layouts using fast ONNX threads
        embeddings_generator = embedding_model.embed(batch_texts)
        batch_embeddings = [vector.tolist() for vector in embeddings_generator]
        
        points_to_upload = []
        for idx, node in enumerate(batch_window):
            # Transform your 16-char string into a valid, standardized 36-char UUID format natively
            valid_uuid = str(uuid.uuid5(NAMESPACE_UUID, node["chunk_id"]))
            
            points_to_upload.append(
                models.PointStruct(
                    id=valid_uuid,  # Serverless engine safely accepts this formatted key string
                    vector={
                        VECTOR_NAME: batch_embeddings[idx]
                    },
                    payload={
                        "page_content": node["content"],
                        "document_id": node["document_id"],
                        "metadata": node["metadata"]
                    }
                )
            )
            
        # Synchronously write batch to database folder layout
        qdrant_client.upsert(
            collection_name=COLLECTION_NAME,
            points=points_to_upload,
            wait=True
        )
        print(f" -> Checkpointed processing index block target: {min(i + batch_size, total_chunks)} / {total_chunks}")
    
    print(f"\n{'='*70}\nINGESTION COMPLETE: Serverless Database Index Successfully Loaded.\n{'='*70}")

# ============================================================
# RUNTIME INTERFACE
# ============================================================
if __name__ == "__main__":
    init_local_collection()
    execute_local_ingestion_pipeline(batch_size=32)
