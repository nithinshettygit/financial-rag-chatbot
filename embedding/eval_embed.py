import json
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

# ============================================================
# INITIALIZE LOCAL EMBEDDED CONTEXTS
# ============================================================
print("Loading Local C++ ONNX Inference Engine for Querying...")
embedding_model = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")

print(f"Connecting to Local Embedded Qdrant Storage: {QDRANT_LOCAL_DB_DIR}")
qdrant_client = QdrantClient(path=str(QDRANT_LOCAL_DB_DIR))

# ============================================================
# SCHEMAS PROVISIONING LAYER
# ============================================================
def verify_local_collection():
    """Verifies that the target collection exists before attempting search queries."""
    if not qdrant_client.collection_exists(collection_name=COLLECTION_NAME):
        raise ValueError(
            f"Collection '{COLLECTION_NAME}' does not exist. "
            "Please run your ingestion script first to populate the vector storage files."
        )
    print(f"Validated active collection index target: '{COLLECTION_NAME}'")

# ============================================================
# QUERY ENGINE LAYER
# ============================================================
def query_financial_rag(query_text: str, top_k: int = 5) -> List[Dict[str, Any]]:
    """Vectorizes user input string and queries Qdrant named vector space using query_points."""
    # FastEmbed returns a generator. Next extracts the embedding array for our single text element.
    query_generator = embedding_model.embed([query_text])
    query_vector = next(query_generator).tolist()
    
    # Corrected: Pass the raw vector list into 'query', and use 'using' for the named vector key
    search_result = qdrant_client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,       # Raw embedding list
        using=VECTOR_NAME,        # Maps to "fastembed-BAAI/bge-small-en-v1.5"
        limit=top_k,
        with_payload=True
    )
    
    results = []
    for hit in search_result.points:
        results.append({
            "score": hit.score,
            "id": hit.id,
            "content": hit.payload.get("page_content"),
            "document_id": hit.payload.get("document_id"),
            "metadata": hit.payload.get("metadata")
        })
        
    return results


# ============================================================
# RUNTIME INTERFACE
# ============================================================
# ============================================================
# RUNTIME INTERFACE
# ============================================================
if __name__ == "__main__":
    # Ensure database records exist locally
    verify_local_collection()
    
    # Execute Test Query
    test_query = "1. What was Duolingo's net income in 2023?"
    search_hits = query_financial_rag(query_text=test_query, top_k=3)
    print(f"\n{'='*70}\nTEST QUERY: {test_query}\n{'='*70}")
    # print(test_query)
    # Print parsing summary layout
    print(f"\n{'='*70}\nSEARCH RESULTS ({len(search_hits)} matches found):\n{'='*70}")
    for idx, hit in enumerate(search_hits, start=1):
        print(f"\n[{idx}] Score: {hit['score']:.4f} | Document Reference: {hit['document_id']}")
        print(f"Content Snapshot: {hit['content'][:250]}...")
        
    # Explicitly close down storage handlers safely BEFORE python tears down modules
    qdrant_client.close()
