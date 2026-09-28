import os
import sys
from pathlib import Path
from typing import List, Dict, Any
from qdrant_client import QdrantClient
from qdrant_client.http import models
from fastembed import TextEmbedding
from groq import Groq
from dotenv import load_dotenv  


load_dotenv() 

# ============================================================
# CONFIGURATION & STORAGE PATHS
# ============================================================
QDRANT_LOCAL_DB_DIR = Path(r"D:\xfcatr\srinivasan_venkataramanan_mentoring\project-financial-rag-chatbot\data\qdrant_storage")
COLLECTION_NAME = "financial_rag_analytics"
VECTOR_NAME = "fastembed-BAAI/bge-small-en-v1.5"

# LLM Inference Targets
GROQ_MODEL = "openai/gpt-oss-20b"

# ============================================================
# INITIALIZATION & CLEAN TEARDOWN FALLBACKS
# ============================================================
# 1. Sweep orphaned database crash locks cleanly before starting up
LOCK_FILE_PATH = QDRANT_LOCAL_DB_DIR / ".lock"
if LOCK_FILE_PATH.exists():
    try:
        os.remove(LOCK_FILE_PATH)
        print("🧹 Swept away an orphaned temporary database lock file.")
    except Exception:
        pass

# 2. Check for required API environment configurations
if not os.environ.get("GROQ_API_KEY"):
    print("❌ ERROR: 'GROQ_API_KEY' environment variable is missing.")
    print("Please set it in your terminal: set GROQ_API_KEY=gsk_...")
    sys.exit(1)

print("Initializing Local Embedding Engine & Groq LLM Client...")
embedding_model = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")
qdrant_client = QdrantClient(path=str(QDRANT_LOCAL_DB_DIR))
groq_client = Groq()

# ============================================================
# RETRIEVAL LAYER
# ============================================================
def retrieve_contexts(query_text: str, top_k: int = 3) -> str:
    """Queries Qdrant to find the most semantically relevant text fragments."""
    if not qdrant_client.collection_exists(collection_name=COLLECTION_NAME):
        raise ValueError(f"Target collection '{COLLECTION_NAME}' does not exist.")
        
    # Generate query vector list array
    query_generator = embedding_model.embed([query_text])
    query_vector = next(query_generator).tolist()
    
    # Query vector database
    search_result = qdrant_client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        using=VECTOR_NAME,
        limit=top_k,
        with_payload=True
    )
    
    # Structure and format the retrieved payload items into a single context string
    context_blocks = []
    for hit in search_result.points:
        doc_id = hit.payload.get("document_id", "Unknown")
        content = hit.payload.get("page_content", "")
        # Extract location safely from nested metadata dictionary
        metadata = hit.payload.get("metadata", {})
        location = metadata.get("location", "N/A")
        
        block = f"--- [Document Reference: {doc_id} | Location: {location}] ---\n{content}"
        context_blocks.append(block)
        
    return "\n\n".join(context_blocks)

# ============================================================
# INFERENCE SYSTEM (GENERATION LOOP)
# ============================================================
def generate_financial_answer(query_text: str):
    """Executes the complete RAG loop, streaming back response analytics directly."""
    # 1. Fetch relevant background documentation from vector storage files
    print("\n[Step 1/2] Retrieving highly contextual background documents...")
    context = retrieve_contexts(query_text=query_text, top_k=3)
    
    if not context.strip():
        print("⚠️ Warning: No context snippets were discovered for this query in the vector index.")
    
    # 2. Structure a secure, financial-grade system context prompt
    system_prompt = (
        "You are an expert financial analysis AI assistant specializing in corporate filing analysis.\n"
        "Your task is to answer the user's inquiry accurately based ONLY on the verified context provided below.\n\n"
        "=== VERIFIED FINANCIAL CONTEXT START ===\n"
        f"{context}\n"
        "=== VERIFIED FINANCIAL CONTEXT END ===\n\n"
        "CRITICAL EXECUTION INSTRUCTIONS:\n"
        "1. Base your metrics, facts, and figures completely on the verified context blocks above.\n"
        "2. If the context does not contain enough data to reliably answer the query, clearly state that you do not have sufficient information.\n"
        "3. Maintain a neutral, professional, and audit-ready analytical tone."
    )
    
    print(f"[Step 2/2] Dispatching context payload to {GROQ_MODEL} stream interface...\n")
    print(f"{'='*70}\n🤖 SYSTEM ANSWER:\n{'='*70}")
    
    # 3. Stream response tokens smoothly over the network connection
    stream = groq_client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": query_text}
        ],
        temperature=0.1,  # Low temperature for highly factual/deterministic auditing output
        stream=True
    )
    
    for chunk in stream:
        token = chunk.choices[0].delta.content
        if token:
            print(token, end="", flush=True)
    print("\n" + "="*70)

# ============================================================
# RUNTIME INTERFACE
# ============================================================
if __name__ == "__main__":
    try:
        print("Welcome to the Financial RAG Chatbot! Type your queries below.")
        # print("Type 'exit' to terminate the session.\n")
    
        while user_query := input("\nEnter your financial query (or type 'exit' to quit): ").strip():
            if user_query.lower() == "exit":
                print("Exiting the financial RAG chatbot. Goodbye!")
                break
            
            try:
                generate_financial_answer(query_text=user_query)
            except Exception as e:
                print(f"❌ An error occurred while processing your query: {e}")     
        # try:
    #     user_query = "What are the key financial performance metrics of Duolingo?"
    #     generate_financial_answer(query_text=user_query)
    except KeyboardInterrupt:
        print("\nSession interrupted by user. Exiting gracefully.")
    
    finally:
        # Guarantee database files release locked access tokens cleanly even on crashes
        qdrant_client.close()
        print("\nDatabase connections discharged safely.")
