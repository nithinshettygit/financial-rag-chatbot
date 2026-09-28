import os
from pathlib import Path
from dotenv import load_dotenv
from groq import Groq

# Ensure safe path discovery anchors are loaded
from vector_retriever import retrieve_vector, client
from bm25_retriever import retrieve_bm25

load_dotenv()

# Secure client authentication checks
api_key = os.getenv("GROQ_API_KEY")
if not api_key:
    raise ValueError("CRITICAL ERROR: Environment variable 'GROQ_API_KEY' is missing.")

groq = Groq(api_key=api_key)
MODEL = "openai/gpt-oss-20b"

# --------------------------------------------------
# 1. HYBRID RETRIEVAL: OPTIMIZED RRF
# --------------------------------------------------
def hybrid_search(query: str, top_k: int = 5):
    """
    Executes dense vector searches and sparse keyword lookups in parallel.
    Fuses results seamlessly using Reciprocal Rank Fusion (RRF).
    """
    # Fetch elements from your local search indices
    vector_results = retrieve_vector(query, top_k=8)
    bm25_results = retrieve_bm25(query, top_k=8)

    fused = {}

    for results in [vector_results, bm25_results]:
        for rank, item in enumerate(results, start=1):
            # Safe fallbacks for unique string indexing matching your chunk payload models
            chunk_id = item.get("chunk_id") or item.get("id")
            if not chunk_id:
                continue

            if chunk_id not in fused:
                fused[chunk_id] = {
                    **item,
                    "rrf_score": 0.0
                }

            # Standard mathematical Reciprocal Rank Fusion constant mapping
            fused[chunk_id]["rrf_score"] += (1.0 / (60.0 + rank))

    # Sort candidates by combined structural visibility metric scores
    ranked = sorted(
        fused.values(),
        key=lambda x: x["rrf_score"],
        reverse=True
    )

    return ranked[:top_k]

# --------------------------------------------------
# 2. CONVERSATIONAL QUERY REWRITING
# --------------------------------------------------
def rewrite_query(query: str, history: list) -> str:
    """
    Analyzes multi-turn dialogue trends to rephrase ambiguous shorthand follow-ups 
    into explicit standalone search queries optimal for database lookup.
    """
    # Performance Optimization: If history is completely empty, skip expensive LLM calls entirely
    if not history:
        return query

    # Format the message payload history string cleanly for the rewriter
    history_context = []
    for turn in history[-4:]:
        role_label = "User" if turn["role"] == "user" else "Assistant"
        history_context.append(f"{role_label}: {turn['content']}")
    history_str = "\n".join(history_context)

    # Simplified system framework instruction to guarantee clean output strings without breaking lookups
    system_prompt = (
        "You are a linguistic query rewriting agent for a financial RAG system.\n"
        "Your sole task is to rephrase the latest user question into a standalone document search query.\n"
        "Use the provided conversation history context only to resolve pronouns or implied references (like years or metrics).\n"
        "CRITICAL RULES:\n"
        "1. Do NOT answer the question under any circumstances.\n"
        "2. Preserve the exact original financial meaning.\n"
        "3. Output ONLY the raw rewritten text search string. No pleasantries, no meta-commentary, no notes."
    )

    response = groq.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {
                "role": "user", 
                "content": f"Conversation History:\n{history_str}\n\nLatest follow-up question: {query}\nStandalone Search Query:"
            }
        ],
        temperature=0.0  # Force absolute deterministic consistency
    )

    rewritten = response.choices[0].message.content.strip()
    print(f"\n[Query Rewriter] Original: '{query}' | Rewritten: '{rewritten}'")
    return rewritten if rewritten else query

# --------------------------------------------------
# 3. GENERATE CONTEXTUAL ANSWER
# --------------------------------------------------
def generate_answer(query: str, history: list) -> str:
    """
    Coordinates query rewrite steps, hybrid search execution, 
    and structured system prompt synthesis with streaming inference hooks.
    """
    # Step 1: Rephrase ambiguous parameters based on active memory layers
    search_query = rewrite_query(query, history)
    print(f"\n[RAG Pipeline] Condensed Standalone Query: '{search_query}'")
    # Step 2: Fetch interleaved text blocks and markdown table fragments via RRF
    results = hybrid_search(search_query, top_k=4)

    # Step 3: Map dictionary keys accurately based on your chunk manifest schemas
    context_parts = []
    for i, item in enumerate(results):
        doc_id = item.get("document_id", "Financial Report")
        meta = item.get("metadata", {})
        page = meta.get("page_number") or item.get("page", "Unknown")
        
        # Safe lookup protection wrapper for matching content payload structures
        raw_content = item.get("page_content") or item.get("content") or ""
        
        context_parts.append(
            f"[S{i+1}] Source: {doc_id} | Location: Page {page}\n"
            f"{raw_content.strip()}"
        )

    context = "\n\n---\n\n".join(context_parts)

    if not context.strip():
        return "I could not find sufficient matching financial records in the data repository index."

    # Step 4: Construct the system framework payload array
    system_instruction = (
        "You are an expert financial analysis QA system. Your assignment is to answer the user's query "
        "using ONLY the verified document context fragments supplied below.\n\n"
        "CRITICAL PRODUCTION RULES:\n"
        "1. Rely strictly on the visual structures of Markdown tables for numbers, metrics, and comparisons.\n"
        "2. Cite your supporting facts explicitly using [S1], [S2], etc., matching the context labels.\n"
        "3. If the context values are insufficient to answer the question, state that clearly. Never hallucinate financial data.\n\n"
        f"=== VERIFIED DOCUMENT CONTEXT ===\n{context}\n================================="
    )

    # Cleanly bundle prompt parameters into a well-formed structured API messages array
    messages = [{"role": "system", "content": system_instruction}]
    
    # Safely append conversational context history blocks to preserve timeline tracking loops
    for turn in history[-6:]:
        messages.append({"role": turn["role"], "content": turn["content"]})
        
    messages.append({"role": "user", "content": query})

    response = groq.chat.completions.create(
        model=MODEL,
        messages=messages,
        temperature=0.1  # Protect numbers tracking accuracy by minimizing creativity parameters
    )

    return response.choices[0].message.content

# --------------------------------------------------
# 4. RUNTIME CHAT INTERFACE LOOP
# --------------------------------------------------
if __name__ == "__main__":
    session_history = []

    print("=" * 60)
    print("Enterprise Financial Hybrid RAG Chatbot (O(1) Local Storage Mode)")
    print("Commands: Type 'clear' to reset memory windows | 'exit' to quit runtime loop.")
    print("=" * 60)

    try:
        while True:
            user_input = input("\nYou: ").strip()

            if user_input.lower() == "exit":
                print("Terminating conversational session context paths...")
                break

            if user_input.lower() == "clear":
                session_history.clear()
                print("System memory reset. All previous conversation context cleared.")
                continue

            if not user_input:
                continue

            try:
                # Fire production orchestration execution layer
                ai_response = generate_answer(user_input, session_history)
                # print(f"\n History : {session_history}")
                print(f"\nAssistant: {ai_response}")

                # Commit data values dynamically to local runtime history memory arrays
                session_history.extend([
                    {"role": "user", "content": user_input},
                    {"role": "assistant", "content": ai_response}
                ])

            except Exception as runtime_error:
                print(f"\nPipeline Execution Error: {runtime_error}")

    finally:
        # Guarantee local file locks are released cleanly upon shell loop exit
        if 'client' in globals() and hasattr(client, 'close'):
            client.close()
            print("Local database connection hooks terminated safely.")
