import os
import sys
import json
from pathlib import Path
from typing import List, Dict, Any, Literal
from typing_extensions import TypedDict

# Core Vector & Client Dependencies
from qdrant_client import QdrantClient
from qdrant_client.http import models
from fastembed import TextEmbedding
from groq import Groq
from dotenv import load_dotenv

# LangGraph Core Architecture Components
from langgraph.graph import StateGraph, START, END

# Load local environment parameters
load_dotenv()

# ============================================================
# CONFIGURATION & STORAGE PATHS
# ============================================================
QDRANT_LOCAL_DB_DIR = Path(r"D:\xfcatr\srinivasan_venkataramanan_mentoring\project-financial-rag-chatbot\data\qdrant_storage")
COLLECTION_NAME = "financial_rag_analytics"
VECTOR_NAME = "fastembed-BAAI/bge-small-en-v1.5"

# LLM Inference Targets
GROQ_MODEL = "openai/gpt-oss-20b"
ROUTER_MODEL = GROQ_MODEL  # Token-optimized lightweight router model variant

# ============================================================
# INITIALIZATION & CLEAN TEARDOWN FALLBACKS
# ============================================================
LOCK_FILE_PATH = QDRANT_LOCAL_DB_DIR / ".lock"
if LOCK_FILE_PATH.exists():
    try:
        os.remove(LOCK_FILE_PATH)
        print("Swept away an orphaned temporary database lock file.")
    except Exception:
        pass

if not os.environ.get("GROQ_API_KEY"):
    print("ERROR: 'GROQ_API_KEY' environment variable is missing.")
    print("Please set it in your terminal: set GROQ_API_KEY=gsk_...")
    sys.exit(1)

print("Initializing Local Embedding Engine, Qdrant Client, & Groq Client...")
embedding_model = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")
qdrant_client = QdrantClient(path=str(QDRANT_LOCAL_DB_DIR))
groq_client = Groq()

# ============================================================
# LANGGRAPH STATE TYPE DEFINITIONS
# ============================================================
class RAGState(TypedDict):
    """Encapsulates execution context passed downstream between LangGraph nodes."""
    user_query: str
    chat_history: List[Dict[str, str]]
    transformed_query: str
    retrieved_context: str
    final_response: str

# ============================================================
# LANGGRAPH PIPELINE IMPLEMENTATION (NODES)
# ============================================================

def _truncate(text: str, max_chars: int = 400) -> str:
    """Caps a single turn's length before it goes into the router prompt.
    Long assistant answers (e.g. big markdown tables) can otherwise confuse
    the router model into malformed JSON or a misclassification."""
    text = text or ""
    return text if len(text) <= max_chars else text[:max_chars] + " ...[truncated]"


def _extract_json(raw: str) -> Dict[str, Any]:
    """Groq's JSON mode should return raw JSON, but some models still wrap it
    in markdown fences. Strip those before parsing so we don't silently fail."""
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`").strip()
        if cleaned.lower().startswith("json"):
            cleaned = cleaned.split("\n", 1)[1] if "\n" in cleaned else ""
    return json.loads(cleaned)


def rewrite_query_node(state: RAGState) -> Dict[str, Any]:
    """Evaluates user intent and dynamically resolves follow-ups into standalone queries."""
    # TOKEN OPTIMIZATION: Slice history to only the last 2 turns to keep input windows tight
    bounded_history = state["chat_history"][-2:] if state["chat_history"] else []

    # Fast-track if this is the first turn of the session
    if not bounded_history:
        print(f"[ROUTER] No prior history. Using raw query: '{state['user_query']}'")
        return {"transformed_query": state["user_query"]}

    history_str = "\n".join(
        f"{m['role'].upper()}: {_truncate(m['content'])}" for m in bounded_history
    )

    system_prompt = (
        "You are an expert financial query routing engine.\n"
        "Analyze the conversation history and the new user question.\n"
        "Determine if the question relies on previous context (e.g., contains pronouns like 'it', 'they', or continuations like 'more about it').\n\n"
        "Return a JSON object containing exactly two keys:\n"
        "1. \"is_context_dependent\": true/false\n"
        "2. \"optimized_query\": \"The fully rephrased standalone search query incorporating historical context\"\n"
        "If the query is a net-new standalone question, set is_context_dependent to false and copy the query text verbatim.\n"
        "Respond with raw JSON only. Do not wrap it in markdown code fences."
    )

    user_prompt = f"=== CONVERSATION HISTORY ===\n{history_str}\n\n=== NEW FOLLOW-UP QUERY ===\n{state['user_query']}"

    try:
        response = groq_client.chat.completions.create(
            model=ROUTER_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.0,
            response_format={"type": "json_object"}
        )

        raw_content = response.choices[0].message.content
        result = _extract_json(raw_content)

        if result.get("is_context_dependent", False):
            search_query = result.get("optimized_query") or state["user_query"]
            print(f"[ROUTER] Context-dependent follow-up detected. Rewritten query for DB: '{search_query}'")
            return {"transformed_query": search_query}

    except Exception as e:
        # BUGFIX: previously a bare `except: pass` swallowed every failure
        # (e.g. markdown-fenced JSON, malformed response) and silently fell
        # back to searching the raw ambiguous follow-up text.
        print(f"[ROUTER] Rewrite failed, falling back to raw query. Error: {e}")

    print(f"[ROUTER] Fresh or standalone query detected. Proceeding directly: '{state['user_query']}'")
    return {"transformed_query": state["user_query"]}


def retrieve_context_node(state: RAGState) -> Dict[str, Any]:
    """Queries your local Qdrant database using the token-optimized search query string."""
    query_text = state["transformed_query"]
    print(f"[DEBUG] Searching Qdrant with transformed_query = '{query_text}'")
    print("[Step 1/2] Searching local Qdrant vector index storage...")

    if not qdrant_client.collection_exists(collection_name=COLLECTION_NAME):
        raise ValueError(f"Target collection '{COLLECTION_NAME}' does not exist.")

    query_generator = embedding_model.embed([query_text])
    query_vector = next(query_generator).tolist()

    search_result = qdrant_client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        using=VECTOR_NAME,
        limit=3,
        with_payload=True
    )

    context_blocks = []
    for hit in search_result.points:
        doc_id = hit.payload.get("document_id", "Unknown")
        content = hit.payload.get("page_content", "")
        metadata = hit.payload.get("metadata", {})
        location = metadata.get("location", "N/A")

        block = f"--- [Document Reference: {doc_id} | Location: {location}] ---\n{content}"
        context_blocks.append(block)

    return {"retrieved_context": "\n\n".join(context_blocks)}


def generate_answer_node(state: RAGState) -> Dict[str, Any]:
    """Streams the final audited corporate insight answer directly to the terminal interface."""
    system_prompt = (
        "You are an expert financial analysis AI assistant specializing in corporate filing analysis.\n"
        "Your task is to answer the user's inquiry accurately based ONLY on the verified context provided below.\n\n"
        "=== VERIFIED FINANCIAL CONTEXT START ===\n"
        f"{state['retrieved_context']}\n"
        "=== VERIFIED FINANCIAL CONTEXT END ===\n\n"
        "CRITICAL EXECUTION INSTRUCTIONS:\n"
        "1. Base your metrics, facts, and figures completely on the verified context blocks above.\n"
        "2. If the context does not contain enough data to reliably answer the query, clearly state that you do not have sufficient information.\n"
        "3. Maintain a neutral, professional, and audit-ready analytical tone."
    )

    print(f"[Step 2/2] Dispatching context payload to {GROQ_MODEL} stream interface...\n")
    print(f"{'='*70}\nSYSTEM ANSWER:\n{'='*70}")

    stream = groq_client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": state["user_query"]}
        ],
        temperature=0.1,
        stream=True
    )

    full_response_text = ""
    for chunk in stream:
        token = chunk.choices[0].delta.content
        if token:
            print(token, end="", flush=True)
            full_response_text += token
    print("\n" + "=" * 70)

    return {"final_response": full_response_text}


def fallback_empty_node(state: RAGState) -> Dict[str, Any]:
    """TOKEN OPTIMIZATION: Blocks downstream LLM generation costs if context is missing."""
    output = "Warning: No matching financial context snippets were discovered for this query in the vector index."
    print(f"{'='*70}\nSYSTEM ANSWER:\n{'='*70}\n{output}\n{'='*70}")
    return {"final_response": output}


# ============================================================
# CONDITIONAL GRAPH ROUTING LOGIC
# ============================================================
def route_after_retrieval(state: RAGState) -> Literal["generate", "fallback"]:
    """Inspects context content thickness prior to committing upstream resources."""
    if not state["retrieved_context"] or not state["retrieved_context"].strip():
        return "fallback"
    return "generate"


# ============================================================
# GRAPH COMPILATION STACK
# ============================================================
workflow = StateGraph(RAGState)

# Add all the processing blocks
workflow.add_node("rewrite_query", rewrite_query_node)
workflow.add_node("retrieve_context", retrieve_context_node)
workflow.add_node("generate_answer", generate_answer_node)
workflow.add_node("fallback_empty", fallback_empty_node)

# Map edge connections
workflow.add_edge(START, "rewrite_query")
workflow.add_edge("rewrite_query", "retrieve_context")

# Connect the token-saving conditional routing router
workflow.add_conditional_edges(
    "retrieve_context",
    route_after_retrieval,
    {
        "generate": "generate_answer",
        "fallback": "fallback_empty"
    }
)

workflow.add_edge("generate_answer", END)
workflow.add_edge("fallback_empty", END)

# Compile into an active application layout
rag_application = workflow.compile()


# ============================================================
# RUNTIME INTERFACE (STATEFUL USER LOOP)
# ============================================================
if __name__ == "__main__":
    # Local variable array maintaining persistent cross-turn session context memory
    session_history: List[Dict[str, str]] = []

    try:
        print("Welcome to the Enterprise Financial RAG Chatbot! Type your queries below.")
        print("Type 'exit' to terminate the session safely.\n")

        while True:
            user_query = input("\nEnter your financial query (or type 'exit' to quit): ").strip()

            if not user_query:
                continue

            if user_query.lower() == "exit":
                print("Exiting the financial RAG chatbot. Goodbye!")
                break

            try:
                # Format state input parameters
                inputs = {
                    "user_query": user_query,
                    "chat_history": session_history,
                    "transformed_query": "",
                    "retrieved_context": "",
                    "final_response": ""
                }

                # Run the complete graph execution loop synchronously
                updated_state = rag_application.invoke(inputs)

                # Append outputs onto memory variables to preserve session continuity
                session_history.append({"role": "user", "content": user_query})
                session_history.append({"role": "assistant", "content": updated_state["final_response"]})

            except Exception as e:
                print(f"An error occurred while processing your query: {e}")

    except KeyboardInterrupt:
        print("\nSession interrupted by user. Exiting gracefully.")

    finally:
        # Guarantee database files release locked access handlers cleanly on system shutdown
        qdrant_client.close()
        print("\nDatabase connections discharged safely.")