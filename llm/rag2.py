import os
import json
import re
from concurrent.futures import ThreadPoolExecutor
from dotenv import load_dotenv
from groq import Groq

from vector_retriever import retrieve_vector, client
from bm25_retriever import retrieve_bm25

# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

load_dotenv()

api_key = os.getenv("GROQ_API_KEY")
if not api_key:
    raise ValueError("GROQ_API_KEY is missing.")

groq = Groq(api_key=api_key)

MODEL = "openai/gpt-oss-20b"

RETRIEVAL_K = 8
FINAL_TOP_K = 4
RRF_K = 60

EMPTY_STATE = {
    "company": None,
    "fiscal_year": None,
    "target_metric": None
}


# --------------------------------------------------
# 1. INTENT ROUTING & QUERY PREPARATION
# --------------------------------------------------

def classify_and_prepare(query: str, history: list, state: dict):
    """
    Determines user intent (RAG vs Conversation Summary vs Chit-chat)
    and rewrites RAG query if needed.
    """
    recent_history = history[-6:]
    history_text = "\n".join(
        f"{turn['role']}: {turn['content']}" for turn in recent_history
    )

    prompt = f"""You are an intelligent assistant routing queries for a financial RAG system.

Current State:
{json.dumps(state)}

Recent Conversation History:
{history_text if history_text else "No prior history"}

Latest User Query:
"{query}"

Analyze the intent of the latest user query:
1. "rag_search": Asking for specific financial numbers, document details, reports, metrics, or follow-ups requiring financial document retrieval.
2. "chat_summary": Asking to summarize, recap, or review the discussion/chat history so far.
3. "general_chat": Greetings, thank yous, or meta-questions about the assistant.

Rules for "rag_search":
- Update company, fiscal_year, and target_metric only if explicitly mentioned/changed.
- Rewrite the query into a fully qualified standalone financial document search query.

Return ONLY a valid JSON object matching this schema:
{{
  "intent": "rag_search" | "chat_summary" | "general_chat",
  "state": {{
    "company": str or null,
    "fiscal_year": str or null,
    "target_metric": str or null
  }},
  "search_query": "standalone search query if rag_search, else empty string"
}}"""

    try:
        response = groq.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            response_format={"type": "json_object"}
        )

        data = json.loads(response.choices[0].message.content)
        intent = data.get("intent", "rag_search")
        updated_state = data.get("state", state)

        # Ensure valid state format
        updated_state = {
            k: updated_state.get(k, state.get(k)) for k in EMPTY_STATE
        }
        search_query = data.get("search_query", "").strip() or query

        return intent, search_query, updated_state

    except Exception as err:
        print(f"[Routing Warning] {err}")
        return "rag_search", query, state.copy()


# --------------------------------------------------
# 2. PARALLEL HYBRID RETRIEVAL + RRF
# --------------------------------------------------

def hybrid_search_parallel(query: str, top_k: int = FINAL_TOP_K):
    """
    Runs vector and BM25 retrieval concurrently for reduced latency.
    """
    with ThreadPoolExecutor(max_workers=2) as executor:
        future_vector = executor.submit(retrieve_vector, query, top_k=RETRIEVAL_K)
        future_bm25 = executor.submit(retrieve_bm25, query, top_k=RETRIEVAL_K)

        vector_results = future_vector.result() or []
        bm25_results = future_bm25.result() or []

    fused = {}
    for results in (vector_results, bm25_results):
        seen_in_list = set()
        for rank, item in enumerate(results, start=1):
            chunk_id = item.get("chunk_id") or item.get("id")
            if not chunk_id or chunk_id in seen_in_list:
                continue

            seen_in_list.add(chunk_id)

            if chunk_id not in fused:
                fused[chunk_id] = {**item, "rrf_score": 0.0}

            fused[chunk_id]["rrf_score"] += 1.0 / (RRF_K + rank)

    ranked = sorted(fused.values(), key=lambda x: x["rrf_score"], reverse=True)
    return ranked[:top_k]


# --------------------------------------------------
# 3. LEXICAL RERANKING
# --------------------------------------------------

def tokenize(text: str) -> set:
    return set(re.findall(r"[a-zA-Z0-9]+", text.lower()))


def rerank_results(query: str, results: list) -> list:
    query_terms = tokenize(query)
    if not query_terms:
        return results

    for item in results:
        content = item.get("page_content") or item.get("content") or ""
        content_terms = tokenize(content)

        if not content_terms:
            item["final_score"] = item["rrf_score"]
            continue

        overlap = len(query_terms & content_terms)
        lexical_score = overlap / len(query_terms)
        item["final_score"] = item["rrf_score"] + (0.01 * lexical_score)

    return sorted(results, key=lambda x: x["final_score"], reverse=True)


# --------------------------------------------------
# 4. CONTEXT & CITATION MANAGEMENT
# --------------------------------------------------

def build_context(results: list):
    context_parts = []
    source_map = {}

    for i, item in enumerate(results, start=1):
        source_id = f"S{i}"
        metadata = item.get("metadata") or {}
        document = item.get("document_id") or item.get("doc_name") or "Form 10-K"
        page = metadata.get("page_number") or item.get("page") or "Unknown"
        section = metadata.get("section_title") or metadata.get("header") or "Financial Statements"
        content = (item.get("page_content") or item.get("content") or "").strip()

        if not content:
            continue

        source_map[source_id] = f"{document}, Page {page}, \"{section}\""
        context_parts.append(
            f"[{source_id}]\nDocument: {document}\nPage: {page}\nSection: {section}\nContent:\n{content}"
        )

    return "\n\n---\n\n".join(context_parts), source_map


def validate_citations(answer: str, source_map: dict) -> str:
    citation_pattern = r"\[(S\d+)\]"

    def replace_citation(match):
        source_id = match.group(1)
        if source_id not in source_map:
            return ""
        return f"[{source_id}: {source_map[source_id]}]"

    return re.sub(citation_pattern, replace_citation, answer)


# --------------------------------------------------
# 5. SPECIALIZED HANDLERS
# --------------------------------------------------
def handle_chat_summary(history: list) -> str:
    if not history:
        return "We haven't discussed anything yet!"

    # Keep only user queries + truncated assistant responses (first 100 chars)
    condensed_history = []
    for msg in history:
        if msg["role"] == "user":
            condensed_history.append(f"User Question: {msg['content']}")
        else:
            # Strip source metadata / citations and keep just the key takeaway prefix
            short_ans = msg["content"].split("\n")[0][:120]
            condensed_history.append(f"Assistant Answered: {short_ans}...")

    formatted_history = "\n".join(condensed_history)

    prompt = f"""Summarize the topics and financial questions discussed in this conversation session:

{formatted_history}

Provide a clean, bulleted executive summary of what was asked and discussed."""

    response = groq.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3
    )
    return response.choices[0].message.content.strip()


def generate_answer(query: str, history: list, session_state: dict):
    """Main routing & answer generation flow."""
    # Step 1: Classify intent & rewrite query
    intent, search_query, updated_state = classify_and_prepare(query, history, session_state)

    print(f"\n[Detected Intent] {intent}")

    # Step 2: Handle non-RAG intents directly
    if intent == "chat_summary":
        summary = handle_chat_summary(history)
        return summary, updated_state

    if intent == "general_chat":
        messages = [
            {"role": "system", "content": "You are a helpful financial assistant. Respond politely to conversational messages."},
            *history[-4:],
            {"role": "user", "content": query}
        ]
        response = groq.chat.completions.create(model=MODEL, messages=messages, temperature=0.7)
        return response.choices[0].message.content.strip(), updated_state

    # Step 3: Execute RAG Pipeline for financial search
    print(f"[Search Query] {search_query}")
    print(f"[Session State] {updated_state}")

    results = hybrid_search_parallel(search_query, top_k=FINAL_TOP_K)
    results = rerank_results(search_query, results)

    context, source_map = build_context(results)

    if not context.strip():
        return "I could not find sufficient matching financial records in the repository.", updated_state

    system_prompt = f"""You are a financial document question-answering assistant.
Answer the question using ONLY the verified document context below.

RULES:
1. Never invent financial facts or figures.
2. Cite sources like [S1], [S2] for every claim.
3. Use Markdown tables for comparative numeric data.

VERIFIED CONTEXT:
{context}"""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": search_query}
    ]

    response = groq.chat.completions.create(
        model=MODEL,
        messages=messages,
        temperature=0
    )

    answer = response.choices[0].message.content.strip()
    answer = validate_citations(answer, source_map)

    return answer, updated_state


# --------------------------------------------------
# 6. TERMINAL CHAT INTERFACE
# --------------------------------------------------

if __name__ == "__main__":
    session_history = []
    session_state = EMPTY_STATE.copy()

    print("=" * 55)
    print("Financial Hybrid RAG Chatbot (Interactive)")
    print("Commands: clear | exit")
    print("=" * 55)

    try:
        while True:
            user_input = input("\nYou: ").strip()

            if user_input.lower() == "exit":
                break

            if user_input.lower() == "clear":
                session_history.clear()
                session_state = EMPTY_STATE.copy()
                print("Conversation reset.")
                continue

            if not user_input:
                continue

            try:
                ai_response, session_state = generate_answer(
                    user_input,
                    session_history,
                    session_state
                )

                print(f"\nAssistant: {ai_response}")

                # Store full history for proper chat summarization
                session_history.append({"role": "user", "content": user_input})
                session_history.append({"role": "assistant", "content": ai_response})

            except Exception as err:
                print(f"\nPipeline Error: {err}")

    finally:
        if client is not None and hasattr(client, "close"):
            client.close()
            print("Database connection closed.")