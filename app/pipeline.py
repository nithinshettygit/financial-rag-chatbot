import re
from concurrent.futures import ThreadPoolExecutor
from groq import Groq
from app.config import settings
from app.schemas import ChatRequest, ChatResponse, ContextChunk, SessionState
from app.utils.router import classify_and_prepare
from app.utils.vector import QdrantVectorRetriever
from app.utils.bm25 import BM25Retriever
from app.utils.logger import logger
from app.utils.tokens import count_tokens
from typing import List, Dict, Any, Tuple

groq_client = Groq(api_key=settings.GROQ_API_KEY)
vector_retriever = QdrantVectorRetriever()
bm25_retriever = BM25Retriever()

def hybrid_search_parallel(query: str, top_k: int = settings.FINAL_TOP_K) -> List[Dict[str, Any]]:
    with ThreadPoolExecutor(max_workers=2) as executor:
        future_vector = executor.submit(vector_retriever.retrieve, query, top_k=settings.RETRIEVAL_K)
        future_bm25 = executor.submit(bm25_retriever.retrieve, query, top_k=settings.RETRIEVAL_K)

        vector_results = future_vector.result() or []
        bm25_results = future_bm25.result() or []

    fused = {}
    for results in (vector_results, bm25_results):
        seen_in_list = set()
        for rank, item in enumerate(results, start=1):
            chunk_id = item.get("chunk_id")
            if not chunk_id or chunk_id in seen_in_list:
                continue

            seen_in_list.add(chunk_id)

            if chunk_id not in fused:
                fused[chunk_id] = {**item, "rrf_score": 0.0}

            fused[chunk_id]["rrf_score"] += 1.0 / (settings.RRF_K + rank)

    ranked = sorted(fused.values(), key=lambda x: x["rrf_score"], reverse=True)
    return ranked[:top_k]

def tokenize_set(text: str) -> set:
    return set(re.findall(r"[a-zA-Z0-9]+", text.lower()))

def rerank_results(query: str, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    query_terms = tokenize_set(query)
    if not query_terms:
        for item in results:
            item["final_score"] = item.get("rrf_score", 0.0)
        return results

    for item in results:
        content = item.get("content") or ""
        content_terms = tokenize_set(content)

        if not content_terms:
            item["final_score"] = item.get("rrf_score", 0.0)
            continue

        overlap = len(query_terms & content_terms)
        lexical_score = overlap / len(query_terms)

        is_table = item.get("chunk_type") == "table"
        table_boost = 0.005 if is_table else 0.0

        item["final_score"] = item.get("rrf_score", 0.0) + (0.01 * lexical_score) + table_boost

    return sorted(results, key=lambda x: x["final_score"], reverse=True)

def build_context(
    results: List[Dict[str, Any]],
    max_tokens: int = settings.MAX_CONTEXT_TOKENS
) -> Tuple[str, Dict[str, str], List[Dict[str, Any]]]:
    context_parts = []
    source_map = {}
    selected_results = []

    for item in results:
        doc_id = item.get("document_id", "Duolingo 10-K")
        page = item.get("page_number", "N/A")
        chunk_type = item.get("chunk_type", "text")
        headings = " > ".join(item.get("headings", [])) if isinstance(item.get("headings"), list) else "Main Section"
        content = item.get("content", "").strip()

        if not content:
            continue

        source_id = f"S{len(selected_results) + 1}"
        context_part = (
            f"[{source_id}]\nDocument: {doc_id} | Page: {page} | Type: {chunk_type}\n"
            f"Heading Hierarchy: {headings}\nContent:\n{content}"
        )
        candidate_context = "\n\n---\n\n".join(context_parts + [context_part])
        if count_tokens(candidate_context) > max_tokens:
            continue

        source_map[source_id] = f"{doc_id}, Page {page}, Header: \"{headings}\""
        context_parts.append(context_part)
        selected_results.append(item)

    return "\n\n---\n\n".join(context_parts), source_map, selected_results

def validate_citations(answer: str, source_map: Dict[str, str]) -> str:
    citation_pattern = r"\[(S\d+)\]"

    def replace_citation(match):
        source_id = match.group(1)
        if source_id not in source_map:
            return ""
        return f"[{source_id}: {source_map[source_id]}]"

    return re.sub(citation_pattern, replace_citation, answer)

def handle_chat_summary(history: List) -> str:
    if not history:
        return "We haven't discussed anything yet!"

    condensed_history = []
    for msg in history:
        if msg.role == "user":
            condensed_history.append(f"User Question: {msg.content}")
        else:
            short_ans = msg.content.split("\n")[0][:120]
            condensed_history.append(f"Assistant Answered: {short_ans}...")

    formatted_history = "\n".join(condensed_history)

    prompt = f"""You are the AI assistant in this conversation. Review what you and the user have discussed so far:
{formatted_history}
Provide a short, 2-3 sentence conversational summary addressing the user directly. 
Speak from your own perspective (use "You asked about..." and "I provided..."). 
Keep it warm, natural, and friendly. Do not use bullets, headers, or a QA format."""

    response = groq_client.chat.completions.create(
        model=settings.LLM_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3
    )
    return response.choices[0].message.content.strip()

def run_rag_pipeline(request: ChatRequest) -> ChatResponse:
    intent, search_query, updated_state = classify_and_prepare(
        request.query, request.history, request.state
    )
    
    logger.info(f"Session {request.session_id} - Intent: {intent}")

    if intent == "chat_summary":
        summary = handle_chat_summary(request.history)
        return ChatResponse(
            session_id=request.session_id,
            answer=summary,
            intent=intent,
            search_query=search_query,
            state=updated_state,
            retrieved_chunks=[]
        )

    if intent == "general_chat":
        messages = [
            {"role": "system", "content": "You are a helpful financial assistant. Respond politely to conversational messages."},
            *[{"role": turn.role, "content": turn.content} for turn in request.history[-4:]],
            {"role": "user", "content": request.query}
        ]
        response = groq_client.chat.completions.create(
            model=settings.LLM_MODEL, 
            messages=messages, 
            temperature=0.7
        )
        return ChatResponse(
            session_id=request.session_id,
            answer=response.choices[0].message.content.strip(),
            intent=intent,
            search_query=search_query,
            state=updated_state,
            retrieved_chunks=[]
        )

    # Hybrid Search & Reranking
    results = hybrid_search_parallel(search_query, top_k=settings.FINAL_TOP_K)
    results = rerank_results(search_query, results)

    context, source_map, results = build_context(results)

    if not context.strip():
        return ChatResponse(
            session_id=request.session_id,
            answer="I could not find matching financial records in the repository.",
            intent=intent,
            search_query=search_query,
            state=updated_state,
            retrieved_chunks=[]
        )

    system_prompt = f"""You are a financial document question-answering assistant.
Answer the user query accurately using ONLY the verified Docling context below.

RULES:
1. Never invent financial facts, metrics, or numbers.
2. Direct citations are required: append [S1], [S2] for every financial figure quoted.
3. When requested for percentages or metrics across years (e.g., 2023), find the row and column in the table context and calculate the percentage (Subscription Revenue / Total Revenue * 100).

VERIFIED CONTEXT:
{context}"""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": search_query}
    ]

    response = groq_client.chat.completions.create(
        model=settings.LLM_MODEL,
        messages=messages,
        temperature=0
    )

    raw_answer = response.choices[0].message.content.strip()
    validated_answer = validate_citations(raw_answer, source_map)

    context_chunks = [
        ContextChunk(
            chunk_id=res["chunk_id"],
            content=res["content"],
            document_id=res["document_id"],
            page_number=res["page_number"],
            chunk_type=res["chunk_type"],
            headings=res["headings"],
            rrf_score=res.get("rrf_score", 0.0),
            final_score=res.get("final_score", 0.0)
        )
        for res in results
    ]

    return ChatResponse(
        session_id=request.session_id,
        answer=validated_answer,
        intent=intent,
        search_query=search_query,
        state=updated_state,
        retrieved_chunks=context_chunks,
        source_map=source_map
    )