import json
import re
from typing import List, Tuple
from groq import Groq
from app.config import settings
from app.schemas import ChatTurn, SessionState
from app.utils.logger import logger

client = Groq(api_key=settings.GROQ_API_KEY)


def is_document_inventory_query(query: str) -> bool:
    normalized_query = query.lower()
    mentions_documents = re.search(
        r"\b(documents?|files?|reports?|(?:data\s+)?sources?|datasets?|knowledge\s+bases?)\b",
        normalized_query,
    )
    asks_for_inventory = re.search(
        r"\b(list|show|display|available|indexed|uploaded)\b", normalized_query
    ) or re.search(
        r"\b(what|which)\b.{0,60}\b(documents?|files?|reports?|sources?|datasets?)\b",
        normalized_query,
    )
    return bool(mentions_documents and asks_for_inventory)


def is_simple_greeting(query: str) -> bool:
    return bool(
        re.fullmatch(
            r"\s*(?:hi|hello|hey|good morning|good afternoon|good evening|"
            r"thanks?(?: a lot| so much)?|thank you(?: so much)?|bye|goodbye)"
            r"(?: there)?[.!?]*\s*",
            query.lower(),
        )
    )


def classify_and_prepare(
    query: str, history: List[ChatTurn], state: SessionState
) -> Tuple[str, str, SessionState]:
    if is_document_inventory_query(query):
        return "document_inventory", query, state
    if is_simple_greeting(query):
        return "general_chat", query, state

    recent_history = history[-min(settings.MAX_HISTORY_TURNS, 4) :]
    history_text = "\n".join(
        f"{turn.role}: {turn.content if turn.role == 'user' else turn.content[:240]}"
        for turn in recent_history
    )

    prompt = f"""You are an intelligent query router and search query rewriter for an enterprise multi-document financial RAG system.

Current Session State:
{json.dumps(state.model_dump())}

Recent Conversation History:
{history_text if history_text else "No prior history"}

Latest User Query:
"{query}"

Tasks:
1. Classify Query Intent:
    - "rag_search": Every factual question or request about financial, company, or business information that must be answered from indexed documents.
   - "chat_summary": Requests to summarize, recap, or review past discussion.
    - "general_chat": Only a standalone greeting or thank-you. Do not classify factual questions as general chat.

2. State Tracking:
   - Extract/update fields: `company`, `fiscal_year`, `target_metric`, and `document_id` if explicitly stated or inferred from context.
   - Preserve existing state attributes if not explicitly overridden by the latest user query.

3. Rewrite Search Query (for "rag_search"):
   - Formulate a clear, self-contained, standalone search query.
   - Include company, document name, fiscal year, and target metrics when present to maximize vector and BM25 retrieval accuracy across multiple PDF documents.
   - Do NOT invent metrics or years not requested or present in state context.

Return ONLY a valid JSON object matching this schema:
{{
  "intent": "rag_search" | "chat_summary" | "general_chat",
  "state": {{
    "company": str or null,
    "fiscal_year": str or null,
    "target_metric": str or null,
    "document_id": str or null
  }},
  "search_query": "standalone search query"
}}"""

    try:
        response = client.chat.completions.create(
            model=settings.LLM_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            response_format={"type": "json_object"},
        )

        data = json.loads(response.choices[0].message.content)
        intent = data.get("intent", "rag_search")
        if intent not in {"rag_search", "chat_summary"}:
            intent = "rag_search"
        updated_state_dict = data.get("state", {})

        # Merge extracted state with current state
        updated_state = SessionState(
            company=updated_state_dict.get("company") or state.company,
            fiscal_year=updated_state_dict.get("fiscal_year") or state.fiscal_year,
            target_metric=updated_state_dict.get("target_metric") or state.target_metric,
        )
        if hasattr(updated_state, "document_id"):
            setattr(
                updated_state,
                "document_id",
                updated_state_dict.get("document_id") or getattr(state, "document_id", None),
            )

        search_query = data.get("search_query", "").strip() or query
        return intent, search_query, updated_state

    except Exception as e:
        logger.error(f"Error in router classification: {e}")
        return "rag_search", query, state