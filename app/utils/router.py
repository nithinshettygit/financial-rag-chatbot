import json
from groq import Groq
from app.config import settings
from app.schemas import ChatTurn, SessionState
from typing import List, Tuple

client = Groq(api_key=settings.GROQ_API_KEY)

def classify_and_prepare(query: str, history: List[ChatTurn], state: SessionState) -> Tuple[str, str, SessionState]:
    recent_history = history[-min(settings.MAX_HISTORY_TURNS, 4):]
    history_text = "\n".join(
        f"{turn.role}: {turn.content if turn.role == 'user' else turn.content[:240]}"
        for turn in recent_history
    )

    prompt = f"""You are an intelligent assistant routing queries for a financial RAG system.

Current State:
{json.dumps(state.model_dump())}

Recent Conversation History:
{history_text if history_text else "No prior history"}

Latest User Query:
"{query}"

Analyze the intent of the latest user query:
1. "rag_search": Asking for financial numbers, document details, reports, metrics, or follow-ups requiring financial document retrieval.
2. "chat_summary": Asking to summarize, recap, or review the discussion/chat history so far.
3. "general_chat": Greetings, thank yous, or meta-questions about the assistant.

Rules for "rag_search":
- Update company, fiscal_year, and target_metric only if explicitly mentioned/changed.
- Rewrite the query into a fully qualified standalone financial document search query.
- For numeric or table questions, include the requested metric and fiscal year when present. Do not add years or metrics the user did not ask about.

Return ONLY a valid JSON object matching this schema:
{{
  "intent": "rag_search" | "chat_summary" | "general_chat",
  "state": {{
    "company": str or null,
    "fiscal_year": str or null,
    "target_metric": str or null
  }},
  "search_query": "standalone search query"
}}"""

    try:
        response = client.chat.completions.create(
            model=settings.LLM_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            response_format={"type": "json_object"}
        )

        data = json.loads(response.choices[0].message.content)
        intent = data.get("intent", "rag_search")
        updated_state_dict = data.get("state", state.model_dump())

        updated_state = SessionState(
            company=updated_state_dict.get("company", state.company),
            fiscal_year=updated_state_dict.get("fiscal_year", state.fiscal_year),
            target_metric=updated_state_dict.get("target_metric", state.target_metric),
        )
        search_query = data.get("search_query", "").strip() or query

        return intent, search_query, updated_state

    except Exception:
        return "rag_search", query, state