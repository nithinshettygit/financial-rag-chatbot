# app/utils/router.py
import asyncio
import json
import re
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from app.config import settings
from app.schemas import ChatTurn, SessionState
from app.utils.llm import chat_with_retry, get_groq_clients
from app.utils.logger import logger

client = get_groq_clients(timeout=20.0)  # on failure we fall back to heuristics

_YEAR_RE = re.compile(r"\b(201\d|202\d|203\d)\b")
_SUFFIXES = {"inc", "incorporated", "corp", "corporation", "company", "co", "ltd", "limited", "plc"}
MAX_SUB_QUERIES = 12

FOLLOWUP_TYPES = {"none", "reference", "elaboration"}
DEFAULT_ASPECTS = [
    "breakdown by segment, product or region",
    "key drivers and management commentary",
    "year-over-year change",
]

# DEGRADED-MODE SAFETY NET ONLY (used when the embedding router is down); the data-driven router is primary.
# "tell me more", "elaborate", "why?", ... -> same topic, wants NEW detail
_ELAB_RE = re.compile(
    r"^\s*(?:and\s+|ok(?:ay)?,?\s+)?(?:please\s+|can you\s+|could you\s+)?"
    r"(?:tell me more|(?:give|show|provide)(?:\s+me)?\s+more|more (?:details?|info\w*)|elaborate|"
    r"expand(?:\s+on)?|go (?:on|deeper)|continue|explain (?:that|it|this|more|further)|"
    r"what else|how so|why\??\s*$)",
    re.I,
)
# pronouns / ellipsis -> needs reference resolution
_REF_RE = re.compile(
    r"\b(?:it|its|that|this|those|these|them|they|their|he|she)\b|^\s*(?:and|also|what about|how about|same for)\b",
    re.I,
)


class QueryFilters(BaseModel):
    years: Optional[List[int]] = None
    company: Optional[str] = None
    document_ids: Optional[List[str]] = None


class SubQuery(BaseModel):
    text: str
    entity: Optional[str] = None  # None = no specific company


class RoutePlan(BaseModel):
    intent: str
    search_query: str                      # standalone question (fully self-contained)
    state: SessionState
    entities: List[str] = Field(default_factory=list)
    years: List[int] = Field(default_factory=list)
    sub_queries: List[SubQuery] = Field(default_factory=list)
    metrics: List[str] = Field(default_factory=list)
    followup_type: str = "none"            # none | reference | elaboration


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def company_key(text: str) -> str:
    terms = re.findall(r"[a-z0-9]+", str(text or "").lower())
    while terms and terms[-1] in _SUFFIXES:
        terms.pop()
    return " ".join(terms)


def detect_known_entities(query: str, known_companies: List[str]) -> List[str]:
    """Deterministic entity detection against the indexed company list (LLM-independent)."""
    q_tokens = [re.sub(r"\'?s$", "", t) for t in re.findall(r"[a-z0-9]+", str(query or "").lower())]
    q = f" {' '.join(q_tokens)} "
    found = []
    for name in known_companies:
        key = company_key(name)
        if key:
            key_stem = re.sub(r"\'?s$", "", key)
            if f" {key} " in q or f" {key_stem} " in q:
                found.append(name)
    return found


def _canonical_entity(entity: str, known: List[str]) -> str:
    key = company_key(entity)
    for name in known:
        k = company_key(name)
        if k and (k == key or k in key or key in k):
            return name
    return entity.strip()


def _dedupe(items: List[str]) -> List[str]:
    seen, out = set(), []
    for item in items:
        k = company_key(item)
        if k and k not in seen:
            seen.add(k)
            out.append(item)
    return out


def _fallback_metrics(query: str, entities: List[str]) -> List[str]:
    """Regex decomposition used only when the LLM router is unavailable."""
    text = re.sub(r"^\s*(?:what\s+(?:were|was|is|are)|show|give me|tell me|compare|list)\s+", "", query.replace("?", " "), flags=re.I)
    text = _YEAR_RE.sub(" ", text)
    for e in entities:
        text = re.sub(re.escape(e) + r"(?:'s)?", " ", text, flags=re.I)
    parts = []
    for p in re.split(r",|&|\band\b", text):
        p = re.sub(r"^\s*(?:the|in|for|of|their|its)\s+", "", p.strip(), flags=re.I).strip()
        if len(p) > 2:
            parts.append(p)
    return parts or [query]


def build_filters(
    years: Optional[List[int]], company: Optional[str], document_ids: Optional[List[str]]
) -> QueryFilters:
    return QueryFilters(years=years or None, company=company, document_ids=document_ids or None)


def classify_followup(query: str, history: List[ChatTurn], llm_type: Optional[str],
                      hint: Optional[str] = None, semantic_available: bool = False) -> str:
    """Follow-up type. Priority: embedding-router hint > LLM verdict > (degraded mode only) regex.
    Never a follow-up without history."""
    if not history:
        return "none"
    if hint in ("elaboration", "reference"):
        # Embeddings are good at "this is a follow-up" but can confuse the two types; the LLM sees the
        # conversation, so when both say follow-up we trust the LLM's type.
        return llm_type if llm_type in ("elaboration", "reference") else hint
    if not semantic_available:  # regexes are only a safety net when the embedding router is down
        if _ELAB_RE.search(query):
            return "elaboration"
        if llm_type in (None, "none") and len(query.split()) <= 6 and _REF_RE.search(query):
            return "reference"
    return llm_type if llm_type in FOLLOWUP_TYPES else "none"


def _last_user_question(history: List[ChatTurn]) -> str:
    for t in reversed(history or []):
        if t.role == "user":
            return (t.content or "").strip()
    return ""


# --------------------------------------------------------------------------- #
# LLM call (sync; executed in a worker thread)
# --------------------------------------------------------------------------- #
def _call_router_llm(query: str, history: List[ChatTurn], state: SessionState, known: List[str]) -> Dict[str, Any]:
    recent = history[-min(settings.MAX_HISTORY_TURNS, 4):] if history else []
    history_text = "\n".join(
        f"{t.role}: {t.content if t.role == 'user' else t.content[:600]}" for t in recent
    )
    prompt = f"""You are a query router for an enterprise multi-document financial RAG system.

Indexed companies: {json.dumps(known)}
Current Session State: {json.dumps(state.model_dump())}
Recent Conversation:
{history_text or "No prior history"}

Latest User Query: "{query}"

Tasks:
1. intent:
   - "rag_search": questions about companies' financial filings (data, metrics, history, risks, comparisons).
   - "chat_summary": the user asks to summarize/recap our conversation.
   - "general_chat": ONLY greetings, thanks, and questions about what this assistant can do.
   - "out_of_scope": EVERYTHING else - requests to perform actions or transactions (payments, transfers,
     purchases, bookings), unrelated tasks (coding, writing, general knowledge, math), personal or investment
     advice, or anything that is not about the indexed filings. When in doubt between "out_of_scope" and
     "rag_search", choose "rag_search" only if the message is plausibly about a company's filings or a follow-up to one.
2. followup_type:
   - "none": the query is fully self-contained.
   - "reference": the query is a NEW question that omits the company/year/metric and relies on the conversation
     (e.g. "what about 2025?", "and for Duolingo?", "what was its net income?").
   - "elaboration": the user wants MORE DETAIL on the topic already answered
     (e.g. "tell me more", "elaborate", "why?", "explain that").
3. entities: ALL companies the latest query is about (resolve pronouns from the conversation/state).
4. metrics: distinct topics asked about (short noun phrases). For "elaboration", the topic being elaborated.
5. fiscal_years: list of years requested (resolve from conversation/state when omitted).
6. search_query: a fully STANDALONE question that makes sense with no conversation.
   - Keep EVERY distinguishing qualifier from the user's wording (e.g. "sources", "details", "breakdown", "growth").
   - "reference": inline the resolved company, year and metric.
   - "elaboration": restate the topic and ask for additional detail not already covered.
7. aspects: ONLY for "elaboration": 2-4 distinct angles NOT yet covered in the previous assistant answer
   (e.g. "breakdown by segment", "regional split", "key drivers", "year-over-year change"). Otherwise [].
8. state: company, fiscal_year, target_metric, document_id.

Return ONLY JSON:
{{"intent": "rag_search"|"chat_summary"|"general_chat"|"out_of_scope",
  "followup_type": "none"|"reference"|"elaboration",
  "entities": [str], "metrics": [str], "fiscal_years": [int],
  "search_query": str, "aspects": [str],
  "state": {{"company": str|null, "fiscal_year": str|null, "target_metric": str|null, "document_id": str|null}}}}"""
    try:
        resp = chat_with_retry(
            client,
            model=settings.LLM_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            response_format={"type": "json_object"},
        )
        data = json.loads(resp.choices[0].message.content)
        return data if isinstance(data, dict) else {}
    except Exception as e:
        logger.error(f"Router LLM failed: {e}")
        return {}


async def plan_query(
    query: str,
    history: List[ChatTurn],
    state: SessionState,
    known_companies: Optional[List[str]] = None,
    semantic: Optional[str] = None,
    semantic_available: bool = False,
) -> RoutePlan:
    known = known_companies or []
    history = history or []

    data = await asyncio.to_thread(_call_router_llm, query, history, state, known)
    intent = data.get("intent", "rag_search")
    if intent not in {"rag_search", "chat_summary", "general_chat", "out_of_scope", "document_inventory"}:
        intent = "rag_search"

    if semantic == "summary" and intent in {"rag_search", "general_chat"}:
        intent = "chat_summary"   # embedding router is confident; it overrides a weaker LLM verdict
    followup_type = (
        classify_followup(query, history, data.get("followup_type"), semantic, semantic_available)
        if intent == "rag_search" else "none"
    )

    llm_state = data.get("state") or {}
    search_query = (data.get("search_query") or "").strip()

    # ---- entities (LLM + deterministic detection) ----
    raw_entities = [str(e) for e in (data.get("entities") or []) if str(e).strip()]
    entities = _dedupe(
        [_canonical_entity(e, known) for e in raw_entities] + detect_known_entities(query, known)
    )

    # ---- years ----
    years = {int(y) for y in _YEAR_RE.findall(query)}
    for y in data.get("fiscal_years") or []:
        try:
            if 1990 <= int(y) <= 2100:
                years.add(int(y))
        except (TypeError, ValueError):
            pass
    years_list = sorted(years)
    if not years_list and state.fiscal_year:
        try:
            years_list = [int(state.fiscal_year)]
        except (TypeError, ValueError):
            pass

    # ---- session state: multiple entities UNLOCK the single-company lock ----
    if len(entities) > 1:
        company, document_id = None, None
    elif len(entities) == 1:
        company = entities[0]
        same_company = company_key(state.company or "") == company_key(company)
        document_id = llm_state.get("document_id") or (state.document_id if same_company else None)
    else:
        company = llm_state.get("company") or state.company
        document_id = llm_state.get("document_id") or state.document_id

    explicit_year = len(years) == 1 and bool(_YEAR_RE.search(query))
    new_state = SessionState(
        company=company,
        fiscal_year=str(years_list[0]) if (explicit_year and years_list) else (llm_state.get("fiscal_year") or state.fiscal_year),
        target_metric=llm_state.get("target_metric") or state.target_metric,
        document_id=document_id,
    )

    # Entities used for retrieval: explicit ones, else the (carried-over) state company.
    effective_entities = entities or ([company] if company else [])

    # ---- metrics ----
    metrics = [str(m).strip() for m in (data.get("metrics") or []) if str(m).strip()]
    if not metrics:
        if followup_type != "none" and state.target_metric:
            metrics = [state.target_metric]
        else:
            metrics = _fallback_metrics(query, effective_entities)

    # ---- standalone query: LLM rewrite + deterministic enforcement ----
    if not search_query:
        if followup_type == "elaboration":
            search_query = f"{_last_user_question(history) or query} additional details"
        elif followup_type == "reference":
            search_query = " ".join(p for p in (state.company, query) if p)
        else:
            search_query = query
    if followup_type != "none":
        for e in effective_entities:
            if company_key(e) not in company_key(search_query):
                search_query = f"{e} {search_query}"
        if years_list and not any(str(y) in search_query for y in years_list):
            search_query = f"{search_query} {' '.join(str(y) for y in years_list)}"

    # ---- sub-queries ----
    year_str = " ".join(str(y) for y in years_list)
    sub_queries: List[SubQuery] = []
    if followup_type == "elaboration":
        # Retrieve DIFFERENT angles than the first answer used, otherwise we re-fetch the same chunks.
        aspects = [str(a).strip() for a in (data.get("aspects") or []) if str(a).strip()] or DEFAULT_ASPECTS
        topic = " ".join(metrics)
        for entity in effective_entities or [None]:
            for aspect in aspects[:4]:
                text = " ".join(p for p in (entity, topic, aspect, year_str) if p)
                sub_queries.append(SubQuery(text=text, entity=entity))
    else:
        for entity in effective_entities or [None]:
            for metric in metrics:
                text = " ".join(p for p in (entity, metric, year_str) if p)
                sub_queries.append(SubQuery(text=text, entity=entity))

    if not sub_queries:
        sub_queries = [SubQuery(text=search_query, entity=effective_entities[0] if effective_entities else None)]
    sub_queries = sub_queries[:MAX_SUB_QUERIES]

    return RoutePlan(
        intent=intent,
        search_query=search_query,
        state=new_state,
        entities=effective_entities,
        years=years_list,
        sub_queries=sub_queries,
        metrics=metrics,
        followup_type=followup_type,
    )
