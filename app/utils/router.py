# app/utils/router.py
import asyncio
import json
import re
from typing import Any, Dict, List, Optional

import logfire
from pydantic import BaseModel, Field

from app.config import settings
from app.schemas import ChatTurn, SessionState
from app.utils.llm import chat_with_retry, extract_completion_text, get_groq_clients
from app.utils.logger import logger
from app.utils.tokens import count_tokens, log_token_usage
from shared.prompts import render_prompt

client = get_groq_clients(timeout=20.0)  # on failure we fall back to heuristics

_YEAR_RE = re.compile(r"\b(201\d|202\d|203\d)\b")
_SUFFIXES = {"inc", "incorporated", "corp", "corporation", "company", "co", "ltd", "limited", "plc"}
MAX_SUB_QUERIES = settings.MAX_SUB_QUERIES

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


def validate_router_response(data: Any) -> Dict[str, Any]:
    if not isinstance(data, dict):
        return {}
    for key in ("intent", "followup_type", "search_query"):
        if key in data and (not isinstance(data[key], str) or len(data[key]) > 2000):
            return {}
    for key in ("entities", "metrics", "aspects"):
        value = data.get(key, [])
        if (
            not isinstance(value, list)
            or len(value) > 20
            or any(not isinstance(item, str) or len(item) > 300 for item in value)
        ):
            return {}
    years = data.get("fiscal_years", [])
    if (
        not isinstance(years, list)
        or len(years) > 20
        or any(type(year) is not int and (not isinstance(year, str) or len(year) > 20) for year in years)
    ):
        return {}
    state = data.get("state", {})
    state_limits = {"company": 200, "fiscal_year": 20, "target_metric": 200, "document_id": 200}
    if state is not None and (
        not isinstance(state, dict)
        or any(key not in state_limits for key in state)
        or any(
            value is not None
            and (not isinstance(value, str) or len(value) > state_limits[key])
            for key, value in state.items()
        )
    ):
        return {}
    return data


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
    prompt = render_prompt(
        "router.yaml",
        "query_router",
        known_companies=json.dumps(known),
        session_state=json.dumps(state.model_dump()),
        history=history_text or "No prior history",
        query=query,
    )
    with logfire.span("Query Router LLM") as router_span:
        estimated_input = count_tokens(prompt)
        router_span.set_attribute("input_tokens_estimate", estimated_input)
        try:
            resp = chat_with_retry(
                client,
                model=settings.LLM_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                response_format={"type": "json_object"},
            )
        except Exception as e:
            router_span.set_attribute("input_tokens", estimated_input)
            router_span.set_attribute("output_tokens", 0)
            router_span.set_attribute("total_tokens", estimated_input)
            router_span.set_attribute("llm_calls", 1)
            router_span.set_attribute("token_count_source", "estimate")
            logger.error("Router LLM failed: %s", type(e).__name__)
            return {}
        try:
            response_text = extract_completion_text(resp)
        except ValueError as e:
            logger.error("Router LLM returned a malformed completion: %s", type(e).__name__)
            return {}
        log_token_usage(router_span, resp, estimated_input, count_tokens(response_text))
        try:
            data = json.loads(response_text)
            validated = validate_router_response(data)
            if not validated:
                logger.error("Router LLM returned an invalid response shape")
            return validated
        except (TypeError, ValueError) as e:
            logger.error(f"Router LLM returned invalid JSON: {e}")
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
