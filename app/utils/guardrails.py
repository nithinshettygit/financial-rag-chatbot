# app/utils/guardrails.py
"""
Thin async wrapper around NeMo Guardrails.

We use NeMo purely as a policy layer (input rails + output rails) via `check_async`,
NOT as the thing that generates answers - the RAG pipeline stays in charge of retrieval/generation.
Requires a recent nemoguardrails (check_async is documented from ~0.21) and langchain-groq.
"""
from __future__ import annotations

import asyncio
import threading
from dataclasses import dataclass
from typing import Optional

import logfire

from app.config import settings
from app.utils import guard_rules as rules

_REFUSAL = "I can only help with questions about the indexed financial filings, and I can't help with that request."
REFUSAL_SCOPE = (
    "I can only help with questions about the indexed financial filings, such as revenue, margins, expenses, "
    "risk factors, and year-over-year changes. I can't perform actions like payments or transfers, or help with "
    "unrelated requests."
)
_REFUSAL_FIGURES = (
    "I couldn't verify the figures in my draft answer against the source documents, so I won't present them. "
    "Please try rephrasing or ask for a specific metric and year."
)

_rails = None
_lock = threading.Lock()


@dataclass
class GuardResult:
    allowed: bool
    text: str                       # possibly modified (e.g. PII-masked) text
    rail: Optional[str] = None      # name of the rail that blocked
    message: Optional[str] = None   # user-facing refusal
    degraded: bool = False          # True when NeMo failed and only deterministic rules ran


def _build_rails():
    from langchain_groq import ChatGroq
    from nemoguardrails import LLMRails, RailsConfig

    config = RailsConfig.from_path(settings.GUARDRAILS_PATH)
    llm = ChatGroq(model=settings.GUARDRAIL_LLM_MODEL, api_key=settings.GROQ_API_KEY, temperature=0)
    return LLMRails(config, llm=llm)


def get_rails():
    global _rails
    with _lock:
        if _rails is None:
            _rails = _build_rails()
    return _rails


def warmup_guardrails() -> None:
    if settings.GUARDRAILS_ENABLED:
        get_rails()


# --------------------------------------------------------------------------- #
# Deterministic fallback (used only if NeMo errors / times out)
# --------------------------------------------------------------------------- #
def _fallback_input(text: str) -> GuardResult:
    verdict = rules.detect_injection(text)
    if verdict != "ok":
        return GuardResult(False, text, rail="check input heuristics", message=_REFUSAL, degraded=True)
    return GuardResult(True, rules.mask_pii(text), degraded=True)


def _fallback_output(answer: str, evidence: str) -> GuardResult:
    if rules.leaks_prompt(answer):
        return GuardResult(False, answer, rail="check prompt leak", message=_REFUSAL, degraded=True)
    if settings.GUARDRAILS_NUMERIC_CHECK and evidence:
        ratio, _ = rules.ungrounded_ratio(answer, evidence)
        if ratio > 0.5:
            return GuardResult(False, answer, rail="check numeric grounding", message=_REFUSAL_FIGURES, degraded=True)
    return GuardResult(True, rules.mask_pii(answer), degraded=True)


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #
async def check_input(text: str) -> GuardResult:
    if not settings.GUARDRAILS_ENABLED:
        return GuardResult(True, text)
    from nemoguardrails.rails.llm.options import RailStatus, RailType

    try:
        res = await asyncio.wait_for(
            get_rails().check_async([{"role": "user", "content": text}], rail_types=[RailType.INPUT]),
            timeout=settings.GUARDRAILS_TIMEOUT_S,
        )
    except Exception as e:  # NeMo error, Groq outage, timeout...
        logfire.error("Input guardrails failed: {err}", err=repr(e))
        return _fallback_input(text) if settings.GUARDRAILS_FAIL_OPEN else GuardResult(
            False, text, rail="guardrail_error", message="I'm temporarily unable to process requests. Please try again.")

    if res.status == RailStatus.BLOCKED:
        logfire.warn("Input blocked by {rail}", rail=res.rail)
        # Prefer the bot message chosen by the Colang flow (rails.co); fall back to the default.
        msg = res.content if res.content and res.content != text else _REFUSAL
        return GuardResult(False, text, rail=res.rail, message=msg)
    return GuardResult(True, res.content if res.status == RailStatus.MODIFIED and res.content else text)


async def check_output(query: str, answer: str, evidence: str) -> GuardResult:
    if not settings.GUARDRAILS_ENABLED:
        return GuardResult(True, answer)
    from nemoguardrails.rails.llm.options import RailStatus, RailType

    messages = [
        {"role": "context", "content": {"relevant_chunks": evidence}},
        {"role": "user", "content": query},
        {"role": "assistant", "content": answer},
    ]
    try:
        res = await asyncio.wait_for(
            get_rails().check_async(messages, rail_types=[RailType.OUTPUT]),
            timeout=settings.GUARDRAILS_TIMEOUT_S,
        )
    except Exception as e:
        logfire.error("Output guardrails failed: {err}", err=repr(e))
        return _fallback_output(answer, evidence) if settings.GUARDRAILS_FAIL_OPEN else GuardResult(
            False, answer, rail="guardrail_error", message="I'm temporarily unable to verify this answer. Please try again.")

    if res.status == RailStatus.BLOCKED:
        logfire.warn("Output blocked by {rail}", rail=res.rail)
        default = _REFUSAL_FIGURES if "numeric" in (res.rail or "") else _REFUSAL
        msg = res.content if res.content and res.content != answer else default
        return GuardResult(False, answer, rail=res.rail, message=msg)
    return GuardResult(True, res.content if res.status == RailStatus.MODIFIED and res.content else answer)
