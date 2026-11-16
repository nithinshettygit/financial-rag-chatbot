"""LLM-as-judge for answer quality.

Design (follows common RAG-eval practice, e.g. RAGAS / G-Eval style rubrics):
  * correctness  = share of expected *atomic facts* the answer states correctly
  * faithfulness = share of the answer's claims that are supported by the retrieved context
  * relevancy    = does the answer address the question (0-2, normalised)
  * abstained / partially_abstained = explicit flags (replaces brittle keyword matching)
A judge failure is reported as `judge_failed=True` and is NOT scored as a 0, so rate limits or
JSON errors never masquerade as quality regressions.

Use a judge model different from the generator (EVAL_JUDGE_MODEL) to limit self-preference bias.
"""
from __future__ import annotations

import json
import logging
import os
import random
import re
import time
from typing import Any, Dict, List, Optional

from groq import Groq

from app.config import settings
from app.utils.llm import chat_with_retry

logger = logging.getLogger(__name__)

JUDGE_MODEL = os.getenv("EVAL_JUDGE_MODEL") or settings.LLM_MODEL
MAX_ATTEMPTS = int(os.getenv("EVAL_JUDGE_RETRIES", "3"))
MAX_CONTEXT_CHARS = int(os.getenv("EVAL_JUDGE_MAX_CONTEXT_CHARS", "12000"))

# Retries are handled here with backoff + jitter, so the SDK's own retries are disabled.
_client = Groq(api_key=settings.GROQ_API_KEY, timeout=60.0, max_retries=0)

SYSTEM_PROMPT = (
    "You are a strict, impartial evaluator of a financial question-answering system. "
    "Use ONLY the material provided. Do not use outside knowledge. Output valid JSON only."
)


def _build_prompt(
    question: str,
    expected_answer: str,
    facts: List[str],
    actual_answer: str,
    context: str,
    answerable: bool,
) -> str:
    facts_block = "\n".join(f"{i}. {f}" for i, f in enumerate(facts, start=1)) or "(none - question is unanswerable)"
    return f"""Evaluate the GENERATED ANSWER.

Definitions:
- facts[].covered: true only if the generated answer states that expected fact with matching values
  (rounding or unit differences such as "$531.1M" vs "531,100 thousand" are acceptable). Otherwise false.
- contradicts_expected: true if the answer states a value that conflicts with an expected fact.
- total_claims: number of distinct factual claims in the generated answer (0 if it only declines to answer).
- unsupported_claims: claims NOT supported by the RETRIEVED CONTEXT (a claim that is true but absent from
  the context still counts as unsupported). Quote each briefly.
- abstained: true if the answer as a whole says the information is unavailable / could not be found.
- partially_abstained: true if it answers some requested parts but says others are unavailable.
- relevancy: 0 = does not address the question, 1 = partially, 2 = directly answers.

The question is {"ANSWERABLE from the documents" if answerable else "UNANSWERABLE from the documents (correct behaviour is to decline)"}.

Return ONLY this JSON:
{{"facts": [{{"id": 1, "covered": true}}], "contradicts_expected": false, "total_claims": 0,
  "unsupported_claims": [], "abstained": false, "partially_abstained": false,
  "relevancy": 0, "reason": "one short sentence"}}

QUESTION:
{question}

EXPECTED ANSWER:
{expected_answer or "(n/a)"}

EXPECTED FACTS:
{facts_block}

GENERATED ANSWER:
{actual_answer}

RETRIEVED CONTEXT:
{context[:MAX_CONTEXT_CHARS]}
"""


def _parse_json(raw: str) -> Dict[str, Any]:
    raw = (raw or "").strip()
    raw = re.sub(r"^```(?:json)?|```$", "", raw, flags=re.MULTILINE).strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", raw, flags=re.DOTALL)
        if not m:
            raise
        return json.loads(m.group(0))


def _to_scores(data: Dict[str, Any], n_facts: int, answerable: bool) -> Dict[str, Any]:
    abstained = bool(data.get("abstained", False))
    partial = bool(data.get("partially_abstained", False)) and not abstained

    verdicts = [v for v in (data.get("facts") or []) if isinstance(v, dict)]
    covered = sum(1 for v in verdicts if v.get("covered") is True)
    covered = min(covered, n_facts)

    if not answerable:
        correctness: Optional[float] = None
    elif n_facts == 0:
        correctness = None
    else:
        correctness = covered / n_facts
        if data.get("contradicts_expected"):
            correctness *= 0.5

    unsupported = [str(c) for c in (data.get("unsupported_claims") or [])]
    total_claims = max(int(data.get("total_claims") or 0), len(unsupported))
    faithfulness = 1.0 if total_claims == 0 else max(0.0, 1.0 - len(unsupported) / total_claims)

    relevancy = max(0, min(2, int(data.get("relevancy", 0)))) / 2.0

    return {
        "correctness": correctness,
        "faithfulness": faithfulness,
        "relevancy": relevancy,
        "abstained": abstained,
        "partially_abstained": partial,
        "unsupported_claims": unsupported,
        "fact_verdicts": verdicts,
        "reason": str(data.get("reason", "")),
        "judge_model": JUDGE_MODEL,
        "judge_failed": False,
    }


def judge_answer(
    question: str,
    expected_answer: str,
    expected_facts: List[str],
    actual_answer: str,
    context: str,
    answerable: bool = True,
) -> Dict[str, Any]:
    """Blocking call - run it via asyncio.to_thread from async code."""
    prompt = _build_prompt(question, expected_answer, expected_facts, actual_answer, context, answerable)
    last_error: Optional[Exception] = None

    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            response = chat_with_retry(
                _client,
                model=JUDGE_MODEL,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                temperature=0,
                response_format={"type": "json_object"},
            )
            data = _parse_json(response.choices[0].message.content)
            return _to_scores(data, len(expected_facts), answerable)
        except Exception as exc:  # rate limit, timeout, malformed JSON
            last_error = exc
            if attempt < MAX_ATTEMPTS:
                delay = min(2 ** attempt, 20) + random.random()
                logger.warning("Judge attempt %d/%d failed (%s); retrying in %.1fs", attempt, MAX_ATTEMPTS, exc, delay)
                time.sleep(delay)

    logger.error("Judge failed after %d attempts: %s", MAX_ATTEMPTS, last_error)
    return {
        "correctness": None,
        "faithfulness": None,
        "relevancy": None,
        "abstained": False,
        "partially_abstained": False,
        "unsupported_claims": [],
        "fact_verdicts": [],
        "reason": f"judge failed: {last_error}",
        "judge_model": JUDGE_MODEL,
        "judge_failed": True,
    }