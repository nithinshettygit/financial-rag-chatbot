# app/utils/guard_rules.py
"""
Pure, dependency-free guardrail checks.

They are used in two places:
  1. as NeMo Guardrails custom actions (app/guardrails_config/actions.py), and
  2. directly as a deterministic fallback when NeMo itself is unavailable.
"""
from __future__ import annotations

import re
from typing import List, Tuple

MAX_INPUT_CHARS = 2000

# --------------------------------------------------------------------------- #
# Input: prompt-injection / jailbreak heuristics
# --------------------------------------------------------------------------- #
_INJECTION_PATTERNS = [
    r"ignore (?:all |any |the )?(?:previous|prior|above|earlier) (?:instructions|prompts?|rules)",
    r"disregard (?:all |any |the )?(?:previous|prior|above|earlier|your) (?:instructions|prompts?|rules)",
    r"(?:reveal|show|print|repeat|output|leak) (?:me )?(?:your|the) (?:system|hidden|initial|original) (?:prompt|instructions|message)",
    r"what (?:is|are) your (?:system prompt|instructions|rules)",
    r"you are now (?:in )?(?:dan|developer mode|an? unrestricted)",
    r"\b(?:jailbreak|do anything now|developer mode)\b",
    r"pretend (?:you (?:are|have)|to be) .{0,40}(?:no|without) (?:restrictions|rules|filters)",
    r"</?(?:system|assistant|instructions?)>",
]
_INJECTION_RE = re.compile("|".join(_INJECTION_PATTERNS), re.I)


def detect_injection(text: str) -> str:
    """Return 'ok' or a short reason code ('empty', 'too_long', 'injection')."""
    text = text or ""
    if not text.strip():
        return "empty"
    if len(text) > MAX_INPUT_CHARS:
        return "too_long"
    if _INJECTION_RE.search(text):
        return "injection"
    return "ok"


# --------------------------------------------------------------------------- #
# PII / secrets masking (input and output) - delegated to Presidio (see pii.py)
# --------------------------------------------------------------------------- #
def mask_pii(text: str, direction: str = "input") -> str:
    from app.utils import pii
    return pii.mask(text, direction)


# --------------------------------------------------------------------------- #
# Output: system-prompt leakage
# --------------------------------------------------------------------------- #
_LEAK_MARKERS = (
    "precise, professional financial qa assistant",
    "using only the context provided below",
    "retrieval focus:",
)


def leaks_prompt(text: str) -> bool:
    low = (text or "").lower()
    return any(m in low for m in _LEAK_MARKERS)


# --------------------------------------------------------------------------- #
# Output: numeric grounding (financial hallucination guard)
# --------------------------------------------------------------------------- #
_NUM = re.compile(r"\d[\d,]*(?:\.\d+)?")
_SCALES = (1.0, 1e3, 1e-3, 1e6, 1e-6)  # "51.4 billion" vs "51,362 (in millions)" etc.
_CITATION = re.compile(r"\[S\d+\]")


def _values(text: str, *, candidates_only: bool) -> List[float]:
    text = _CITATION.sub(" ", text or "")
    out: List[float] = []
    for m in _NUM.finditer(text):
        raw = m.group()
        try:
            v = float(raw.replace(",", ""))
        except ValueError:
            continue
        if candidates_only:
            is_pct = text[m.end(): m.end() + 1] == "%"
            has_fmt = "," in raw or "." in raw
            is_year = raw.isdigit() and 1900 <= v <= 2100
            if is_year or not (is_pct or has_fmt or v >= 1000):
                continue
        out.append(v)
    return out


def ungrounded_ratio(answer: str, evidence: str) -> Tuple[float, List[float]]:
    """Share of 'material' figures in the answer that cannot be found in the evidence
    (within 1% tolerance, allowing thousand/million unit rescaling)."""
    cands = _values(answer, candidates_only=True)
    if not cands:
        return 0.0, []
    ctx = _values(evidence, candidates_only=False)
    bad = [
        v for v in cands
        if not any(abs(v - c * s) <= 0.01 * abs(v) for c in ctx for s in _SCALES)
    ]
    return len(bad) / len(cands), bad
