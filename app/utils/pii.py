# app/utils/pii.py
"""
PII masking. Uses Microsoft Presidio (NER + validated recognizers: phone numbers via libphonenumber,
Luhn-checked cards, emails, IBAN, ...) instead of hand-written patterns per data type.
Which entity types are masked is configuration (settings.PII_INPUT_ENTITIES / PII_OUTPUT_ENTITIES).

If Presidio or the spaCy model is unavailable we degrade to a tiny regex masker rather than failing.
"""
from __future__ import annotations

import re
import threading
from typing import List, Optional

import logfire

from app.config import settings

_analyzer = None
_analyzer_failed = False
_lock = threading.Lock()


def _build_analyzer():
    from presidio_analyzer import AnalyzerEngine, RecognizerRegistry
    from presidio_analyzer.nlp_engine import NlpEngineProvider

    nlp = NlpEngineProvider(nlp_configuration={
        "nlp_engine_name": "spacy",
        "models": [{"lang_code": "en", "model_name": settings.PII_SPACY_MODEL}],
    }).create_engine()
    registry = RecognizerRegistry()
    registry.load_predefined_recognizers(nlp_engine=nlp, languages=["en"])
    try:  # India-specific IDs are not in Presidio's default registry
        from presidio_analyzer.predefined_recognizers import InAadhaarRecognizer, InPanRecognizer
        registry.add_recognizer(InAadhaarRecognizer())
        registry.add_recognizer(InPanRecognizer())
    except Exception:
        pass
    return AnalyzerEngine(nlp_engine=nlp, registry=registry, supported_languages=["en"])


def get_analyzer():
    """Lazy singleton. Returns None if Presidio is unusable (caller falls back to regex)."""
    global _analyzer, _analyzer_failed
    if _analyzer is not None or _analyzer_failed or settings.PII_ENGINE != "presidio":
        return _analyzer
    with _lock:
        if _analyzer is None and not _analyzer_failed:
            try:
                _analyzer = _build_analyzer()
            except Exception as e:
                _analyzer_failed = True
                logfire.warn("Presidio unavailable, using regex PII fallback: {err}", err=repr(e))
    return _analyzer


def warmup_pii() -> None:
    a = get_analyzer()
    if a:
        a.analyze(text="warmup 555-123-4567 a@b.com", language="en", entities=["PHONE_NUMBER", "EMAIL_ADDRESS"])


# --- fallback: a few high-precision structured patterns + secrets -------------------------------
_RX = [
    (re.compile(r"\b\d{3}-\d{2}-\d{4}\b"), "[US_SSN]"),
    (re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b"), "[EMAIL_ADDRESS]"),
    (re.compile(r"(?<![\d,.])(?:\+\d{1,3}[\s-]?)?\d{10,13}(?![\d,.])"), "[PHONE_NUMBER]"),
]
_SECRET = re.compile(r"\b(?:gsk_[A-Za-z0-9]{20,}|sk-[A-Za-z0-9_-]{20,}|AKIA[0-9A-Z]{16})\b")


def _luhn(digits: str) -> bool:
    total, alt = 0, False
    for ch in reversed(digits):
        d = int(ch)
        if alt:
            d = d * 2 - 9 if d * 2 > 9 else d * 2
        total += d
        alt = not alt
    return total % 10 == 0


def _regex_mask(text: str) -> str:
    def card(m):
        digits = re.sub(r"\D", "", m.group())
        return "[CREDIT_CARD]" if 13 <= len(digits) <= 19 and _luhn(digits) else m.group()

    text = re.sub(r"\b(?:\d[ -]?){13,19}\b", card, text)
    for rx, label in _RX:
        text = rx.sub(label, text)
    return text


def mask(text: str, direction: str = "input") -> str:
    """Mask PII in `text`. direction: 'input' (user text) or 'output' (model text)."""
    text = text or ""
    if not text.strip():
        return text
    text = _SECRET.sub("[SECRET]", text)  # secrets are a separate class from PII; Presidio doesn't cover them
    analyzer = get_analyzer()
    if analyzer is None:
        return _regex_mask(text)

    entities: List[str] = settings.PII_INPUT_ENTITIES if direction == "input" else settings.PII_OUTPUT_ENTITIES
    try:
        results = analyzer.analyze(text=text, language="en", entities=entities,
                                   score_threshold=settings.PII_SCORE_THRESHOLD)
    except Exception as e:
        logfire.warn("Presidio analyze failed, using regex fallback: {err}", err=repr(e))
        return _regex_mask(text)

    # replace right-to-left so offsets stay valid; skip overlaps
    out, last_start = text, len(text) + 1
    for r in sorted(results, key=lambda r: (r.start, -r.score), reverse=True):
        if r.end > last_start:
            continue
        out = out[:r.start] + f"[{r.entity_type}]" + out[r.end:]
        last_start = r.start
    return out
