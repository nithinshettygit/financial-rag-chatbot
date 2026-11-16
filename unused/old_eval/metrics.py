"""Deterministic evaluation metrics (no LLM calls).

Ground truth model
------------------
Each question may list one or more *gold sources*:
    {"document": "duolingo", "pages": [45, 46], "section": "Consolidated Statements of Operations"}
A retrieved chunk matches a gold source when the document matches (if given) AND
(page matches OR section heading matches). If a gold source has neither page nor
section, any chunk from that document matches (document-level gold).
Multi-metric / comparative questions list several gold sources, so recall@k means
"what fraction of the required sources reached the top-k", not just "any hit".
"""
from __future__ import annotations

import math
import re
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set

K_VALUES = (1, 3, 5, 10)

_SUFFIXES = {"inc", "incorporated", "corp", "corporation", "company", "co", "ltd", "limited", "plc"}


# --------------------------------------------------------------------------- #
# Normalisation helpers
# --------------------------------------------------------------------------- #
def _norm(text: Any) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(text or "").lower()).strip()


def _entity_key(text: Any) -> str:
    terms = _norm(text).split()
    while terms and terms[-1] in _SUFFIXES:
        terms.pop()
    return " ".join(terms)


def _parse_pages(value: Any) -> Set[int]:
    if value is None:
        return set()
    if isinstance(value, (list, tuple, set)):
        out: Set[int] = set()
        for v in value:
            out |= _parse_pages(v)
        return out
    s = str(value).strip()
    if not s or s.upper() == "N/A":
        return set()
    m = re.fullmatch(r"(\d+)\s*[-–]\s*(\d+)", s)
    if m:
        a, b = int(m.group(1)), int(m.group(2))
        return set(range(a, b + 1)) if 0 <= b - a < 50 else {a, b}
    return {int(x) for x in re.findall(r"\d+", s)}


def normalize_gold_sources(item: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Accept the new `gold_sources` list or the legacy source_page/source_reference fields."""
    raw = item.get("gold_sources")
    sources: List[Dict[str, Any]] = []
    if raw:
        for s in raw:
            if not isinstance(s, dict):
                continue
            sources.append(
                {
                    "document": s.get("document") or s.get("company"),
                    "pages": sorted(_parse_pages(s.get("pages", s.get("page")))),
                    "section": s.get("section") or s.get("reference"),
                }
            )
        return sources

    pages = _parse_pages(item.get("source_page"))
    ref = item.get("source_reference")
    has_ref = bool(ref) and str(ref).strip().upper() != "N/A"
    if pages or has_ref:
        sources.append(
            {"document": item.get("document"), "pages": sorted(pages), "section": ref if has_ref else None}
        )
    return sources


# --------------------------------------------------------------------------- #
# Chunk <-> gold matching
# --------------------------------------------------------------------------- #
def _doc_matches(document: Optional[str], chunk: Dict[str, Any]) -> bool:
    if not document:
        return True
    needle = f" {_entity_key(document)} "
    hay = f" {_norm(str(chunk.get('document_id', '')) + ' ' + str(chunk.get('source_file', '')))} "
    return needle in hay


def _section_matches(section: Optional[str], headings: Optional[List[str]]) -> bool:
    ref = _norm(section)
    if not ref or not headings:
        return False
    for h in headings:
        hn = _norm(h)
        if hn and (ref in hn or (len(hn) >= 6 and hn in ref)):
            return True
    return False


def chunk_matches_source(chunk: Dict[str, Any], src: Dict[str, Any]) -> bool:
    if not _doc_matches(src.get("document"), chunk):
        return False
    pages = set(src.get("pages") or [])
    section = src.get("section")
    has_page, has_section = bool(pages), bool(_norm(section))
    if not has_page and not has_section:
        return True
    page_ok = has_page and bool(_parse_pages(chunk.get("page_number")) & pages)
    sec_ok = has_section and _section_matches(section, chunk.get("headings"))
    return page_ok or sec_ok


# --------------------------------------------------------------------------- #
# Retrieval metrics (source-level, order-aware)
# --------------------------------------------------------------------------- #
def _ndcg(matched: List[Set[int]], n_gold: int, k: int) -> float:
    seen: Set[int] = set()
    dcg = 0.0
    for rank, m in enumerate(matched[:k], start=1):
        if m - seen:  # gain only for the first chunk that covers a new gold source
            dcg += 1.0 / math.log2(rank + 1)
        seen |= m
    ideal = sum(1.0 / math.log2(r + 1) for r in range(1, min(k, n_gold) + 1))
    return dcg / ideal if ideal else 0.0


def calculate_retrieval_metrics(
    chunks: List[Dict[str, Any]],
    gold: List[Dict[str, Any]],
    k_values: Sequence[int] = K_VALUES,
) -> Dict[str, float]:
    """hit/recall/precision/ndcg @k plus *@ctx (everything the generator actually saw)."""
    n_gold = len(gold)
    matched = [{i for i, s in enumerate(gold) if chunk_matches_source(c, s)} for c in chunks]
    out: Dict[str, float] = {}

    def _at(label: str, k: int) -> None:
        top = matched[:k]
        covered: Set[int] = set().union(*top) if top else set()
        out[f"hit@{label}"] = float(bool(covered))
        out[f"recall@{label}"] = len(covered) / n_gold if n_gold else 0.0
        out[f"precision@{label}"] = (sum(1 for m in top if m) / len(top)) if top else 0.0
        out[f"ndcg@{label}"] = _ndcg(matched, n_gold, k) if n_gold else 0.0

    for k in k_values:
        _at(str(k), k)
    _at("ctx", max(len(chunks), 1))

    out["mrr"] = next((1.0 / r for r, m in enumerate(matched, start=1) if m), 0.0)
    return out


def entity_coverage(chunks: List[Dict[str, Any]], entities: Iterable[str]) -> Optional[float]:
    """Fraction of expected companies that appear in the retrieved context (comparative queries)."""
    entities = [e for e in entities if e]
    if not entities:
        return None
    hit = sum(1 for e in entities if any(_doc_matches(e, c) for c in chunks))
    return hit / len(entities)


# --------------------------------------------------------------------------- #
# Deterministic numeric fact check (cross-check for the LLM judge)
# --------------------------------------------------------------------------- #
_NUM_RE = re.compile(r"\d[\d,]*(?:\.\d+)?")
_SCALES = (1.0, 1e3, 1e6, 1e9, 1e-3, 1e-6, 1e-9)


def _numbers(text: str) -> List[float]:
    out = []
    for m in _NUM_RE.findall(str(text)):
        try:
            out.append(float(m.replace(",", "")))
        except ValueError:
            pass
    return out


def _value_in_answer(value: Any, answer: str, rel_tol: float = 1e-3) -> bool:
    targets = _numbers(str(value))
    if not targets:  # non-numeric value -> case-insensitive substring
        return _norm(value) in _norm(answer) if _norm(value) else False
    target = targets[0]
    scales = _SCALES if target >= 100 else (1.0,)  # allow "531.1 million" vs "531,100" only for big numbers
    for a in _numbers(answer):
        for s in scales:
            if abs(a * s - target) <= rel_tol * abs(target) + 1e-9:
                return True
    return False


def numeric_fact_recall(answer: str, facts: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """A fact passes if ANY of its listed `values` appears in the answer (unit-scale tolerant)."""
    checkable = [f for f in facts if f.get("values")]
    if not checkable:
        return None
    details = []
    for f in checkable:
        ok = any(_value_in_answer(v, answer) for v in f["values"])
        details.append({"fact": f["fact"], "matched": ok})
    matched = sum(d["matched"] for d in details)
    return {"checked": len(details), "matched": matched, "recall": matched / len(details), "details": details}


# --------------------------------------------------------------------------- #
# Abstention (regex fallback; the LLM judge is primary)
# --------------------------------------------------------------------------- #
_REFUSAL_PATTERNS = (
    "do not contain", "does not contain", "not available", "couldn't find", "could not find",
    "cannot find", "can't find", "unable to find", "no information", "not provided",
    "not found in the", "documents do not", "context does not", "doesn't provide", "does not provide",
    "not specified", "not mentioned", "no indexed documents", "insufficient information",
    "not disclosed", "missing from the context",
)


def is_abstention(answer: str) -> bool:
    text = (answer or "").lower()
    return bool(text) and any(p in text for p in _REFUSAL_PATTERNS)


# --------------------------------------------------------------------------- #
# Aggregation utilities
# --------------------------------------------------------------------------- #
def mean(values: Iterable[Optional[float]]) -> Optional[float]:
    vals = [v for v in values if v is not None]
    return sum(vals) / len(vals) if vals else None


def percentile(values: Iterable[float], p: float) -> Optional[float]:
    vals = sorted(values)
    if not vals:
        return None
    pos = (len(vals) - 1) * p / 100.0
    lo, hi = math.floor(pos), math.ceil(pos)
    return vals[lo] + (vals[hi] - vals[lo]) * (pos - lo)


def latency_stats(values: List[float]) -> Dict[str, Optional[float]]:
    return {
        "mean_s": mean(values),
        "p50_s": percentile(values, 50),
        "p95_s": percentile(values, 95),
        "max_s": max(values) if values else None,
    }


def aggregate_retrieval_metrics(results: List[Dict[str, Any]]) -> Dict[str, float]:
    """Mean of every retrieval metric over answerable questions that have gold sources."""
    rows = [r["retrieval_metrics"] for r in results if r.get("retrieval_metrics")]
    if not rows:
        return {}
    keys = sorted({k for row in rows for k in row})
    return {k: sum(row.get(k, 0.0) for row in rows) / len(rows) for k in keys}