# app/utils/intent_router.py
"""
Embedding-based intent routing (the "semantic router" pattern).

Instead of regexes per phrasing, we embed a labelled set of example phrases (data/intent_examples.yml)
once at startup and classify an incoming message by cosine similarity to the closest examples.
It reuses the bge-small model the cache already loads, so there is no extra model to host.

Only HIGH-CONFIDENCE matches trigger a fast path. Anything ambiguous returns label=None and falls through
to the LLM router, so a miss costs latency, never correctness.
"""
from __future__ import annotations

import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np
import yaml

from app.config import settings

FAST_LABELS = {"smalltalk", "elaboration", "reference", "summary", "out_of_scope"}
SHORT_ONLY = {"smalltalk", "elaboration", "reference", "summary"}  # these are always short utterances
MAX_WORDS_SHORT, MAX_WORDS_ANY = 12, 40


@dataclass
class IntentDecision:
    label: Optional[str]   # None => no confident fast-path match
    top: str               # best-scoring class regardless of confidence (useful for logs / eval)
    score: float
    margin: float


def _unit(m: np.ndarray) -> np.ndarray:
    return m / np.clip(np.linalg.norm(m, axis=-1, keepdims=True), 1e-9, None)


class IntentRouter:
    def __init__(self, embed_many: Callable[[Sequence[str]], List[List[float]]],
                 examples: Dict[str, List[str]],
                 min_score: Optional[float] = None, min_margin: Optional[float] = None):
        self.min_score = settings.INTENT_MIN_SCORE if min_score is None else min_score
        self.min_margin = settings.INTENT_MIN_MARGIN if min_margin is None else min_margin
        self._embed_many = embed_many
        self._labels: List[str] = []
        texts: List[str] = []
        for label, items in examples.items():
            for t in items:
                self._labels.append(label)
                texts.append(str(t))
        self._classes = sorted(set(self._labels))
        self._label_arr = np.array(self._labels)
        self._matrix = _unit(np.asarray(embed_many(texts), dtype=np.float32))

    def scores(self, text: str) -> Dict[str, float]:
        v = _unit(np.asarray(self._embed_many([text]), dtype=np.float32))[0]
        sims = self._matrix @ v
        return {c: float(sims[self._label_arr == c].max()) for c in self._classes}

    def decide(self, text: str) -> IntentDecision:
        words = len(text.split())
        ranked = sorted(self.scores(text).items(), key=lambda kv: kv[1], reverse=True)
        (top, s1), s2 = ranked[0], (ranked[1][1] if len(ranked) > 1 else 0.0)
        margin = s1 - s2
        label = None
        if top in FAST_LABELS and s1 >= self.min_score and margin >= self.min_margin:
            limit = MAX_WORDS_SHORT if top in SHORT_ONLY else MAX_WORDS_ANY
            if words <= limit:
                label = top
        return IntentDecision(label=label, top=top, score=s1, margin=margin)


# --------------------------------------------------------------------------- #
# Singleton wired to the cache's bge-small vectorizer
# --------------------------------------------------------------------------- #
_router: Optional[IntentRouter] = None
_failed = False
_lock = threading.Lock()


def load_examples(path: Optional[str] = None) -> Dict[str, List[str]]:
    with open(path or settings.INTENT_EXAMPLES_PATH, encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_intent_router() -> Optional[IntentRouter]:
    """None if the router could not be built (callers then use the LLM router only)."""
    global _router, _failed
    if _router is not None or _failed or not settings.INTENT_ROUTER_ENABLED:
        return _router
    with _lock:
        if _router is None and not _failed:
            try:
                from app.utils.cache import get_vectorizer
                vec = get_vectorizer()
                _router = IntentRouter(lambda texts: vec.embed_many(list(texts)), load_examples())
            except Exception as e:
                import logfire
                _failed = True
                logfire.warn("Intent router unavailable, falling back to LLM router: {err}", err=repr(e))
    return _router


def warmup_intent_router() -> None:
    r = get_intent_router()
    if r:
        r.decide("warmup")
