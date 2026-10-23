# app/utils/cache.py
"""
Two-tier cache for the RAG pipeline.

Principle: a cached answer must be a pure function of
    (standalone question, entity scope, year scope, intent, corpus version[, conversation context])
so anything that depends on the conversation is either keyed by that context or not cached at all.

  L1  exact match   : Redis string, key = hash(scope + normalised text). No embeddings, microseconds.
  L2  semantic      : RedisVL vector search, filtered by scope tags, then VERIFIED lexically
                      (same numbers, high content-word overlap) so "revenue details" != "revenue sources".
"""
from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Optional

import logfire
from redis import Redis
from redisvl.index import SearchIndex
from redisvl.query import VectorQuery
from redisvl.query.filter import Tag
from redisvl.schema import IndexSchema
from redisvl.utils.vectorize import HFTextVectorizer

from app.config import settings

CACHE_INDEX_NAME = "financial_semantic_cache_v5"
CACHE_PREFIX = "fincache_v5"
L1_PREFIX = "fincache_l1"
EMBED_DIMS = 384  # bge-small-en-v1.5

_REDIS_CLIENT: Optional[Redis] = None
_VECTORIZER = None
_INDEX: Optional[SearchIndex] = None


# --------------------------------------------------------------------------- #
# Scope
# --------------------------------------------------------------------------- #
def _slug(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(s).lower()).strip("_")


@dataclass(frozen=True)
class CacheScope:
    intent: str
    entity: str = "global"
    year: str = "any"
    corpus: str = "0"   # changes when documents are (re)indexed -> old entries become unreachable
    ctx: str = ""       # conversation context hash; only for context-dependent (elaboration) entries

    def fingerprint(self) -> str:
        return "|".join((self.intent, self.entity, self.year, self.corpus, self.ctx))


def build_scope(intent: str, entities: Iterable[str], years: Iterable[int], corpus: str, ctx: str = "") -> CacheScope:
    ent = "__".join(sorted({_slug(e) for e in entities if e})) or "global"
    yrs = "__".join(str(y) for y in sorted(set(years))) or "any"
    return CacheScope(intent=intent or "rag_search", entity=ent, year=yrs, corpus=corpus, ctx=ctx)


# --------------------------------------------------------------------------- #
# Text normalisation + lexical verification
# --------------------------------------------------------------------------- #
_STOP = {
    "a", "an", "the", "of", "for", "in", "on", "to", "is", "are", "was", "were", "be", "s", "me", "my", "i",
    "what", "whats", "which", "tell", "show", "give", "please", "can", "could", "you", "about", "its", "their",
    "and", "or", "with", "do", "does", "did", "us", "it",
}


def normalize_query(text: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", str(text or "").lower()))


def _stem(t: str) -> str:
    return t[:-1] if len(t) > 3 and t.endswith("s") else t


def _terms(text: str) -> set:
    return {_stem(t) for t in normalize_query(text).split() if t not in _STOP}


def terms_compatible(a: str, b: str) -> bool:
    """Cheap guard against embedding false-positives: numbers must match, content words must overlap."""
    ta, tb = _terms(a), _terms(b)
    if not ta or not tb:
        return False
    if {t for t in ta if t.isdigit()} != {t for t in tb if t.isdigit()}:
        return False
    return len(ta & tb) / len(ta | tb) >= settings.CACHE_MIN_TERM_OVERLAP


# --------------------------------------------------------------------------- #
# Clients
# --------------------------------------------------------------------------- #
def get_vectorizer():
    global _VECTORIZER
    if _VECTORIZER is None:
        _VECTORIZER = HFTextVectorizer(model="BAAI/bge-small-en-v1.5")
    return _VECTORIZER


def get_redis_client() -> Redis:
    global _REDIS_CLIENT
    if _REDIS_CLIENT is None:
        _REDIS_CLIENT = Redis.from_url(os.getenv("REDIS_URL", settings.REDIS_URL), decode_responses=True)
    return _REDIS_CLIENT


def get_cache_schema(dim: int = EMBED_DIMS) -> IndexSchema:
    return IndexSchema.from_dict({
        "index": {"name": CACHE_INDEX_NAME, "prefix": CACHE_PREFIX, "storage_type": "hash"},
        "fields": [
            {"name": "prompt", "type": "text"},
            {"name": "prompt_vector", "type": "vector", "attrs": {
                "algorithm": "flat", "datatype": "float32", "dims": int(dim), "distance_metric": "cosine"}},
            {"name": "response", "type": "text"},
            {"name": "entity", "type": "tag"},
            {"name": "year", "type": "tag"},
            {"name": "intent", "type": "tag"},
            {"name": "corpus", "type": "tag"},
        ],
    })


def _get_index() -> SearchIndex:
    global _INDEX
    if _INDEX is None:
        _INDEX = SearchIndex(get_cache_schema(), redis_client=get_redis_client())
    return _INDEX


def init_financial_cache() -> None:
    """Create the L2 index if missing and pre-load the embedding model + Redis connection."""
    try:
        get_vectorizer()
        index = _get_index()
        if not index.exists():
            index.create(overwrite=False)
            logfire.info("Created semantic cache index: {name}", name=CACHE_INDEX_NAME)
        else:
            logfire.info("Using existing semantic cache index: {name}", name=CACHE_INDEX_NAME)
    except Exception as e:
        logfire.error("Failed to set up cache index", error=str(e))
        raise


# --------------------------------------------------------------------------- #
# L1 - exact
# --------------------------------------------------------------------------- #
def _l1_key(scope: CacheScope, text: str) -> str:
    digest = hashlib.sha256(f"{scope.fingerprint()}||{normalize_query(text)}".encode()).hexdigest()
    return f"{L1_PREFIX}:{scope.corpus}:{digest}"


def _l1_get(scope: CacheScope, text: str) -> Optional[Dict[str, Any]]:
    raw = get_redis_client().get(_l1_key(scope, text))
    return json.loads(raw) if raw else None


def _l1_set(scope: CacheScope, text: str, payload: Dict[str, Any], ttl: int) -> None:
    get_redis_client().set(_l1_key(scope, text), json.dumps(payload), ex=ttl)


# --------------------------------------------------------------------------- #
# L2 - semantic (filtered by scope, verified lexically)
# --------------------------------------------------------------------------- #
def _l2_get(scope: CacheScope, text: str) -> Optional[Dict[str, Any]]:
    f_expr = (
        (Tag("entity") == scope.entity)
        & (Tag("year") == scope.year)
        & (Tag("intent") == scope.intent)
        & (Tag("corpus") == scope.corpus)
    )
    query = VectorQuery(
        vector=get_vectorizer().embed(text),
        vector_field_name="prompt_vector",
        return_fields=["prompt", "response", "vector_distance"],
        num_results=1,
        filter_expression=f_expr,
    )
    results = _get_index().query(query)
    if not results:
        return None
    hit = results[0]
    distance = float(hit.get("vector_distance", 1.0))
    if distance > settings.CACHE_MAX_DISTANCE:
        logfire.info("Cache L2 near-miss", query=text, distance=distance)
        return None
    if not terms_compatible(text, hit.get("prompt", "")):
        logfire.info("Cache L2 rejected by lexical guard", query=text, cached=hit.get("prompt"), distance=distance)
        return None
    raw = hit.get("response", "{}")
    return json.loads(raw) if isinstance(raw, str) else raw


def _l2_set(scope: CacheScope, text: str, payload: Dict[str, Any], ttl: int) -> None:
    entry_id = hashlib.md5(f"{scope.fingerprint()}||{normalize_query(text)}".encode()).hexdigest()
    _get_index().load(
        [{
            "id": entry_id,
            "prompt": text,
            "prompt_vector": get_vectorizer().embed(text, as_buffer=True),
            "response": json.dumps(payload),
            "entity": scope.entity,
            "year": scope.year,
            "intent": scope.intent,
            "corpus": scope.corpus,
        }],
        id_field="id",
        ttl=ttl,
    )


# --------------------------------------------------------------------------- #
# Public API (blocking; call via asyncio.to_thread)
# --------------------------------------------------------------------------- #
def cache_lookup(text: str, scope: CacheScope, semantic: bool = True) -> Optional[Dict[str, Any]]:
    try:
        hit = _l1_get(scope, text)
        if hit:
            logfire.info("Cache HIT (L1)", query=text, scope=scope.fingerprint())
            return {**hit, "_tier": "l1"}
        if semantic:
            hit = _l2_get(scope, text)
            if hit:
                _l1_set(scope, text, hit, settings.CACHE_TTL)  # promote for next time
                logfire.info("Cache HIT (L2)", query=text, scope=scope.fingerprint())
                return {**hit, "_tier": "l2"}
    except Exception as e:
        logfire.error("Cache read error", error=str(e))
    logfire.info("Cache MISS", query=text, scope=scope.fingerprint())
    return None


def _to_dict(c: Any) -> Any:
    if hasattr(c, "model_dump"):
        return c.model_dump()
    return c.dict() if hasattr(c, "dict") else c


def cache_store(text: str, scope: CacheScope, payload: Dict[str, Any], semantic: bool = True,
                ttl: Optional[int] = None) -> None:
    """payload keys: answer, retrieved_chunks (models or dicts), intent, search_query, document_ids."""
    try:
        ttl = ttl or settings.CACHE_TTL
        body = {**payload, "retrieved_chunks": [_to_dict(c) for c in payload.get("retrieved_chunks", [])]}
        _l1_set(scope, text, body, ttl)
        if semantic:
            _l2_set(scope, text, body, ttl)
        logfire.info("Cache STORE", query=text, scope=scope.fingerprint(), semantic=semantic)
    except Exception as e:
        logfire.error("Cache write error", error=str(e))