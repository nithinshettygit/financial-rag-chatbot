# app/utils/cache.py
from __future__ import annotations
import logfire
from redisvl.extensions.llmcache import SemanticCache
from redisvl.utils.vectorize import FastEmbedVectorizer
from app.config import settings

_cache_instance = None

def get_financial_cache() -> SemanticCache | None:
    global _cache_instance
    if _cache_instance is not None:
        return _cache_instance
    
    try:
        # Uses FastEmbed with your exact project embedding model
        vectorizer = FastEmbedVectorizer(model=settings.EMBEDDING_MODEL)
        
        _cache_instance = SemanticCache(
            name="financial_rag_semantic_cache",
            redis_url=settings.REDIS_URL,
            vectorizer=vectorizer,
            distance_threshold=1.0 - settings.SEMANTIC_CACHE_THRESHOLD # RedisVL uses distance
        )
        logfire.info("Redis Semantic Cache initialized successfully.")
    except Exception as e:
        logfire.error("Failed to initialize Redis Semantic Cache", error=str(e))
        _cache_instance = None
        
    return _cache_instance

def get_cached_rag_response(query: str) -> dict | None:
    cache = get_financial_cache()
    if not cache:
        return None
    try:
        results = cache.search(query, return_fields=["response", "retrieved_chunks", "intent", "search_query"])
        if results:
            logfire.info("Cache HIT", query=query)
            hit = results[0]
            return {
                "answer": hit.get("response"),
                "retrieved_chunks": eval(hit.get("retrieved_chunks", "[]")),
                "intent": hit.get("intent", "search"),
                "search_query": hit.get("search_query", query)
            }
    except Exception as e:
        logfire.error("Cache read error", error=str(e))
    
    logfire.info("Cache MISS", query=query)
    return None

def set_cached_rag_response(query: str, answer: str, retrieved_chunks: list, intent: str, search_query: str):
    cache = get_financial_cache()
    if not cache:
        return
    try:
        cache.store(
            prompts=[query],
            responses=[{
                "response": answer,
                "retrieved_chunks": str([c.dict() if hasattr(c, "dict") else c for c in retrieved_chunks]),
                "intent": intent,
                "search_query": search_query
            }],
            ttl=settings.CACHE_TTL
        )
    except Exception as e:
        logfire.error("Cache write error", error=str(e))