# app/utils/cache.py
from __future__ import annotations
import json
import traceback
import logfire
from redisvl.extensions.llmcache import SemanticCache
from redisvl.utils.vectorize import HFTextVectorizer
from app.config import settings

_cache_instance = None

def init_financial_cache() -> SemanticCache | None:
    global _cache_instance
    if _cache_instance is not None:
        return _cache_instance
    
    try:
        logfire.info("Initializing Redis Semantic Cache...", model=settings.EMBEDDING_MODEL)
        vectorizer = HFTextVectorizer(model=settings.EMBEDDING_MODEL)
        
        _cache_instance = SemanticCache(
            name="financial_rag_semantic_cache",
            redis_url=settings.REDIS_URL,
            vectorizer=vectorizer,
            distance_threshold=1.0 - settings.SEMANTIC_CACHE_THRESHOLD
        )
        
        try:
            _cache_instance.set_up_index()
            logfire.info("Redis Semantic Cache index verified/created.")
        except Exception as idx_err:
            logfire.info("Index note (may already exist)", detail=str(idx_err))
            
    except Exception as e:
        logfire.error("Failed to initialize Redis Semantic Cache", error=str(e), traceback=traceback.format_exc())
        _cache_instance = None
        
    return _cache_instance

def get_financial_cache() -> SemanticCache | None:
    if _cache_instance is None:
        return init_financial_cache()
    return _cache_instance

def get_cached_rag_response(query: str) -> dict | None:
    cache = get_financial_cache()
    if not cache:
        return None
    try:
        results = cache.check(prompt=query, return_fields=["response", "retrieved_chunks", "intent", "search_query"])
        if results:
            logfire.info("Cache HIT", query=query)
            hit = results[0]
            raw_response = hit.get("response", "{}")
            
            # Parse the stored JSON string back into a dictionary
            if isinstance(raw_response, str):
                try:
                    payload = json.loads(raw_response)
                except json.JSONDecodeError:
                    payload = {"response": raw_response, "retrieved_chunks": [], "intent": "search", "search_query": query}
            else:
                payload = raw_response

            return {
                "answer": payload.get("response"),
                "retrieved_chunks": payload.get("retrieved_chunks", []),
                "intent": payload.get("intent", "search"),
                "search_query": payload.get("search_query", query)
            }
    except Exception as e:
        print(f"CACHE READ ERROR EXCEPTION: {e}")
        logfire.error("Cache read error", error=str(e))
    
    logfire.info("Cache MISS", query=query)
    return None

def set_cached_rag_response(query: str, answer: str, retrieved_chunks: list, intent: str, search_query: str):
    cache = get_financial_cache()
    if not cache:
        return
    try:
        # Convert chunks to serializable dicts
        serialized_chunks = [c.dict() if hasattr(c, "dict") else c for c in retrieved_chunks]
        
        # Bundle everything into a dict and serialize it to a JSON string for Pydantic/RedisVL string validation
        response_payload = json.dumps({
            "response": answer,
            "retrieved_chunks": serialized_chunks,
            "intent": intent,
            "search_query": search_query
        })
        
        cache.store(
            prompt=query,
            response=response_payload,
            ttl=settings.CACHE_TTL
        )
        logfire.info("Cache STORE SUCCESS", query=query)
    except Exception as e:
        print(f"CACHE WRITE ERROR EXCEPTION: {e}")
        logfire.error("Cache write error", error=str(e))