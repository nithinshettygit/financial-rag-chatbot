# app/utils/cache.py
import json
import os
import hashlib
import logfire
from redis import Redis
from redisvl.index import SearchIndex
from redisvl.schema import IndexSchema
from redisvl.query import VectorQuery
from redisvl.query.filter import Tag
from redisvl.utils.vectorize import HFTextVectorizer
from app.config import settings

_REDIS_CLIENT = None
_VECTORIZER = None

CACHE_INDEX_NAME = "financial_semantic_cache_v4"
CACHE_PREFIX = "fincache_v4"

def get_vectorizer():
    global _VECTORIZER
    if _VECTORIZER is None:
        _VECTORIZER = HFTextVectorizer(model="BAAI/bge-small-en-v1.5")
    return _VECTORIZER

def get_redis_client() -> Redis:
    global _REDIS_CLIENT
    if _REDIS_CLIENT is None:
        redis_url = os.getenv("REDIS_URL", "redis://localhost:6379")
        _REDIS_CLIENT = Redis.from_url(redis_url, decode_responses=True)
    return _REDIS_CLIENT

def get_cache_schema(dim: int = 384) -> IndexSchema:
    # Ensure dim is an explicit integer (bge-small-en-v1.5 is always 384)
    resolved_dim = int(dim) if dim and isinstance(dim, int) else 384
    
    return IndexSchema.from_dict({
        "index": {
            "name": CACHE_INDEX_NAME,
            "prefix": CACHE_PREFIX,
            "storage_type": "hash"
        },
        "fields": [
            {"name": "prompt", "type": "text"},
            {"name": "prompt_vector", "type": "vector", "attrs": {
                "algorithm": "flat",
                "datatype": "float32",
                "dims": resolved_dim,
                "distance_metric": "cosine"
            }},

            {"name": "response", "type": "text"},
            {"name": "entity", "type": "tag"},
            {"name": "year", "type": "tag"},
            {"name": "intent", "type": "tag"}
        ]
    })

def init_financial_cache():
    """Initializes the RedisVL Search Index for semantic caching with metadata tags."""
    try:
        client = get_redis_client()
        vectorizer = get_vectorizer()
        
        # Safely resolve dimensions with hardcoded fallback to 384
        dims = 384
        try:
            if hasattr(vectorizer, "dims") and vectorizer.dims:
                dims = int(vectorizer.dims)
        except Exception:
            pass
        
        schema = get_cache_schema(dims)
        index = SearchIndex(schema, redis_client=client)
        
        # if index.exists():
        #     try:
        #         index.delete(drop=True)
        #         logfire.info("Dropped existing conflicting cache index: {name}", name=CACHE_INDEX_NAME)
        #     except Exception as drop_err:
        #         logfire.warning("Could not drop existing index: {err}", err=str(drop_err))

        # index.create(overwrite=True)
        # logfire.info("Successfully created/verified financial semantic cache index: {name}", name=CACHE_INDEX_NAME)
        if not index.exists():
            index.create(overwrite=False)
            logfire.info("Created new financial semantic cache index: {name}", name=CACHE_INDEX_NAME)
        else:
            logfire.info("Using existing financial semantic cache index: {name}", name=CACHE_INDEX_NAME)    
            
    except Exception as e:
        logfire.error("Failed to set up cache index", error=str(e))
        raise

def get_cached_rag_response(
    query: str, 
    entity: str | None = None, 
    year: str | int | None = None, 
    intent: str | None = None
) -> dict | None:
    try:
        client = get_redis_client()
        vectorizer = get_vectorizer()
        vector = vectorizer.embed(query)
        
        f_expr = None
        if entity:
            f_expr = Tag("entity") == str(entity).lower()
        if year:
            year_tag = Tag("year") == str(year)
            f_expr = (f_expr & year_tag) if f_expr else year_tag
        if intent:
            intent_tag = Tag("intent") == str(intent)
            f_expr = (f_expr & intent_tag) if f_expr else intent_tag

        v_query = VectorQuery(
            vector=vector,
            vector_field_name="prompt_vector",
            return_fields=["response", "retrieved_chunks", "intent", "search_query", "vector_distance"],
            num_results=1,
            filter_expression=f_expr
        )
        
        dims = 384
        try:
            if hasattr(vectorizer, "dims") and vectorizer.dims:
                dims = int(vectorizer.dims)
        except Exception:
            pass

        schema = get_cache_schema(dims)
        index = SearchIndex(schema, redis_client=client)
        results = index.query(v_query)
        
        if results:
            hit = results[0]
            distance = float(hit.get("vector_distance", 1.0))
            # if distance <= 0.15:
            if distance <= 0.075:
                logfire.info("Cache HIT", query=query, distance=distance, entity=entity, year=year)
                raw_response = hit.get("response", "{}")
                payload = json.loads(raw_response) if isinstance(raw_response, str) else raw_response
                return {
                    "answer": payload.get("response"),
                    "retrieved_chunks": payload.get("retrieved_chunks", []),
                    "intent": payload.get("intent", "search"),
                    "search_query": payload.get("search_query", query)
                }
    except Exception as e:
        logfire.error("Cache read error", error=str(e))
    
    logfire.info("Cache MISS", query=query, entity=entity, year=year)
    return None

def set_cached_rag_response(
    query: str, 
    answer: str, 
    retrieved_chunks: list, 
    intent: str, 
    search_query: str, 
    entity: str | None = None, 
    year: str | int | None = None,
    document_ids: list[str] | None = None
):
    try:
        client = get_redis_client()
        vectorizer = get_vectorizer()
        vector = vectorizer.embed(query, as_buffer=True)
        
        serialized_chunks = [c.dict() if hasattr(c, "dict") else c for c in retrieved_chunks]
        response_payload = json.dumps({
            "response": answer,
            "retrieved_chunks": serialized_chunks,
            "intent": intent,
            "search_query": search_query,
            "entity": entity or "",
            "year": str(year) if year else "",
            "document_ids": document_ids or []
        })
        
        dims = 384
        try:
            if hasattr(vectorizer, "dims") and vectorizer.dims:
                dims = int(vectorizer.dims)
        except Exception:
            pass

        schema = get_cache_schema(dims)
        index = SearchIndex(schema, redis_client=client)
        
        entry_id = hashlib.md5(query.encode()).hexdigest()
        
        data = [{
            "id": entry_id,
            "prompt": query,
            "prompt_vector": vector,
            "response": response_payload,
            "entity": str(entity).lower() if entity else "global",
            "year": str(year) if year else "any",
            "intent": intent or "search"
        }]
        
        index.load(data, ttl=settings.CACHE_TTL)
        logfire.info("Cache STORE SUCCESS", query=query, entity=entity, year=year)
    except Exception as e:
        logfire.error("Cache write error", error=str(e))