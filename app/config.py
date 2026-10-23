# app/config.py
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    APP_NAME: str = "Docling Financial Hybrid RAG Service"
    GROQ_API_KEY: str
    FALLBACK_GROQ_API_KEY: str | None = None
    LLM_MODEL: str = "openai/gpt-oss-20b"
    EMBEDDING_MODEL: str = "BAAI/bge-small-en-v1.5"
    VECTOR_NAME: str = "fastembed-BAAI/bge-small-en-v1.5"
    COLLECTION_NAME: str = "financial_rag_analytics"

    # FlashRank & Logfire
    RERANKER_MODEL: str = "ms-marco-MiniLM-L-12-v2"
    LOGFIRE_TOKEN: str | None = None

    QDRANT_URL: str | None = None
    QDRANT_API_KEY: str | None = None
    QDRANT_STORAGE_PATH: str = "data/qdrant_storage"
    BM25_MANIFEST_PATH: str = "data/manifests"

    RETRIEVAL_K: int = 30
    FINAL_TOP_K: int = 5
    RRF_K: int = 60
    MAX_HISTORY_TURNS: int = 6
    MAX_CONTEXT_TOKENS: int = 4096

    # ------------------------------------------------------------------ #
    # Cache (two tiers: L1 exact-match, L2 semantic + lexical verification)
    # ------------------------------------------------------------------ #
    REDIS_URL: str = "redis://localhost:6379"
    CACHE_TTL: int = 86400                 # 24h for answers
    CACHE_GREETING_TTL: int = 604800       # 7d for greetings / small talk
    CACHE_MAX_DISTANCE: float = 0.04       # cosine distance for an L2 candidate (tune from logs)
    CACHE_MIN_TERM_OVERLAP: float = 0.7    # Jaccard overlap of content words required on an L2 hit

    # ------------------------------------------------------------------ #
    # NeMo Guardrails
    # ------------------------------------------------------------------ #
    GUARDRAILS_ENABLED: bool = True
    GUARDRAILS_PATH: str = "guardrails_config"
    # Small NON-reasoning model for yes/no policy checks (reasoning models burn the tiny
    # max_tokens budget NeMo gives self-check tasks). Confirm it is available on your Groq account.
    GUARDRAIL_LLM_MODEL: str = "qwen/qwen3.8-27b"
    GUARDRAILS_TIMEOUT_S: float = 8.0
    GUARDRAILS_FAIL_OPEN: bool = True      # if NeMo itself errors, fall back to deterministic rules only
    GUARDRAILS_NUMERIC_CHECK: bool = True  # block answers whose figures are not found in the retrieved context

    class Config:
        env_file = ".env"


settings = Settings()