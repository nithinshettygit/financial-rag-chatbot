from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    APP_NAME: str = "Docling Financial Hybrid RAG Service"

    GROQ_API_KEY: str
    FALLBACK_GROQ_API_KEY: str | None = None
    LLM_MODEL: str = "openai/gpt-oss-20b"
    EMBEDDING_MODEL: str = "BAAI/bge-small-en-v1.5"
    VECTOR_NAME: str = "fastembed-BAAI-bge-small-en-v1-5"
    COLLECTION_NAME: str = "financial_rag_analytics"

    # ------------------------------------------------------------------ #
    # Browser session authentication
    # ------------------------------------------------------------------ #
    SESSION_HMAC_SECRET: str = Field(min_length=32)
    SESSION_TTL_SECONDS: int = Field(default=86400, gt=0)
    SESSION_COOKIE_SECURE: bool = True
    SESSION_COOKIE_SAMESITE: str = "lax"

    # ------------------------------------------------------------------ #
    # Distributed Redis rate limiting
    # ------------------------------------------------------------------ #
    CHAT_SESSION_RATE_LIMIT_REQUESTS: int = Field(default=20, gt=0)
    CHAT_IP_RATE_LIMIT_REQUESTS: int = Field(default=60, gt=0)
    CHAT_RATE_LIMIT_WINDOW_SECONDS: int = Field(default=60, gt=0)

    # ------------------------------------------------------------------ #
    # Distributed RAG concurrency / waiting queue
    # ------------------------------------------------------------------ #
    MAX_CONCURRENT_RAG: int = Field(default=2, gt=0)
    MAX_QUEUE_WAITERS: int = Field(default=10, gt=0)
    RAG_QUEUE_MAX_WAIT_SECONDS: int = Field(default=15, gt=0)
    RAG_QUEUE_POLL_INTERVAL_SECONDS: float = Field(default=0.25, gt=0)
    RAG_QUEUE_RETRY_AFTER_SECONDS: int = Field(default=2, gt=0)
    RAG_CONCURRENCY_LEASE_SECONDS: int = Field(default=60, gt=0)

    # FlashRank & Logfire
    RERANKER_MODEL: str = "ms-marco-MiniLM-L-12-v2"
    LOGFIRE_TOKEN: str | None = None

    QDRANT_URL: str | None = None
    QDRANT_API_KEY: str | None = None
    BM25_MANIFEST_PATH: str = "data/manifests"

    RETRIEVAL_K: int = 30
    FINAL_TOP_K: int = 5
    RRF_K: int = 60
    MAX_HISTORY_TURNS: int = 6
    MAX_CONTEXT_TOKENS: int = 4096

    # ------------------------------------------------------------------ #
    # Cache
    # ------------------------------------------------------------------ #
    REDIS_URL: str = "redis://localhost:6379"
    CACHE_TTL: int = 86400
    CACHE_GREETING_TTL: int = 604800
    CACHE_MAX_DISTANCE: float = 0.04
    CACHE_MIN_TERM_OVERLAP: float = 0.7

    # ------------------------------------------------------------------ #
    # NeMo Guardrails
    # ------------------------------------------------------------------ #
    GUARDRAILS_ENABLED: bool = True
    GUARDRAILS_PATH: str = "app/guardrails_config"
    GUARDRAIL_LLM_MODEL: str = "openai/gpt-oss-20b"
    GUARDRAILS_TIMEOUT_S: float = 8.0
    GUARDRAILS_FAIL_OPEN: bool = True
    GUARDRAILS_NUMERIC_CHECK: bool = True

    # ------------------------------------------------------------------ #
    # Intent router
    # ------------------------------------------------------------------ #
    INTENT_ROUTER_ENABLED: bool = True
    INTENT_EXAMPLES_PATH: str = "app/data/intent_examples.yml"
    INTENT_MIN_SCORE: float = 0.85
    INTENT_MIN_MARGIN: float = 0.03

    # ------------------------------------------------------------------ #
    # PII detection
    # ------------------------------------------------------------------ #
    PII_ENGINE: str = "presidio"
    PII_SPACY_MODEL: str = "en_core_web_sm"
    PII_SCORE_THRESHOLD: float = 0.4
    PII_INPUT_ENTITIES: list[str] = [
        "PHONE_NUMBER", "EMAIL_ADDRESS", "CREDIT_CARD", "US_SSN", "IBAN_CODE",
        "IP_ADDRESS", "US_BANK_NUMBER", "US_PASSPORT", "US_DRIVER_LICENSE",
        "IN_AADHAAR", "IN_PAN",
    ]
    PII_OUTPUT_ENTITIES: list[str] = [
        "EMAIL_ADDRESS", "CREDIT_CARD", "US_SSN", "IBAN_CODE", "US_BANK_NUMBER",
        "IN_AADHAAR", "IN_PAN",
    ]

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


settings = Settings()
