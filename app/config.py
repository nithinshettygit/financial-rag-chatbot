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
    GUARDRAILS_PATH: str = "app/guardrails_config"
    # Small NON-reasoning model for yes/no policy checks (reasoning models burn the tiny
    # max_tokens budget NeMo gives self-check tasks). Confirm it is available on your Groq account.
    GUARDRAIL_LLM_MODEL: str = "openai/gpt-oss-20b"
    GUARDRAILS_TIMEOUT_S: float = 8.0
    GUARDRAILS_FAIL_OPEN: bool = True      # if NeMo itself errors, fall back to deterministic rules only
    GUARDRAILS_NUMERIC_CHECK: bool = True  # block answers whose figures are not found in the retrieved context

    # Embedding-based intent router (fast paths for small talk / follow-ups / summary / out-of-scope)
    INTENT_ROUTER_ENABLED: bool = True
    INTENT_EXAMPLES_PATH: str = "app/data/intent_examples.yml"
    INTENT_MIN_SCORE: float = 0.85     # cosine similarity to the closest example; TUNE with eval/run_eval.py
    INTENT_MIN_MARGIN: float = 0.03    # lead over the runner-up class

    # PII detection (Presidio; falls back to a small regex masker if Presidio / the spaCy model is missing)
    PII_ENGINE: str = "presidio"           # "presidio" | "regex"
    PII_SPACY_MODEL: str = "en_core_web_sm"
    PII_SCORE_THRESHOLD: float = 0.4
    # What to mask in user input vs. model output. Output deliberately skips PHONE_NUMBER because
    # filings legitimately contain investor-relations phone numbers.
    PII_INPUT_ENTITIES: list[str] = ["PHONE_NUMBER", "EMAIL_ADDRESS", "CREDIT_CARD", "US_SSN", "IBAN_CODE",
                                     "IP_ADDRESS", "US_BANK_NUMBER", "US_PASSPORT", "US_DRIVER_LICENSE",
                                     "IN_AADHAAR", "IN_PAN"]
    PII_OUTPUT_ENTITIES: list[str] = ["EMAIL_ADDRESS", "CREDIT_CARD", "US_SSN", "IBAN_CODE", "US_BANK_NUMBER",
                                      "IN_AADHAAR", "IN_PAN"]

    class Config:
        env_file = ".env"


settings = Settings()
