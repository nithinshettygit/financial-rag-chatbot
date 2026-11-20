"""Shared non-secret application settings used by the API and admin UI."""

from pathlib import Path
from typing import Any

import yaml
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


DEFAULTS_PATH = Path(__file__).resolve().parents[1] / "config" / "defaults.yaml"


def _read_defaults() -> dict[str, Any]:
    with DEFAULTS_PATH.open(encoding="utf-8") as defaults_file:
        defaults = yaml.safe_load(defaults_file)
    if not isinstance(defaults, dict):
        raise ValueError(f"Expected a YAML mapping in {DEFAULTS_PATH}")
    return defaults


_DEFAULTS = _read_defaults()


class GeneralSettings(BaseSettings):
    APP_NAME: str = _DEFAULTS["APP_NAME"]

    LLM_MODEL: str = _DEFAULTS["LLM_MODEL"]
    EMBEDDING_MODEL: str = _DEFAULTS["EMBEDDING_MODEL"]
    VECTOR_NAME: str = _DEFAULTS["VECTOR_NAME"]
    EMBEDDING_DIMENSIONS: int = Field(default=_DEFAULTS["EMBEDDING_DIMENSIONS"], gt=0)
    COLLECTION_NAME: str = Field(default=_DEFAULTS["COLLECTION_NAME"], min_length=1)
    QDRANT_URL: str | None = _DEFAULTS["QDRANT_URL"]
    QDRANT_STORAGE_PATH: str = _DEFAULTS["QDRANT_STORAGE_PATH"]
    DATA_DIR: str = _DEFAULTS["DATA_DIR"]
    TEMP_UPLOAD_DIR: str = _DEFAULTS["TEMP_UPLOAD_DIR"]
    PROMPTS_PATH: str = _DEFAULTS["PROMPTS_PATH"]

    SESSION_TTL_SECONDS: int = Field(default=_DEFAULTS["SESSION_TTL_SECONDS"], gt=0)
    SESSION_COOKIE_SECURE: bool = _DEFAULTS["SESSION_COOKIE_SECURE"]
    SESSION_COOKIE_SAMESITE: str = _DEFAULTS["SESSION_COOKIE_SAMESITE"]

    CHAT_SESSION_RATE_LIMIT_REQUESTS: int = Field(
        default=_DEFAULTS["CHAT_SESSION_RATE_LIMIT_REQUESTS"], gt=0
    )
    CHAT_IP_RATE_LIMIT_REQUESTS: int = Field(
        default=_DEFAULTS["CHAT_IP_RATE_LIMIT_REQUESTS"], gt=0
    )
    CHAT_RATE_LIMIT_WINDOW_SECONDS: int = Field(
        default=_DEFAULTS["CHAT_RATE_LIMIT_WINDOW_SECONDS"], gt=0
    )

    MAX_CONCURRENT_RAG: int = Field(default=_DEFAULTS["MAX_CONCURRENT_RAG"], gt=0)
    MAX_QUEUE_WAITERS: int = Field(default=_DEFAULTS["MAX_QUEUE_WAITERS"], gt=0)
    RAG_QUEUE_MAX_WAIT_SECONDS: int = Field(
        default=_DEFAULTS["RAG_QUEUE_MAX_WAIT_SECONDS"], gt=0
    )
    RAG_QUEUE_POLL_INTERVAL_SECONDS: float = Field(
        default=_DEFAULTS["RAG_QUEUE_POLL_INTERVAL_SECONDS"], gt=0
    )
    RAG_QUEUE_RETRY_AFTER_SECONDS: int = Field(
        default=_DEFAULTS["RAG_QUEUE_RETRY_AFTER_SECONDS"], gt=0
    )
    RAG_CONCURRENCY_LEASE_SECONDS: int = Field(
        default=_DEFAULTS["RAG_CONCURRENCY_LEASE_SECONDS"], gt=0
    )
    REQUEST_TIMEOUT_S: float = Field(default=_DEFAULTS["REQUEST_TIMEOUT_S"], gt=0)

    LLM_MAX_ATTEMPTS: int = Field(default=_DEFAULTS["LLM_MAX_ATTEMPTS"], gt=0)
    LLM_MAX_WAIT_S: float = Field(default=_DEFAULTS["LLM_MAX_WAIT_S"], ge=0)
    LLM_REASONING_EFFORT: str | None = _DEFAULTS["LLM_REASONING_EFFORT"]

    BM25_MANIFEST_PATH: str = _DEFAULTS["BM25_MANIFEST_PATH"]
    MANIFEST_METADATA_SUBDIR: str = Field(
        default=_DEFAULTS["MANIFEST_METADATA_SUBDIR"], min_length=1
    )
    RETRIEVAL_K: int = Field(default=_DEFAULTS["RETRIEVAL_K"], gt=0)
    RERANK_CANDIDATES: int = Field(default=_DEFAULTS["RERANK_CANDIDATES"], gt=0)
    FINAL_TOP_K: int = Field(default=_DEFAULTS["FINAL_TOP_K"], gt=0)
    MAX_FINAL_TOP_K: int = Field(default=_DEFAULTS["MAX_FINAL_TOP_K"], gt=0)
    MIN_STRICT_YIELD: int = Field(default=_DEFAULTS["MIN_STRICT_YIELD"], gt=0)
    RRF_K: int = Field(default=_DEFAULTS["RRF_K"], gt=0)
    MAX_HISTORY_TURNS: int = Field(default=_DEFAULTS["MAX_HISTORY_TURNS"], gt=0)
    MAX_CONTEXT_TOKENS: int = Field(default=_DEFAULTS["MAX_CONTEXT_TOKENS"], gt=0)
    CATALOG_TTL_S: int = Field(default=_DEFAULTS["CATALOG_TTL_S"], gt=0)
    MAX_CONCURRENT_SEARCHES: int = Field(
        default=_DEFAULTS["MAX_CONCURRENT_SEARCHES"], gt=0
    )
    MAX_SUB_QUERIES: int = Field(default=_DEFAULTS["MAX_SUB_QUERIES"], gt=0)

    RERANKER_MODEL: str = _DEFAULTS["RERANKER_MODEL"]
    RERANKER_TIMEOUT_S: int = Field(default=_DEFAULTS["RERANKER_TIMEOUT_S"], gt=0)
    FASTEMBED_CACHE_DIR: str | None = _DEFAULTS["FASTEMBED_CACHE_DIR"]
    FASTEMBED_LOCAL_FILES_ONLY: bool = _DEFAULTS["FASTEMBED_LOCAL_FILES_ONLY"]

    REDIS_URL: str = _DEFAULTS["REDIS_URL"]
    CACHE_TTL: int = Field(default=_DEFAULTS["CACHE_TTL"], ge=0)
    CACHE_GREETING_TTL: int = Field(default=_DEFAULTS["CACHE_GREETING_TTL"], ge=0)
    CACHE_MAX_DISTANCE: float = Field(default=_DEFAULTS["CACHE_MAX_DISTANCE"], ge=0)
    CACHE_MIN_TERM_OVERLAP: float = Field(
        default=_DEFAULTS["CACHE_MIN_TERM_OVERLAP"], ge=0, le=1
    )

    GUARDRAILS_ENABLED: bool = _DEFAULTS["GUARDRAILS_ENABLED"]
    GUARDRAILS_PATH: str = _DEFAULTS["GUARDRAILS_PATH"]
    GUARDRAIL_LLM_MODEL: str = _DEFAULTS["GUARDRAIL_LLM_MODEL"]
    GUARDRAILS_TIMEOUT_S: float = Field(default=_DEFAULTS["GUARDRAILS_TIMEOUT_S"], gt=0)
    GUARDRAILS_FAIL_OPEN: bool = _DEFAULTS["GUARDRAILS_FAIL_OPEN"]
    GUARDRAILS_NUMERIC_CHECK: bool = _DEFAULTS["GUARDRAILS_NUMERIC_CHECK"]
    MAX_INPUT_CHARS: int = Field(default=_DEFAULTS["MAX_INPUT_CHARS"], gt=0)

    INTENT_ROUTER_ENABLED: bool = _DEFAULTS["INTENT_ROUTER_ENABLED"]
    INTENT_EXAMPLES_PATH: str = _DEFAULTS["INTENT_EXAMPLES_PATH"]
    INTENT_MIN_SCORE: float = Field(
        default=_DEFAULTS["INTENT_MIN_SCORE"], ge=0, le=1
    )
    INTENT_MIN_MARGIN: float = Field(
        default=_DEFAULTS["INTENT_MIN_MARGIN"], ge=0, le=1
    )
    MAX_WORDS_SHORT: int = Field(default=_DEFAULTS["MAX_WORDS_SHORT"], gt=0)
    MAX_WORDS_ANY: int = Field(default=_DEFAULTS["MAX_WORDS_ANY"], gt=0)

    PII_ENGINE: str = _DEFAULTS["PII_ENGINE"]
    PII_SPACY_MODEL: str = _DEFAULTS["PII_SPACY_MODEL"]
    PII_SCORE_THRESHOLD: float = Field(
        default=_DEFAULTS["PII_SCORE_THRESHOLD"], ge=0, le=1
    )
    PII_INPUT_ENTITIES: list[str] = _DEFAULTS["PII_INPUT_ENTITIES"]
    PII_OUTPUT_ENTITIES: list[str] = _DEFAULTS["PII_OUTPUT_ENTITIES"]

    MAX_EMBEDDING_TOKENS: int = Field(
        default=_DEFAULTS["MAX_EMBEDDING_TOKENS"], gt=0
    )
    MIN_TEXT_CHARS: int = Field(default=_DEFAULTS["MIN_TEXT_CHARS"], gt=0)
    MIN_TEXT_WORDS: int = Field(default=_DEFAULTS["MIN_TEXT_WORDS"], gt=0)
    OCR_FALLBACK_MISSING_PAGE_RATIO: float = Field(
        default=_DEFAULTS["OCR_FALLBACK_MISSING_PAGE_RATIO"], ge=0, le=1
    )

    model_config = SettingsConfigDict(
        env_file=".env",
        env_ignore_empty=True,
        extra="ignore",
        validate_default=True,
    )


class AdminSettings(GeneralSettings):
    ADMIN_USERNAME: str | None = None
    ADMIN_PASSWORD: str | None = None
    QDRANT_API_KEY: str | None = None
    LOGFIRE_TOKEN: str | None = None


general_settings = GeneralSettings()
admin_settings = AdminSettings()
