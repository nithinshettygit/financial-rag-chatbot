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
    
    # New: FlashRank & Logfire Settings
    RERANKER_MODEL: str = "ms-marco-MiniLM-L-12-v2"
    LOGFIRE_TOKEN: str | None = None
    
    QDRANT_URL: str | None = None
    QDRANT_API_KEY: str | None = None
    QDRANT_STORAGE_PATH: str = "data/qdrant_storage"
    BM25_MANIFEST_PATH: str = "data/manifests"
    
    RETRIEVAL_K: int = 30
    FINAL_TOP_K: int = 6
    RRF_K: int = 60
    MAX_HISTORY_TURNS: int = 6
    MAX_CONTEXT_TOKENS: int = 2048

    class Config:
        env_file = ".env"

settings = Settings()