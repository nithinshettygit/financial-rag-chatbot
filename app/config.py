# app/config.py
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    APP_NAME: str = "Docling Financial Hybrid RAG Service"
    GROQ_API_KEY: str
    
    # Models & Collection
    LLM_MODEL: str = "openai/gpt-oss-20b"
    EMBEDDING_MODEL: str = "BAAI/bge-small-en-v1.5"
    VECTOR_NAME: str = "fastembed-BAAI/bge-small-en-v1.5"
    COLLECTION_NAME: str = "financial_rag_analytics"
    
    # Qdrant Paths
    QDRANT_URL: str | None = None
    QDRANT_API_KEY: str | None = None
    QDRANT_STORAGE_PATH: str = "data/qdrant_storage"
    
    # BM25 Path
    BM25_MANIFEST_PATH: str = "data/chunks_manifest.json"
    
    # Hyperparameters
    RETRIEVAL_K: int = 12
    FINAL_TOP_K: int = 4
    RRF_K: int = 60
    MAX_HISTORY_TURNS: int = 6
    MAX_CONTEXT_TOKENS: int = 1800

    class Config:
        env_file = ".env"

settings = Settings()