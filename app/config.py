from pydantic import Field

from shared.settings import GeneralSettings


class Settings(GeneralSettings):
    GROQ_API_KEY: str
    FALLBACK_GROQ_API_KEY: str | None = None
    VOYAGE_API_KEY: str | None = None
    SESSION_HMAC_SECRET: str = Field(min_length=32)
    LOGFIRE_TOKEN: str | None = None
    QDRANT_API_KEY: str | None = None


settings = Settings()
