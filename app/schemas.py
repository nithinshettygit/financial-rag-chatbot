from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator


class ChatTurn(BaseModel):
    model_config = ConfigDict(extra="forbid")
    role: Literal["user", "assistant"]
    content: str = Field(..., min_length=1, max_length=2000)

    @field_validator("content")
    @classmethod
    def strip_content(cls, content: str) -> str:
        content = content.strip()
        if not content:
            raise ValueError("Conversation content must not be blank")
        return content


class SessionState(BaseModel):
    model_config = ConfigDict(extra="forbid")
    company: Optional[str] = Field(default=None, max_length=200)
    fiscal_year: Optional[str] = Field(default=None, max_length=20)
    target_metric: Optional[str] = Field(default=None, max_length=200)
    document_id: Optional[str] = Field(default=None, max_length=200)


class ChatRequest(BaseModel):
    """Client payload. Session identity is intentionally not client-controlled."""

    model_config = ConfigDict(extra="forbid")

    # Internal field populated by the API from the signed HttpOnly cookie.
    # exclude=True keeps it out of response serialization and the client payload.
    session_id: Optional[str] = Field(default=None, exclude=True)
    query: str = Field(..., min_length=1, max_length=2000)
    history: Optional[List[ChatTurn]] = Field(
        default=None,
        max_length=12,
        description="At most 12 recent conversation messages",
    )
    state: Optional[SessionState] = Field(default_factory=SessionState)

    @field_validator("query")
    @classmethod
    def strip_query(cls, query: str) -> str:
        query = query.strip()
        if not query:
            raise ValueError("Query must not be blank")
        return query


class ContextChunk(BaseModel):
    model_config = ConfigDict(extra="forbid")
    chunk_id: str
    content: str
    document_id: str = "doc_default"
    source_file: Optional[str] = "document.pdf"
    page_number: Any = "N/A"
    chunk_type: str = "text"
    headings: List[str] = Field(default_factory=list)
    rrf_score: float = 0.0
    final_score: float = 0.0


class ChatResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")
    session_id: str
    answer: str
    intent: str
    search_query: str
    state: SessionState
    retrieved_chunks: List[ContextChunk]
    source_map: Dict[str, str] = Field(default_factory=dict)
