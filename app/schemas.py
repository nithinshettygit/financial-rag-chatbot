from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field


class ChatTurn(BaseModel):
    model_config = ConfigDict(extra="forbid")
    role: str = Field(..., description="'user' or 'assistant'")
    content: str


class SessionState(BaseModel):
    model_config = ConfigDict(extra="forbid")
    company: Optional[str] = None
    fiscal_year: Optional[str] = None
    target_metric: Optional[str] = None
    document_id: Optional[str] = None


class ChatRequest(BaseModel):
    """Client payload. Session identity is intentionally not client-controlled."""

    model_config = ConfigDict(extra="forbid")

    # Internal field populated by the API from the signed HttpOnly cookie.
    # exclude=True keeps it out of response serialization and the client payload.
    session_id: Optional[str] = Field(default=None, exclude=True)
    query: str = Field(..., min_length=1, max_length=2000)
    history: Optional[List[ChatTurn]] = Field(default=None, description="Past conversation turns")
    state: Optional[SessionState] = Field(default_factory=SessionState)


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
