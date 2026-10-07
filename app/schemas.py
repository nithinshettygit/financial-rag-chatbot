from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any

class ChatTurn(BaseModel):
    role: str = Field(..., description="'user' or 'assistant'")
    content: str

class SessionState(BaseModel):
    company: Optional[str] = None
    fiscal_year: Optional[str] = None
    target_metric: Optional[str] = None
    document_id: Optional[str] = None
    
class ChatRequest(BaseModel):
    session_id: str = Field(..., example="sess_12345")
    query: str = Field(..., min_length=1, max_length=2000)
    history: Optional[List[ChatTurn]] = Field(default=[], description="Past conversation turns")
    state: Optional[SessionState] = Field(default_factory=SessionState)

class ContextChunk(BaseModel):
    chunk_id: str
    content: str
    document_id: str = "doc_default"
    source_file: Optional[str] = "document.pdf"
    page_number: Any = "N/A"
    chunk_type: str = "text"
    headings: List[str] = []
    rrf_score: float = 0.0
    final_score: float = 0.0

class ChatResponse(BaseModel):
    session_id: str
    answer: str
    intent: str
    search_query: str
    state: SessionState
    retrieved_chunks: List[ContextChunk]
    source_map: Dict[str, str] = {}