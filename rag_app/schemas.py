from __future__ import annotations

from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, Field


class IndexRequest(BaseModel):
    force: bool = False


class IndexResponse(BaseModel):
    collection: str
    chunks: int
    indexed: bool
    message: str


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1)
    session_id: UUID | None = None


class Source(BaseModel):
    chunk_id: int
    source_ref: str
    doc_title: str
    h1: str = ""
    h2: str = ""
    source_file: str = ""
    snippet: str
    doc_url: str | None = None
    dense_score: float | None = None
    sparse_score: float | None = None
    fused_score: float
    rerank_score: float | None = None


class ResponseStats(BaseModel):
    latency_ms: int
    input_tokens: int = 0
    output_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


class ChatResponse(BaseModel):
    answer: str
    sources: list[Source]
    cited_source_indices: list[int] = Field(default_factory=list)
    session_id: UUID
    call_id: UUID
    model: str
    latency_ms: int
    stats: ResponseStats


class HealthResponse(BaseModel):
    status: str
    chunks_loaded: int
    collection: str
    model: str


class SessionSummary(BaseModel):
    id: UUID
    title: str
    created_at: datetime
    updated_at: datetime


class StoredCall(BaseModel):
    id: UUID
    session_id: UUID
    user_message: str
    answer: str
    sources: list[dict]
    cited_source_indices: list[int] = Field(default_factory=list)
    model: str
    latency_ms: int
    stats: ResponseStats | None = None
    created_at: datetime
