from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    question: str = Field(min_length=1, max_length=8000)
    collection: str = Field(default="docs", min_length=1, max_length=64)
    user_id: str = Field(default="demo", min_length=1, max_length=80)

    top_k: int | None = Field(default=None, ge=1, le=50)
    bm25_k: int | None = Field(default=None, ge=1, le=200)

    rerank: bool = True
    return_context: bool = False
    # Optional source routing: restrict retrieval to chunks whose ingested `source` path contains this substring.
    source_hint: str | None = Field(default=None, max_length=512)
    # If True (and source_hint is not set), pick the best-matching document filename from the index.
    auto_source: bool = False


class SourceChunk(BaseModel):
    doc_id: str
    chunk_id: str
    source: str
    score: float
    text: str | None = None


class ClaimEvidence(BaseModel):
    claim: str
    status: str  # "verified" | "uncertain" | "hallucinated"
    source: SourceChunk | None = None
    rationale: str | None = None


class SkippedFile(BaseModel):
    filename: str
    reason: str


class IngestResponse(BaseModel):
    collection: str
    chunk_count: int = Field(ge=0)
    sources: list[str] = Field(default_factory=list)
    skipped: list[SkippedFile] = Field(default_factory=list)


class QueryResponse(BaseModel):
    question: str
    collection: str
    user_id: str
    answer: str

    query_type: str
    confidence: float = Field(ge=0.0, le=1.0)

    sources: list[SourceChunk] = Field(default_factory=list)
    claims: list[ClaimEvidence] = Field(default_factory=list)

    used_llm: bool
    retries: dict[str, int] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)
    request_id: str | None = None
    router: dict[str, Any] | None = None

