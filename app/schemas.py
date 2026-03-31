from __future__ import annotations

from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    question: str = Field(min_length=1, max_length=8000)
    collection: str = Field(default="docs", min_length=1, max_length=64)

    top_k: int | None = Field(default=None, ge=1, le=50)
    bm25_k: int | None = Field(default=None, ge=1, le=200)

    rerank: bool = True
    return_context: bool = False


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


class QueryResponse(BaseModel):
    question: str
    collection: str
    answer: str

    query_type: str
    confidence: float = Field(ge=0.0, le=1.0)

    sources: list[SourceChunk] = Field(default_factory=list)
    claims: list[ClaimEvidence] = Field(default_factory=list)

    used_llm: bool
    retries: dict[str, int] = Field(default_factory=dict)

